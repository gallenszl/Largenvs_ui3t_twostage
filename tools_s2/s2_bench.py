# Stage-2 speed benchmark (plan G8): full training steps (forward + backward + AdamW) on cached real training
# batches, same process / same GPU for every leg: {P8, P4} x {sdpa, flex}.  Drops the warmup steps, reports
# median / p90 step time and peak memory.  Decision rule (plan): use flex only if its whole step is >= 10 %
# faster than sdpa and the GPU equivalence tests of flex passed.
#   python tools_s2/s2_bench.py --configs configs/S2P8_...yaml,configs/S2P4_...yaml --backends sdpa,flex \
#       --warmup 100 --steps 200 --out ~/tmp/s2_bench.json

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def load_cfg(path, batch):
    from easydict import EasyDict as edict
    from omegaconf import OmegaConf
    c = edict(OmegaConf.to_container(OmegaConf.load(path), resolve=True))
    c.training.batch_size_per_gpu = batch
    return c


def cached_batches(cfg, n):
    import importlib
    from torch.utils.data import DataLoader
    mod, cls = cfg.training.dataset_name.rsplit(".", 1)
    ds = importlib.import_module(mod).__dict__[cls](cfg)
    dl = DataLoader(ds, batch_size=cfg.training.batch_size_per_gpu, shuffle=True, num_workers=8, drop_last=True)
    out = []
    for b in dl:
        out.append({k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()})
        if len(out) == n:
            return out
    return out


def run_leg(model, opt, batches, warmup, steps):
    model.train()
    times = []
    torch.cuda.reset_peak_memory_stats()
    for i in range(warmup + steps):
        b = batches[i % len(batches)]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(b)
        out.loss_metrics.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    t = np.array(times)
    return dict(median_s=float(np.median(t)), p90_s=float(np.percentile(t, 90)), mean_s=float(t.mean()),
                peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30, steps=int(len(t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True)
    ap.add_argument("--backends", default="sdpa,flex")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n_batches", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(29400 + int(os.environ.get("SLURM_JOB_ID", "1")) % 400))
        dist.init_process_group("gloo", rank=0, world_size=1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.chdir(REPO)
    from model_s2.stage2_wrapper import Stage2LagerNVS
    from utils.training_utils import create_optimizer
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for cpath in args.configs.split(","):
        cfg = load_cfg(cpath, args.batch)
        tag = os.path.basename(cpath).split("_")[0]
        todo = [bk for bk in args.backends.split(",") if f"{tag}_{bk}" not in res]
        if not todo:
            continue
        batches = cached_batches(cfg, args.n_batches)
        model = Stage2LagerNVS(cfg).cuda()
        opt, _, _ = create_optimizer(model, cfg.training.weight_decay, cfg.training.lr,
                                     (cfg.training.beta1, cfg.training.beta2), fused=True)
        for bk in todo:
            model.masked_backend = bk
            t0 = time.time()
            r = run_leg(model, opt, batches, args.warmup, args.steps)
            r["wall_s"] = time.time() - t0
            res[f"{tag}_{bk}"] = r
            print(f"[s2-bench] {tag} {bk}: median {r['median_s']:.3f}s p90 {r['p90_s']:.3f}s "
                  f"peak {r['peak_gb']:.1f}GB", flush=True)
            json.dump(res, open(args.out + ".tmp", "w"), indent=1)
            os.replace(args.out + ".tmp", args.out)
        del model, opt
        torch.cuda.empty_cache()
    print("[s2-bench] DONE", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
