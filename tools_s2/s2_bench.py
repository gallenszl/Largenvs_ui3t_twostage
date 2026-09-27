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


def run_leg(model, opt, batches, warmup, steps, label):
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
        dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt)
        if (i + 1) % 25 == 0:                         # progress for the 5-minute watchdog
            print(f"[s2-bench] {label} step {i + 1}/{warmup + steps} last {dt:.2f}s "
                  f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f}GB", flush=True)
    t = np.array(times)
    return dict(median_s=float(np.median(t)), p90_s=float(np.percentile(t, 90)), mean_s=float(t.mean()),
                peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30, steps=int(len(t)))


def breakdown(model, opt, batches, n=10):
    """GPU time per step of each part (CUDA events; parts called several times per step are summed)."""
    import model_s2.geometry as geo
    import model_s2.stage2_wrapper as w
    acc = {}

    def wrap(obj, name, label):
        fn = getattr(obj, name)

        def timed(*a, **k):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            out = fn(*a, **k)
            e.record()
            acc.setdefault(label, []).append((s, e))
            return out
        setattr(obj, name, timed)
        return obj, name, fn

    patches = [wrap(model.stage1, "pass1", "stage-1 pass 1 (no grad)"),
               wrap(model.stage1, "pass2", "stage-1 pass 2 (no grad)"),
               wrap(geo, "build_layout", "layout + mask tables"),
               wrap(geo, "build_forward_table", "layout + mask tables"),
               wrap(geo, "build_reverse_table", "layout + mask tables"),
               wrap(w, "MaskTable", "layout + mask tables"),
               wrap(model.renderer, "forward", "stage-2 renderer forward"),
               wrap(w, "render_color", "heads forward"),
               wrap(w, "render_points", "heads forward"),
               wrap(model.loss_computer, "forward", "loss forward")]
    whole = {"forward (all)": [], "backward": [], "optimizer": []}
    try:
        model.train()
        for i in range(n):
            b = batches[i % len(batches)]
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            ev[0].record()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(b)
            ev[1].record()
            out.loss_metrics.loss.backward()
            ev[2].record()
            opt.step()
            opt.zero_grad(set_to_none=True)
            ev[3].record()
            torch.cuda.synchronize()
            whole["forward (all)"].append(ev[0].elapsed_time(ev[1]))
            whole["backward"].append(ev[1].elapsed_time(ev[2]))
            whole["optimizer"].append(ev[2].elapsed_time(ev[3]))
    finally:
        import types
        for obj, name, fn in patches:
            if isinstance(obj, types.ModuleType):
                setattr(obj, name, fn)                  # module-level function / class: put the original back
            else:
                vars(obj).pop(name, None)               # instance attribute shadowing the method: remove it
    res = {k: float(np.mean(v)) for k, v in whole.items()}
    for k, v in acc.items():
        res[k] = float(sum(s.elapsed_time(e) for s, e in v) / n)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True)
    ap.add_argument("--backends", default="sdpa,flex")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n_batches", type=int, default=12)
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
            r = run_leg(model, opt, batches, args.warmup, args.steps, f"{tag} {bk}")
            r["wall_s"] = time.time() - t0
            r["breakdown_ms"] = breakdown(model, opt, batches)
            print(f"[s2-bench] {tag} {bk} breakdown (ms/step): "
                  + ", ".join(f"{k} {v:.0f}" for k, v in r["breakdown_ms"].items()), flush=True)
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
