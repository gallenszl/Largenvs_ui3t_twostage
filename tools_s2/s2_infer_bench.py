# Stage-2 inference speed: one scene at a time (batch 1, 4 input views -> 10 target views, the evaluation set-up),
# eval mode, no grad, bf16 autocast; time from the input batch to the target RGB and point maps (the loss is
# replaced by a constant, no metrics).  Legs, all in this process on this GPU:
#   stage 1 only        the wrapper with stage 2 off (prepare + stage-1 forward = LagerNVSInRnG's computation)
#   S2P8 / S2P4 x sdpa / flex   stage 1 + stage 2
# Scenes are cycled; the first passes over all scenes are warmup (FlexAttention compiles one kernel per padding
# size).  Reports mean / median / p10 / p90 latency (host wall clock after cuda synchronize), peak memory, and the
# GPU time of each part on one extra pass.
#   python tools_s2/s2_infer_bench.py --configs configs/S2P8_...yaml,configs/S2P4_...yaml --out x.json
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def val_scenes(cfg, n):
    from data.dataset_gso_ours import GSODataset_ours
    ds = GSODataset_ours(cfg)
    out = []
    for i in range(n):
        s = ds[i]
        out.append({k: (v[None].cuda() if torch.is_tensor(v) else [v]) for k, v in s.items()})
    return out


def infer(model, b):
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return model(b, target_has_input=False, is_valid=True)


def parts(model, scenes):
    """GPU time (ms) per part on one pass over the scenes (CUDA events around each part, summed per scene)."""
    import model_s2.geometry as geo
    import model_s2.stage2_wrapper as w
    import types
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
    patches = [wrap(model.stage1, "pass1", "stage-1 forward (pass 1)"),
               wrap(model.stage1, "pass2", "stage-1 input-view pass (pass 2)"),
               wrap(geo, "build_layout", "layout + mask tables"),
               wrap(geo, "build_forward_table", "layout + mask tables"),
               wrap(geo, "build_reverse_table", "layout + mask tables"),
               wrap(w, "MaskTable", "layout + mask tables"),
               wrap(model.renderer, "forward", "stage-2 renderer"),
               wrap(w, "render_color", "heads"),
               wrap(w, "render_points", "heads")]
    try:
        for b in scenes:
            infer(model, b)
        torch.cuda.synchronize()
    finally:
        for obj, name, fn in patches:
            if isinstance(obj, types.ModuleType):
                setattr(obj, name, fn)
            else:
                vars(obj).pop(name, None)
    return {k: float(sum(s.elapsed_time(e) for s, e in v) / len(scenes)) for k, v in acc.items()}


def leg(model, scenes, warm_passes, passes, label):
    for _ in range(warm_passes):
        for b in scenes:
            infer(model, b)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(passes):
        for b in scenes:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            infer(model, b)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
    t = np.array(ts)
    r = dict(mean_ms=float(t.mean()), median_ms=float(np.median(t)), p10_ms=float(np.percentile(t, 10)),
             p90_ms=float(np.percentile(t, 90)), n=int(len(t)), peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
    r["parts_ms"] = parts(model, scenes)
    print(f"[infer] {label}: median {r['median_ms']:.1f} ms mean {r['mean_ms']:.1f} p10 {r['p10_ms']:.1f} "
          f"p90 {r['p90_ms']:.1f} peak {r['peak_gb']:.1f} GB | parts " +
          ", ".join(f"{k} {v:.1f}" for k, v in r["parts_ms"].items()), flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True)
    ap.add_argument("--n_scenes", type=int, default=12)
    ap.add_argument("--warm_passes", type=int, default=2)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(29100 + int(os.environ.get("SLURM_JOB_ID", "9")) % 400))
        dist.init_process_group("gloo", rank=0, world_size=1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.chdir(REPO)
    from easydict import EasyDict as edict
    from omegaconf import OmegaConf
    from model_s2.stage2_wrapper import Stage2LagerNVS
    res = {}
    for ci, cpath in enumerate(args.configs.split(",")):
        cfg = edict(OmegaConf.to_container(OmegaConf.load(cpath), resolve=True))
        tag = os.path.basename(cpath).split("_")[0]
        model = Stage2LagerNVS(cfg).cuda().eval()
        model.val_cam_cond_zero_p = 0.0
        zero = torch.zeros((), device="cuda")
        model.loss_computer.forward = lambda *a, **k: edict(loss=zero)
        scenes = val_scenes(cfg, args.n_scenes)
        if ci == 0:
            model.enabled = False
            res["stage1_only"] = leg(model, scenes, args.warm_passes, args.passes, "stage 1 only")
            model.enabled = True
        for bk in ("sdpa", "flex"):
            model.masked_backend = bk
            res[f"{tag}_{bk}"] = leg(model, scenes, args.warm_passes, args.passes, f"{tag} {bk} (stage 1 + 2)")
            json.dump(res, open(args.out, "w"), indent=1)
        del model, scenes
        torch.cuda.empty_cache()
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"[infer] DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
