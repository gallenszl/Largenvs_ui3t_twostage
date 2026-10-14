# Why is a fg_dual training step ~2x slower than the current structure (09-28, G8 job 6909972 / training 6910008)?
#   A. one profiled training step (batch 8, real data, after warm-up): the CUDA ops with the largest total time
#   B. the suspect alone, at the training shapes: padding packed rows into the fixed-width window layout by
#      gathering from one shared zero row (x_z[src], what model_s2/blocks.fg_window does) vs writing them into a
#      zeros tensor at their destination rows (index_copy); forward + backward, same values.  In the gather form every
#      pad position's gradient is accumulated into the same zero row.
#   python tools_s2/s2_fgdual_profile.py --config configs/S2P8fgdual_...yaml --out x.json
import argparse
import json
import os
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def timed(fn, reps=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(28600 + int(os.environ.get("SLURM_JOB_ID", "3")) % 400))
        dist.init_process_group("gloo", rank=0, world_size=1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.chdir(REPO)
    from torch.profiler import ProfilerActivity, profile
    from model_s2.stage2_wrapper import Stage2LagerNVS
    from tools_s2.s2_bench import cached_batches, load_cfg
    from utils.training_utils import create_optimizer
    cfg = load_cfg(args.config, 8)
    batches = cached_batches(cfg, 3)
    model = Stage2LagerNVS(cfg).cuda()
    opt, _, _ = create_optimizer(model, cfg.training.weight_decay, cfg.training.lr,
                                 (cfg.training.beta1, cfg.training.beta2), fused=True)
    model.train()
    shapes = {}

    def step(b):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(b)
        out.loss_metrics.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return out

    import model_s2.blocks as Bk
    orig = Bk.fg_window

    def spy(q, k, v, win, key, ctx, direction):
        BV, Lq, Lk = win["BV"], win[key + "_Lq"], win[key + "_Lk"]
        nk = k.shape[0] if win[key + "_kv_src"] is None else win[key + "_kv_src"].numel()
        shapes.setdefault(direction, dict(q=list(q.shape), k=[nk] + list(k.shape[1:]), qsrc=[BV, Lq], ksrc=[BV, Lk],
                                          q_pad_rows=BV * Lq - q.shape[0], k_pad_rows=BV * Lk - nk))
        return orig(q, k, v, win, key, ctx, direction)
    Bk.fg_window = spy
    for i in range(6):
        step(batches[i % len(batches)])
    Bk.fg_window = orig
    torch.cuda.synchronize()
    print(f"[prof] window shapes: {json.dumps(shapes)}", flush=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        step(batches[0])
        torch.cuda.synchronize()
    rows = []
    for e in prof.key_averages():
        ct = getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)
        rows.append((ct / 1000.0, e.count, e.key))
    rows.sort(reverse=True)
    print("[prof] A. top CUDA time per op (ms, calls), one fg_dual training step:", flush=True)
    for ms, n, k in rows[:25]:
        print(f"[prof]   {ms:9.1f} ms  {n:6d}  {k[:110]}", flush=True)

    # B. the padding alone, at the recorded training shapes
    res = dict(window_shapes=shapes, top_ops=[dict(ms=r[0], calls=r[1], op=r[2]) for r in rows[:40]], micro={})
    for direction, sh in shapes.items():
        for side in ("k", "q"):
            n_src = sh[side][0]
            src_shape = sh["ksrc"] if side == "k" else sh["qsrc"]
            BV, W = src_shape
            H, D = sh[side][1], sh[side][2]
            n_pad = sh["k_pad_rows"] if side == "k" else sh["q_pad_rows"]
            # a padded layout with the same shapes and pad count: the first (W*BV - n_pad) slots real, rest pad
            flat = torch.full((BV * W,), n_src, dtype=torch.long, device="cuda")
            real = BV * W - n_pad
            real = min(real, n_src)
            pos = torch.randperm(BV * W, device="cuda")[:real]
            flat[pos] = torch.arange(real, device="cuda")
            src = flat.view(BV, W)
            x = torch.randn(n_src, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            g = torch.randn(BV, W, H, D, device="cuda", dtype=torch.bfloat16)

            def gather_form():
                xz = torch.cat([x, x.new_zeros(1, H, D)])
                y = xz[src]
                y.backward(g)

            dst = pos
            keep = flat[pos]

            def scatter_form():
                y = x.new_zeros(BV * W, H, D).index_copy(0, dst, x[keep]).view(BV, W, H, D)
                y.backward(g)
            a, b = timed(gather_form), timed(scatter_form)
            res["micro"][f"{direction}_{side}"] = dict(BV=BV, W=W, pad_rows=n_pad, gather_ms=a, scatter_ms=b)
            print(f"[prof] B. {direction} {side}: padded [{BV}, {W}], pad rows {n_pad} -> gather from a shared zero row "
                  f"{a:7.2f} ms vs index_copy {b:7.2f} ms (forward + backward, one call)", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"[prof] DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
