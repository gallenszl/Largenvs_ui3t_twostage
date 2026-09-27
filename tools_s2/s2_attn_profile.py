# Stage-2 attention profile: on real training batches (the training data pipeline, batch 8, the training-mode
# posed / unposed draw), take the per-step context the wrapper builds (layout + the two boolean mask tables) and
#   1. mask density: share of allowed (query, key) pairs among the real ones, and share of 128 x 128 tiles that
#      contain at least one allowed pair (= the tiles FlexAttention computes; SDPA with an additive bias computes
#      every tile), plus what the tile share would be without the always-allowed register rows / summary columns;
#   2. time of every attention type of the stage-2 renderer on those exact shapes and masks (random bf16 q / k / v,
#      12 heads x 64): forward and forward+backward, SDPA-with-bias vs flex vs dense attention without a mask, and
#      the dense FA3 block-diagonal parts.  Per training step every block runs its forward twice (forward and the
#      checkpoint recompute) and its backward once, so per-step time of a call = forward + (forward + backward).
#   python tools_s2/s2_attn_profile.py --config configs/S2P8_...yaml [--batch 8] [--n_batches 3] --out x.json
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
H, D, TILE = 12, 64, 128


class _Captured(Exception):
    pass


def capture_ctx(model, batch):
    store = {}

    def fwd(rays6, mask1, rec_rep, ctx):
        store["ctx"] = ctx
        raise _Captured()
    model.renderer.forward = fwd
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(batch)
    except _Captured:
        pass
    finally:
        del model.renderer.forward
    return store["ctx"]


def tiles(T):
    B, L1, L2 = T.shape
    t = T.view(B, L1 // TILE, TILE, L2 // TILE, TILE).any(4).any(2)
    return float(t.float().mean()), int(t.sum()), int(t.numel())


def densities(ctx, S):
    lay = ctx.layout
    Ft, Rt = ctx.fwd_mask.table, ctx.rev_mask.table
    BV, Lq, s_pad = Ft.shape
    Kr = Rt.shape[2]
    dev = Ft.device
    n = torch.tensor(lay.n, device=dev)
    r = torch.arange(Lq, device=dev)[None]
    fg_row = (r >= 4) & (r < 4 + n[:, None])                                 # [BV, Lq]
    per_tok = Ft[:, :, :S].sum(-1)[fg_row].float()                            # allowed scene tokens per fg token
    out = dict(BV=BV, Lq=Lq, s_pad=s_pad, Kr=Kr, S=S, n_fg=int(n.sum()), max_n=int(n.max()),
               fg_share_of_grid=float(n.float().mean()) / (lay.g * lay.g))
    out["fwd_pair_density_fg"] = float(per_tok.sum()) / (float(fg_row.sum()) * S)
    out["fwd_keys_per_fg_token_mean"] = float(per_tok.mean())
    out["fwd_keys_per_fg_token_p90"] = float(per_tok.quantile(0.9))
    out["fwd_tile_share"], out["fwd_tiles_on"], out["fwd_tiles_all"] = tiles(Ft)
    Fn = Ft.clone()
    Fn[:, :4, :] = False                                                     # what if registers went elsewhere
    out["fwd_tile_share_without_register_rows"] = tiles(Fn)[0]
    c = torch.arange(Kr, device=dev)[None]
    fg_col = (c >= 128) & (c < 128 + n[:, None])                              # [BV, Kr]
    Rf = Rt[:, :S, :] & fg_col[:, None, :]
    vis = Rf.any(-1)                                                          # scene token sees any fg token
    out["rev_visible_scene_share"] = float(vis.float().mean())
    per_scene = Rf.sum(-1)[vis].float()
    out["rev_fg_keys_per_visible_scene_token_mean"] = float(per_scene.mean()) if per_scene.numel() else 0.0
    out["rev_pair_density_fg_cols"] = float(Rf.sum()) / (S * float(fg_col.sum()))
    out["rev_tile_share"], out["rev_tiles_on"], out["rev_tiles_all"] = tiles(Rt)
    Rn = Rt.clone()
    Rn[:, :, :128] = False                                                   # what if summaries / registers went elsewhere
    out["rev_tile_share_without_first_128_cols"] = tiles(Rn)[0]
    return out


def timeit(fn, reps=15, warm=4):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return float(np.median(ts))


def f_and_fb(call, shapes_q, shapes_kv, dev):
    """forward ms and forward+backward ms of call(q, k, v) on random bf16 tensors of the given shapes."""
    q = torch.randn(*shapes_q, H, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(*shapes_kv, H, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(*shapes_kv, H, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    with torch.no_grad():
        go = torch.randn_like(call(q, k, v))

    def fwd():
        with torch.no_grad():
            call(q, k, v)

    def fb():
        q.grad = k.grad = v.grad = None
        call(q, k, v).backward(go)
    return timeit(fwd), timeit(fb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n_batches", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(29200 + int(os.environ.get("SLURM_JOB_ID", "5")) % 400))
        dist.init_process_group("gloo", rank=0, world_size=1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.chdir(REPO)
    from model_s2.masked_attention import MaskTable, attend_blockdiag, attend_masked
    from model_s2.stage2_wrapper import Stage2LagerNVS
    from tools_s2.s2_bench import cached_batches, load_cfg
    cfg = load_cfg(args.config, args.batch)
    model = Stage2LagerNVS(cfg).cuda().train()
    model.masked_backend = "flex"                         # keeps the boolean tables in the captured context
    batches = cached_batches(cfg, args.n_batches)
    dev = torch.device("cuda")
    res = []
    for bi, b in enumerate(batches):
        torch.manual_seed(bi)
        ctx = capture_ctx(model, b)
        S = ctx.n_views * ctx.scene_g * ctx.scene_g
        d = densities(ctx, S)
        lay = ctx.layout
        BV, Lq, s_pad, Kr = d["BV"], d["Lq"], d["s_pad"], d["Kr"]
        T = int(sum(ctx.q_seqlens))
        nblk = ctx.n_views * ((ctx.scene_g + ctx.scene_block - 1) // ctx.scene_block) ** 2
        tf, tr = ctx.fwd_mask.table, ctx.rev_mask.table
        m = {"fwd": (MaskTable(tf, "sdpa"), MaskTable(tf, "flex"), (BV, Lq), (BV, s_pad)),
             "rev": (MaskTable(tr, "sdpa"), MaskTable(tr, "flex"), (BV, s_pad), (BV, Kr))}
        t = {}
        for name, (ms, mf, sq, skv) in m.items():
            t[f"{name}_sdpa"] = f_and_fb(lambda q, k, v, ms=ms: attend_masked(q, k, v, ms, "sdpa"), sq, skv, dev)
            t[f"{name}_flex"] = f_and_fb(lambda q, k, v, mf=mf: attend_masked(q, k, v, mf, "flex"), sq, skv, dev)
            t[f"{name}_dense_nomask"] = f_and_fb(
                lambda q, k, v: F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                                               v.transpose(1, 2)).transpose(1, 2), sq, skv, dev)
            del ms, mf
        t["self_fa3"] = f_and_fb(lambda q, k, v: attend_blockdiag(q, k, v, ctx.q_seqlens, ctx.q_seqlens, "fa3"),
                                 (T,), (T,), dev)
        t["compressed_fa3"] = f_and_fb(
            lambda q, k, v: attend_blockdiag(q, k, v, ctx.q_seqlens, [nblk] * BV, "fa3"), (T,), (BV * nblk,), dev)
        # per training step: 12 blocks x (self + compressed + selected), 11 blocks x reverse; forward twice + backward
        n_calls = {"self_fa3": 12, "compressed_fa3": 12, "fwd": 12, "rev": 11}
        step = {}
        for be in ("sdpa", "flex", "dense_nomask"):
            tot = 0.0
            for key, cnt in n_calls.items():
                kk = key if key.endswith("fa3") else f"{key}_{be}"
                f, fb = t[kk]
                tot += cnt * (f + fb)
            step[be] = tot
        d.update(T=T, times_ms={k: {"fwd": v[0], "fwd_bwd": v[1]} for k, v in t.items()}, per_step_attention_ms=step)
        res.append(d)
        print(f"[attn] batch {bi}: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                                    for k, v in d.items() if k not in ("times_ms",)}), flush=True)
        for k, v in t.items():
            print(f"[attn]   {k:22s} fwd {v[0]:8.2f} ms   fwd+bwd {v[1]:8.2f} ms", flush=True)
        del ctx, m
        torch.cuda.empty_cache()
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"[attn] DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
