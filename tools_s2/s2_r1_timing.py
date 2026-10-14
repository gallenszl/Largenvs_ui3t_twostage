# r0 / r1 inference time of fg_dense with the refined tiles from the GT target mask (S2P8fgdense_...) against the
# refined tiles from the stage-1 render (S2P8fgdense_s1sel1_..., user 09-28: "开训前先跑一个约 15 分钟的测速").
# One process, one GPU, batch 1, 1 and 10 target views, the first --n_scenes GSO v1 validation entries, the real code
# path: Stage2LagerNVS.make_ctx builds the selection and the layout exactly as training and inference do.  Weights at
# init: every stage-2 branch output is zero-initialised, and the kernel times depend on the token counts (from the GT
# mask or from the frozen stage-1 render), not on the weight values.  Per config and target-view count:
#   tiles   refined target tiles per view (r1's query tokens; r0 always renders all 1024 tiles of a view)
#   ctx     Stage2LagerNVS.make_ctx = selection + layout + scene-token subset (eager; one host sync, the tile counts)
#   r0      the stage-1 renderer on the target views
#   r1      the stage-2 renderer
# Modes: eager with bf16 weights and the three host syncs of tools_s2/s2_stage_timing.py removed (same values), and
# CUDA Graph replay with bf16 weights = pure GPU time (s2_stage_timing's G_bf16).  Median of --reps after --warm
# calls, then the mean over scenes.
#   python tools_s2/s2_r1_timing.py --configs a.yaml,b.yaml [--n_scenes 4] --out x.json
import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from tools_s2.s2_graph_bench import capture, patch_aggregator_zeros, patch_rope_max_position, to_bf16  # noqa: E402
from tools_s2.s2_stage_timing import timed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True)
    ap.add_argument("--n_scenes", type=int, default=4)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--vts", default="1,10")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(28500 + int(os.environ.get("SLURM_JOB_ID", "3")) % 400))
        dist.init_process_group("gloo", rank=0, world_size=1)
    torch.backends.cuda.matmul.allow_tf32 = True          # production inference setting (config use_tf32 true)
    torch.backends.cudnn.allow_tf32 = True
    os.chdir(REPO)
    import xformers.ops as xops
    from xformers.ops.fmha.attn_bias import BlockDiagonalMask
    import model_s2.blocks as Bk
    from model_s2.masked_attention import _fa3_ops
    from model_s2.stage2_wrapper import Stage2LagerNVS
    from tools_s2.s2_bench import load_cfg
    from tools_s2.s2_infer_bench import val_scenes

    # the host syncs removed as in s2_stage_timing (same values): block-diagonal masks cached per sequence lengths,
    # aggregator zeros built on the GPU, the RoPE max position read once per shape
    cache = {}

    def attend_blockdiag_cached(q, k, v, q_seqlens, kv_seqlens, backend="fa3"):
        key = (tuple(q_seqlens), tuple(kv_seqlens))
        if key not in cache:
            cache[key] = BlockDiagonalMask.from_seqlens(list(q_seqlens), list(kv_seqlens), device=q.device)
        return xops.memory_efficient_attention(q[None], k[None], v[None], attn_bias=cache[key], op=_fa3_ops())[0]
    Bk.attend_blockdiag = attend_blockdiag_cached
    patch_aggregator_zeros()
    patch_rope_max_position()

    vts = [int(x) for x in args.vts.split(",")]
    res = dict(gpu=torch.cuda.get_device_name(0), n_scenes=args.n_scenes, warm=args.warm, reps=args.reps, rows={})
    for cpath in args.configs.split(","):
        cfg = load_cfg(cpath, 1)
        name = str(cfg.exp_name)
        model = Stage2LagerNVS(cfg).cuda().eval()
        model.val_cam_cond_zero_p = 0.0
        to_bf16(model)
        ed = model.stage1.model.model
        scenes = val_scenes(cfg, args.n_scenes)
        pvd = model.stage1.model.process_val_data.config.training
        print(f"[r1t] {name}: target_select {model.target_select} (dilate {model.select_dilate}), mask_input "
              f"{model.mask_input}", flush=True)
        for vt in vts:
            pvd.num_target_views = vt
            rows = []
            for b in scenes:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    inp, tgt, images, rays, cam, posed, vin = model.stage1.prepare(b, True, False, True, False, 0.0)
                    p1 = model.stage1.pass1(images, rays, cam, vin)
                trays = rays[:, vin:]

                def ctx_fn():
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        return model.make_ctx(p1, inp, tgt, posed, vin)
                ctx, mask_in, _, _ = ctx_fn()
                mask1 = mask_in.to(trays.dtype)

                def r0():
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        return ed.renderer(p1["rec_rep"], trays, return_intermediates=True)

                def r1():
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        return model.renderer(trays, mask1, p1["rec_rep"], ctx)
                t = dict(tiles=float(sum(ctx.layout.n)) / ctx.layout.BV, ctx=timed(ctx_fn, args.warm, args.reps),
                         r0_eager=timed(r0, args.warm, args.reps), r1_eager=timed(r1, args.warm, args.reps))
                for k, fn in (("r0_graph", r0), ("r1_graph", r1)):
                    try:
                        g, _ = capture(fn)
                        t[k] = timed(g.replay, args.warm, args.reps)
                        del g
                    except Exception as e:                                                # noqa: BLE001
                        t[k] = None
                        print(f"[r1t]   {name} vt={vt}: {k} capture failed: {type(e).__name__}: "
                              f"{str(e).splitlines()[0][:160]}", flush=True)
                rows.append(t)
                del ctx, mask_in, mask1, p1
                torch.cuda.empty_cache()
            agg = {k: (float(np.mean([r[k] for r in rows])) if all(r[k] is not None for r in rows) else None)
                   for k in rows[0]}
            res["rows"][f"{name}|vt{vt}"] = dict(mean=agg, per_scene=rows)
            f = lambda x: "  n/a " if x is None else f"{x:6.1f}"  # noqa: E731
            print(f"[r1t] {name} vt={vt:2d}: tiles/view {agg['tiles']:6.1f} | ctx {f(agg['ctx'])} ms | r0 eager "
                  f"{f(agg['r0_eager'])} graph {f(agg['r0_graph'])} ms | r1 eager {f(agg['r1_eager'])} graph "
                  f"{f(agg['r1_graph'])} ms", flush=True)
            json.dump(res, open(args.out, "w"), indent=1)
        del model, scenes
        torch.cuda.empty_cache()
    print(f"[r1t] DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
