# Where the time goes, stage 1 vs stage 2, part by part (CUDA events around every sub-module / helper, summed per
# forward, averaged over scenes), for two shapes:
#   infer  batch 1, 4 input -> 10 target views (the evaluation set-up)
#   train  batch 8, 4 input -> 6 target views (48 target views, forward only, no grad)
# plus
#   gpu-busy   torch.profiler: summed kernel time vs wall time of one forward (stage 1 only, stage 1 + 2), i.e.
#              whether the stage-2 renderer leaves the GPU idle between many small kernels
#   merge      one stage-1 renderer + point-head call on 14 views (10 targets + the 4 input cameras) vs the current
#              two calls (10, then 4): time, and max |difference| of the 10 target outputs
#   python tools_s2/s2_infer_profile.py --config configs/S2P8_...yaml --out x.json
import argparse
import json
import os
import sys
import time
import types

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


class Timers:
    def __init__(self):
        self.acc = {}
        self.patches = []

    def wrap(self, obj, name, label):
        fn = getattr(obj, name)
        acc = self.acc

        def timed(*a, **k):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            out = fn(*a, **k)
            e.record()
            acc.setdefault(label, []).append((s, e))
            return out
        setattr(obj, name, timed)
        self.patches.append((obj, name, fn))

    def restore(self):
        for obj, name, fn in reversed(self.patches):
            if isinstance(obj, types.ModuleType):
                setattr(obj, name, fn)
            else:
                vars(obj).pop(name, None)
        self.patches = []

    def result(self, n):
        torch.cuda.synchronize()
        out = {k: float(sum(s.elapsed_time(e) for s, e in v) / n) for k, v in self.acc.items()}
        self.acc = {}
        return out


def instrument(model, t):
    import model_s2.blocks as B
    s1 = model.stage1
    m = s1.model
    ed = m.model
    t.wrap(s1, "pass1", "S1 pass 1 (all)")
    t.wrap(s1, "pass2", "S1 pass 2 (input cameras, all)")
    t.wrap(ed.reconstructor, "forward", "S1 reconstructor (DINOv2 stem + VGGT aggregator)")
    agg = ed.reconstructor.vggt.aggregator
    t.wrap(agg.patch_embed, "forward", "S1   DINOv2 stem")
    t.wrap(ed.renderer, "forward", "S1 renderer call (all blocks + final layer)")
    for i, blk in enumerate(ed.renderer.renderer_core.renderer_blocks):
        names = ("self_attn", "cross_attn_x", "cross_attn_rec", "mlp_x", "mlp_rec") if hasattr(blk, "mlp_rec") \
            else ("self_attn", "cross_attn", "mlp")
        for nm in names:
            lab = {"self_attn": "target self-attn", "cross_attn_x": "target->scene attn (dense)",
                   "cross_attn": "target->scene attn (dense)", "cross_attn_rec": "scene->target attn (dense)",
                   "mlp_x": "target MLP", "mlp": "target MLP", "mlp_rec": "scene MLP"}[nm]
            t.wrap(getattr(blk, nm), "forward", f"S1   {lab}")
    t.wrap(m.camera_head, "forward", "S1 camera head")
    t.wrap(m.point_head, "forward", "S1 point head (DPT) call")
    # stage 2
    import model_s2.geometry as geo
    import model_s2.stage2_wrapper as w
    for nm in ("build_layout", "build_forward_table", "build_reverse_table"):
        t.wrap(geo, nm, "S2 layout + mask tables")
    t.wrap(w, "MaskTable", "S2 layout + mask tables")
    t.wrap(model.renderer, "forward", "S2 renderer (all)")
    for blk in model.renderer.blocks:
        for nm in ("inj_x", "inj_rec"):
            t.wrap(getattr(blk, nm), "forward", "S2   injection (x + scene)")
        for nm in ("mlp_x", "mlp"):
            if hasattr(blk, nm):
                t.wrap(getattr(blk, nm), "forward", "S2   target MLP")
        if hasattr(blk, "mlp_rec"):
            t.wrap(blk.mlp_rec, "forward", "S2   scene MLP")
        for nm in ("comp_k", "comp_v", "rcomp_k", "rcomp_v"):
            if hasattr(blk, nm):
                t.wrap(getattr(blk, nm), "forward", "S2     compression ResBlocks")
        t.wrap(blk.gate, "forward", "S2     gate")
    t.wrap(B, "self_attention_packed", "S2   target self-attn (all)")
    t.wrap(B, "target_to_scene", "S2   target->scene (all)")
    t.wrap(B, "scene_to_target", "S2   scene->target (all)")
    t.wrap(B, "scene_block_mean", "S2     scene block mean")
    t.wrap(B, "target_block_mean", "S2     target block mean")
    t.wrap(B, "attend_blockdiag", "S2     dense attn (FA3: self + compressed)")
    t.wrap(B, "attend_masked", "S2     masked attn (flex/sdpa: selected + reverse)")
    t.wrap(B, "_pad_rows", "S2     pad rows (copies)")
    t.wrap(w, "render_color", "S2 heads")
    t.wrap(w, "render_points", "S2 heads")


def run(model, batches, is_valid):
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for b in batches:
            model(b, target_has_input=False, is_valid=is_valid)


def gpu_busy(model, b, is_valid):
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t0 = time.perf_counter()
        run(model, [b], is_valid)
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) * 1000.0
    kern = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    busy = sum(e.time_range.elapsed_us() for e in kern) / 1000.0          # each GPU event's own duration
    return dict(wall_ms=wall, kernel_ms=busy, n_kernels=len(kern))


def merge_test(model, batches, is_valid=True):
    """one renderer + point-head call on 14 views vs two calls (10, 4); target outputs compared."""
    import einops
    from model_s2.geometry import plucker_rays
    s1 = model.stage1
    m = s1.model
    ed = m.model
    t_sep, t_mrg, dmax_img, dmax_pts = [], [], 0.0, 0.0
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for rep in range(3):
            for b in batches:
                inp, tgt, images, rays, cam, posed, vin = s1.prepare(b, True, False, is_valid, False, 0.0)
                rec_tok, agg_last, _ = ed.reconstructor(images[:, :vin], cam[:, :vin], return_tokens=True)
                rec0 = einops.rearrange(rec_tok, "b v p c -> b (v p) c")
                tr = rays[:, vin:]
                ri = plucker_rays(inp.c2w.float(), inp.fxfycxcy.float(), 256, 256).to(tr.dtype)
                B, vt = tr.shape[:2]

                def call(rv):
                    rr = einops.repeat(rec0, "b np d -> (b v) np d", v=rv.shape[1])
                    img, inter, psi = ed.renderer(rr, rv, return_intermediates=True)
                    with torch.autocast("cuda", enabled=False):
                        tok = [einops.rearrange(x, "(b v) n c -> b v n c", b=B) for x in inter]
                        pts, _ = m.point_head(tok, rv.float(), patch_token_start=psi)
                    return img, pts
                torch.cuda.synchronize()
                e = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
                e[0].record()
                img_t, pts_t = call(tr)
                call(ri)
                e[1].record()
                e[2].record()
                img_m, pts_m = call(torch.cat([tr, ri], dim=1))
                e[3].record()
                torch.cuda.synchronize()
                if rep > 0:                                   # rep 0 = warmup
                    t_sep.append(e[0].elapsed_time(e[1]))
                    t_mrg.append(e[2].elapsed_time(e[3]))
                dmax_img = max(dmax_img, float((img_m[:, :vt].float() - img_t.float()).abs().max()))
                dmax_pts = max(dmax_pts, float((pts_m[:, :vt].float() - pts_t.float()).abs().max()))
    return dict(separate_ms=float(np.median(t_sep)), merged_ms=float(np.median(t_mrg)),
                max_abs_diff_target_rgb=dmax_img, max_abs_diff_target_points=dmax_pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_scenes", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--merge_only", action="store_true")
    args = ap.parse_args()
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(29000 + int(os.environ.get("SLURM_JOB_ID", "7")) % 400))
        dist.init_process_group("gloo", rank=0, world_size=1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.chdir(REPO)
    from easydict import EasyDict as edict
    from model_s2.stage2_wrapper import Stage2LagerNVS
    from tools_s2.s2_bench import cached_batches, load_cfg
    from tools_s2.s2_infer_bench import val_scenes
    cfg = load_cfg(args.config, 8)
    model = Stage2LagerNVS(cfg).cuda().eval()
    model.val_cam_cond_zero_p = 0.0
    zero = torch.zeros((), device="cuda")
    model.loss_computer.forward = lambda *a, **k: edict(loss=zero)
    scenes = val_scenes(cfg, args.n_scenes)
    train_b = cached_batches(cfg, 2)
    res = {}
    if "--merge_only" in sys.argv:
        model.enabled = True
        res["merge_train"] = merge_test(model, train_b, is_valid=False)
        print(f"[prof] merge (train shape): {json.dumps(res['merge_train'])}", flush=True)
        json.dump(res, open(args.out, "w"), indent=1)
        return
    for shape, batches, is_valid in (("infer", scenes, True), ("train", train_b, False)):
        for s2_on in (False, True):
            model.enabled = s2_on
            run(model, batches, is_valid)                                  # warmup (flex compiles)
            run(model, batches, is_valid)
            t = Timers()
            instrument(model, t)
            try:
                run(model, batches, is_valid)
                r = t.result(len(batches))
            finally:
                t.restore()
            r["gpu_busy"] = gpu_busy(model, batches[0], is_valid)
            key = f"{shape}_{'stage1+2' if s2_on else 'stage1_only'}"
            res[key] = r
            print(f"[prof] {key}:", flush=True)
            for k in sorted(r):
                if k != "gpu_busy":
                    print(f"[prof]   {k:58s} {r[k]:9.2f} ms", flush=True)
            g = r["gpu_busy"]
            print(f"[prof]   one forward: wall {g['wall_ms']:.1f} ms, GPU kernel time {g['kernel_ms']:.1f} ms "
                  f"({100 * g['kernel_ms'] / max(g['wall_ms'], 1e-9):.0f} % busy), {g['n_kernels']} kernels", flush=True)
    model.enabled = True
    res["merge_infer"] = merge_test(model, scenes)
    print(f"[prof] merge (infer): {json.dumps(res['merge_infer'])}", flush=True)
    res["merge_train"] = merge_test(model, train_b, is_valid=False)
    print(f"[prof] merge (train shape): {json.dumps(res['merge_train'])}", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"[prof] DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
