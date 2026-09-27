# Stage-2 renderer debug (inference, batch 1, 4 input views): is the stage-2 renderer faster than stage 1's renderer,
# per target view?  Everything is patched inside this process only (no repository code changes).
#   A. for 1 and 10 target views: GPU time per scene of VGGT, the stage-1 renderer (target views / input-camera
#      pass), the mask tables, the stage-2 renderer and heads, for these stage-2 variants:
#        current     the implementation as trained
#        C_fused     same math, the compression path (ResBlock + LayerNorm + 3x3 block mean) compiled into fused
#                    kernels (torch.compile)
#        D1_mean1st  timing only, not the approved design: block mean first, ResBlock on the summaries
#        D2_static   timing only: no scene update in stage 2 (no scene injection, reverse attention, scene MLP)
#        D3_shared   timing only: D2 and the scene side computed once and shared by all target views
#   B. at 10 views, inside both renderers: attention kernels, projections / norms / MLPs split by scene side vs
#      target side, compression, copies, and the unlabelled rest
#   C. max |difference| of the fused compression path vs the eager one on real tensors
#   python tools_s2/s2_renderer_debug.py --config configs/S2P8_...yaml --out x.json
import argparse
import json
import os
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
ACC = {}


def ev():
    return torch.cuda.Event(enable_timing=True)


def timed(fn, label):
    def w(*a, **k):
        s, e = ev(), ev()
        s.record()
        out = fn(*a, **k)
        e.record()
        ACC.setdefault(label() if callable(label) else label, []).append((s, e))
        return out
    return w


def patch(ps, obj, name, new):
    ps.append((obj, name, getattr(obj, name), name in vars(obj) if not isinstance(obj, types.ModuleType) else True))
    setattr(obj, name, new)


def unpatch(ps):
    for obj, name, old, was_own in reversed(ps):
        if isinstance(obj, (types.ModuleType, type)) or was_own:
            setattr(obj, name, old)
        else:
            vars(obj).pop(name, None)
    ps.clear()


def collect(n):
    torch.cuda.synchronize()
    out = {k: float(sum(s.elapsed_time(e) for s, e in v) / n) for k, v in ACC.items()}
    ACC.clear()
    return out


def fwd(model, b):
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return model(b, target_has_input=False, is_valid=True)


# ------------------------------------------------------------------------------------------ variants
def comp_mean_eager(comp, t, g, nv, blk):
    from model_s2.blocks import scene_block_mean
    return scene_block_mean(comp(t), g, nv, blk)


def _comp_mean_fn(t, w1, b1, w2, b2, g, nv, blk):
    from model_s2.blocks import scene_block_mean
    h = F.linear(F.silu(F.linear(t, w1, b1)), w2, b2)
    y = F.layer_norm((t + h).float(), (t.shape[-1],), eps=1e-5).to(t.dtype)
    return scene_block_mean(y, g, nv, blk)


_comp_mean_c = None


def comp_mean_fused(comp, t, g, nv, blk):
    global _comp_mean_c
    if _comp_mean_c is None:
        _comp_mean_c = torch.compile(_comp_mean_fn, dynamic=False)
    a = comp.act_layers
    return _comp_mean_c(t, a[0].weight, a[0].bias, a[2].weight, a[2].bias, g, nv, blk)


def make_t2s(mode, B_of):
    """target_to_scene with the compression path per variant; mode in current|fused|mean1st|shared"""
    import model_s2.blocks as Bk

    def t2s(attn, gate, comp_k, comp_v, xn, rn, ctx):
        H = attn.num_heads
        lay = ctx.layout
        BV = lay.BV
        q = attn.q_norm(Bk._heads(attn.q_proj(xn), H))
        src = rn[:B_of()] if mode == "shared" else rn                     # shared: one scene copy per sample
        kr = Bk._heads(attn.k_proj(src), H)
        vr = Bk._heads(attn.v_proj(src), H)
        g, nv, blk = ctx.scene_g, ctx.n_views, ctx.scene_block
        if mode == "fused":
            kc = attn.k_norm(comp_mean_fused(comp_k, kr, g, nv, blk))
            vc = comp_mean_fused(comp_v, vr, g, nv, blk)
        elif mode == "mean1st":
            kc = attn.k_norm(comp_k(Bk.scene_block_mean(kr, g, nv, blk)))
            vc = comp_v(Bk.scene_block_mean(vr, g, nv, blk))
        else:
            kc = attn.k_norm(Bk.scene_block_mean(comp_k(kr), g, nv, blk))
            vc = Bk.scene_block_mean(comp_v(vr), g, nv, blk)
        ks = Bk._pad_rows(attn.k_norm(kr), ctx.s_pad)
        vs = Bk._pad_rows(vr, ctx.s_pad)
        if mode == "shared":
            rep = BV // kc.shape[0]
            kc, vc = kc.repeat_interleave(rep, 0), vc.repeat_interleave(rep, 0)
            ks, vs = ks.expand(BV, -1, -1, -1), vs.expand(BV, -1, -1, -1)
        nblk = kc.shape[1]
        o_c = Bk.attend_blockdiag(q, kc.flatten(0, 1), vc.flatten(0, 1), ctx.q_seqlens, [nblk] * BV, ctx.dense_backend)
        qz = torch.cat([q, q.new_zeros(1, *q.shape[1:])])
        q_pad = qz[lay.pack_src]
        o_s = Bk.attend_masked(q_pad, ks, vs, ctx.fwd_mask, ctx.masked_backend)
        o_s = o_s.reshape(-1, *o_s.shape[2:])[lay.pad_dst]
        g_c, g_s = torch.sigmoid(gate(xn)).chunk(2, dim=-1)
        return attn.attn_fc_dropout(attn.proj(g_c * o_c.flatten(-2) + g_s * o_s.flatten(-2)))
    return t2s


def bidir_static(self, x, rec, ctx):
    import model_s2.blocks as Bk
    x = x + self.inj_x(ctx.x_prior)
    x = x + Bk.self_attention_packed(self.self_attn, self.norm1_x(x), ctx)
    xn = self.norm2_x(x)
    rn = self.norm1_rec(rec)
    x = x + Bk.target_to_scene(self.cross_attn_x, self.gate, self.comp_k, self.comp_v, xn, rn, ctx)
    x = x + self.mlp_x(self.norm3_x(x))
    return x, rec


def final_static(self, x, rec, ctx):
    import model_s2.blocks as Bk
    x = x + self.inj_x(ctx.x_prior)
    x = x + Bk.self_attention_packed(self.self_attn, self.norm1(x), ctx)
    xn = self.norm2(x)
    rn = self.norm2_kv(rec)
    x = x + Bk.target_to_scene(self.cross_attn, self.gate, self.comp_k, self.comp_v, xn, rn, ctx)
    x = x + self.mlp(self.norm_ffn(x))
    return x


def apply_variant(ps, name, B_of):
    import model_s2.blocks as Bk
    if name == "C_fused":
        patch(ps, Bk, "target_to_scene", make_t2s("fused", B_of))
    elif name == "D1_mean1st":
        patch(ps, Bk, "target_to_scene", make_t2s("mean1st", B_of))
    elif name == "D2_static":
        patch(ps, Bk.S2BidirBlock, "forward", bidir_static)
        patch(ps, Bk.S2FinalBlock, "forward", final_static)
    elif name == "D3_shared":
        patch(ps, Bk.S2BidirBlock, "forward", bidir_static)
        patch(ps, Bk.S2FinalBlock, "forward", final_static)
        patch(ps, Bk, "target_to_scene", make_t2s("shared", B_of))


# ------------------------------------------------------------------------------------------ instrumentation
def coarse(ps, model, state):
    import model_s2.geometry as geo
    import model_s2.stage2_wrapper as w
    s1 = model.stage1
    m = s1.model
    ed = m.model
    patch(ps, ed.reconstructor, "forward", timed(ed.reconstructor.forward, "VGGT (DINOv2 + aggregator)"))
    orig_p2 = s1.pass2

    def p2(*a, **k):
        state["p2"] = True
        try:
            return orig_p2(*a, **k)
        finally:
            state["p2"] = False
    patch(ps, s1, "pass2", p2)
    patch(ps, ed.renderer, "forward", timed(ed.renderer.forward, lambda: "stage-1 renderer (input-camera pass)"
                                              if state["p2"] else "stage-1 renderer (target views)"))
    patch(ps, m.point_head, "forward", timed(m.point_head.forward, lambda: "stage-1 point head (input-camera pass)"
                                               if state["p2"] else "stage-1 point head (target views)"))
    patch(ps, m.camera_head, "forward", timed(m.camera_head.forward, "stage-1 camera head"))
    for nm in ("build_layout", "build_forward_table", "build_reverse_table"):
        patch(ps, geo, nm, timed(getattr(geo, nm), "stage-2 mask tables"))
    patch(ps, w, "MaskTable", timed(w.MaskTable, "stage-2 mask tables"))
    patch(ps, model.renderer, "forward", timed(model.renderer.forward, "stage-2 renderer"))
    patch(ps, w, "render_color", timed(w.render_color, "stage-2 heads"))
    patch(ps, w, "render_points", timed(w.render_points, "stage-2 heads"))


SIDE = {  # (attention module name, sub-module) -> side
    ("self_attn", "q_proj"): "target", ("self_attn", "k_proj"): "target", ("self_attn", "v_proj"): "target",
    ("self_attn", "proj"): "target", ("self_attn", "q_norm"): "target", ("self_attn", "k_norm"): "target",
    ("cross_attn_x", "q_proj"): "target", ("cross_attn_x", "proj"): "target", ("cross_attn_x", "q_norm"): "target",
    ("cross_attn_x", "k_proj"): "scene", ("cross_attn_x", "v_proj"): "scene", ("cross_attn_x", "k_norm"): "scene",
    ("cross_attn", "q_proj"): "target", ("cross_attn", "proj"): "target", ("cross_attn", "q_norm"): "target",
    ("cross_attn", "k_proj"): "scene", ("cross_attn", "v_proj"): "scene", ("cross_attn", "k_norm"): "scene",
    ("cross_attn_rec", "q_proj"): "scene", ("cross_attn_rec", "proj"): "scene", ("cross_attn_rec", "q_norm"): "scene",
    ("cross_attn_rec", "k_proj"): "target", ("cross_attn_rec", "v_proj"): "target",
    ("cross_attn_rec", "k_norm"): "target"}
NORM_SIDE = {"norm1_x": "target", "norm2_x": "target", "norm3_x": "target", "norm1": "target", "norm2": "target",
             "norm_ffn": "target", "norm1_rec": "scene", "norm2_rec": "scene", "norm2_kv": "scene"}


def detailed(ps, model, which):
    """leaf-level labels inside one renderer (which = 's1' or 's2'); the renderer total is timed too."""
    import xformers.ops as xops
    import model_s2.blocks as Bk
    s1 = model.stage1
    ed = s1.model.model
    if which == "s1":
        rnd = ed.renderer
        blocks = list(ed.renderer.renderer_core.renderer_blocks)
    else:
        rnd = model.renderer
        blocks = list(model.renderer.blocks)
    state = {"in": False}
    orig = rnd.forward

    def inside(*a, **k):
        state["in"] = True
        try:
            return orig(*a, **k)
        finally:
            state["in"] = False
    patch(ps, rnd, "forward", timed(inside, f"{which} renderer total"))
    xo = xops.memory_efficient_attention

    def xo_w(*a, **k):                    # count FA3 calls made inside the renderer under test only
        if not state["in"]:
            return xo(*a, **k)
        s, e = ev(), ev()
        s.record()
        out = xo(*a, **k)
        e.record()
        ACC.setdefault("attention kernel, dense (FA3)", []).append((s, e))
        return out
    patch(ps, xops, "memory_efficient_attention", xo_w)
    for blk in blocks:
        for an, sub in SIDE:
            if hasattr(blk, an) and hasattr(getattr(blk, an), sub):
                mod = getattr(getattr(blk, an), sub)
                kind = "q/k norms" if sub.endswith("norm") else "projections"
                patch(ps, mod, "forward", timed(mod.forward, f"{SIDE[(an, sub)]}-side {kind}"))
        for nn_, side in NORM_SIDE.items():
            if hasattr(blk, nn_):
                mod = getattr(blk, nn_)
                patch(ps, mod, "forward", timed(mod.forward, f"{side}-side block norms"))
        for nn_, lab in (("mlp_x", "target-side MLP"), ("mlp", "target-side MLP"), ("mlp_rec", "scene-side MLP"),
                         ("inj_x", "target-side injection"), ("inj_rec", "scene-side injection"),
                         ("gate", "target-side gate"), ("comp_k", "compression ResBlocks (scene side)"),
                         ("comp_v", "compression ResBlocks (scene side)"),
                         ("rcomp_k", "compression ResBlocks (target side)"),
                         ("rcomp_v", "compression ResBlocks (target side)")):
            if hasattr(blk, nn_):
                mod = getattr(blk, nn_)
                patch(ps, mod, "forward", timed(mod.forward, lab))
    if which == "s2":
        patch(ps, Bk, "attend_masked", timed(Bk.attend_masked, "attention kernel, masked (flex)"))
        patch(ps, Bk, "scene_block_mean", timed(Bk.scene_block_mean, "block mean (scene side)"))
        patch(ps, Bk, "target_block_mean", timed(Bk.target_block_mean, "block mean (target side)"))
        patch(ps, Bk, "_pad_rows", timed(Bk._pad_rows, "padding copies"))


def gpu_busy(model, b):
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t0 = time.perf_counter()
        fwd(model, b)
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) * 1000.0
    kern = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    return dict(wall_ms=wall, kernel_ms=sum(e.time_range.elapsed_us() for e in kern) / 1000.0, n_kernels=len(kern))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_scenes", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(28900 + int(os.environ.get("SLURM_JOB_ID", "3")) % 400))
        dist.init_process_group("gloo", rank=0, world_size=1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.chdir(REPO)
    from easydict import EasyDict as edict
    from model_s2.stage2_wrapper import Stage2LagerNVS
    from tools_s2.s2_bench import load_cfg
    from tools_s2.s2_infer_bench import val_scenes
    import model_s2.blocks as Bk
    cfg = load_cfg(args.config, 1)
    model = Stage2LagerNVS(cfg).cuda().eval()
    model.val_cam_cond_zero_p = 0.0
    zero = torch.zeros((), device="cuda")
    model.loss_computer.forward = lambda *a, **k: edict(loss=zero)
    scenes = val_scenes(cfg, args.n_scenes)
    pvd = model.stage1.model.process_val_data.config.training
    res = {}
    B_of = lambda: 1                                                                  # batch 1 throughout
    # ---- C: fused vs eager compression on real tensors (block 0, 10 target views)
    pvd.num_target_views = 10
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        cap = {}
        ps = []
        patch(ps, model.renderer, "forward", lambda r6, m1, rr, ctx: cap.update(rr=rr, ctx=ctx) or {})
        try:
            fwd(model, scenes[0])
        except Exception:
            pass
        unpatch(ps)
        blk0 = model.renderer.blocks[0]
        rn = blk0.norm1_rec(cap["rr"])
        kr = Bk._heads(blk0.cross_attn_x.k_proj(rn), 12)
        c = cap["ctx"]
        a = comp_mean_eager(blk0.comp_k, kr, c.scene_g, c.n_views, c.scene_block)
        b = comp_mean_fused(blk0.comp_k, kr, c.scene_g, c.n_views, c.scene_block)
        res["C_equivalence"] = dict(max_abs_diff=float((a.float() - b.float()).abs().max()),
                                    max_abs_value=float(a.float().abs().max()), shape=list(a.shape))
        print(f"[dbg] C fused vs eager compression: {res['C_equivalence']}", flush=True)
        del cap, rn, kr, a, b
    # ---- A: per target-view count and variant
    for vt in (1, 10):
        pvd.num_target_views = vt
        for var in ("stage1_only", "current", "C_fused", "D1_mean1st", "D2_static", "D3_shared"):
            vps = []
            model.enabled = var != "stage1_only"
            if var not in ("stage1_only", "current"):
                apply_variant(vps, var, B_of)
            try:
                for _ in range(2):
                    for sc in scenes:
                        fwd(model, sc)
                torch.cuda.synchronize()
                walls = []
                for _ in range(3):
                    for sc in scenes:
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        fwd(model, sc)
                        torch.cuda.synchronize()
                        walls.append((time.perf_counter() - t0) * 1000.0)
                ps, state = [], {"p2": False}
                coarse(ps, model, state)
                try:
                    for sc in scenes:
                        fwd(model, sc)
                    parts = collect(len(scenes))
                finally:
                    unpatch(ps)
                busy = gpu_busy(model, scenes[0])
                r = dict(wall_median_ms=float(np.median(walls)), parts_ms=parts, gpu_busy=busy)
            except Exception as e:                                                     # noqa: BLE001
                r = dict(error=f"{type(e).__name__}: {e}")
            finally:
                unpatch(vps)
            res[f"vt{vt}_{var}"] = r
            if "error" in r:
                print(f"[dbg] vt={vt} {var}: ERROR {r['error']}", flush=True)
                continue
            print(f"[dbg] vt={vt} {var}: wall median {r['wall_median_ms']:.1f} ms | GPU busy "
                  f"{100 * busy['kernel_ms'] / busy['wall_ms']:.0f} % ({busy['n_kernels']} kernels) | "
                  + ", ".join(f"{k} {v:.1f}" for k, v in sorted(parts.items())), flush=True)
    # ---- B: leaf-level split inside both renderers at 10 target views, current implementation
    pvd.num_target_views = 10
    for which in ("s1", "s2"):
        model.enabled = which == "s2"
        ps = []
        detailed(ps, model, which)
        try:
            for sc in scenes:
                fwd(model, sc)
            parts = collect(len(scenes))
        finally:
            unpatch(ps)
        tot = parts.get(f"{which} renderer total", 0.0)
        lab = sum(v for k, v in parts.items() if k != f"{which} renderer total")
        parts["unlabelled rest (residual adds, reshapes, gathers, embed, final layer)"] = tot - lab
        res[f"B_{which}"] = parts
        print(f"[dbg] B {which} renderer at 10 views (ms per scene):", flush=True)
        for k, v in sorted(parts.items(), key=lambda kv: -kv[1]):
            print(f"[dbg]   {k:72s} {v:8.2f}", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"[dbg] DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
