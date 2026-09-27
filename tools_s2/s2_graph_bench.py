# Inference speed-up at batch 1 (1 and 10 target views): (a) the weights of every Linear / Conv / RMSNorm that runs
# under bf16 autocast stored in bf16 -- exactly the values autocast casts to on every call, so the maths does not
# change -- (b) CUDA Graphs: every capturable segment recorded once per scene and replayed.  Segments:
#   G1  stage-1 pass 1 (VGGT + stage-1 renderer + camera / point heads)   G1r  the stage-1 renderer call alone
#   G2  stage-1 input-camera pass                                          G3   stage-2 renderer   G4  stage-2 heads
# not capturable (a device -> host read of the foreground counts): data preparation, pose -> c2w, layout + mask
# tables; they stay eager and are timed.  Outputs of every path are compared with the unmodified eager path
# (max |diff|, PSNR / LPIPS against the ground truth).  Stage 2 = block 8 with the smoke-trained checkpoint (step 90)
# so that its residual is not zero.
#   python tools_s2/s2_graph_bench.py --config configs/S2P8_...yaml --s2_ckpt <ckpt> --out x.json
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def sync_time(fn, reps=5):
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts))


def to_bf16(model):
    s1 = model.stage1.model.model
    trees = [s1.reconstructor, s1.renderer, model.renderer, model.color_head]
    n = 0
    for t in trees:
        for mod in t.modules():
            if isinstance(mod, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                mod.to(torch.bfloat16)
                n += 1
            elif type(mod).__name__ == "RMSNorm":
                mod.weight.data = mod.weight.data.to(torch.bfloat16)
                n += 1
    return n


def from_bf16(model):
    """undo to_bf16 on exactly the modules it converted (everything else keeps its own dtype, e.g. the stage-1
    register tokens that are bf16 by design)."""
    s1 = model.stage1.model.model
    for t in (s1.reconstructor, s1.renderer, model.renderer, model.color_head):
        for mod in t.modules():
            if isinstance(mod, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                mod.to(torch.float32)
            elif type(mod).__name__ == "RMSNorm":
                mod.weight.data = mod.weight.data.to(torch.float32)


def patch_aggregator_zeros():
    """In-process only (the repository is not changed): vggt/models/aggregator.py builds the special-token
    positions with torch.zeros(...) on the CPU and copies them to the GPU (a synchronising copy that a CUDA Graph
    cannot record).  Build the same zeros directly on the GPU; values are identical."""
    import inspect
    import textwrap
    import vggt.models.aggregator as A
    src = textwrap.dedent(inspect.getsource(A.Aggregator.forward))
    old = "torch.zeros(B * S, self.patch_start_idx, 2)"
    assert src.count(old) == 1, "aggregator source changed"
    src = src.replace(old, "torch.zeros(B * S, self.patch_start_idx, 2, device=images.device)")
    ns = {}
    exec(compile(src, A.__file__, "exec"), A.__dict__, ns)
    A.Aggregator.forward = ns["forward"]


def patch_rope_max_position():
    """In-process only: vggt/layers/rope.py reads int(positions.max()) + 1 on the host at every call (a
    device -> host synchronisation).  The positions only depend on the patch grid, so the value is read once per
    positions shape (during the eager warm-up) and reused; the frequency table it selects is unchanged."""
    import inspect
    import textwrap
    import vggt.layers.rope as R
    src = textwrap.dedent(inspect.getsource(R.RotaryPositionEmbedding2D.forward))
    old = "max_position = int(positions.max()) + 1"
    assert src.count(old) == 1, "rope source changed"
    src = src.replace(old, "max_position = _rope_max_position(positions)")
    cache = {}

    def _rope_max_position(positions):
        key = (tuple(positions.shape), positions.device)
        if key not in cache:
            cache[key] = int(positions.max()) + 1
        return cache[key]
    R.__dict__["_rope_max_position"] = _rope_max_position
    ns = {}
    exec(compile(src, R.__file__, "exec"), R.__dict__, ns)
    R.RotaryPositionEmbedding2D.forward = ns["forward"]


def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    return g, out


class Pipe:
    """the stage-2 forward of Stage2LagerNVS split into capturable segments (same calls, same order)."""

    def __init__(self, model, batch):
        self.m, self.b = model, batch

    def prep(self):
        m = self.m
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            r = m.stage1.prepare(self.b, True, False, True, False, m.val_cam_cond_zero_p)
        self.inp, self.tgt, self.images, self.rays, self.cam, self.posed, self.vin = r

    def g1(self):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return self.m.stage1.pass1(self.images, self.rays, self.cam, self.vin)

    def g1r(self):
        ed = self.m.stage1.model.model
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return ed.renderer(self.p1["rec_rep"], self.rays[:, self.vin:], return_intermediates=True)

    def glue1(self):
        import model_s2.geometry as geo
        B = self.tgt.image.shape[0]
        H = self.m.img
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            c2w_pred = geo.c2w_from_pose_enc(self.p1["pose_enc_list"][-1], H)
            c2w_in = torch.where(self.posed.view(B, 1, 1, 1), self.inp.c2w.float(), c2w_pred)
            K_in = self.inp.fxfycxcy.float()
        return c2w_in, K_in

    def g2(self):
        H = self.m.img
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return self.m.stage1.pass2(self.p1["rec0"], self.c2w_in, self.K_in, H, H)

    def glue2(self):
        import model_s2.geometry as geo
        from model_s2.blocks import S2Ctx
        from model_s2.heads_s2 import build_x_prior
        from model_s2.masked_attention import MaskTable
        from model_s2.stage2_wrapper import SCENE_G
        m = self.m
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            alpha_t = self.tgt.alpha_mask.float()
            alpha_in = self.inp.alpha_mask.float()
            lay = geo.build_layout(alpha_t, m.patch, m.pad_bucket, m.img, m.tblock_px)
            S = self.vin * SCENE_G * SCENE_G
            s_pad = geo.roundup(S + 1, 128)
            P_t = self.p1["points"].float()
            Ft = geo.build_forward_table(P_t, alpha_t, lay, self.c2w_in, self.K_in, self.P_in2.float(), alpha_in,
                                         m.patch, SCENE_G, m.img, radius=m.scene_radius, tau=m.tau, s_pad=s_pad)
            Rt = geo.build_reverse_table(self.P_in2.float(), alpha_in, lay, self.tgt.c2w.float(),
                                         self.tgt.fxfycxcy.float(), P_t, alpha_t, m.patch, SCENE_G, m.img,
                                         radius=m.target_radius, tau=m.tau, s_pad=s_pad)
            fm, rm = MaskTable(Ft, m.masked_backend), MaskTable(Rt, m.masked_backend)
            x_prior = build_x_prior(self.p1["T1"][11], lay, m.patch, m.s1_patch)
        ctx = S2Ctx(layout=lay, q_seqlens=[geo.N_REG + n for n in lay.n], fwd_mask=fm, rev_mask=rm, x_prior=x_prior,
                    rec_prior=self.p1["rec10"], scene_g=SCENE_G, n_views=self.vin, scene_block=m.scene_block,
                    s_pad=s_pad, dense_backend=m.dense_backend, masked_backend=m.masked_backend)
        mask1 = (alpha_t > 0.5).to(self.rays.dtype)
        return ctx, mask1

    def g3(self):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return self.m.renderer(self.rays[:, self.vin:], self.mask1, self.p1["rec_rep"], self.ctx)

    def g4(self):
        import model_s2.geometry as geo
        from model_s2.heads_s2 import assemble_tokens, render_color, render_points
        m = self.m
        B, Vt = self.tgt.image.shape[:2]
        lay = self.ctx.layout
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            Ts = {k: assemble_tokens(self.p1["T1"][k], r, lay, m.patch, m.s1_patch) for k, r in self.res.items()}
            has_regs = m.patch == m.s1_patch
            render = render_color(m.color_head, Ts[11], m.patch, B, Vt, m.img, m.img, has_regs)
            points, _ = render_points(m.point_head, Ts, self.rays[:, self.vin:], B, geo.N_REG if has_regs else 0)
        return render, points


def psnr(a, b):
    return float(-10.0 * torch.log10(((a.float() - b.float()) ** 2).mean().clamp_min(1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--s2_ckpt", required=True)
    ap.add_argument("--n_scenes", type=int, default=8)
    ap.add_argument("--vts", default="1,10", help="target-view counts; run one per process to isolate a failed capture")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(28300 + int(os.environ.get("SLURM_JOB_ID", "3")) % 400))
        dist.init_process_group("gloo", rank=0, world_size=1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.chdir(REPO)
    import lpips
    import xformers.ops as xops
    from easydict import EasyDict as edict
    from xformers.ops.fmha.attn_bias import BlockDiagonalMask
    import model_s2.blocks as Bk
    from model_s2.masked_attention import _fa3_ops
    from model_s2.stage2_wrapper import Stage2LagerNVS
    from tools_s2.s2_bench import load_cfg
    from tools_s2.s2_infer_bench import val_scenes

    # block-diagonal masks built once per segment layout (xformers builds them with a host -> device copy, which a
    # CUDA Graph cannot record); same mask, same kernel
    cache = {}

    def attend_blockdiag_cached(q, k, v, q_seqlens, kv_seqlens, backend="fa3"):
        key = (tuple(q_seqlens), tuple(kv_seqlens))
        if key not in cache:
            cache[key] = BlockDiagonalMask.from_seqlens(list(q_seqlens), list(kv_seqlens), device=q.device)
        return xops.memory_efficient_attention(q[None], k[None], v[None], attn_bias=cache[key], op=_fa3_ops())[0]
    Bk.attend_blockdiag = attend_blockdiag_cached
    patch_aggregator_zeros()
    patch_rope_max_position()

    cfg = load_cfg(args.config, 1)
    model = Stage2LagerNVS(cfg).cuda().eval()
    model.load_ckpt(args.s2_ckpt)
    model.val_cam_cond_zero_p = 0.0
    zero = torch.zeros((), device="cuda")
    model.loss_computer.forward = lambda *a, **k: edict(loss=zero)
    lp = lpips.LPIPS(net="vgg").cuda().eval()
    scenes = val_scenes(cfg, args.n_scenes)
    pvd = model.stage1.model.process_val_data.config.training

    def wrapper_out(b):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return model(b, target_has_input=False, is_valid=True)

    def quality(img, gt):
        with torch.no_grad():
            x = img.flatten(0, 1).float() * 2 - 1
            y = gt.flatten(0, 1).float() * 2 - 1
            return psnr(img, gt), float(lp(x, y).mean())

    def section_b(vt, ref):
        n = to_bf16(model)
        d1, d2, t1s, t2s = 0.0, 0.0, [], []
        for b, r in zip(scenes, ref):
            model.enabled = False
            o1 = wrapper_out(b)
            t1s.append(sync_time(lambda: wrapper_out(b)))
            model.enabled = True
            o2 = wrapper_out(b)
            t2s.append(sync_time(lambda: wrapper_out(b)))
            d1 = max(d1, float((o1.render - r["s1"][0]).abs().max()), float((o1.points - r["s1"][1]).abs().max()))
            d2 = max(d2, float((o2.render - r["s2"][0]).abs().max()), float((o2.points - r["s2"][1]).abs().max()))
        out = dict(converted_modules=n, stage1_ms=float(np.mean(t1s)), stage2_ms=float(np.mean(t2s)),
                   max_abs_diff_stage1=d1, max_abs_diff_stage2=d2)
        print(f"[graph] vt={vt} eager bf16 weights ({n} modules): stage 1 {np.mean(t1s):.1f} ms | stage 1+2 "
              f"{np.mean(t2s):.1f} ms | max |diff| vs fp32 weights: stage 1 {d1:.3e}, stage 1+2 {d2:.3e}", flush=True)
        return out

    def section_c(vt, ref):
        model.enabled = True
        rows = []
        for b, r in zip(scenes, ref):
            p = Pipe(model, b)
            try:
                p.prep()
                gr1, p.p1 = capture(p.g1)
                gr1.replay()
                grr, _ = capture(p.g1r)
                p.c2w_in, p.K_in = p.glue1()
                gr2, p.P_in2 = capture(p.g2)
                gr2.replay()
                p.ctx, p.mask1 = p.glue2()
                gr3, p.res = capture(p.g3)
                gr3.replay()
                gr4, (render, points) = capture(p.g4)
                gr1.replay(); gr2.replay(); gr3.replay(); gr4.replay()
                torch.cuda.synchronize()
                s1_img = p.p1["render"]
                dd1 = max(float((s1_img - r["s1"][0]).abs().max()), float((p.p1["points"] - r["s1"][1]).abs().max()))
                dd2 = max(float((render - r["s2"][0]).abs().max()), float((points - r["s2"][1]).abs().max()))
                q = dict(ref1=quality(r["s1"][0], r["gt"]), g1=quality(s1_img, r["gt"]),
                         ref2=quality(r["s2"][0], r["gt"]), g2=quality(render, r["gt"]))

                def stage1_only():
                    p.prep()
                    gr1.replay()

                def stage2_full():
                    p.prep()
                    gr1.replay()
                    p.glue1()
                    gr2.replay()
                    p.glue2()
                    gr3.replay()
                    gr4.replay()
                rows.append(dict(t_stage1=sync_time(stage1_only), t_stage2=sync_time(stage2_full),
                                 t_prep=sync_time(p.prep), t_g1=sync_time(gr1.replay), t_g1r=sync_time(grr.replay),
                                 t_glue1=sync_time(p.glue1), t_g2=sync_time(gr2.replay),
                                 t_glue2=sync_time(p.glue2), t_g3=sync_time(gr3.replay), t_g4=sync_time(gr4.replay),
                                 diff_stage1=dd1, diff_stage2=dd2, q=q))
                del gr1, grr, gr2, gr3, gr4, p
            except Exception as e:                                                     # noqa: BLE001
                import traceback
                traceback.print_exc()
                rows.append(dict(error=f"{type(e).__name__}: {e}"))
                state["capture_failed"] = True   # a failed capture can leave the allocator mid-capture:
                break                            # no empty_cache() / further CUDA work after this
            torch.cuda.empty_cache()
        ok = [x for x in rows if "error" not in x]
        agg = {k: float(np.mean([x[k] for x in ok])) for k in ok[0] if k.startswith("t_")} if ok else {}
        if ok:
            agg["max_diff_stage1"] = max(x["diff_stage1"] for x in ok)
            agg["max_diff_stage2"] = max(x["diff_stage2"] for x in ok)
            for key in ("ref1", "g1", "ref2", "g2"):
                agg[f"psnr_{key}"] = float(np.mean([x["q"][key][0] for x in ok]))
                agg[f"lpips_{key}"] = float(np.mean([x["q"][key][1] for x in ok]))
        out = dict(scenes=len(ok), errors=[x["error"] for x in rows if "error" in x], **agg)
        print(f"[graph] vt={vt} bf16 + CUDA Graphs: " + json.dumps(
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}), flush=True)
        return out

    def sync_check(vt):
        """every capturable segment run eagerly (after warm-up) under set_sync_debug_mode('error'): the first
        host-device synchronisation inside a segment raises and is reported with its source location."""
        import traceback
        pvd.num_target_views = vt
        p = Pipe(model, scenes[0])
        p.prep()
        for _ in range(2):                           # warm-up: RoPE / position / mask caches, flex kernels
            p.p1 = p.g1()
            p.c2w_in, p.K_in = p.glue1()
            p.P_in2 = p.g2()
            p.ctx, p.mask1 = p.glue2()
            p.res = p.g3()
            p.g4()
            p.g1r()
        torch.cuda.synchronize()
        out = {}
        for name in ("g1", "g1r", "g2", "g3", "g4"):
            torch.cuda.set_sync_debug_mode("error")
            try:
                getattr(p, name)()
                out[name] = "sync-free"
            except Exception as e:                                                     # noqa: BLE001
                tb = traceback.format_exc().strip().splitlines()
                frames = [ln.strip() for ln in tb if ln.strip().startswith("File ") and "site-packages/torch/" not in ln]
                out[name] = [f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"] + frames[-6:]
            finally:
                torch.cuda.set_sync_debug_mode(0)
            torch.cuda.synchronize()
            # (the exception text is kept in the JSON only, so the log watchdog does not read it as a crash)
            print(f"[graph] vt={vt} sync check {name}: "
                  f"{'sync-free' if out[name] == 'sync-free' else 'host-device sync found at'}", flush=True)
            if out[name] != "sync-free":
                for ln in out[name][1:]:
                    print(f"[graph]     {ln}", flush=True)
        del p
        return out

    res = {}
    state = {"capture_failed": False}
    for vt in [int(x) for x in args.vts.split(",")]:
        res[f"vt{vt}"] = {"sync_check": sync_check(vt)}
        graph_ok = all(v == "sync-free" for v in res[f"vt{vt}"]["sync_check"].values())
        pvd.num_target_views = vt
        # A. eager, fp32 weights (the unmodified path): reference outputs and times
        ref = []
        for b in scenes:
            model.enabled = False
            o1 = wrapper_out(b)
            t1 = sync_time(lambda: wrapper_out(b))
            model.enabled = True
            o2 = wrapper_out(b)
            t2 = sync_time(lambda: wrapper_out(b))
            ref.append(dict(s1=(o1.render.clone(), o1.points.clone()), s2=(o2.render.clone(), o2.points.clone()),
                            gt=o2.target.image.clone(), t_s1=t1, t_s2=t2))
        res[f"vt{vt}"]["eager_fp32"] = dict(stage1_ms=float(np.mean([r["t_s1"] for r in ref])),
                                            stage2_ms=float(np.mean([r["t_s2"] for r in ref])))
        print(f"[graph] vt={vt} eager fp32 weights: stage 1 {res[f'vt{vt}']['eager_fp32']['stage1_ms']:.1f} ms | "
              f"stage 1+2 {res[f'vt{vt}']['eager_fp32']['stage2_ms']:.1f} ms", flush=True)
        json.dump(res, open(args.out, "w"), indent=1)
        saved = {k: v.detach().clone() for k, v in model.state_dict().items()}
        s1_saved = {k: v.detach().clone() for k, v in model.stage1.model.state_dict().items()}
        sections = [("eager_bf16", section_b)] + ([("graph_bf16", section_c)] if graph_ok else [])
        if not graph_ok:
            res[f"vt{vt}"]["graph_bf16"] = dict(skipped="host-device synchronisation left in a segment, see sync_check")
            print(f"[graph] vt={vt} graph section skipped: a segment still synchronises (see sync check)", flush=True)
        for name, fn in sections:
            try:
                res[f"vt{vt}"][name] = fn(vt, ref)
            except Exception as e:                                                     # noqa: BLE001
                import traceback
                traceback.print_exc()
                res[f"vt{vt}"][name] = dict(error=f"{type(e).__name__}: {e}")
                print(f"[graph] vt={vt} {name} failed: {res[f'vt{vt}'][name]['error']}", flush=True)
                break
            json.dump(res, open(args.out, "w"), indent=1)
        json.dump(res, open(args.out, "w"), indent=1)
        if state["capture_failed"]:
            print("[graph] a capture failed: stopping this process without further CUDA work", flush=True)
            break
        # restore the fp32 weights (exact values) for the next view count
        from_bf16(model)
        model.load_state_dict(saved)
        model.stage1.model.load_state_dict(s1_saved)
        bad = [k for k, v in model.state_dict().items() if v.dtype != saved[k].dtype or not torch.equal(v, saved[k])]
        bad += [k for k, v in model.stage1.model.state_dict().items()
                if v.dtype != s1_saved[k].dtype or not torch.equal(v, s1_saved[k])]
        assert not bad, f"weights not restored: {bad[:5]}"
        del ref
        torch.cuda.empty_cache()
        json.dump(res, open(args.out, "w"), indent=1)
    print(f"[graph] DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
