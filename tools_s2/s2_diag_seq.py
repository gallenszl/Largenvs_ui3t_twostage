# Diagnostic (G4 / G7 follow-up), one GPU:
#  A. resume keys: which state_dict keys a trainable-only checkpoint must carry for the real P8 model
#     (aliases, persistent buffers) and what the P8 smoke ckpt_60 is missing.
#  B. replays the GPU init-equivalence tests in unittest order on one fixture and re-checks
#     "P8 stage 2 at init == stage 1" before and after each test, to find the test that changes shared state.
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from tests.test_s2_init_equivalence_gpu import InitEquivalenceTests  # noqa: E402

SMOKE_CKPT = ("/mnt/data-alpha-sg-01/team-camera/home/z50057756/moe_experiments/SMOKE/"
              "S2P8_uni3t70k_b32t6_lr35_a21k26k_SMOKE_137837/ckpt_0000000000000060.pt")


def part_a(m):
    from utils.training_utils import trainable_state_keys
    sd = m.state_dict()
    pn = dict(m.named_parameters(remove_duplicate=False))
    pn_dedup = dict(m.named_parameters())
    bufs = [k for k in sd if k not in pn]
    keep = trainable_state_keys(m)
    print(f"[A] state_dict keys {len(sd)} | params (all names) {len(pn)} | params (dedup) {len(pn_dedup)} | "
          f"persistent buffers {len(bufs)}", flush=True)
    print(f"[A] persistent buffers: {bufs[:20]}", flush=True)
    print(f"[A] alias names of params: {sorted(set(pn) - set(pn_dedup))[:6]} ...", flush=True)
    print(f"[A] trainable-only keep set {len(keep)}; buffers kept {[k for k in keep if k in bufs][:10]}", flush=True)
    if os.path.exists(SMOKE_CKPT):
        ck = torch.load(SMOKE_CKPT, map_location="cpu", weights_only=True, mmap=True)["model"]
        missing = [k for k in sd if k not in ck]
        unexpected = [k for k in ck if k not in sd]
        bad = [k for k in missing if k in keep]
        print(f"[A] smoke ckpt_60: {len(ck)} keys, missing {len(missing)} (of which in keep set: {len(bad)} "
              f"{bad[:6]}), unexpected {len(unexpected)} {unexpected[:6]}", flush=True)
        del ck


def main():
    InitEquivalenceTests.setUpClass()
    t = InitEquivalenceTests()
    m = t.m8
    part_a(m)
    snap = {k: v.detach().clone() for k, v in m.state_dict().items()}
    bsnap = {n: b.detach().clone() for n, b in m.named_buffers()}
    store = {}
    if "--lazy-hook" in sys.argv:              # keep references only: no host sync inside the forward
        m.renderer.register_forward_hook(lambda mod, inp, out: store.update(out=out))
    else:
        m.renderer.register_forward_hook(lambda mod, inp, out: store.update(
            res={k: (bool(torch.isfinite(v).all()), float(v.float().abs().max())) for k, v in out.items()}))

    def check(tag):
        for zp in (0.0, 1.0):
            r, s = t._pair(zp)
            if "out" in store:
                store["res"] = {k: (bool(torch.isfinite(v).all()), float(v.float().abs().max()))
                                for k, v in store.pop("out").items()}
            d = (s.render.float() - r.render.float()).abs()
            dp = (s.points.float() - r.points.float()).abs()
            msg = (f"[B] {tag:22s} zp={zp} render_eq={torch.equal(s.render, r.render)} "
                   f"s1_eq={torch.equal(s.render_s1, r.render)} points_eq={torch.equal(s.points, r.points)} "
                   f"render max {float(d.max()):.3e} frac {float((d > 0).float().mean()):.3e} "
                   f"nan {int(torch.isnan(s.render).sum())} | points max {float(dp.max()):.3e} "
                   f"| s2 residual {store.get('res')}")
            try:
                am = s.target.alpha_mask.float()
                fg = (am.view(am.shape[0], am.shape[1], 1, *d.shape[-2:]) > 0.5).expand_as(d)
                msg += (f" | differing px fg {int(((d > 0) & fg).sum())} bg {int(((d > 0) & ~fg).sum())}")
            except Exception as e:                                        # noqa: BLE001
                msg += f" | fg split n/a ({type(e).__name__})"
            print(msg, flush=True)
            if not torch.equal(s.render, r.render):
                head_level(zp)
        sd = m.state_dict()
        diff = [k for k in snap if not torch.equal(sd[k], snap[k])]
        bdiff = [n for n, b in m.named_buffers() if n in bsnap and not torch.equal(b, bsnap[n])]
        print(f"[B] {tag:22s} state keys differing from start: {len(diff)} {diff[:6]} | buffers differing "
              f"{bdiff[:6]} | out_lin max|w| {[float(l.weight.abs().max()) for l in m.renderer.out_lin]} "
              f"| training {m.training}", flush=True)

    def head_level(zp):
        from model_s2.heads_s2 import assemble_tokens, render_color
        from model_s2 import geometry as geo
        s1 = m.stage1
        r1 = s1.model.model.renderer
        eqw = all(torch.equal(a, b) for a, b in zip(m.color_head.parameters(), r1.final_layer.parameters()))
        m.val_cam_cond_zero_p = zp
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            from tests.test_s2_init_equivalence_gpu import seed_all
            seed_all(123)
            inp, tgt, images, rays, cam, posed, vin = s1.prepare(t.batch, True, False, True, False, zp)
            p1 = s1.pass1(images, rays, cam, vin)
            B, Vt = tgt.image.shape[:2]
            T11 = p1["T1"][11]
            a = render_color(m.color_head, T11, 8, B, Vt, 256, 256, True)
            b = render_color(r1.final_layer, T11, 8, B, Vt, 256, 256, True)
            lay = geo.build_layout(tgt.alpha_mask.float(), 8, 128, 256)
            zero = torch.zeros(sum(lay.n), T11.shape[-1], device="cuda", dtype=torch.bfloat16)
            Tt = assemble_tokens(T11, zero, lay, 8, 8)
            c = render_color(m.color_head, Tt, 8, B, Vt, 256, 256, True)
        print(f"[H] zp={zp} color weights eq {eqw} | copied head on T1 == s1 head on T1 {torch.equal(a, b)} "
              f"| == pass1 render {torch.equal(b, p1['render'])} | assemble(T1,0)==T1 {torch.equal(Tt, T11)} "
              f"| head on assembled == on T1 {torch.equal(c, a)} | T11 {T11.dtype} contig {T11.is_contiguous()} "
              f"stride {T11.stride()} ptr%512 {T11.data_ptr() % 512} Tt ptr%512 {Tt.data_ptr() % 512}", flush=True)

    if "--no-fresh" not in sys.argv:           # unittest order: the first forward of m8 is the training step
        check("fresh")
    for name in ("test_one_step_updates_only_stage2", "test_p4_forward_is_finite",
                 "test_pass2_depth_and_tables_are_sane"):
        getattr(t, name)()
        check(f"after {name[5:20]}")
    print("[B] DONE", flush=True)


if __name__ == "__main__":
    main()
