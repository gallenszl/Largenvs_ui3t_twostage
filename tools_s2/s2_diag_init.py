# Diagnostic (G4 follow-up): where does the P8 stage-2 output at init diverge from stage 1?
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from tests.test_s2_init_equivalence_gpu import AMP, S2_CFG, load_cfg, seed_all, val_batch  # noqa: E402


def main():
    import torch.distributed as dist
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29877")
    dist.init_process_group("gloo", rank=0, world_size=1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.chdir(REPO)
    from model_s2 import geometry as geo
    from model_s2.heads_s2 import assemble_tokens, render_color, render_points
    from model_s2.stage2_wrapper import Stage2LagerNVS
    cfg = load_cfg(S2_CFG[8])
    m = Stage2LagerNVS(cfg).cuda().eval()
    batch = val_batch(cfg, 0)
    s1 = m.stage1
    r1 = s1.model.model.renderer
    print("color head weights equal:", all(torch.equal(a, b) for a, b in zip(m.color_head.parameters(), r1.final_layer.parameters())))
    print("point head weights equal:", all(torch.equal(a, b) for a, b in zip(m.point_head.parameters(), s1.model.point_head.parameters())))
    print("out_lin max |w|:", [float(l.weight.abs().max()) for l in m.renderer.out_lin])
    m.val_cam_cond_zero_p = 0.0
    with torch.no_grad(), torch.autocast(**AMP):
        seed_all(1)
        out = m(batch, target_has_input=False, is_valid=True)
        seed_all(1)
        inp, tgt, images, rays, cam, posed, vin = s1.prepare(batch, True, False, True, False, 0.0)
        p1 = s1.pass1(images, rays, cam, vin)
        B, Vt = tgt.image.shape[:2]
        print("render_s1 equal (two calls):", torch.equal(out.render_s1, p1["render"]))
        d = (out.render - out.render_s1).abs()
        print(f"render vs render_s1: max {float(d.max()):.3e}  frac differing {float((d > 0).float().mean()):.3e}  "
              f"nan s2 {int(torch.isnan(out.render).sum())} nan s1 {int(torch.isnan(out.render_s1).sum())}")
        dp = (out.points - out.points_s1).abs()
        print(f"points vs points_s1: max {float(dp.max()):.3e}  frac differing {float((dp > 0).float().mean()):.3e}")
        # the colour head on T1 directly, both head objects
        T11 = p1["T1"][11]
        a = render_color(m.color_head, T11, 8, B, Vt, 256, 256, True)
        b = render_color(r1.final_layer, T11, 8, B, Vt, 256, 256, True)
        print("copied head on T1 == stage-1 head on T1:", torch.equal(a, b), " == pass1 render:", torch.equal(b, p1["render"]))
        x = r1.final_layer(T11[:, 4:])
        y = r1.final_layer(T11[:, 4:].contiguous())
        print("final layer on slice vs contiguous equal:", torch.equal(x, y), x.dtype)
        # zero residual on T1
        lay = geo.build_layout(tgt.alpha_mask.float(), 8, 128, 256)
        zero = torch.zeros(sum(lay.n), 768, device="cuda", dtype=torch.bfloat16)
        Tt = assemble_tokens(T11, zero, lay, 8, 8)
        print("assemble(T1, 0) == T1:", torch.equal(Tt, T11), "dtype", Tt.dtype, T11.dtype, "contig", T11.is_contiguous())
        c = render_color(m.color_head, Tt, 8, B, Vt, 256, 256, True)
        print("head on assembled == head on T1:", torch.equal(c, a))


if __name__ == "__main__":
    main()
