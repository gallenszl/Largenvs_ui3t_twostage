"""CPU tests of the stage-2 modules on a tiny stage-1 renderer (plan G3).

Run:  python -m unittest tests.test_s2_modules -v
"""
import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_s2 import geometry as geo  # noqa: E402
from model_s2.blocks import S2Ctx  # noqa: E402
from model_s2.heads_s2 import (assemble_tokens, build_x_prior, make_color_head, make_point_head,  # noqa: E402
                               render_color, render_points)
from model_s2.masked_attention import MaskTable  # noqa: E402
from model_s2.renderer_s2 import Stage2Renderer  # noqa: E402
from models.layers.dense_point_head import DensePointHead  # noqa: E402
from models.renderer import Renderer  # noqa: E402
from tests.test_s2_geometry import IMG, SG, TAU, make_scene  # noqa: E402

C, NH = 32, 2


def tiny_stage1():
    torch.manual_seed(0)
    r = Renderer(12, C, 8, NH, attention_to_features_type="bidirectional_cross_attention")
    with torch.no_grad():                                   # a trained-looking stage 1 (nonzero head)
        r.final_layer.linear.weight.normal_(0, 0.05)
        r.per_view_register_tokens.normal_(0, 0.1)
    ph = DensePointHead(dim_in=C, patch_size=8, intermediate_layer_idx=[2, 5, 8, 11])
    r.requires_grad_(False)
    ph.requires_grad_(False)
    return r, ph


def s2_setup(patch, sc, drop_view_fg=False):
    A_t = sc["A_t"].clone()
    if drop_view_fg:
        A_t[0, 1] = 0.0                                     # one target view without foreground
    lay = geo.build_layout(A_t, patch, bucket=8, img=IMG, tblock_px=16)
    S = 2 * SG * SG
    s_pad = geo.roundup(S + 1, 8)
    Ft = geo.build_forward_table(sc["P_t"].float(), A_t.float(), lay, sc["c2w_i"].float(), sc["K_i"].float(),
                                 sc["P_i"].float(), sc["A_i"].float(), patch, SG, IMG, radius=1, tau=TAU, s_pad=s_pad)
    Rt = geo.build_reverse_table(sc["P_i"].float(), sc["A_i"].float(), lay, sc["c2w_t"].float(), sc["K_t"].float(),
                                 sc["P_t"].float(), A_t.float(), patch, SG, IMG, radius=1, tau=TAU, s_pad=s_pad)
    return A_t, lay, S, s_pad, Ft, Rt


class ModuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.sc = make_scene(2, B=1, Vt=2, Vi=2)
        import torch.distributed as dist
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29731")
            dist.init_process_group("gloo", rank=0, world_size=1)

    def _run(self, patch, drop_view_fg=False, seed=0):
        torch.manual_seed(seed)
        r1, ph1 = tiny_stage1()
        A_t, lay, S, s_pad, Ft, Rt = s2_setup(patch, self.sc, drop_view_fg)
        r2 = Stage2Renderer(r1, patch, num_heads=NH)
        ch = make_color_head(r1.final_layer, patch, 8)
        ph = make_point_head(ph1, patch, 8)
        B, Vt = A_t.shape[:2]
        BV = B * Vt
        T1 = {m: torch.randn(BV, 4 + (IMG // 8) ** 2, C) for m in (2, 5, 8, 11)}
        ctx = S2Ctx(layout=lay, q_seqlens=[geo.N_REG + n for n in lay.n], fwd_mask=MaskTable(Ft, "ref"),
                    rev_mask=MaskTable(Rt, "ref"), x_prior=build_x_prior(T1[11], lay, patch, 8),
                    rec_prior=torch.randn(BV, S, C), scene_g=SG, n_views=2, scene_block=2, s_pad=s_pad,
                    dense_backend="ref", masked_backend="ref")
        rays = torch.randn(B, Vt, 6, IMG, IMG)
        mask1 = (A_t > 0.5).float()
        rec_rep = torch.randn(BV, S, C)
        return r1, ph1, r2, ch, ph, lay, T1, ctx, rays, mask1, rec_rep

    def test_p8_copies_and_new_init(self):
        r1, ph1, r2, ch, ph, *_ = self._run(8)
        self.assertTrue(torch.equal(r2.tgt_proj.weight[:, :6], r1.tgt_embedder.proj.weight))
        self.assertAlmostEqual(float(r2.tgt_proj.weight[:, 6:].std()), 0.02, delta=0.006)
        self.assertTrue(torch.equal(r2.registers, r1.per_view_register_tokens.float()))
        for i, (b1, b2) in enumerate(zip(r1.renderer_core.renderer_blocks, r2.blocks)):
            for k, v in b1.state_dict().items():
                self.assertTrue(torch.equal(b2.state_dict()[k], v), f"block {i} {k}")
            self.assertAlmostEqual(float(b2.gate.weight.std()), 0.02, delta=0.004)
            self.assertAlmostEqual(float(b2.inj_x.weight.std()), 0.02, delta=0.004)
            self.assertEqual(float(b2.comp_k.act_layers[0].bias.abs().max()), 0.0)
        for lin in r2.out_lin:
            self.assertEqual(float(lin.weight.abs().max()), 0.0)
            self.assertEqual(float(lin.bias.abs().max()), 0.0)
        for k, v in ph1.state_dict().items():
            self.assertTrue(torch.equal(ph.state_dict()[k], v), k)
        self.assertTrue(all(p.dtype == torch.float32 for p in list(r2.parameters()) + list(ch.parameters()) + list(ph.parameters())))
        self.assertTrue(all(p.requires_grad for p in list(r2.parameters()) + list(ch.parameters()) + list(ph.parameters())))

    def test_p8_starts_exactly_at_stage1(self):
        r1, ph1, r2, ch, ph, lay, T1, ctx, rays, mask1, rec_rep = self._run(8)
        B, Vt = rays.shape[:2]
        res = r2(rays, mask1, rec_rep, ctx)
        Ts = {m: assemble_tokens(T1[m], res[m], lay, 8, 8) for m in res}
        for m in Ts:
            self.assertTrue(torch.equal(Ts[m], T1[m]))
        img2 = render_color(ch, Ts[11], 8, B, Vt, IMG, IMG, True)
        img1 = render_color(r1.final_layer, T1[11], 8, B, Vt, IMG, IMG, True)
        self.assertTrue(torch.equal(img2, img1))
        p2, c2 = render_points(ph, Ts, rays, B, 4)
        p1, c1 = render_points(ph1, T1, rays, B, 4)
        self.assertTrue(torch.equal(p2, p1) and torch.equal(c2, c1))

    def test_p4_shapes_and_background(self):
        r1, ph1, r2, ch, ph, lay, T1, ctx, rays, mask1, rec_rep = self._run(4)
        B, Vt = rays.shape[:2]
        self.assertEqual(lay.g, IMG // 4)
        res = r2(rays, mask1, rec_rep, ctx)
        Ts = {m: assemble_tokens(T1[m], res[m], lay, 4, 8) for m in res}
        g = lay.g
        self.assertEqual(tuple(Ts[11].shape), (B * Vt, g * g, C))
        # background children equal their replicated stage-1 parent
        for bv in range(B * Vt):
            bg = torch.nonzero(~lay.fg_tok[bv]).flatten()[:5]
            for t in bg.tolist():
                r, c = divmod(t, g)
                parent = 4 + (r // 2) * (IMG // 8) + (c // 2)
                self.assertTrue(torch.equal(Ts[11][bv, t], T1[11][bv, parent]))
        img = render_color(ch, Ts[11], 4, B, Vt, IMG, IMG, False)
        self.assertEqual(float((img - 0.5).abs().max()), 0.0)          # zero-init colour head: grey 0.5
        pts, conf = render_points(ph, Ts, rays, B, 0)
        self.assertEqual(tuple(pts.shape), (B, Vt, 3, IMG, IMG))
        self.assertTrue(bool(torch.isfinite(pts).all()) and bool(torch.isfinite(conf).all()))

    def test_every_trainable_param_gets_a_gradient(self):
        for patch in (8, 4):
            r1, ph1, r2, ch, ph, lay, T1, ctx, rays, mask1, rec_rep = self._run(patch, drop_view_fg=True, seed=3)
            self.assertIn(0, lay.n)                                   # a view without foreground
            r2.train()
            B, Vt = rays.shape[:2]
            # make the zero-init output layers nonzero so gradients reach every upstream parameter
            with torch.no_grad():
                for lin in r2.out_lin:
                    lin.weight.normal_(0, 0.02)
            res = r2(rays, mask1, rec_rep, ctx)
            Ts = {m: assemble_tokens(T1[m], res[m], lay, patch, 8) for m in res}
            img = render_color(ch, Ts[11], patch, B, Vt, IMG, IMG, patch == 8)
            pts, conf = render_points(ph, Ts, rays, B, 4 if patch == 8 else 0)
            (img.square().mean() + pts.square().mean() + conf.mean()).backward()
            for name, p in list(r2.named_parameters()) + list(ch.named_parameters()) + list(ph.named_parameters()):
                self.assertIsNotNone(p.grad, f"patch {patch}: {name} has no gradient")
                self.assertTrue(bool(torch.isfinite(p.grad).all()), name)

    def test_weight_decay_groups(self):
        from utils.training_utils import create_optimizer
        r1, ph1, r2, ch, ph, *_ = self._run(8)
        model = nn.ModuleDict(dict(renderer=r2, color_head=ch, point_head=ph))
        opt, optd, _ = create_optimizer(model, 0.05, 3.5e-5, (0.9, 0.95))
        decay = {id(p) for p in opt.param_groups[0]["params"]}
        for name, p in model.named_parameters():
            self.assertEqual(id(p) in decay, p.dim() > 1, name)
        self.assertEqual(opt.param_groups[0]["weight_decay"], 0.05)
        self.assertEqual(opt.param_groups[1]["weight_decay"], 0.0)


if __name__ == "__main__":
    unittest.main()
