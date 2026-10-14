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
from model_s2.blocks import NEW_MODULES, S2Ctx  # noqa: E402
from model_s2.fg_scene import build_fg_scene, build_fg_windows  # noqa: E402
from model_s2.heads_s2 import (assemble_tokens, build_x_prior, make_color_head, make_point_head,  # noqa: E402
                               render_color, render_points)
from model_s2.masked_attention import MaskTable  # noqa: E402
from model_s2.renderer_s2 import Stage2Renderer  # noqa: E402
from models.layers.dense_point_head import DensePointHead  # noqa: E402
from models.renderer import Renderer  # noqa: E402
from tests.test_s2_geometry import IMG, SG, TAU, make_scene  # noqa: E402

C, NH = 32, 2
STRUCTURES = ("current", "fg_dense", "fg_dual")


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

    def _run(self, patch, drop_view_fg=False, seed=0, structure="current"):
        torch.manual_seed(seed)
        r1, ph1 = tiny_stage1()
        A_t, lay, S, s_pad, Ft, Rt = s2_setup(patch, self.sc, drop_view_fg)
        r2 = Stage2Renderer(r1, patch, num_heads=NH, structure=structure)
        ch = make_color_head(r1.final_layer, patch, 8)
        ph = make_point_head(ph1, patch, 8)
        B, Vt = A_t.shape[:2]
        BV = B * Vt
        T1 = {m: torch.randn(BV, 4 + (IMG // 8) ** 2, C) for m in (2, 5, 8, 11)}
        rec_prior = torch.randn(BV, S, C)
        kw = dict(fwd_mask=MaskTable(Ft, "ref"), rev_mask=MaskTable(Rt, "ref"), rec_prior=rec_prior)
        if structure != "current":
            fgs = build_fg_scene(self.sc["A_i"], Vt, SG)
            win = build_fg_windows(Ft, Rt, lay, fgs, "ref") if structure == "fg_dual" else None
            kw = dict(fwd_mask=None, rev_mask=None, rec_prior=rec_prior.reshape(-1, C)[fgs.idx], structure=structure,
                      s_seqlens=fgs.seq, scene_idx=fgs.idx, win=win)
        ctx = S2Ctx(layout=lay, q_seqlens=[geo.N_REG + n for n in lay.n], x_prior=build_x_prior(T1[11], lay, patch, 8),
                    scene_g=SG, n_views=2, scene_block=2, s_pad=s_pad, dense_backend="ref", masked_backend="ref", **kw)
        rays = torch.randn(B, Vt, 6, IMG, IMG)
        mask1 = (A_t > 0.5).float()
        rec_rep = torch.randn(BV, S, C)
        return r1, ph1, r2, ch, ph, lay, T1, ctx, rays, mask1, rec_rep

    def test_p8_copies_and_new_init(self):
        for structure in STRUCTURES:
            r1, ph1, r2, ch, ph, *_ = self._run(8, structure=structure)
            self.assertTrue(torch.equal(r2.tgt_proj.weight[:, :6], r1.tgt_embedder.proj.weight))
            self.assertAlmostEqual(float(r2.tgt_proj.weight[:, 6:].std()), 0.02, delta=0.006)
            self.assertTrue(torch.equal(r2.registers, r1.per_view_register_tokens.float()))
            last = len(r2.blocks) - 1
            for i, (b1, b2) in enumerate(zip(r1.renderer_core.renderer_blocks, r2.blocks)):
                for k, v in b1.state_dict().items():
                    self.assertTrue(torch.equal(b2.state_dict()[k], v), f"{structure} block {i} {k}")
                want = NEW_MODULES[("final" if i == last else "bidir", structure)]
                new = {k.split(".")[0] for k in b2.state_dict() if k not in b1.state_dict()}
                self.assertEqual(new, set(want), f"{structure} block {i}: new modules")
                for name in ("inj_x", "inj_rec", "gate", "rgate"):
                    if name in want:
                        self.assertAlmostEqual(float(getattr(b2, name).weight.std()), 0.02, delta=0.004,
                                               msg=f"{structure} block {i} {name}")
                if "comp_k" in want:
                    self.assertEqual(float(b2.comp_k.act_layers[0].bias.abs().max()), 0.0)
            for lin in r2.out_lin:
                self.assertEqual(float(lin.weight.abs().max()), 0.0)
                self.assertEqual(float(lin.bias.abs().max()), 0.0)
            for k, v in ph1.state_dict().items():
                self.assertTrue(torch.equal(ph.state_dict()[k], v), k)
            params = list(r2.parameters()) + list(ch.parameters()) + list(ph.parameters())
            self.assertTrue(all(p.dtype == torch.float32 for p in params))
            self.assertTrue(all(p.requires_grad for p in params))

    def test_p8_starts_exactly_at_stage1(self):
        for structure in STRUCTURES:
            r1, ph1, r2, ch, ph, lay, T1, ctx, rays, mask1, rec_rep = self._run(8, structure=structure)
            B, Vt = rays.shape[:2]
            res = r2(rays, mask1, rec_rep, ctx)
            Ts = {m: assemble_tokens(T1[m], res[m], lay, 8, 8) for m in res}
            for m in Ts:
                self.assertTrue(torch.equal(Ts[m], T1[m]), structure)
            img2 = render_color(ch, Ts[11], 8, B, Vt, IMG, IMG, True)
            img1 = render_color(r1.final_layer, T1[11], 8, B, Vt, IMG, IMG, True)
            self.assertTrue(torch.equal(img2, img1), structure)
            p2, c2 = render_points(ph, Ts, rays, B, 4)
            p1, c1 = render_points(ph1, T1, rays, B, 4)
            self.assertTrue(torch.equal(p2, p1) and torch.equal(c2, c1), structure)

    def test_fg_scene_and_windows_brute_force(self):
        """kept scene tokens and the fg_dual window tables against explicit per-token loops"""
        import math
        A_t, lay, S, s_pad, Ft, Rt = s2_setup(8, self.sc)
        # the small scene's foreground covers every 5 x 5 scene token once dilated: clear the left half of the
        # input alpha so that tokens really are dropped (the window tables are cut from the full Ft / Rt either way)
        A_i = self.sc["A_i"].clone()
        A_i[..., : IMG // 2] = 0.0
        B, Vi = A_i.shape[:2]
        Vt = A_t.shape[1]
        g = SG
        keep = []                                               # per scene: kept flat scene indices (raster)
        for b in range(B):
            tok = torch.zeros(Vi, g, g, dtype=torch.bool)
            for v in range(Vi):
                for r in range(g):
                    for c in range(g):
                        y0, y1 = (r * IMG) // g, math.ceil((r + 1) * IMG / g)
                        x0, x1 = (c * IMG) // g, math.ceil((c + 1) * IMG / g)
                        tok[v, r, c] = bool(A_i[b, v, 0, y0:y1, x0:x1].max() > 0.5)
            dil = torch.zeros_like(tok)
            for v in range(Vi):
                for r in range(g):
                    for c in range(g):
                        dil[v, r, c] = bool(tok[v, max(0, r - 1):r + 2, max(0, c - 1):c + 2].any())
            ks = [j for j in range(Vi * g * g) if dil.view(-1)[j]]
            keep.append(ks if ks else list(range(Vi * g * g)))
        fgs = build_fg_scene(A_i, Vt, g)
        self.assertEqual(fgs.S, Vi * g * g)
        want_idx = [bv * fgs.S + j for bv in range(B * Vt) for j in keep[bv // Vt]]
        self.assertEqual(fgs.idx.tolist(), want_idx)
        self.assertEqual(fgs.seq, [len(keep[bv // Vt]) for bv in range(B * Vt)])
        self.assertTrue(0.0 < fgs.share < 1.0)
        # a scene without foreground keeps every scene token
        empty = build_fg_scene(torch.zeros_like(A_i), Vt, g)
        self.assertEqual(empty.seq, [fgs.S] * (B * Vt))
        # windows
        w = build_fg_windows(Ft, Rt, lay, fgs, "ref")
        Fg, Rn = w["f_mask"].table, w["r_mask"].table
        for bv in range(lay.BV):
            ks, n = keep[bv // Vt], lay.n[bv]
            for row in range(Fg.shape[1]):
                allowed = [kk for kk in range(len(ks)) if row >= geo.N_REG and bool(Ft[bv, row, ks[kk]])]
                got = torch.nonzero(Fg[bv, row, :len(ks)]).flatten().tolist()
                self.assertEqual(got, allowed, f"forward window bv {bv} row {row}")
                self.assertFalse(bool(Fg[bv, row, len(ks):-1].any()))
                self.assertEqual(bool(Fg[bv, row, -1]), not allowed)
            for kk in range(Rn.shape[1]):
                allowed = ([t for t in range(n) if bool(Rt[bv, ks[kk], geo.REV_FG_OFF + t])] if kk < len(ks) else [])
                got = torch.nonzero(Rn[bv, kk, :-1]).flatten().tolist()
                self.assertEqual(got, allowed, f"reverse window bv {bv} scene slot {kk}")
                self.assertEqual(bool(Rn[bv, kk, -1]), not allowed)
        # the padded layouts fg_window builds: packed row p carries the value p + 1, pad slots must be 0
        from model_s2.blocks import _scatter_pad
        BV, Lq, Ns, Kp = lay.BV, Fg.shape[1], Fg.shape[2], Rn.shape[2]
        tgt = (torch.arange(lay.T, dtype=torch.float32) + 1).view(-1, 1, 1)
        scn = (torch.arange(fgs.idx.numel(), dtype=torch.float32) + 1).view(-1, 1, 1)
        fq = _scatter_pad(tgt, None, w["f_q_dst"], BV * w["f_Lq"]).view(BV, Lq)
        fkv = _scatter_pad(scn, w["f_kv_src"], w["f_kv_dst"], BV * w["f_Lk"]).view(BV, Ns)
        rq = _scatter_pad(scn, None, w["r_q_dst"], BV * w["r_Lq"]).view(BV, Ns)
        rkv = _scatter_pad(tgt, w["r_kv_src"], w["r_kv_dst"], BV * w["r_Lk"]).view(BV, Kp)
        soff = 0
        for bv in range(BV):
            n, sq = lay.n[bv], fgs.seq[bv]
            for row in range(Lq):
                self.assertEqual(float(fq[bv, row]), lay.offsets[bv] + row + 1 if row < geo.N_REG + n else 0.0)
            for kk in range(Ns):
                want = soff + kk + 1 if kk < sq else 0.0
                self.assertEqual(float(fkv[bv, kk]), want, f"forward keys bv {bv} slot {kk}")
                self.assertEqual(float(rq[bv, kk]), want, f"reverse queries bv {bv} slot {kk}")
            for t in range(Kp):
                self.assertEqual(float(rkv[bv, t]), lay.offsets[bv] + geo.N_REG + t + 1 if t < n else 0.0)
            soff += sq
        self.assertTrue(torch.equal(w["f_out"], lay.pad_dst))
        packed_bv = torch.repeat_interleave(torch.arange(BV), torch.tensor(fgs.seq))
        packed_k = torch.cat([torch.arange(x) for x in fgs.seq])
        self.assertTrue(torch.equal(w["r_out"], packed_bv * Ns + packed_k))

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
        for patch, structure in [(p, s) for p in (8, 4) for s in STRUCTURES]:
            r1, ph1, r2, ch, ph, lay, T1, ctx, rays, mask1, rec_rep = self._run(patch, drop_view_fg=True, seed=3,
                                                                                 structure=structure)
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
                self.assertIsNotNone(p.grad, f"patch {patch} {structure}: {name} has no gradient")
                self.assertTrue(bool(torch.isfinite(p.grad).all()), f"{structure} {name}")

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



class Stage1SelectionTests(unittest.TestCase):
    """user 09-28: the refined target tiles come from the stage-1 render (tiles holding a pixel with a colour channel
    more than tau below white, grown by one tile) instead of the GT target mask.  Checked against explicit per-pixel /
    per-tile loops that do not use the functions under test."""

    def test_painted_tiles_and_layout_brute_force(self):
        gen = torch.Generator().manual_seed(7)
        B, V, H, P, tau = 2, 3, 64, 8, 0.1
        g = H // P
        render = torch.ones(B, V, 3, H, H)
        # dark blobs, single-channel tints on both sides of the threshold, one view left blank
        for b in range(B):
            for v in range(V - (1 if b == 1 else 0)):
                y0, x0 = [int(t) for t in torch.randint(0, H - 12, (2,), generator=gen)]
                render[b, v, :, y0:y0 + 11, x0:x0 + 7] = 0.3
                for _ in range(6):
                    y, x, c = [int(t) for t in torch.randint(0, H, (3,), generator=gen)]
                    render[b, v, c % 3, y, x] = 1.0 - tau + (0.03 if (y + x) % 2 else -0.03)
        painted = geo.stage1_painted(render, tau)
        self.assertEqual(tuple(painted.shape), (B, V, 1, H, H))
        for b in range(B):
            for v in range(V):
                for y in range(H):
                    for x in range(H):
                        want = any(float(render[b, v, c, y, x]) < 1.0 - tau for c in range(3))
                        self.assertEqual(bool(painted[b, v, 0, y, x] > 0.5), want, (b, v, y, x))
        for dilate in (0, 1, 2):
            sel = geo.tiles_from_painted(painted, P, dilate)
            lay = geo.build_layout(sel, P, bucket=8, img=H, tblock_px=16)
            fg_tok = lay.fg_tok.reshape(B, V, g, g)
            for b in range(B):
                for v in range(V):
                    tile = [[bool(painted[b, v, 0, i * P:(i + 1) * P, j * P:(j + 1) * P].any()) for j in range(g)]
                            for i in range(g)]
                    for i in range(g):
                        for j in range(g):
                            want = any(tile[ii][jj] for ii in range(max(0, i - dilate), min(g, i + dilate + 1))
                                       for jj in range(max(0, j - dilate), min(g, j + dilate + 1)))
                            blk = sel[b, v, 0, i * P:(i + 1) * P, j * P:(j + 1) * P]
                            self.assertTrue(bool((blk == float(want)).all()), (dilate, b, v, i, j))
                            self.assertEqual(bool(fg_tok[b, v, i, j]), want, (dilate, b, v, i, j))
                    self.assertEqual(lay.n[b * V + v], int(fg_tok[b, v].sum()))
        self.assertEqual(lay.n[1 * V + V - 1], 0)                  # the blank view selects nothing

    def test_wrapper_rejects_stage1_selection_outside_fg_dense(self):
        from easydict import EasyDict as edict
        from omegaconf import OmegaConf
        from model_s2.stage2_wrapper import Stage2LagerNVS
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cfg = edict(OmegaConf.to_container(OmegaConf.load(
            os.path.join(root, "configs/S2P8fgdense_s1sel1_s1mask_uni3t70k_b32t6_lr35_a21k26k.yaml")), resolve=True))
        cfg.model.stage2.structure = "fg_dual"
        with self.assertRaises(ValueError):                         # raised before stage 1 is loaded
            Stage2LagerNVS(cfg)
        cfg.model.stage2.structure = "fg_dense"
        cfg.model.stage2.mask_input = "none"
        with self.assertRaises(ValueError):
            Stage2LagerNVS(cfg)


if __name__ == "__main__":
    unittest.main()
