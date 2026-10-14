"""GPU test with the real stage-1 checkpoint (plan G4): the wrapper's stage-1 path equals
LagerNVSInRnG.forward bitwise, and the P8 stage-2 model starts exactly at stage 1.

Needs: a GPU, the ckpt_70000 of PLN2uni3t_all287k_b32t6_fp32lr35_wsd60k_a10k, the GSO val data.
Run on a GPU node:  python -m unittest tests.test_s2_init_equivalence_gpu -v
"""
import os
import random
import sys
import unittest

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

CUDA = torch.cuda.is_available()
S1_CFG = os.path.join(REPO, "configs/RnGUP_lagernvs_uni3t_b32t6_fp32lr35_wsd60k_a10k_all287k.yaml")
S2_CFG = {8: os.path.join(REPO, "configs/S2P8_uni3t70k_b32t6_lr35_a21k26k.yaml"),
          4: os.path.join(REPO, "configs/S2P4_uni3t70k_b32t6_lr35_a21k26k.yaml")}
AMP = dict(device_type="cuda", dtype=torch.bfloat16)


def load_cfg(path):
    from easydict import EasyDict as edict
    from omegaconf import OmegaConf
    return edict(OmegaConf.to_container(OmegaConf.load(path), resolve=True))


def val_batch(cfg, idx=0):
    from data.dataset_gso_ours import GSODataset_ours
    ds = GSODataset_ours(cfg)
    s = ds[idx]
    return {k: (v[None].cuda() if torch.is_tensor(v) else [v]) for k, v in s.items()}


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def tensor_hash(ts):
    return [float(t.double().sum()) for t in ts]


@unittest.skipUnless(CUDA, "needs a GPU")
class InitEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch.distributed as dist
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", str(29500 + int(os.environ.get("SLURM_JOB_ID", "7")) % 400))
            dist.init_process_group("gloo", rank=0, world_size=1)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        os.chdir(REPO)
        from model.lagernvs_wrapper import LagerNVSInRnG
        from model_s2.stage2_wrapper import Stage2LagerNVS
        cls.s1cfg = load_cfg(S1_CFG)
        cls.s2cfg = load_cfg(S2_CFG[8])
        ck = cls.s2cfg.model.stage2.stage1_ckpt
        ref = LagerNVSInRnG(cls.s1cfg)
        sd = torch.load(ck, map_location="cpu", weights_only=True, mmap=True)["model"]
        missing, unexpected = ref.load_state_dict(sd, strict=False)
        assert all(k.startswith("loss_computer.") for k in list(missing) + list(unexpected)), (missing, unexpected)
        cls.ref = ref.cuda().eval()
        cls.m8 = Stage2LagerNVS(cls.s2cfg).cuda().eval()
        cls.batch = val_batch(cls.s2cfg, 0)
        # optional: keep references to the stage-2 renderer output for failure reports.  Off by default -- keeping
        # them alive changes the allocation history, and the test must see the same memory behaviour as training
        cls.s2out = {}
        if os.environ.get("S2_TEST_KEEP_S2OUT") == "1":
            cls.m8.renderer.register_forward_hook(lambda mod, inp, out: cls.s2out.update(out=out))

    def _pair(self, zero_p, training=False, seed=123):
        self.ref.val_cam_cond_zero_p = zero_p
        self.m8.val_cam_cond_zero_p = zero_p
        with torch.no_grad(), torch.autocast(**AMP):
            seed_all(seed)
            r = self.ref(self.batch, target_has_input=False, is_valid=not training)
            seed_all(seed)
            s = self.m8(self.batch, target_has_input=False, is_valid=not training)
        return r, s

    def _report(self, s, r):
        """why the P8 output differs from stage 1 (only built when an equality check fails)."""
        d = (s.render.float() - r.render.float()).abs()
        out = self.s2out.get("out", {})
        res = {k: (bool(torch.isfinite(v).all()), float(v.float().abs().max())) for k, v in out.items()}
        lin = [(float(l.weight.abs().max()), float(l.bias.abs().max())) for l in self.m8.renderer.out_lin]
        ds1 = (s.render.float() - s.render_s1.float()).abs()
        dp = (s.points.float() - r.points.float()).abs()
        msg = (f"render max|diff| {float(d.max()):.3e} frac {float((d > 0).float().mean()):.3e} "
               f"nan {int(torch.isnan(s.render).sum())} inf {int(torch.isinf(s.render).sum())} | vs own render_s1 "
               f"max {float(ds1.max()):.3e} | points max|diff| {float(dp.max()):.3e} nan "
               f"{int(torch.isnan(s.points).sum())} | s2 residual (finite, max) {res} | Lin_m (w, b) {lin}")
        try:
            am = s.target.alpha_mask.float()
            fg = (am.view(am.shape[0], am.shape[1], 1, *d.shape[-2:]) > 0.5).expand_as(d)
            msg += f" | differing px fg {int(((d > 0) & fg).sum())} bg {int(((d > 0) & ~fg).sum())}"
        except Exception as e:                                                # noqa: BLE001
            msg += f" | fg split n/a ({type(e).__name__})"
        return msg

    def test_stage1_path_and_p8_init_equal_stage1(self):
        for zp in (0.0, 1.0):
            r, s = self._pair(zp)
            self.assertTrue(torch.equal(s.render_s1, r.render), f"zero_p {zp}: stage-1 render")
            self.assertTrue(torch.equal(s.points_s1, r.points), f"zero_p {zp}: stage-1 points")
            for a, b in zip(s.camera, r.camera):
                self.assertTrue(torch.equal(a, b), f"zero_p {zp}: camera")
            if not torch.equal(s.render, r.render):
                self.fail(f"zero_p {zp}: P8 init render != stage 1: {self._report(s, r)}")
            self.assertTrue(torch.equal(s.points, r.points), f"zero_p {zp}: P8 init points != stage 1")
            print(f"[s2-gpu] zero_p {zp}: bitwise equal (render {tuple(s.render.shape)})", flush=True)

    def test_training_mode_draw_matches(self):
        self.ref.train()
        self.m8.train()
        try:
            r, s = self._pair(0.4, training=True, seed=7)
        finally:
            self.ref.eval()
            self.m8.eval()
        self.assertTrue(torch.equal(s.render_s1, r.render))
        if not torch.equal(s.render, r.render):
            self.fail(f"training draw: P8 init render != stage 1: {self._report(s, r)}")

    def test_pass2_depth_and_tables_are_sane(self):
        from model_s2 import geometry as geo
        m = self.m8
        s1 = m.stage1
        with torch.no_grad(), torch.autocast(**AMP):
            m.val_cam_cond_zero_p = 0.0
            inp, tgt, images, rays, cam, posed, vin = s1.prepare(self.batch, True, False, True, False, 0.0)
            p1 = s1.pass1(images, rays, cam, vin)
            P_in2 = s1.pass2(p1["rec0"], inp.c2w.float(), inp.fxfycxcy.float(), 256, 256)
        D2 = geo.depth_in_camera(P_in2.float(), inp.c2w.float())
        Dgt = inp.depth_map.float().view_as(D2)
        fg = (inp.alpha_mask.view_as(D2) > 0.5) & (Dgt > 0)
        absrel = float(((D2 - Dgt).abs() / Dgt)[fg].mean())
        print(f"[s2-gpu] pass-2 input depth abs_rel {absrel:.4f}", flush=True)
        self.assertLess(absrel, 0.03)
        lay = geo.build_layout(tgt.alpha_mask.float(), 8, 128, 256)
        Ft = geo.build_forward_table(p1["points"].float(), tgt.alpha_mask.float(), lay, inp.c2w.float(),
                                     inp.fxfycxcy.float(), P_in2.float(), inp.alpha_mask.float(), 8, 37, 256, radius=2)
        S = 4 * 1369
        nonempty = [float(Ft[bv, 4:4 + lay.n[bv], :S].any(-1).float().mean()) for bv in range(lay.BV) if lay.n[bv]]
        share = float(np.mean(nonempty))
        print(f"[s2-gpu] fg target tokens with a non-empty window: {share:.3f}", flush=True)
        self.assertGreater(share, 0.8)

    def test_one_step_updates_only_stage2(self):
        from utils.training_utils import create_optimizer
        m = self.m8
        s1_params = [p for _, p in list(m.stage1.model.named_parameters())[::50]]
        before = tensor_hash(s1_params)
        # the class fixture must stay at init for the other tests (unittest runs them alphabetically)
        saved = {k: v.detach().clone() for k, v in m.state_dict().items()}
        opt, _, _ = create_optimizer(m, 0.05, 3.5e-5, (0.9, 0.95))
        m.train()
        try:
            with torch.autocast(**AMP):
                seed_all(3)
                out = m(self.batch, target_has_input=False, is_valid=False)
            loss = out.loss_metrics.loss
            self.assertTrue(bool(torch.isfinite(loss)))
            loss.backward()
            for n, p in m.named_parameters():
                if p.requires_grad:
                    self.assertIsNotNone(p.grad, n)
                    self.assertTrue(bool(torch.isfinite(p.grad).all()), n)
            # Lin_m is zero at init: its gradient must be nonzero (the chain starts there)
            self.assertGreater(float(m.renderer.out_lin[3].weight.grad.abs().max()), 0.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            self.assertGreater(float(m.renderer.out_lin[3].weight.abs().max()), 0.0)   # the step moved Lin_m
        finally:
            m.load_state_dict(saved)
            m.eval()
        self.assertEqual(before, tensor_hash(s1_params))
        self.assertTrue(all(p.dtype == torch.float32 for p in m.parameters() if p.requires_grad))

    def test_p4_forward_is_finite(self):
        from model_s2.stage2_wrapper import Stage2LagerNVS
        m4 = Stage2LagerNVS(load_cfg(S2_CFG[4])).cuda().eval()
        with torch.no_grad(), torch.autocast(**AMP):
            out = m4(self.batch, target_has_input=False, is_valid=True)
        self.assertTrue(bool(torch.isfinite(out.render).all()) and bool(torch.isfinite(out.points).all()))
        self.assertTrue(bool(torch.isfinite(out.loss_metrics.loss)))
        print(f"[s2-gpu] P4 forward ok, fg tokens/view {out.loss_metrics.s2_fg_tokens:.0f}", flush=True)
        del m4


S2_FG_CFG = {"fg_dense": os.path.join(REPO, "configs/S2P8fgdense_uni3t70k_b32t6_lr35_a21k26k.yaml"),
             "fg_dual": os.path.join(REPO, "configs/S2P8fgdual_uni3t70k_b32t6_lr35_a21k26k.yaml"),
             "s1sel_s1mask": os.path.join(REPO, "configs/S2P8fgdense_s1sel1_s1mask_uni3t70k_b32t6_lr35_a21k26k.yaml"),
             "s1sel_nomask": os.path.join(REPO, "configs/S2P8fgdense_s1sel1_nomask_uni3t70k_b32t6_lr35_a21k26k.yaml")}


@unittest.skipUnless(CUDA, "needs a GPU")
class FgStructureInitTests(unittest.TestCase):
    """fg_dense / fg_dual, block 8 (user 09-28): step 0 equals the stock stage 1 bitwise (posed and unposed); one
    AdamW step gives every trainable parameter a finite gradient, moves Lin_m and leaves stage 1 untouched.
    s1sel_*: fg_dense with the refined tiles taken from the stage-1 render (painted tiles + one ring) instead of the
    GT target mask: the tile count equals an explicit numpy recount from the stage-1 render; at the second step the
    mask channel's weights get exactly zero gradient when the channel is all zero (nomask) and a nonzero one
    otherwise."""

    @classmethod
    def setUpClass(cls):
        import torch.distributed as dist
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", str(29500 + int(os.environ.get("SLURM_JOB_ID", "7")) % 400))
            dist.init_process_group("gloo", rank=0, world_size=1)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        os.chdir(REPO)
        from model.lagernvs_wrapper import LagerNVSInRnG
        cfg = load_cfg(S2_FG_CFG["fg_dense"])
        ref = LagerNVSInRnG(load_cfg(S1_CFG))
        sd = torch.load(cfg.model.stage2.stage1_ckpt, map_location="cpu", weights_only=True, mmap=True)["model"]
        missing, unexpected = ref.load_state_dict(sd, strict=False)
        assert all(k.startswith("loss_computer.") for k in list(missing) + list(unexpected)), (missing, unexpected)
        cls.ref = ref.cuda().eval()
        cls.batch = val_batch(cfg, 0)

    def _check(self, key):
        from model_s2.stage2_wrapper import Stage2LagerNVS
        from utils.training_utils import create_optimizer
        structure = "fg_dense" if key.startswith("s1sel") else key
        m = Stage2LagerNVS(load_cfg(S2_FG_CFG[key])).cuda().eval()
        self.assertEqual(m.structure, structure)
        self.assertEqual((m.target_select, m.mask_input), {"s1sel_s1mask": ("stage1", "stage1"),
                                                           "s1sel_nomask": ("stage1", "zero")}.get(key, ("gt", "gt")))
        try:
            for zp in (0.0, 1.0):
                self.ref.val_cam_cond_zero_p = zp
                m.val_cam_cond_zero_p = zp
                with torch.no_grad(), torch.autocast(**AMP):
                    seed_all(123)
                    r = self.ref(self.batch, target_has_input=False, is_valid=True)
                    seed_all(123)
                    s = m(self.batch, target_has_input=False, is_valid=True)
                self.assertTrue(torch.equal(s.render, r.render), f"{key} zero_p {zp}: init render != stage 1")
                self.assertTrue(torch.equal(s.points, r.points), f"{key} zero_p {zp}: init points != stage 1")
                if key.startswith("s1sel"):
                    # explicit recount from the stage-1 render: pixel painted if a channel < 0.9, 8x8 tiles, one ring
                    rs = s.render_s1.float().cpu().numpy()[0]                        # [Vt, 3, H, W]
                    counts = []
                    for v in range(rs.shape[0]):
                        pt = (rs[v] < 0.9).any(0)                                    # [H, W]
                        g = pt.shape[0] // 8
                        tile = pt.reshape(g, 8, g, 8).any(axis=(1, 3))
                        grown = np.zeros_like(tile)
                        for i in range(g):
                            for j in range(g):
                                grown[i, j] = tile[max(0, i - 1):i + 2, max(0, j - 1):j + 2].any()
                        counts.append(int(grown.sum()))
                    self.assertEqual(float(s.loss_metrics.s2_fg_tokens), float(np.mean(counts)), key)
                    print(f"[s2-gpu] {key} zero_p {zp}: refined tiles per view {np.mean(counts):.1f} (explicit recount "
                          f"matches), GT pixels outside {float(s.loss_metrics.s2_sel_gt_miss):.4f}", flush=True)
                print(f"[s2-gpu] {key} zero_p {zp}: bitwise equal to stage 1, scene fg share "
                      f"{s.loss_metrics.s2_scene_fg_share:.3f}", flush=True)
            s1_params = [p for _, p in list(m.stage1.model.named_parameters())[::50]]
            before = tensor_hash(s1_params)
            opt, _, _ = create_optimizer(m, 0.05, 3.5e-5, (0.9, 0.95))
            m.train()
            with torch.autocast(**AMP):
                seed_all(3)
                out = m(self.batch, target_has_input=False, is_valid=False)
            loss = out.loss_metrics.loss
            self.assertTrue(bool(torch.isfinite(loss)))
            loss.backward()
            for n, p in m.named_parameters():
                if p.requires_grad:
                    self.assertIsNotNone(p.grad, f"{structure}: {n} has no gradient")
                    self.assertTrue(bool(torch.isfinite(p.grad).all()), f"{structure}: {n}")
            self.assertGreater(float(m.renderer.out_lin[3].weight.grad.abs().max()), 0.0)
            opt.step()
            self.assertGreater(float(m.renderer.out_lin[3].weight.abs().max()), 0.0)
            self.assertEqual(before, tensor_hash(s1_params))
            # step 2: at init the zero Lin_m outputs block every gradient upstream of them (the target embedding's
            # included -- that is what makes step 0 equal stage 1); after one step they are nonzero, so the target
            # embedding now gets a gradient: nonzero on the ray channels, and on the mask channel exactly zero when
            # that channel is all zero (nomask), nonzero when it carries a mask
            opt.zero_grad(set_to_none=True)
            with torch.autocast(**AMP):
                seed_all(4)
                out2 = m(self.batch, target_has_input=False, is_valid=False)
            out2.loss_metrics.loss.backward()
            gray = float(m.renderer.tgt_proj.weight.grad[:, :6].abs().max())
            gmask = float(m.renderer.tgt_proj.weight.grad[:, 6:].abs().max())
            self.assertGreater(gray, 0.0, key)
            if key == "s1sel_nomask":
                self.assertEqual(gmask, 0.0, "all-zero mask channel must get no gradient")
            else:
                self.assertGreater(gmask, 0.0, key)
            print(f"[s2-gpu] {key}: step-2 target-embedding gradient max |g| rays {gray:.3e}, mask channel "
                  f"{gmask:.3e}", flush=True)
            print(f"[s2-gpu] {key}: one AdamW step ok, trainable params "
                  f"{sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6:.1f}M", flush=True)
        finally:
            del m
            torch.cuda.empty_cache()

    def test_fg_dense(self):
        self._check("fg_dense")

    def test_fg_dual(self):
        self._check("fg_dual")

    def test_s1sel_s1mask(self):
        self._check("s1sel_s1mask")

    def test_s1sel_nomask(self):
        self._check("s1sel_nomask")


if __name__ == "__main__":
    unittest.main()
