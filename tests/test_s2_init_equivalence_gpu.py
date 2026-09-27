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
        assert not unexpected and all(k.startswith("loss_computer.") for k in missing)
        cls.ref = ref.cuda().eval()
        cls.m8 = Stage2LagerNVS(cls.s2cfg).cuda().eval()
        cls.batch = val_batch(cls.s2cfg, 0)

    def _pair(self, zero_p, training=False, seed=123):
        self.ref.val_cam_cond_zero_p = zero_p
        self.m8.val_cam_cond_zero_p = zero_p
        with torch.no_grad(), torch.autocast(**AMP):
            seed_all(seed)
            r = self.ref(self.batch, target_has_input=False, is_valid=not training)
            seed_all(seed)
            s = self.m8(self.batch, target_has_input=False, is_valid=not training)
        return r, s

    def test_stage1_path_and_p8_init_equal_stage1(self):
        for zp in (0.0, 1.0):
            r, s = self._pair(zp)
            self.assertTrue(torch.equal(s.render_s1, r.render), f"zero_p {zp}: stage-1 render")
            self.assertTrue(torch.equal(s.points_s1, r.points), f"zero_p {zp}: stage-1 points")
            for a, b in zip(s.camera, r.camera):
                self.assertTrue(torch.equal(a, b), f"zero_p {zp}: camera")
            self.assertTrue(torch.equal(s.render, r.render), f"zero_p {zp}: P8 init render != stage 1")
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
        self.assertTrue(torch.equal(s.render, r.render))

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
        finally:
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


if __name__ == "__main__":
    unittest.main()
