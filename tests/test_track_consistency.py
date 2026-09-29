"""CPU tests of model/track_consistency.py against an independent numpy reference.

Scene: a sphere seen by 4 cameras.  GT depth is the analytic ray-sphere intersection through each
pixel centre; GT points are its unprojection (pixel-centre convention).  The reference correspondences
and loss are computed with plain numpy loops written from the definition, never with the module.
"""
import os
import sys
import unittest

import numpy as np
import torch
from easydict import EasyDict as edict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.track_consistency import build_tracks, sample_tracks, track_consistency_loss  # noqa: E402

H = W = 32
F = 30.0
CX = CY = 16.0          # continuous principal point (image centre)
RADIUS = 0.6            # apparent radius 30 * 0.6 / 1.6 = 11.25 px: the sphere fills most of the image
DIST = 1.6
TAU = 0.01
NB = {"exact": [(0, 0)], "omega4": [(0, 0), (0, 1), (1, 0), (1, 1)], "centre4": [(0, 0), (0, 1), (1, 0), (1, 1)]}


def look_at(cam_pos):
    """OpenCV camera (x right, y down, z forward) at cam_pos looking at the origin -> c2w [4, 4]."""
    z = -cam_pos / np.linalg.norm(cam_pos)
    up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(z, up)) > 0.99:
        up = np.array([1.0, 0.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, cam_pos
    return c2w


def render_depth(c2w):
    """z-depth of the sphere through every pixel centre; 0 where the ray misses."""
    ys, xs = np.mgrid[:H, :W]
    d = np.stack([(xs + 0.5 - CX) / F, (ys + 0.5 - CY) / F, np.ones_like(xs, dtype=np.float64)], -1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    w2c = np.linalg.inv(c2w)
    c = w2c[:3, 3]                       # sphere centre (world origin) in camera coords
    b = -2.0 * (d @ c)                   # |s d - c|^2 = r^2  ->  s^2 + b s + (|c|^2 - r^2) = 0
    cc = c @ c - RADIUS ** 2
    disc = b * b - 4.0 * cc
    hit = disc > 0
    s = np.where(hit, (-b - np.sqrt(np.clip(disc, 0, None))) / 2.0, 0.0)
    return np.where(hit, s * d[..., 2], 0.0)


def unproject(depth, c2w):
    ys, xs = np.mgrid[:H, :W]
    pc = np.stack([(xs + 0.5 - CX) / F * depth, (ys + 0.5 - CY) / F * depth, depth], -1)
    P = pc @ c2w[:3, :3].T + c2w[:3, 3]
    return np.where((depth > 0)[..., None], P, 0.0)


def make_scene(azimuths=(0.0, 35.0, 110.0, 200.0), elev=20.0):
    c2ws, depths, points = [], [], []
    for az in azimuths:
        a, e = np.radians(az), np.radians(elev)
        pos = DIST * np.array([np.cos(e) * np.sin(a), np.sin(e), np.cos(e) * np.cos(a)])
        c2w = look_at(pos)
        depth = render_depth(c2w)
        c2ws.append(c2w)
        depths.append(depth)
        points.append(unproject(depth, c2w))
    K = np.tile(np.array([F, F, CX, CY]), (len(azimuths), 1))
    return np.stack(c2ws), np.stack(depths), np.stack(points), K


def reference_tracks(depths, points, c2ws, K, tau=TAU, neighbours="omega4", border=1):
    """numpy loops from the definition: for every (a, b, pixel of a) the matched flat pixel in b and validity."""
    V = depths.shape[0]
    idx = np.zeros((V, V, H, W), dtype=np.int64)
    valid = np.zeros((V, V, H, W), dtype=bool)
    depth_rejected = 0
    for a in range(V):
        for b in range(V):
            if a == b:
                continue
            w2c = np.linalg.inv(c2ws[b])
            for y in range(H):
                for x in range(W):
                    if depths[a, y, x] <= 0:
                        continue
                    pc = w2c[:3, :3] @ points[a, y, x] + w2c[:3, 3]
                    if pc[2] <= 1e-6:
                        continue
                    u = K[b, 0] * pc[0] / pc[2] + K[b, 2]
                    v = K[b, 1] * pc[1] / pc[2] + K[b, 3]
                    if neighbours == "centre4":
                        u, v = u - 0.5, v - 0.5
                    bx, by = int(np.floor(u)), int(np.floor(v))
                    if not (border <= bx <= W - 1 - border and border <= by <= H - 1 - border):
                        continue
                    best = None
                    for dy, dx in NB[neighbours]:
                        xi, yi = min(bx + dx, W - 1), min(by + dy, H - 1)
                        D = depths[b, yi, xi]
                        if D <= 0:
                            continue
                        gap = abs(pc[2] - D)
                        if best is None or gap < best[0]:
                            best = (gap, D, yi * W + xi)
                    if best is None:
                        continue
                    gap, D, flat = best
                    idx[a, b, y, x] = flat
                    if gap < tau * D and gap < tau * pc[2]:
                        valid[a, b, y, x] = True
                    else:
                        depth_rejected += 1
    return idx, valid, depth_rejected


def reference_loss(pred, points, idx, valid):
    """mean over valid entries x 3 coordinates of |(pred_a - gt_a) - (pred_b - gt_b)|."""
    V = pred.shape[0]
    tot, n = 0.0, 0
    for a in range(V):
        for b in range(V):
            for y in range(H):
                for x in range(W):
                    if not valid[a, b, y, x]:
                        continue
                    fb = idx[a, b, y, x]
                    ea = pred[a, y, x] - points[a, y, x]
                    eb = pred[b, fb // W, fb % W] - points[b, fb // W, fb % W]
                    tot += np.abs(ea - eb).sum()
                    n += 1
    return tot / (3 * n), n


def as_torch(c2ws, depths, points, K):
    return (torch.tensor(points, dtype=torch.float32).permute(0, 3, 1, 2)[None],
            torch.tensor(depths, dtype=torch.float32)[None],
            torch.tensor(c2ws, dtype=torch.float32)[None],
            torch.tensor(K, dtype=torch.float32)[None])


class TrackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c2ws, cls.depths, cls.points, cls.K = make_scene()
        cls.pts_gt, cls.depth_t, cls.c2w_t, cls.K_t = as_torch(cls.c2ws, cls.depths, cls.points, cls.K)

    def test_scene_is_not_trivial(self):
        fg = (self.depths > 0).sum(axis=(1, 2))
        self.assertTrue((fg > 80).all(), fg)
        _, valid, rejected = reference_tracks(self.depths, self.points, self.c2ws, self.K)
        self.assertGreater(valid.sum(), 500)
        self.assertGreater(rejected, 50, "the depth test never rejects anything: no occlusion in the scene")

    def test_tracks_match_reference_all_neighbour_sets(self):
        for nb in ("exact", "omega4", "centre4"):
            ref_idx, ref_valid, _ = reference_tracks(self.depths, self.points, self.c2ws, self.K, neighbours=nb)
            idx, valid = build_tracks(self.pts_gt, self.depth_t, self.c2w_t, self.K_t, tau=TAU, neighbours=nb, border=1)
            valid = valid[0].numpy()
            idx = idx[0].numpy()
            self.assertEqual(int((valid != ref_valid).sum()), 0, nb)
            self.assertEqual(int((idx[ref_valid] != ref_idx[ref_valid]).sum()), 0, nb)

    def test_border_and_query_views(self):
        _, v1 = build_tracks(self.pts_gt, self.depth_t, self.c2w_t, self.K_t, border=1)
        _, v8 = build_tracks(self.pts_gt, self.depth_t, self.c2w_t, self.K_t, border=8)   # 8: reaches the sphere
        self.assertEqual(int((v8 & ~v1).sum()), 0)
        self.assertLess(int(v8.sum()), int(v1.sum()))
        _, va = build_tracks(self.pts_gt, self.depth_t, self.c2w_t, self.K_t)
        _, vf = build_tracks(self.pts_gt, self.depth_t, self.c2w_t, self.K_t, query_views="first")
        self.assertEqual(int(vf[:, 1:].sum()), 0)
        self.assertTrue(torch.equal(vf[:, 0], va[:, 0]))

    def test_loss_matches_reference_and_invariances(self):
        rng = np.random.default_rng(0)
        pred = self.points + 0.02 * rng.standard_normal(self.points.shape)
        ref_idx, ref_valid, _ = reference_tracks(self.depths, self.points, self.c2ws, self.K, neighbours="centre4", border=4)
        ref, n = reference_loss(pred, self.points, ref_idx, ref_valid)   # module defaults: centre4, border 4
        pe = torch.tensor(pred, dtype=torch.float32).permute(0, 3, 1, 2)[None].requires_grad_(True)
        out = track_consistency_loss(pe, self.pts_gt, self.depth_t, self.c2w_t, self.K_t, tau=TAU)
        self.assertAlmostEqual(float(out["loss_consistency"]), ref, places=5)
        self.assertEqual(int(out["consistency_valid"]), n)
        out["loss_consistency"].backward()
        self.assertTrue(torch.isfinite(pe.grad).all() and float(pe.grad.abs().sum()) > 0)
        # perfect prediction -> 0
        z = track_consistency_loss(self.pts_gt.clone(), self.pts_gt, self.depth_t, self.c2w_t, self.K_t)
        self.assertEqual(float(z["loss_consistency"]), 0.0)
        # the same error in every view -> 0 (common-mode invariance)
        shift = self.pts_gt + torch.tensor([0.05, -0.02, 0.03]).view(1, 1, 3, 1, 1)
        s = track_consistency_loss(shift, self.pts_gt, self.depth_t, self.c2w_t, self.K_t)
        self.assertLess(float(s["loss_consistency"]), 1e-6)
        # one view shifted -> positive, equal to the reference
        one = self.points.copy()
        one[1] += np.array([0.03, 0.0, 0.0]) * (self.depths[1] > 0)[..., None]
        ref1, _ = reference_loss(one, self.points, ref_idx, ref_valid)
        o1 = track_consistency_loss(torch.tensor(one, dtype=torch.float32).permute(0, 3, 1, 2)[None],
                                    self.pts_gt, self.depth_t, self.c2w_t, self.K_t)
        self.assertGreater(float(o1["loss_consistency"]), 0.0)
        self.assertAlmostEqual(float(o1["loss_consistency"]), ref1, places=5)

    def test_sampling(self):
        _, valid = build_tracks(self.pts_gt, self.depth_t, self.c2w_t, self.K_t)
        fg = self.depth_t > 0
        g = torch.Generator().manual_seed(0)
        k = 40
        for scheme in ("omega", "uniform"):
            sel = sample_tracks(valid, fg, k, scheme, g)
            self.assertEqual(int((sel & ~valid).sum()), 0)                  # only ever removes entries
            chosen = sel.any(2)[0]                                           # [Va, H, W] query pixels kept
            count = valid.sum(2)[0]                                          # [Va, H, W] other views seeing the pixel
            for a in range(chosen.shape[0]):
                pixels = chosen[a] & fg[0, a]
                self.assertLessEqual(int(pixels.sum()), k)
                if scheme == "uniform":
                    self.assertTrue(bool((count[a][pixels] >= 1).all()))
                else:
                    n_fg = int(fg[0, a].sum())
                    if n_fg // 2 > k:            # the reference keeps the top half by visible count
                        pool = torch.sort(count[a][fg[0, a]], descending=True).values[: n_fg // 2]
                        self.assertTrue(bool((count[a][pixels] >= pool.min()).all()))
        dense = track_consistency_loss(self.pts_gt.clone(), self.pts_gt, self.depth_t, self.c2w_t, self.K_t)
        sub = track_consistency_loss(self.pts_gt.clone(), self.pts_gt, self.depth_t, self.c2w_t, self.K_t,
                                     num_tracks=k, generator=g)
        self.assertLess(int(sub["consistency_valid"]), int(dense["consistency_valid"]))

    def test_agrees_with_stage2_geometry(self):
        """Same pixel choice as model_s2/geometry.py (stage-2 tables) on GT: index-unit round == continuous floor;
        the visibility masks differ only where the two tolerance formulas differ (|z - D| between tau*z and tau*D)."""
        from model_s2 import geometry as g2
        V = self.pts_gt.shape[1]
        HW = H * W
        X = self.pts_gt.permute(0, 1, 3, 4, 2).reshape(1, V, 1, HW, 3)
        u, v, z = g2.project(X, self.c2w_t[:, None], self.K_t[:, None])           # [1, Va, Vb, HW]
        ins = g2.inside_image(u, v, z, H, W)
        vi, ui = g2.nearest_pixel(u, v, H, W)
        pix = vi * W + ui
        idx, valid = build_tracks(self.pts_gt, self.depth_t, self.c2w_t, self.K_t, tau=TAU, neighbours="exact", border=0)
        idx, valid = idx.reshape(1, V, V, HW), valid.reshape(1, V, V, HW)
        fgq = (self.depth_t > 0).reshape(1, V, 1, HW)
        D = self.depth_t.reshape(1, V, HW)[torch.arange(1)[:, None, None, None], torch.arange(V)[None, None, :, None], pix]
        eye = torch.eye(V, dtype=torch.bool)[None, :, :, None]
        care = ins & fgq & (D > 0) & ~eye          # build_tracks only records a pixel where b's depth is foreground
        self.assertGreater(int(care.sum()), 1000)
        self.assertEqual(int(((idx != pix) & care).sum()), 0)                      # identical pixel choice
        vis2 = care & ((z - D).abs() <= TAU * D)                                    # stage-2 formula (tables use tau 0.03)
        gap = (z - D).abs()
        band = (gap >= TAU * z) & (gap <= TAU * D)                                  # where the two formulas may differ
        self.assertEqual(int(((vis2 ^ valid) & ~band).sum()), 0)
        self.assertGreater(int((vis2 & valid).sum()), 0.99 * int(vis2.sum()))

    def test_min_valid_and_nan_guard(self):
        pred = self.pts_gt.clone()
        pred[0, 0, 0, 5, 5] = float("inf")
        out = track_consistency_loss(pred, self.pts_gt, self.depth_t, self.c2w_t, self.K_t)
        self.assertTrue(torch.isfinite(out["loss_consistency"]))
        gated = track_consistency_loss(pred, self.pts_gt, self.depth_t, self.c2w_t, self.K_t, min_valid=10 ** 9)
        self.assertEqual(float(gated["loss_consistency"]), 0.0)
        self.assertTrue(gated["loss_consistency"].requires_grad is False or gated["loss_consistency"].grad_fn is not None)


class WiringTests(unittest.TestCase):
    """MultiTaskLossComputer: no key -> bitwise unchanged; key -> weight x loss_consistency added."""

    def _inputs(self):
        torch.manual_seed(0)
        c2ws, depths, points, K = make_scene()
        pts_gt, depth_t, c2w_t, K_t = as_torch(c2ws, depths, points, K)
        b, v = 1, pts_gt.shape[1]
        rendering = torch.rand(b, v, 3, H, W)
        target = torch.rand(b, v, 3, H, W)
        pts_est = pts_gt + 0.02 * torch.randn_like(pts_gt)
        pts_conf = torch.rand(b, v, 1, H, W) + 1.0
        eye = torch.eye(4).view(1, 1, 4, 4).repeat(b, v, 1, 1)
        intr = torch.eye(3).view(1, 1, 3, 3).repeat(b, v, 1, 1)
        intr[..., 0, 0] = intr[..., 1, 1] = 100.0
        pose_enc_list = [torch.randn(b, v, 9) for _ in range(4)]
        return dict(rendering=rendering, target=target, exclude_bg=False, pts_est=pts_est, pts_conf=pts_conf,
                    pts_gt=pts_gt, pose_enc_list=pose_enc_list, extrinsics=eye, intrinsics=intr,
                    image_hw=(H, W)), dict(target_depth=depth_t, target_c2w=c2w_t, target_K=K_t)

    @staticmethod
    def _config(weight_consistency=None, **consistency):
        tr = dict(l2_loss_weight=1.0, lpips_loss_weight=0.0, perceptual_loss_weight=0.0,
                  weight_camera=0.0, weight_point=0.2)
        if weight_consistency is not None:
            tr["weight_consistency"] = weight_consistency
        if consistency:
            tr["consistency"] = consistency
        return edict(training=edict(tr))

    def test_no_key_is_bitwise_unchanged_and_key_adds_weighted_term(self):
        from model.loss import MultiTaskLossComputer
        base_in, extra = self._inputs()
        torch.manual_seed(1)
        ref = MultiTaskLossComputer(self._config())(**base_in)
        torch.manual_seed(1)
        same = MultiTaskLossComputer(self._config())(**base_in, **extra)
        self.assertEqual(set(ref.keys()), set(same.keys()))
        self.assertTrue(torch.equal(ref["loss"], same["loss"]))
        torch.manual_seed(1)
        on = MultiTaskLossComputer(self._config(weight_consistency=0.2, tau=TAU))(**base_in, **extra)
        self.assertIn("loss_consistency", on)
        self.assertGreater(float(on["loss_consistency"]), 0.0)
        self.assertAlmostEqual(float(on["loss"]), float(ref["loss"]) + 0.2 * float(on["loss_consistency"]), places=5)


if __name__ == "__main__":
    unittest.main()
