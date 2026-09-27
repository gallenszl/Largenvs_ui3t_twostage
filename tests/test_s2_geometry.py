"""CPU tests for model_s2/geometry.py (plan G1).

The references below are written independently with numpy float64 per-pixel loops; they share only the
documented conventions (pixel centres, index-unit projection, token formula, 3 % depth test), never the
code under test.  Run from the repo root:  python -m unittest tests.test_s2_geometry -v
"""
import math
import os
import sys
import unittest

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_s2 import geometry as geo  # noqa: E402

IMG, PATCH, SG = 32, 4, 5          # small scene: 32 px, target grid 8, scene grid 5
TAU = 0.03


def look_at(eye, target=(0.0, 0.0, 0.0)):
    """OpenCV c2w looking from eye at target (x right, y down, z forward)."""
    eye = np.asarray(eye, dtype=np.float64)
    fwd = np.asarray(target, dtype=np.float64) - eye
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, -1.0, 0.0])
    right = np.cross(fwd, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, down, fwd, eye
    return c2w


SPHERES = [((0.0, 0.0, 0.0), 0.35), ((0.25, -0.1, -0.35), 0.15)]


def render_points(c2w, K, H=IMG, W=IMG):
    """ray-cast the spheres: world points [3, H, W] (0 on background) and alpha [1, H, W]."""
    fx, fy, cx, cy = K
    P = np.zeros((3, H, W))
    A = np.zeros((1, H, W))
    o = c2w[:3, 3]
    for y in range(H):
        for x in range(W):
            d = c2w[:3, :3] @ np.array([(x + 0.5 - cx) / fx, (y + 0.5 - cy) / fy, 1.0])
            d /= np.linalg.norm(d)
            best = None
            for c, r in SPHERES:
                oc = o - np.asarray(c)
                bq = np.dot(oc, d)
                disc = bq * bq - (np.dot(oc, oc) - r * r)
                if disc < 0:
                    continue
                tt = -bq - math.sqrt(disc)
                if tt > 0 and (best is None or tt < best):
                    best = tt
            if best is not None:
                P[:, y, x] = o + best * d
                A[0, y, x] = 1.0
    return P, A


def make_scene(seed, B=2, Vt=2, Vi=2, noise=0.004):
    rng = np.random.default_rng(seed)
    K = np.array([36.0, 36.0, IMG / 2, IMG / 2])
    c2w_t, c2w_i, P_t, A_t, P_i, A_i = [], [], [], [], [], []
    for b in range(B):
        ct, ci, pt, at, pi, ai = [], [], [], [], [], []
        for v in range(Vt + Vi):
            az, el = rng.uniform(0, 2 * math.pi), rng.uniform(-0.6, 0.6)
            rad = rng.uniform(1.2, 1.6)
            eye = rad * np.array([math.cos(el) * math.cos(az), math.sin(el), math.cos(el) * math.sin(az)])
            c = look_at(eye, rng.normal(0, 0.03, 3))
            P, A = render_points(c, K)
            P = P + rng.normal(0, noise, P.shape) * A          # "predicted" points: GT + noise on fg
            if v < Vt:
                ct.append(c); pt.append(P); at.append(A)
            else:
                ci.append(c); pi.append(P); ai.append(A)
        c2w_t.append(ct); c2w_i.append(ci); P_t.append(pt); A_t.append(at); P_i.append(pi); A_i.append(ai)
    T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64)
    Kt = torch.tensor(K, dtype=torch.float64).view(1, 1, 4)
    return dict(c2w_t=T(c2w_t), c2w_i=T(c2w_i), P_t=T(P_t), A_t=T(A_t), P_i=T(P_i), A_i=T(A_i),
                K_t=Kt.expand(B, Vt, 4).clone(), K_i=Kt.expand(B, Vi, 4).clone(), Knp=K)


def ref_project(X, c2w, K):
    pc = c2w[:3, :3].T @ (X - c2w[:3, 3])
    z = pc[2]
    zs = 1e-9 if abs(z) < 1e-9 else z
    return K[0] * pc[0] / zs + K[2] - 0.5, K[1] * pc[1] / zs + K[3] - 0.5, z


def ref_tok(c, g):
    return int(min(max(math.floor((c + 0.5) * g / IMG), 0), g - 1))


def ref_depth(P, c2w, y, x):
    return (c2w[:3, :3].T @ (P[:, y, x] - c2w[:3, 3]))[2]


def ref_forward(sc, radius):
    """dict (b, t, token) -> set of allowed scene columns."""
    B, Vt = sc["P_t"].shape[:2]
    Vi = sc["P_i"].shape[1]
    g = IMG // PATCH
    Pt, Pi = sc["P_t"].numpy(), sc["P_i"].numpy()
    At, Ai = sc["A_t"].numpy(), sc["A_i"].numpy()
    ct, ci, K = sc["c2w_t"].numpy(), sc["c2w_i"].numpy(), sc["Knp"]
    out = {}
    for b in range(B):
        for t in range(Vt):
            for y in range(IMG):
                for x in range(IMG):
                    if At[b, t, 0, y, x] <= 0.5:
                        continue
                    tok = (y // PATCH) * g + (x // PATCH)
                    s = out.setdefault((b, t, tok), set())
                    X = Pt[b, t, :, y, x]
                    for k in range(Vi):
                        u, v, z = ref_project(X, ci[b, k], K)
                        if not (u > -0.5 and u < IMG - 0.5 and v > -0.5 and v < IMG - 0.5 and z > 0):
                            continue
                        ui = int(min(max(np.round(u), 0), IMG - 1))
                        vi = int(min(max(np.round(v), 0), IMG - 1))
                        D = ref_depth(Pi[b, k], ci[b, k], vi, ui)
                        if not (Ai[b, k, 0, vi, ui] > 0.5 and D > 0 and abs(z - D) <= TAU * D):
                            continue
                        su, sv = ref_tok(u, SG), ref_tok(v, SG)
                        for dy in range(-radius, radius + 1):
                            for dx in range(-radius, radius + 1):
                                r_, c_ = sv + dy, su + dx
                                if 0 <= r_ < SG and 0 <= c_ < SG:
                                    s.add(k * SG * SG + r_ * SG + c_)
    return out


def ref_reverse(sc, radius):
    """dict (b, t, scene_token) -> set of allowed target tokens (only covisible scene tokens appear)."""
    B, Vt = sc["P_t"].shape[:2]
    Vi = sc["P_i"].shape[1]
    g = IMG // PATCH
    Pt, Pi = sc["P_t"].numpy(), sc["P_i"].numpy()
    At, Ai = sc["A_t"].numpy(), sc["A_i"].numpy()
    ct, K = sc["c2w_t"].numpy(), sc["Knp"]
    out = {}
    for b in range(B):
        for t in range(Vt):
            for k in range(Vi):
                for y in range(IMG):
                    for x in range(IMG):
                        if Ai[b, k, 0, y, x] <= 0.5:
                            continue
                        u, v, z = ref_project(Pi[b, k, :, y, x], ct[b, t], K)
                        if not (u > -0.5 and u < IMG - 0.5 and v > -0.5 and v < IMG - 0.5 and z > 0):
                            continue
                        ui = int(min(max(np.round(u), 0), IMG - 1))
                        vi = int(min(max(np.round(v), 0), IMG - 1))
                        D = ref_depth(Pt[b, t], ct[b, t], vi, ui)
                        if not (At[b, t, 0, vi, ui] > 0.5 and D > 0 and abs(z - D) <= TAU * D):
                            continue
                        s_tok = k * SG * SG + ref_tok(y, SG) * SG + ref_tok(x, SG)
                        tv, tu = ref_tok(v, g), ref_tok(u, g)
                        st = out.setdefault((b, t, s_tok), set())
                        for dy in range(-radius, radius + 1):
                            for dx in range(-radius, radius + 1):
                                r_, c_ = tv + dy, tu + dx
                                if 0 <= r_ < g and 0 <= c_ < g:
                                    st.add(r_ * g + c_)
    return out


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.sc = make_scene(0)

    def test_tok_index_formula(self):
        for g in (5, 8, 37, 32, 64):
            img = 32 if g in (5, 8) else 256
            x = torch.arange(img, dtype=torch.float64)
            ref = torch.tensor([min(int(math.floor((i + 0.5) * g / img)), g - 1) for i in range(img)])
            self.assertTrue(torch.equal(geo.tok_index(x, g, img), ref), g)

    def test_project_roundtrip_and_principal_point(self):
        c2w = torch.tensor(look_at((0.4, -0.3, -1.3)))
        K = torch.tensor([36.0, 36.0, 16.0, 16.0], dtype=torch.float64)
        y, x = torch.meshgrid(torch.arange(IMG, dtype=torch.float64), torch.arange(IMG, dtype=torch.float64), indexing="ij")
        d = 1.0 + torch.rand(IMG, IMG, dtype=torch.float64)
        pc = torch.stack([(x + 0.5 - 16) / 36 * d, (y + 0.5 - 16) / 36 * d, d], -1).reshape(-1, 3)
        X = pc @ c2w[:3, :3].T + c2w[:3, 3]
        u, v, z = geo.project(X[None], c2w[None], K[None])
        self.assertLess((u[0] - x.reshape(-1)).abs().max().item(), 1e-9)
        self.assertLess((v[0] - y.reshape(-1)).abs().max().item(), 1e-9)
        self.assertLess((z[0] - d.reshape(-1)).abs().max().item(), 1e-9)
        u0, v0, _ = geo.project((c2w[:3, 3] + 2.0 * c2w[:3, 2])[None, None], c2w[None], K[None])
        self.assertAlmostEqual(float(u0), 15.5, places=9)   # principal point in index units = cx - 0.5
        self.assertAlmostEqual(float(v0), 15.5, places=9)

    def test_morton_rank_is_bijection_and_z_order(self):
        for g in (2, 8, 32, 37, 64):
            r = geo.morton_rank(g)
            self.assertTrue(torch.equal(torch.sort(r).values, torch.arange(g * g)))
        r = geo.morton_rank(4)
        order = torch.argsort(r)
        self.assertEqual(order[:4].tolist(), [0, 1, 4, 5])          # the first 2x2 quad first

    def test_dilate_matches_maxpool(self):
        m = torch.rand(3, 2, 9, 11) > 0.93
        for rad in (1, 2, 3):
            ref = F.max_pool2d(m.float().view(-1, 1, 9, 11), 2 * rad + 1, 1, rad).view(3, 2, 9, 11) > 0
            self.assertTrue(torch.equal(geo.dilate_bool(m, rad), ref), rad)

    def test_plucker_matches_compute_rays(self):
        from utils.data_utils import ProcessData
        B, V = 2, 3
        c2w = torch.stack([torch.stack([torch.tensor(look_at((math.cos(a), 0.3 * b, math.sin(a))), dtype=torch.float32)
                                        for a in (0.3, 1.1, 2.5)]) for b in range(B)])
        K = torch.tensor([36.0, 36.0, 16.0, 16.0]).view(1, 1, 4).expand(B, V, 4).contiguous()
        ro, rd = ProcessData.compute_rays(None, c2w, K, IMG, IMG, device="cpu")
        ref = torch.cat([torch.cross(ro, rd, dim=2), rd], dim=2)
        got = geo.plucker_rays(c2w, K, IMG, IMG)
        self.assertLess((got - ref).abs().max().item(), 1e-5)

    def test_layout_pack_pad_roundtrip(self):
        sc = self.sc
        lay = geo.build_layout(sc["A_t"].float(), PATCH, bucket=16, img=IMG)
        B, Vt = sc["A_t"].shape[:2]
        self.assertEqual(lay.T, sum(geo.N_REG + n for n in lay.n))
        x = torch.randn(lay.T, 5)
        xz = torch.cat([x, torch.zeros(1, 5)])
        pad = xz[lay.pack_src]                                            # [BV, Lq, 5]
        back = pad.reshape(-1, 5)[lay.pad_dst]
        self.assertTrue(torch.equal(back, x))
        g = IMG // PATCH
        for bv in range(lay.BV):
            fg = lay.fg_tok[bv]
            slots = lay.slot2tok[bv, :lay.n[bv]]
            self.assertTrue(bool(fg[slots].all()))
            self.assertEqual(int(fg.sum()), lay.n[bv])
            self.assertTrue(torch.equal(lay.tok2slot[bv, slots], torch.arange(lay.n[bv])))
            rk = geo.morton_rank(g)[slots]
            self.assertTrue(bool((rk[1:] > rk[:-1]).all()))                # slots follow Morton order
        # fg bookkeeping
        self.assertTrue(torch.equal(lay.fg_tok_flat, lay.slot2tok[lay.fg_bv, torch.cat([torch.arange(n) for n in lay.n])]))

    def _tables(self, radius_f=2, radius_r=1):
        sc = self.sc
        lay = geo.build_layout(sc["A_t"], PATCH, bucket=16, img=IMG)
        Ft = geo.build_forward_table(sc["P_t"], sc["A_t"], lay, sc["c2w_i"], sc["K_i"], sc["P_i"], sc["A_i"],
                                     PATCH, SG, IMG, radius=radius_f, tau=TAU)
        Rt = geo.build_reverse_table(sc["P_i"], sc["A_i"], lay, sc["c2w_t"], sc["K_t"], sc["P_t"], sc["A_t"],
                                     PATCH, SG, IMG, radius=radius_r, tau=TAU)
        return lay, Ft, Rt

    def test_forward_table_matches_bruteforce(self):
        lay, Ft, _ = self._tables()
        ref = ref_forward(self.sc, 2)
        B, Vt = self.sc["P_t"].shape[:2]
        Vi = self.sc["P_i"].shape[1]
        S = Vi * SG * SG
        nonempty = 0
        for bv in range(lay.BV):
            b, t = divmod(bv, Vt)
            self.assertTrue(bool(Ft[bv, :geo.N_REG, :S].all()))              # registers see everything
            self.assertFalse(bool(Ft[bv, :geo.N_REG, S:].any()))
            for j in range(lay.n[bv]):
                tok = int(lay.slot2tok[bv, j])
                row = Ft[bv, geo.N_REG + j]
                got = set(torch.nonzero(row[:S]).flatten().tolist())
                want = ref.get((b, t, tok), set())
                self.assertEqual(got, want, f"bv {bv} token {tok}")
                self.assertEqual(bool(row[S]), len(want) == 0)                  # null key iff empty
                self.assertFalse(bool(row[S + 1:].any()))
                nonempty += len(want) > 0
            for r in range(geo.N_REG + lay.n[bv], lay.Lq):                      # pad rows: null only
                self.assertEqual(torch.nonzero(Ft[bv, r]).flatten().tolist(), [S])
        self.assertGreater(nonempty, 20)                                        # the test is not vacuous
        self.assertTrue(bool(Ft.any(-1).all()))                                 # no empty row anywhere

    def test_reverse_table_matches_bruteforce(self):
        lay, _, Rt = self._tables()
        ref = ref_reverse(self.sc, 1)
        Vt = self.sc["P_t"].shape[1]
        Vi = self.sc["P_i"].shape[1]
        S = Vi * SG * SG
        visible = 0
        for bv in range(lay.BV):
            b, t = divmod(bv, Vt)
            slot_of = {int(lay.slot2tok[bv, j]): j for j in range(lay.n[bv])}
            self.assertTrue(bool(Rt[bv, :, geo.SUM_COLS:geo.SUM_COLS + geo.N_REG].all()))
            self.assertFalse(bool(Rt[bv, :, geo.SUM_COLS + geo.N_REG:geo.REV_FG_OFF].any()))
            self.assertTrue(torch.equal(Rt[bv, :, :geo.SUM_COLS], lay.tblock_valid[bv].view(1, -1).expand(Rt.shape[1], -1)))
            for s in range(S):
                got = set((torch.nonzero(Rt[bv, s, geo.REV_FG_OFF:]).flatten()).tolist())
                want_tok = ref.get((b, t, s), set())
                want = {slot_of[tk] for tk in want_tok if tk in slot_of}      # only fg target tokens are keys
                self.assertEqual(got, want, f"bv {bv} scene {s}")
                visible += len(want) > 0
            self.assertFalse(bool(Rt[bv, S:, geo.REV_FG_OFF:].any()))           # pad rows see no fg token
        self.assertGreater(visible, 10)

    def test_occlusion_changes_the_table(self):
        """tightening tau to ~0 must remove entries: the depth test is live, not vacuous."""
        sc = self.sc
        lay = geo.build_layout(sc["A_t"], PATCH, bucket=16, img=IMG)
        F1 = geo.build_forward_table(sc["P_t"], sc["A_t"], lay, sc["c2w_i"], sc["K_i"], sc["P_i"], sc["A_i"],
                                     PATCH, SG, IMG, radius=0, tau=TAU)
        F0 = geo.build_forward_table(sc["P_t"], sc["A_t"], lay, sc["c2w_i"], sc["K_i"], sc["P_i"], sc["A_i"],
                                     PATCH, SG, IMG, radius=0, tau=1e-6)
        S = sc["P_i"].shape[1] * SG * SG
        self.assertLess(int(F0[:, geo.N_REG:, :S].sum()), int(F1[:, geo.N_REG:, :S].sum()))

    def test_block_ids(self):
        ids, nblk = geo.scene_block_ids(37, 4, 3)
        self.assertEqual(nblk, 4 * 13 * 13)
        self.assertEqual(ids.numel(), 4 * 1369)
        cnt = torch.bincount(ids, minlength=nblk).view(4, 13, 13)
        self.assertEqual(int(cnt[0, 0, 0]), 9)
        self.assertEqual(int(cnt[0, 12, 0]), 3)
        self.assertEqual(int(cnt[0, 12, 12]), 1)


if __name__ == "__main__":
    unittest.main()
