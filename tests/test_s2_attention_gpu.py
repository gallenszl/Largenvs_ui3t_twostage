"""GPU tests of the stage-2 attention backends and table builders at realistic sizes (plan G4).

Pass criteria (plan section 三): vs an fp64 reference -- forward rel-L2 <= 1e-2 and max-abs <= 3e-2,
gradients rel-L2 <= 2e-2; rows whose only allowed key is the null key give output and dq exactly 0.
Run on a GPU node:  python -m unittest tests.test_s2_attention_gpu -v
"""
import math
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_s2 import geometry as geo  # noqa: E402
from model_s2.masked_attention import MaskTable, attend_blockdiag, attend_masked  # noqa: E402

CUDA = torch.cuda.is_available()
H, D = 12, 64


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def window_table(B, Lq, S, s_pad, n_views=4, g=37, radius=3, n_empty=6, n_pad=40, seed=0):
    """registers see all; fg rows see (2r+1)^2 windows in up to 4 views; some empty rows (null key) and
    trailing pad rows (null key only)."""
    gen = torch.Generator().manual_seed(seed)
    T = torch.zeros(B, Lq, s_pad, dtype=torch.bool)
    T[:, :4, :S] = True
    for b in range(B):
        for r in range(4, Lq - n_pad):
            if r < 4 + n_empty:
                continue
            for k in range(n_views):
                if torch.rand(1, generator=gen).item() < 0.25:
                    continue
                cy, cx = torch.randint(0, g, (2,), generator=gen).tolist()
                ys = slice(max(0, cy - radius), min(g, cy + radius + 1))
                xs = slice(max(0, cx - radius), min(g, cx + radius + 1))
                blk = torch.zeros(g, g, dtype=torch.bool)
                blk[ys, xs] = True
                T[b, r, k * g * g:(k + 1) * g * g] |= blk.flatten()
    T[:, :, S] = ~T[:, :, :S].any(-1)
    return T


def ref_masked(q, k, v, table):
    s = torch.einsum("bqhd,bkhd->bhqk", q.double(), k.double()) / math.sqrt(q.shape[-1])
    s = s.masked_fill(~table[:, None], float("-inf"))
    return torch.einsum("bhqk,bkhd->bqhd", torch.softmax(s, -1), v.double())


@unittest.skipUnless(CUDA, "needs a GPU")
class MaskedBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.B, cls.Lq, cls.S = 3, 384, 4 * 37 * 37
        cls.s_pad = geo.roundup(cls.S + 1, 128)
        cls.table = window_table(cls.B, cls.Lq, cls.S, cls.s_pad).cuda()
        dev = "cuda"
        q = torch.randn(cls.B, cls.Lq, H, D, device=dev)
        k = torch.randn(cls.B, cls.s_pad, H, D, device=dev)
        v = torch.randn(cls.B, cls.s_pad, H, D, device=dev)
        k[:, cls.S:] = 0
        v[:, cls.S:] = 0
        cls.qkv = [t.to(torch.bfloat16) for t in (q, k, v)]
        cls.dout = torch.randn(cls.B, cls.Lq, H, D, device=dev).to(torch.bfloat16)
        qd, kd, vd = [t.double().requires_grad_(True) for t in cls.qkv]
        o = ref_masked(qd, kd, vd, cls.table)
        o.backward(cls.dout.double())
        cls.ref = (o.detach(), qd.grad, kd.grad, vd.grad)
        cls.empty_rows = [(b, r) for b in range(cls.B) for r in range(4, cls.Lq)
                          if not bool(cls.table[b, r, :cls.S].any())]

    def _check(self, backend):
        mask = MaskTable(self.table, backend)
        q, k, v = [t.clone().requires_grad_(True) for t in self.qkv]
        o = attend_masked(q, k, v, mask, backend)
        o.backward(self.dout)
        ro, rq, rk, rv = self.ref
        self.assertLessEqual(rel(o, ro), 1e-2, backend)
        self.assertLessEqual(float((o.double() - ro).abs().max()), 3e-2, backend)
        for g_, r_, name in ((q.grad, rq, "dq"), (k.grad, rk, "dk"), (v.grad, rv, "dv")):
            self.assertLessEqual(rel(g_, r_), 2e-2, f"{backend} {name}")
            self.assertTrue(bool(torch.isfinite(g_).all()), f"{backend} {name} finite")
        self.assertTrue(len(self.empty_rows) > 0)
        for b, r in self.empty_rows:
            self.assertEqual(float(o[b, r].abs().max()), 0.0, f"{backend} empty row output")
            self.assertEqual(float(q.grad[b, r].abs().max()), 0.0, f"{backend} empty row dq")
        return o

    def test_sdpa_efficient(self):
        self._check("sdpa")

    def test_flex(self):
        o_flex = self._check("flex")
        o_sdpa = attend_masked(*self.qkv, MaskTable(self.table, "sdpa"), "sdpa")
        self.assertLessEqual(rel(o_flex, o_sdpa), 1e-2)
        # the BlockMask is exactly the block-any reduction of the table
        bm = MaskTable(self.table, "flex").block_mask
        B, Lq, Lk = self.table.shape
        want = self.table.view(B, Lq // 128, 128, Lk // 128, 128).any(4).any(2)
        dense = bm.to_dense()[:, 0].bool()
        self.assertTrue(torch.equal(dense, want))


@unittest.skipUnless(CUDA, "needs a GPU")
class BlockDiagTests(unittest.TestCase):
    def test_fa3_blockdiag_vs_ref(self):
        torch.manual_seed(1)
        qlens = [4 + n for n in (0, 37, 400, 211, 5)]
        for kvlens in (qlens, [676] * len(qlens)):
            q = torch.randn(sum(qlens), H, D, device="cuda").to(torch.bfloat16).requires_grad_(True)
            k = torch.randn(sum(kvlens), H, D, device="cuda").to(torch.bfloat16).requires_grad_(True)
            v = torch.randn(sum(kvlens), H, D, device="cuda").to(torch.bfloat16).requires_grad_(True)
            o = attend_blockdiag(q, k, v, qlens, kvlens, "fa3")
            dout = torch.randn_like(o)
            o.backward(dout)
            qd, kd, vd = [t.detach().double().requires_grad_(True) for t in (q, k, v)]
            ro = attend_blockdiag(qd, kd, vd, qlens, kvlens, "ref")
            ro.backward(dout.double())
            self.assertLessEqual(rel(o, ro), 1e-2)
            for g_, r_ in ((q.grad, qd.grad), (k.grad, kd.grad), (v.grad, vd.grad)):
                self.assertLessEqual(rel(g_, r_), 2e-2)


@unittest.skipUnless(CUDA, "needs a GPU")
class BuilderSyncTests(unittest.TestCase):
    def test_table_builders_are_host_sync_free(self):
        torch.manual_seed(2)
        B, Vt, Vi, img = 2, 3, 4, 256
        dev = "cuda"
        ang = torch.linspace(0, 5.5, Vt + Vi)
        c2w = torch.eye(4, device=dev).repeat(B, Vt + Vi, 1, 1)
        for i, a in enumerate(ang.tolist()):
            ca, sa = math.cos(a), math.sin(a)
            R = torch.tensor([[ca, 0, -sa], [0, 1, 0], [sa, 0, ca]], device=dev)
            c2w[:, i, :3, :3] = R
            c2w[:, i, :3, 3] = -1.4 * R[:, 2]
        K = torch.tensor([351.7, 351.7, 128.0, 128.0], device=dev).view(1, 1, 4)
        P = torch.randn(B, Vt + Vi, 3, img, img, device=dev) * 0.2
        A = (torch.rand(B, Vt + Vi, 1, img, img, device=dev) > 0.6).float()
        lay = geo.build_layout(A[:, :Vt], 8, 128, img)                   # the one allowed host sync
        torch.cuda.set_sync_debug_mode("error")
        try:
            Ft = geo.build_forward_table(P[:, :Vt], A[:, :Vt], lay, c2w[:, Vt:], K.expand(B, Vi, 4), P[:, Vt:],
                                         A[:, Vt:], 8, 37, img, radius=2)
            Rt = geo.build_reverse_table(P[:, Vt:], A[:, Vt:], lay, c2w[:, :Vt], K.expand(B, Vt, 4), P[:, :Vt],
                                         A[:, :Vt], 8, 37, img, radius=2)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        self.assertEqual(tuple(Ft.shape), (B * Vt, lay.Lq, geo.roundup(Vi * 1369 + 1, 128)))
        self.assertEqual(tuple(Rt.shape), (B * Vt, geo.roundup(Vi * 1369 + 1, 128), lay.Kr))


if __name__ == "__main__":
    unittest.main()
