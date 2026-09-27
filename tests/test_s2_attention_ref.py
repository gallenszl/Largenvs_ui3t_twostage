"""CPU tests for model_s2/blocks.py + masked_attention.py reference backends (plan G2).

The expected outputs come from an independent float64 per-row re-implementation written here from the
module parameters (RMSNorm, LayerNorm, ResBlock, softmax over the allowed set of each row); it does not
call any helper of the code under test.  Run:  python -m unittest tests.test_s2_attention_ref -v
"""
import math
import os
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_s2 import geometry as geo  # noqa: E402
from model_s2.blocks import S2BidirBlock, S2FinalBlock, S2Ctx  # noqa: E402
from model_s2.masked_attention import MaskTable  # noqa: E402
from models.layers.renderer_blocks import BidirectionalCrossAttentionBlock, CrossAttentionBlock  # noqa: E402
from tests.test_s2_geometry import IMG, PATCH, SG, TAU, make_scene  # noqa: E402

C, NH, SBLK, TBLK_PX = 32, 2, 2, 16


def lin(m, t):
    y = t @ m.weight.T
    return y + m.bias if m.bias is not None else y


def ln(m, t):
    return F.layer_norm(t, (t.shape[-1],), m.weight, m.bias, m.eps)


def rms(m, t):
    return t / torch.sqrt((t ** 2).mean(-1, keepdim=True) + m.eps) * m.weight


def heads(t):
    return t.reshape(*t.shape[:-1], NH, t.shape[-1] // NH)


def resblock(rb, t):
    l1, l2 = rb.act_layers[0], rb.act_layers[2]
    y = t + lin(l2, F.silu(lin(l1, t)))
    return F.layer_norm(y, (y.shape[-1],), None, None, rb.norm.eps)


def mlp(m, t):
    return lin(m.fc2, F.gelu(lin(m.fc1, t)))


def attend_rows(q, K, V, allowed):
    """q [H, D]; K, V [N, H, D]; allowed list of column ids -> [H, D]"""
    idx = torch.tensor(sorted(allowed))
    s = torch.einsum("hd,nhd->hn", q, K[idx]) / math.sqrt(q.shape[-1])
    return torch.einsum("hn,nhd->hd", torch.softmax(s, -1), V[idx])


def ref_target_to_scene(attn, gate, ck, cv, xn, rn, lay, Ft, S, s_blocks):
    """returns dx [T, C]"""
    T = xn.shape[0]
    out = torch.zeros(T, C, dtype=xn.dtype)
    for bv in range(lay.BV):
        kraw = heads(lin(attn.k_proj, rn[bv]))                 # [S, H, D]
        v = heads(lin(attn.v_proj, rn[bv]))
        ksel = rms(attn.k_norm, kraw)
        kc_in, vc_in = resblock(ck, kraw), resblock(cv, v)
        kc = torch.stack([rms(attn.k_norm, kc_in[b].mean(0)) for b in s_blocks])
        vc = torch.stack([vc_in[b].mean(0) for b in s_blocks])
        Kz = torch.cat([ksel, torch.zeros(1, NH, C // NH, dtype=xn.dtype)])   # null key at S
        Vz = torch.cat([v, torch.zeros(1, NH, C // NH, dtype=xn.dtype)])
        off = lay.offsets[bv]
        for r in range(geo.N_REG + lay.n[bv]):
            i = off + r
            q = rms(attn.q_norm, heads(lin(attn.q_proj, xn[i])))
            o_c = attend_rows(q, kc, vc, range(len(s_blocks)))
            allowed = torch.nonzero(Ft[bv, r, :S + 1]).flatten().tolist()
            o_s = attend_rows(q, Kz, Vz, allowed)
            g = torch.sigmoid(lin(gate, xn[i]))
            out[i] = lin(attn.proj, g[:C] * o_c.flatten() + g[C:] * o_s.flatten())
    return out


def ref_scene_to_target(attn, rk, rv, rn, xn, lay, Rt, S):
    """returns drec [BV, S, C]"""
    out = torch.zeros(lay.BV, S, C, dtype=xn.dtype)
    for bv in range(lay.BV):
        off = lay.offsets[bv]
        n = lay.n[bv]
        kt = heads(lin(attn.k_proj, xn[off:off + geo.N_REG + n]))
        vt = heads(lin(attn.v_proj, xn[off:off + geo.N_REG + n]))
        keys, vals, cols = [], [], []
        # summaries: blocks of fg tokens
        blk_of = lay.tblock_id[bv][lay.slot2tok[bv, :n]]
        rkt, rvt = resblock(rk, kt[geo.N_REG:]), resblock(rv, vt[geo.N_REG:])
        for b in range(geo.SUM_COLS):
            m = (blk_of == b)
            if m.any():
                keys.append(rms(attn.k_norm, rkt[m].mean(0))); vals.append(rvt[m].mean(0)); cols.append(b)
        for r in range(geo.N_REG):
            keys.append(rms(attn.k_norm, kt[r])); vals.append(vt[r]); cols.append(geo.SUM_COLS + r)
        for j in range(n):
            keys.append(rms(attn.k_norm, kt[geo.N_REG + j])); vals.append(vt[geo.N_REG + j]); cols.append(geo.REV_FG_OFF + j)
        K, V = torch.stack(keys), torch.stack(vals)
        pos = {c: i for i, c in enumerate(cols)}
        for s in range(S):
            q = rms(attn.q_norm, heads(lin(attn.q_proj, rn[bv, s])))
            allowed_cols = torch.nonzero(Rt[bv, s]).flatten().tolist()
            allowed = [pos[c] for c in allowed_cols]
            self_c = set(allowed_cols) - set(pos)
            assert not self_c, f"table allows columns without keys: {self_c}"
            out[bv, s] = lin(attn.proj, attend_rows(q, K, V, allowed).flatten())
    return out


class BlockRefTests(unittest.TestCase):
    """float64 comparisons.  Stage-1's RMSNorm (models/layers/attention.py) always computes in fp32
    (x.float()), which would cap agreement at ~1e-6; it is not code under test here, so for the duration
    of these tests it computes in the input dtype and the block wiring is checked to fp64 precision."""

    @classmethod
    def setUpClass(cls):
        from models.layers.attention import RMSNorm
        cls._rms_fwd = RMSNorm.forward
        RMSNorm.forward = lambda self, x: self._norm(x) * self.weight.type_as(x)
        torch.set_num_threads(2)
        torch.manual_seed(0)
        sc = make_scene(1, B=1, Vt=2, Vi=2)
        cls.lay = geo.build_layout(sc["A_t"], PATCH, bucket=8, img=IMG, tblock_px=TBLK_PX)
        cls.S = 2 * SG * SG
        cls.s_pad = geo.roundup(cls.S + 1, 8)
        cls.Ft = geo.build_forward_table(sc["P_t"], sc["A_t"], cls.lay, sc["c2w_i"], sc["K_i"], sc["P_i"], sc["A_i"],
                                         PATCH, SG, IMG, radius=1, tau=TAU, s_pad=cls.s_pad)
        cls.Rt = geo.build_reverse_table(sc["P_i"], sc["A_i"], cls.lay, sc["c2w_t"], sc["K_t"], sc["P_t"], sc["A_t"],
                                         PATCH, SG, IMG, radius=1, tau=TAU, s_pad=cls.s_pad)
        ids, nblk = geo.scene_block_ids(SG, 2, SBLK)
        cls.s_blocks = [torch.nonzero(ids == b).flatten() for b in range(nblk)]

    @classmethod
    def tearDownClass(cls):
        from models.layers.attention import RMSNorm
        RMSNorm.forward = cls._rms_fwd

    def _ctx(self, dt):
        lay = self.lay
        T = lay.T
        ctx = S2Ctx(layout=lay, q_seqlens=[geo.N_REG + n for n in lay.n],
                    fwd_mask=MaskTable(self.Ft, "ref"), rev_mask=MaskTable(self.Rt, "ref"),
                    x_prior=torch.randn(T, C, dtype=dt), rec_prior=torch.randn(lay.BV, self.S, C, dtype=dt),
                    scene_g=SG, n_views=2, scene_block=SBLK, s_pad=self.s_pad,
                    dense_backend="ref", masked_backend="ref")
        return ctx

    def _randomize(self, blk):
        # stage-1 blocks start from small weights; use a larger random init so every path matters
        with torch.no_grad():
            for p in blk.parameters():
                p.copy_(torch.randn_like(p) * 0.3 + (1.0 if p.dim() == 1 else 0.0))

    def test_bidir_block_matches_reference(self):
        dt = torch.float64
        blk = S2BidirBlock(C, NH).to(dt)
        self._randomize(blk)
        ctx = self._ctx(dt)
        lay = self.lay
        x = torch.randn(lay.T, C, dtype=dt)
        rec = torch.randn(lay.BV, self.S, C, dtype=dt)
        with torch.no_grad():
            got_x, got_rec = blk(x, rec, ctx)
            # reference
            x1 = x + lin(blk.inj_x, ctx.x_prior)
            r1 = rec + lin(blk.inj_rec, ctx.rec_prior)
            h = ln(blk.norm1_x, x1)
            a = blk.self_attn
            sa = torch.zeros_like(x1)
            for bv in range(lay.BV):
                o, L = lay.offsets[bv], geo.N_REG + lay.n[bv]
                q = rms(a.q_norm, heads(lin(a.q_proj, h[o:o + L])))
                k = rms(a.k_norm, heads(lin(a.k_proj, h[o:o + L])))
                v = heads(lin(a.v_proj, h[o:o + L]))
                for i in range(L):
                    sa[o + i] = lin(a.proj, attend_rows(q[i], k, v, range(L)).flatten())
            x2 = x1 + sa
            xn, rn = ln(blk.norm2_x, x2), ln(blk.norm1_rec, r1)
            dx = ref_target_to_scene(blk.cross_attn_x, blk.gate, blk.comp_k, blk.comp_v, xn, rn, lay, self.Ft, self.S, self.s_blocks)
            drec = ref_scene_to_target(blk.cross_attn_rec, blk.rcomp_k, blk.rcomp_v, rn, xn, lay, self.Rt, self.S)
            x3, r3 = x2 + dx, r1 + drec
            want_x = x3 + mlp(blk.mlp_x, ln(blk.norm3_x, x3))
            want_rec = r3 + mlp(blk.mlp_rec, ln(blk.norm2_rec, r3))
        self.assertLess((got_x - want_x).abs().max().item(), 1e-9)
        self.assertLess((got_rec - want_rec).abs().max().item(), 1e-9)

    def test_final_block_matches_reference(self):
        dt = torch.float64
        blk = S2FinalBlock(C, NH).to(dt)
        self._randomize(blk)
        ctx = self._ctx(dt)
        lay = self.lay
        x = torch.randn(lay.T, C, dtype=dt)
        rec = torch.randn(lay.BV, self.S, C, dtype=dt)
        with torch.no_grad():
            got = blk(x, rec, ctx)
            x1 = x + lin(blk.inj_x, ctx.x_prior)
            r1 = rec + lin(blk.inj_rec, ctx.rec_prior)
            h = ln(blk.norm1, x1)
            a = blk.self_attn
            sa = torch.zeros_like(x1)
            for bv in range(lay.BV):
                o, L = lay.offsets[bv], geo.N_REG + lay.n[bv]
                q = rms(a.q_norm, heads(lin(a.q_proj, h[o:o + L])))
                k = rms(a.k_norm, heads(lin(a.k_proj, h[o:o + L])))
                v = heads(lin(a.v_proj, h[o:o + L]))
                for i in range(L):
                    sa[o + i] = lin(a.proj, attend_rows(q[i], k, v, range(L)).flatten())
            x2 = x1 + sa
            xn, rn = ln(blk.norm2, x2), ln(blk.norm2_kv, r1)
            dx = ref_target_to_scene(blk.cross_attn, blk.gate, blk.comp_k, blk.comp_v, xn, rn, lay, self.Ft, self.S, self.s_blocks)
            x3 = x2 + dx
            want = x3 + mlp(blk.mlp, ln(blk.norm_ffn, x3))
        self.assertLess((got - want).abs().max().item(), 1e-9)

    def test_empty_rows_and_registers(self):
        lay, Ft = self.lay, self.Ft
        S = self.S
        empty = [(bv, r) for bv in range(lay.BV) for r in range(geo.N_REG, geo.N_REG + lay.n[bv])
                 if not bool(Ft[bv, r, :S].any())]
        self.assertTrue(all(bool(Ft[bv, r, S]) for bv, r in empty))
        self.assertTrue(bool(Ft[:, :geo.N_REG, :S].all()))
        # the null key makes the selected-branch output of an empty row exactly zero, with zero dq
        from model_s2.masked_attention import attend_masked
        q = torch.randn(lay.BV, lay.Lq, NH, C // NH, dtype=torch.float64, requires_grad=True)
        k = torch.cat([torch.randn(lay.BV, S, NH, C // NH, dtype=torch.float64),
                       torch.zeros(lay.BV, self.s_pad - S, NH, C // NH, dtype=torch.float64)], 1)
        v = torch.cat([torch.randn(lay.BV, S, NH, C // NH, dtype=torch.float64),
                       torch.zeros(lay.BV, self.s_pad - S, NH, C // NH, dtype=torch.float64)], 1)
        o = attend_masked(q, k, v, MaskTable(Ft, "ref"), "ref")
        o.sum().backward()
        for bv, r in empty:
            self.assertEqual(float(o[bv, r].abs().max()), 0.0)
            self.assertEqual(float(q.grad[bv, r].abs().max()), 0.0)

    def test_stage1_state_dict_loads_into_s2_blocks(self):
        for s1, s2 in ((BidirectionalCrossAttentionBlock(C, NH), S2BidirBlock(C, NH)),
                       (CrossAttentionBlock(C, NH), S2FinalBlock(C, NH))):
            missing, unexpected = s2.load_state_dict(s1.state_dict(), strict=False)
            self.assertEqual(unexpected, [])
            self.assertTrue(all(k.split(".")[0] in s2.new_module_names() for k in missing), missing)
            for k, v in s1.state_dict().items():
                self.assertTrue(torch.equal(s2.state_dict()[k], v), k)


if __name__ == "__main__":
    unittest.main()
