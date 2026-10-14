# Stage-2 renderer blocks (plan 2026-09-27, section 一.4-5 / 二 "注意力实现").
#
# S2BidirBlock / S2FinalBlock subclass stage 1's BidirectionalCrossAttentionBlock / CrossAttentionBlock,
# so every stage-1 submodule keeps its name and the stage-1 block state_dict loads into them as is.
# New modules per block (LSRM-style, trunc-normal std 0.02, no bias unless stated):
#   inj_x, inj_rec   Linear(C, C)       per-layer injection of the frozen stage-1 final tokens
#   gate             Linear(C, 2C)      Sigmoid gate of the compressed / selected branches
#   comp_k, comp_v   CompressionResBlock(64) on the scene keys / values (forward compressed branch)
#   rcomp_k, rcomp_v CompressionResBlock(64) on the target keys / values (reverse summaries; bidir only)
# Target stream x is packed [T, C] (per view: 4 registers + foreground tokens in Morton slot order);
# scene stream rec is [BV, S, C] in raster view-major order.
#
# structure (config model.stage2.structure; user 09-28, prototypes in tools_s2/s2_fgdense_whatif.py):
#   current   the modules above: compressed + selected (window) target->scene, filtered reverse attention
#   fg_dense  scene stream = foreground scene tokens only, packed [Ts, C] (model_s2/fg_scene.py); both directions
#             dense attention inside each target view (FA3 block-diagonal): no compression, no windows, no gate
#   fg_dual   fg_dense's foreground-global branch + a geometric-window branch (flex, windows cut to the foreground
#             scene tokens) in both directions, gated sums: gate(x) forward, rgate(rec) reverse (bidir only)
# New modules per structure: current inj_x, inj_rec, gate, comp_k, comp_v (+ rcomp_k, rcomp_v bidir);
# fg_dense inj_x, inj_rec; fg_dual inj_x, inj_rec, gate (+ rgate bidir).

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from model_s2.geometry import N_REG, REV_FG_OFF, SUM_COLS
from model_s2.masked_attention import MaskTable, attend_blockdiag, attend_masked
from models.layers.renderer_blocks import BidirectionalCrossAttentionBlock, CrossAttentionBlock


def _up(t):
    """at least fp32: bf16/fp16 -> fp32, fp32/fp64 unchanged (LSRM LayerNorm32 semantics)."""
    return t.float() if t.dtype in (torch.float16, torch.bfloat16) else t


class CompressionResBlock(nn.Module):
    """LSRM SparseResBlock on one 64-d head vector (shared by all heads): Linear-SiLU-Linear, residual,
    LayerNorm without affine computed in fp32 (sparse/module/residual_block.py)."""

    def __init__(self, dim):
        super().__init__()
        self.act_layers = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, x):
        h = self.act_layers(x)
        return self.norm(_up(x + h)).type(x.dtype)


def init_new_linear(m, std=0.02):
    nn.init.trunc_normal_(m.weight, std=std)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


def init_compression(block):
    for lin in block.act_layers:
        if isinstance(lin, nn.Linear):
            init_new_linear(lin)


@dataclass
class S2Ctx:
    """per-step context, built once outside activation checkpointing"""
    layout: object
    q_seqlens: list                 # per view N_REG + n
    fwd_mask: MaskTable             # [BV, Lq, S_pad]
    rev_mask: MaskTable             # [BV, S_pad, Kr]
    x_prior: torch.Tensor           # [T, C] frozen stage-1 final target tokens (packed)
    rec_prior: torch.Tensor         # [BV, S, C] frozen stage-1 scene tokens after block 10 ([Ts, C] packed for fg_*)
    scene_g: int
    n_views: int
    scene_block: int
    s_pad: int
    dense_backend: str = "fa3"
    masked_backend: str = "sdpa"
    cache: dict = field(default_factory=dict)
    structure: str = "current"
    s_seqlens: list = None          # fg_*: kept scene tokens per target view
    scene_idx: torch.Tensor = None  # fg_*: [Ts] rows of rec_rep.reshape(BV * S, C) kept
    win: dict = None                # fg_dual: window tables and gather indices (model_s2/fg_scene.build_fg_windows)


def _heads(t, n_heads):
    return t.unflatten(-1, (n_heads, t.shape[-1] // n_heads))


def _pad_rows(t, n):
    """[BV, L, H, D] -> [BV, n, H, D] with zero rows appended (null key / padding)."""
    if t.shape[1] == n:
        return t
    return torch.cat([t, t.new_zeros(t.shape[0], n - t.shape[1], *t.shape[2:])], dim=1)


def scene_block_mean(t, scene_g, n_views, block):
    """[BV, S, H, D] (raster, view-major) -> block means [BV, n_views * nb * nb, H, D] (fp32 sums)."""
    BV, S, H, D = t.shape
    nb = (scene_g + block - 1) // block
    P = nb * block
    x = _up(t).reshape(BV, n_views, scene_g, scene_g, H * D)
    x = torch.nn.functional.pad(x, (0, 0, 0, P - scene_g, 0, P - scene_g))
    x = x.reshape(BV, n_views, nb, block, nb, block, H * D).sum(dim=(3, 5))
    cnt = torch.full((P, P), 1.0, device=t.device, dtype=x.dtype)
    cnt[scene_g:, :] = 0
    cnt[:, scene_g:] = 0
    cnt = cnt.reshape(nb, block, nb, block).sum(dim=(1, 3))
    x = x / cnt.view(1, 1, nb, nb, 1)
    return x.reshape(BV, n_views * nb * nb, H, D).to(t.dtype)


def target_block_mean(t_fg, layout):
    """foreground rows [nfg, H, D] -> per-view 32-px block means [BV, SUM_COLS, H, D] (0 where empty)."""
    lay = layout
    nfg, H, D = t_fg.shape
    blk = lay.tblock_id[lay.fg_bv, lay.fg_tok_flat]                           # [nfg]
    idx = lay.fg_bv * SUM_COLS + blk
    tf = _up(t_fg)
    acc = tf.new_zeros(lay.BV * SUM_COLS, H, D)
    acc = acc.index_add(0, idx, tf)
    cnt = lay.tblock_cnt.reshape(-1).clamp_min(1).to(tf.dtype).view(-1, 1, 1)
    return (acc / cnt).view(lay.BV, SUM_COLS, H, D).to(t_fg.dtype)


def self_attention_packed(attn, h, ctx):
    H = attn.num_heads
    q = attn.q_norm(_heads(attn.q_proj(h), H))
    k = attn.k_norm(_heads(attn.k_proj(h), H))
    v = _heads(attn.v_proj(h), H)
    o = attend_blockdiag(q, k, v, ctx.q_seqlens, ctx.q_seqlens, ctx.dense_backend)
    return attn.attn_fc_dropout(attn.proj(o.flatten(-2)))


def target_to_scene(attn, gate, comp_k, comp_v, xn, rn, ctx):
    """gated compressed + selected cross-attention of the packed target stream to the scene stream."""
    H = attn.num_heads
    lay = ctx.layout
    BV = lay.BV
    q = attn.q_norm(_heads(attn.q_proj(xn), H))                               # [T, H, D]
    kr = _heads(attn.k_proj(rn), H)                                           # [BV, S, H, D]
    vr = _heads(attn.v_proj(rn), H)
    # compressed branch: ResBlock on the raw keys / values, block mean, then the copied k-norm (LSRM order)
    kc = attn.k_norm(scene_block_mean(comp_k(kr), ctx.scene_g, ctx.n_views, ctx.scene_block))
    vc = scene_block_mean(comp_v(vr), ctx.scene_g, ctx.n_views, ctx.scene_block)
    nblk = kc.shape[1]
    o_c = attend_blockdiag(q, kc.flatten(0, 1), vc.flatten(0, 1), ctx.q_seqlens, [nblk] * BV, ctx.dense_backend)
    # selected branch: exact per-token table (registers allow every scene token)
    ks = _pad_rows(attn.k_norm(kr), ctx.s_pad)
    vs = _pad_rows(vr, ctx.s_pad)
    qz = torch.cat([q, q.new_zeros(1, *q.shape[1:])])
    q_pad = qz[lay.pack_src]                                                  # [BV, Lq, H, D]
    o_s = attend_masked(q_pad, ks, vs, ctx.fwd_mask, ctx.masked_backend)
    o_s = o_s.reshape(-1, *o_s.shape[2:])[lay.pad_dst]                        # [T, H, D]
    g_c, g_s = torch.sigmoid(gate(xn)).chunk(2, dim=-1)
    return attn.attn_fc_dropout(attn.proj(g_c * o_c.flatten(-2) + g_s * o_s.flatten(-2)))


def scene_to_target(attn, rcomp_k, rcomp_v, rn, xn, ctx):
    """visibility-filtered reverse cross-attention: scene tokens -> [summaries | registers | fg tokens]."""
    H = attn.num_heads
    lay = ctx.layout
    BV, S = rn.shape[0], rn.shape[1]
    qr = attn.q_norm(_heads(attn.q_proj(rn), H))                              # [BV, S, H, D]
    kt = _heads(attn.k_proj(xn), H)                                           # [T, H, D]
    vt = _heads(attn.v_proj(xn), H)
    ksum = attn.k_norm(target_block_mean(rcomp_k(kt[lay.fg_packed]), lay))    # [BV, 64, H, D]
    vsum = target_block_mean(rcomp_v(vt[lay.fg_packed]), lay)
    kn = attn.k_norm(kt)
    Hh, D = kn.shape[1], kn.shape[2]
    knz = torch.cat([kn, kn.new_zeros(1, Hh, D)])
    vtz = torch.cat([vt, vt.new_zeros(1, Hh, D)])
    gap = REV_FG_OFF - SUM_COLS - N_REG
    K = torch.cat([ksum, kn[lay.reg_packed].view(BV, N_REG, Hh, D), kn.new_zeros(BV, gap, Hh, D),
                   knz[lay.revkey_src]], dim=1)                               # [BV, Kr, H, D]
    V = torch.cat([vsum, vt[lay.reg_packed].view(BV, N_REG, Hh, D), vt.new_zeros(BV, gap, Hh, D),
                   vtz[lay.revkey_src]], dim=1)
    o = attend_masked(_pad_rows(qr, ctx.s_pad), K, V, ctx.rev_mask, ctx.masked_backend)[:, :S]
    return attn.attn_fc_dropout(attn.proj(o.flatten(-2)))


# ------------------------------------------------------------------------------------ fg_dense / fg_dual
def _qkv(attn, qin, kvin):
    H = attn.num_heads
    return (attn.q_norm(_heads(attn.q_proj(qin), H)), attn.k_norm(_heads(attn.k_proj(kvin), H)),
            _heads(attn.v_proj(kvin), H))


def fg_global(q, k, v, q_seqlens, kv_seqlens, ctx, direction):
    """dense attention of each target view's queries to that view's keys only (packed segments, FA3).
    direction ('t2s' / 's2t') only labels the call for instrumentation."""
    return attend_blockdiag(q, k, v, q_seqlens, kv_seqlens, ctx.dense_backend)


def _scatter_pad(x, src, dst, rows):
    """packed rows x [N, H, D] (optionally the subset x[src]) written into a zeros [rows, H, D] at the unique
    destinations dst.  Not x_zero_row[gather_index]: its backward accumulates the gradient of every pad slot into the
    one shared zero row (atomics on one address; 09-28 profile job 6910150: 149 ms vs 2.1 ms per scene-side call,
    69 % of a fg_dual training step); index_copy's backward is a plain gather."""
    if src is not None:
        x = torch.index_select(x, 0, src)
    return x.new_zeros(rows, *x.shape[1:]).index_copy(0, dst, x)


def fg_window(q, k, v, win, key, ctx, direction):
    """geometric-window branch: packed q [Nq, H, D] / k, v [Nk, H, D] written into the window table's padded
    layout [BV, Lq] / [BV, Lk] (pad slots zero), per-token masked attention, the packed query rows selected back.
    win / key ('f' forward, 'r' reverse): model_s2/fg_scene.build_fg_windows."""
    BV, Lq, Lk = win["BV"], win[key + "_Lq"], win[key + "_Lk"]
    qp = _scatter_pad(q, None, win[key + "_q_dst"], BV * Lq).view(BV, Lq, *q.shape[1:])
    kp = _scatter_pad(k, win[key + "_kv_src"], win[key + "_kv_dst"], BV * Lk).view(BV, Lk, *k.shape[1:])
    vp = _scatter_pad(v, win[key + "_kv_src"], win[key + "_kv_dst"], BV * Lk).view(BV, Lk, *v.shape[1:])
    o = attend_masked(qp, kp, vp, win[key + "_mask"], ctx.masked_backend)
    return torch.index_select(o.reshape(BV * Lq, *o.shape[2:]), 0, win[key + "_out"])


def fg_cross(attn, gate, qin, kvin, q_seq, kv_seq, win, key, ctx, direction):
    """fg_dense: the foreground-global branch; fg_dual: + the window branch, sigmoid-gated per channel"""
    q, k, v = _qkv(attn, qin, kvin)
    o = fg_global(q, k, v, q_seq, kv_seq, ctx, direction).flatten(-2)
    if gate is not None:
        o_w = fg_window(q, k, v, win, key, ctx, direction).flatten(-2)
        g_g, g_w = torch.sigmoid(gate(qin)).chunk(2, dim=-1)
        o = g_g * o + g_w * o_w
    return attn.attn_fc_dropout(attn.proj(o))


NEW_MODULES = {("bidir", "current"): ("inj_x", "inj_rec", "gate", "comp_k", "comp_v", "rcomp_k", "rcomp_v"),
               ("bidir", "fg_dense"): ("inj_x", "inj_rec"),
               ("bidir", "fg_dual"): ("inj_x", "inj_rec", "gate", "rgate"),
               ("final", "current"): ("inj_x", "inj_rec", "gate", "comp_k", "comp_v"),
               ("final", "fg_dense"): ("inj_x", "inj_rec"),
               ("final", "fg_dual"): ("inj_x", "inj_rec", "gate")}


def _make_new(block, kind, structure, hidden_dim, num_heads):
    if (kind, structure) not in NEW_MODULES:
        raise ValueError(f"unknown stage-2 structure {structure!r}")
    block.structure = structure
    block._new = NEW_MODULES[(kind, structure)]
    d = hidden_dim // num_heads
    for name in block._new:
        if name in ("inj_x", "inj_rec"):
            setattr(block, name, nn.Linear(hidden_dim, hidden_dim, bias=False))
        elif name in ("gate", "rgate"):
            setattr(block, name, nn.Linear(hidden_dim, 2 * hidden_dim, bias=False))
        else:
            setattr(block, name, CompressionResBlock(d))


def _init_new(block):
    for name in block._new:
        m = getattr(block, name)
        if isinstance(m, CompressionResBlock):
            init_compression(m)
        else:
            init_new_linear(m)


class S2BidirBlock(BidirectionalCrossAttentionBlock):
    def __init__(self, hidden_dim, num_heads, structure="current"):
        super().__init__(hidden_dim=hidden_dim, num_heads=num_heads)
        _make_new(self, "bidir", structure, hidden_dim, num_heads)

    def new_module_names(self):
        return self._new

    def init_new(self):
        _init_new(self)

    def forward(self, x, rec, ctx):
        x = x + self.inj_x(ctx.x_prior)
        rec = rec + self.inj_rec(ctx.rec_prior)
        x = x + self_attention_packed(self.self_attn, self.norm1_x(x), ctx)
        xn = self.norm2_x(x)
        rn = self.norm1_rec(rec)
        if self.structure == "current":
            dx = target_to_scene(self.cross_attn_x, self.gate, self.comp_k, self.comp_v, xn, rn, ctx)
            drec = scene_to_target(self.cross_attn_rec, self.rcomp_k, self.rcomp_v, rn, xn, ctx)
        else:
            dual = self.structure == "fg_dual"
            dx = fg_cross(self.cross_attn_x, self.gate if dual else None, xn, rn, ctx.q_seqlens, ctx.s_seqlens,
                          ctx.win, "f", ctx, "t2s")
            drec = fg_cross(self.cross_attn_rec, self.rgate if dual else None, rn, xn, ctx.s_seqlens, ctx.q_seqlens,
                            ctx.win, "r", ctx, "s2t")
        x = x + dx
        rec = rec + drec
        x = x + self.mlp_x(self.norm3_x(x))
        rec = rec + self.mlp_rec(self.norm2_rec(rec))
        return x, rec


class S2FinalBlock(CrossAttentionBlock):
    def __init__(self, hidden_dim, num_heads, structure="current"):
        super().__init__(hidden_dim=hidden_dim, num_heads=num_heads)
        _make_new(self, "final", structure, hidden_dim, num_heads)

    def new_module_names(self):
        return self._new

    def init_new(self):
        _init_new(self)

    def forward(self, x, rec, ctx):
        x = x + self.inj_x(ctx.x_prior)
        rec = rec + self.inj_rec(ctx.rec_prior)
        x = x + self_attention_packed(self.self_attn, self.norm1(x), ctx)
        xn = self.norm2(x)
        rn = self.norm2_kv(rec)
        if self.structure == "current":
            x = x + target_to_scene(self.cross_attn, self.gate, self.comp_k, self.comp_v, xn, rn, ctx)
        else:
            x = x + fg_cross(self.cross_attn, self.gate if self.structure == "fg_dual" else None, xn, rn,
                             ctx.q_seqlens, ctx.s_seqlens, ctx.win, "f", ctx, "t2s")
        x = x + self.mlp(self.norm_ffn(x))
        return x
