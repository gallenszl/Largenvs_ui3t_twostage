# Foreground-only scene side of the fg_dense / fg_dual stage-2 structures (user 09-27 / 09-28; the timing-only
# prototypes they come from are tools_s2/s2_fgdense_whatif.py).
#
# The scene stream keeps, for every target view, only the scene tokens whose input-view patch holds foreground:
# GT input alpha max-pooled to the scene token grid (> 0.5), dilated by one token.  Every target view of a scene
# keeps the same set (its own copy, updated every block, as the scene copies of the current structure).  Tokens
# are packed view-major, raster order inside a view: [sum_bv Ns_bv, C].  A scene whose input views have no
# foreground token at all keeps every scene token (so no view ever has an empty key segment).
# Built once per step, outside the renderer and activation checkpointing (one host read: the counts).
#
# fg_dual also needs geometric-window tables restricted to those tokens; they are cut out of the full forward /
# reverse tables of model_s2/geometry.py (same windows, same visibility test), registers excluded (they are in the
# foreground-global branch), every row with an all-zero null key as its last column.

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from model_s2.geometry import N_REG, REV_FG_OFF, roundup
from model_s2.masked_attention import MaskTable


@dataclass
class FgScene:
    idx: torch.Tensor           # [Ts] row of every kept scene token in rec_rep.reshape(BV * S, C)
    seq: list                   # kept scene tokens per target view (host ints)
    S: int                      # scene tokens per view before the cut (n_views * g * g)
    share: float                # kept fraction of all scene tokens (host)


def build_fg_scene(alpha_in, Vt, scene_g):
    """alpha_in [B, Vin, 1, H, W] (GT input alpha) -> FgScene over the BV = B * Vt target views."""
    B, Vin = alpha_in.shape[:2]
    dev = alpha_in.device
    g = scene_g
    S = Vin * g * g
    a = F.adaptive_max_pool2d(alpha_in.reshape(B * Vin, 1, *alpha_in.shape[-2:]).float(), g) > 0.5
    a = F.max_pool2d(a.float(), 3, 1, 1) > 0                                          # one-token dilation
    fg = a.reshape(B, S)
    fg = fg | (fg.sum(1, keepdim=True) == 0)                                            # no foreground: keep all
    BV = B * Vt
    fgv = fg.repeat_interleave(Vt, 0)                                                   # [BV, S]
    rows = torch.arange(BV, device=dev).view(BV, 1) * S + torch.arange(S, device=dev).view(1, S)
    idx = rows[fgv]
    seq = [int(v) for v in fgv.sum(1).tolist()]                                         # the host read
    return FgScene(idx=idx, seq=seq, S=S, share=float(sum(seq)) / float(BV * S))


def build_fg_windows(Ft, Rt, layout, fgs, backend):
    """Full tables Ft [BV, Lq, s_pad] / Rt [BV, s_pad, Kr] (model_s2.geometry) -> the window branch of fg_dual on
    the foreground scene tokens of fgs:
      forward : rows = target [registers | fg | pad] (registers: null only), cols = kept scene tokens | pad | null
      reverse : rows = kept scene tokens | pad, cols = fg target tokens (slot order) | pad | null
    plus the unique destination slots of the packed q / k / v rows in those layouts (model_s2/blocks.fg_window)."""
    lay = layout
    dev = fgs.idx.device
    BV, S, seq = lay.BV, fgs.S, fgs.seq
    Lq, s_pad = Ft.shape[1], Ft.shape[2]
    assert s_pad - 1 > S, "column s_pad - 1 must be a pad column (never allowed), not the null key S"
    # the kept-scene axis is padded to the full scene width s_pad (fixed), NOT to max(seq): FlexAttention compiles one
    # kernel per shape, and a per-batch width recompiled at every new batch until dynamo's limit (smoke 6909938 hit
    # recompile_limit 64 and fell back to the unfused path).  The kept tokens sit first; flex skips the all-pad tiles.
    Ns_pad = s_pad
    seq_t = torch.tensor(seq, device=dev)
    soff = torch.cumsum(seq_t, 0) - seq_t
    bv = fgs.idx // S
    j = fgs.idx % S
    k = torch.arange(fgs.idx.numel(), device=dev) - soff[bv]                            # slot of the kept token
    col = torch.full((BV, Ns_pad), s_pad - 1, device=dev, dtype=torch.long)             # s_pad - 1: never allowed
    col[bv, k] = j
    Fg = torch.gather(Ft, 2, col.view(BV, 1, Ns_pad).expand(BV, Lq, Ns_pad)).clone()
    Fg[:, :N_REG, :] = False
    Fg[:, :, Ns_pad - 1] = ~Fg[:, :, :Ns_pad - 1].any(-1)
    max_n = max(lay.n)
    Kp = roundup(max_n + 1, 128)
    Rg = torch.gather(Rt, 1, col.view(BV, Ns_pad, 1).expand(BV, Ns_pad, Rt.shape[2]))
    Rn = torch.zeros(BV, Ns_pad, Kp, dtype=torch.bool, device=dev)
    Rn[:, :, :max_n] = Rg[:, :, REV_FG_OFF:REV_FG_OFF + max_n]
    Rn &= (torch.arange(Ns_pad, device=dev).view(1, -1) < seq_t.view(-1, 1)).view(BV, Ns_pad, 1)
    Rn[:, :, Kp - 1] = ~Rn[:, :, :Kp - 1].any(-1)
    off_t = torch.tensor(lay.offsets, device=dev)
    fg_slot = lay.fg_packed - off_t[lay.fg_bv] - N_REG                                  # slot of every fg target token
    sc_dst = bv * Ns_pad + k                                                            # packed scene row -> padded slot
    # destinations (unique) instead of gather indices with a shared zero row: see model_s2/blocks._scatter_pad
    return dict(BV=BV,
                f_Lq=Lq, f_Lk=Ns_pad, f_q_dst=lay.pad_dst, f_kv_src=None, f_kv_dst=sc_dst, f_out=lay.pad_dst,
                f_mask=MaskTable(Fg, backend),
                r_Lq=Ns_pad, r_Lk=Kp, r_q_dst=sc_dst, r_kv_src=lay.fg_packed, r_kv_dst=lay.fg_bv * Kp + fg_slot,
                r_out=sc_dst, r_mask=MaskTable(Rn, backend))
