# Stage-2 output side (plan 2026-09-27, section 一.6): T~(m) = T1(m) + Lin_m(t2(m)) on the foreground
# tokens, then the colour head and the DPT point head.
#   P8: trainable copies of stage 1's heads on stage 1's own token layout (registers kept) -> at init the
#       outputs are bitwise stage 1's.
#   P4: T1 of each patch-8 parent replicated to its four 4x4 children (no registers); a new colour head
#       (LayerNorm + Linear 768 -> 48, zero-init) and the stage-1 DPT copied with patch_size 8 (so its
#       12-channel output layer and pixel-shuffle 2 carry over), then re-pointed at the 64x64 grid with
#       new resize layers x2 / x1 / x1/2 / x1/4 so the four level maps stay 128 / 64 / 32 / 16.

import copy
import math

import einops
import torch
import torch.nn as nn

from model_s2.geometry import N_REG
from models.layers.final_layer import FinalLayer


def make_color_head(s1_final_layer, patch, s1_patch):
    if patch == s1_patch:
        return copy.deepcopy(s1_final_layer).requires_grad_(True)
    C = s1_final_layer.norm_final.normalized_shape[0]
    head = FinalLayer(hidden_size=C, patch_size=patch, out_channels=3)
    nn.init.zeros_(head.linear.weight)
    return head.to(s1_final_layer.linear.weight.device)


def make_point_head(s1_point_head, patch, s1_patch):
    head = copy.deepcopy(s1_point_head).requires_grad_(True)
    if patch == s1_patch:
        return head
    ch = [p.out_channels for p in head.projects]                              # [256, 512, 1024, 1024]
    dev = head.proj.weight.device
    head.patch_size = patch                                                  # token-grid reshape only
    head.resize_layers = nn.ModuleList([
        nn.ConvTranspose2d(ch[0], ch[0], kernel_size=2, stride=2, padding=0),   # 64 -> 128
        nn.Identity(),                                                          # 64
        nn.Conv2d(ch[2], ch[2], kernel_size=3, stride=2, padding=1),            # 64 -> 32
        nn.Conv2d(ch[3], ch[3], kernel_size=4, stride=4, padding=0),            # 64 -> 16
    ]).to(dev)
    return head


def parent_index(g, g1):
    """token index on the stage-1 g1 x g1 grid (without registers) of every child on the g x g grid."""
    f = g // g1
    r = torch.arange(g).view(g, 1).expand(g, g) // f
    c = torch.arange(g).view(1, g).expand(g, g) // f
    return (r * g1 + c).reshape(-1)


def assemble_tokens(T1_m, res_m, lay, patch, s1_patch):
    """T1_m [BV, N_REG + g1*g1, C] frozen stage-1 tokens, res_m [nfg, C] -> T~(m)."""
    BV, N1, C = T1_m.shape
    if patch == s1_patch:
        base = T1_m.reshape(-1, C)
        idx = lay.fg_bv * N1 + N_REG + lay.fg_tok_flat
        return base.index_add(0, idx, res_m.to(base.dtype)).view(BV, N1, C)
    g1 = int(round(math.sqrt(N1 - N_REG)))
    g = lay.g
    par = parent_index(g, g1).to(T1_m.device) + N_REG
    base = T1_m[:, par].reshape(-1, C)                                       # [BV * g*g, C]
    idx = lay.fg_bv * (g * g) + lay.fg_tok_flat
    return base.index_add(0, idx, res_m.to(base.dtype)).view(BV, g * g, C)


def build_x_prior(T1f, lay, patch, s1_patch):
    """packed [T, C] prior for the target stream: registers <- final registers, fg <- final token (parent)."""
    BV, N1, C = T1f.shape
    flat = T1f.reshape(-1, C)
    if patch == s1_patch:
        fg = flat[lay.fg_bv * N1 + N_REG + lay.fg_tok_flat]
    else:
        g1 = int(round(math.sqrt(N1 - N_REG)))
        par = parent_index(lay.g, g1).to(T1f.device)[lay.fg_tok_flat]
        fg = flat[lay.fg_bv * N1 + N_REG + par]
    x = T1f.new_zeros(lay.T, C)
    x[lay.reg_packed] = T1f[:, :N_REG].reshape(-1, C)
    x[lay.fg_packed] = fg
    return x


def render_color(color_head, T11, patch, B, Vt, H, W, has_registers):
    x = T11[:, N_REG:] if has_registers else T11
    x = torch.sigmoid(color_head(x))
    img = einops.rearrange(x, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)", b=B, v=Vt,
                           h=H // patch, w=W // patch, p1=patch, p2=patch, c=3)
    return img.float().clamp(0.0, 1.0)


def render_points(point_head, Ts, rays6, B, patch_token_start, n_layers=12):
    tok = [None] * n_layers
    for m, t in Ts.items():
        tok[m] = einops.rearrange(t, "(b v) n c -> b v n c", b=B)
    with torch.autocast("cuda", enabled=False):
        pts, conf = point_head(tok, rays6.float(), patch_token_start=patch_token_start)
    return (einops.rearrange(pts, "b v h w c -> b v c h w", c=3),
            einops.rearrange(conf, "b v h w -> b v 1 h w"))
