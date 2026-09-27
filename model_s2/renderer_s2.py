# Stage-2 renderer (plan 2026-09-27, section 一.4 / 二): target-token embedding (rays + GT mask, 7 channels),
# copied registers / tgt_norm / 12 blocks, per-layer injection via the blocks, and the layer-aligned
# residual outputs Lin_m(t2(m)) for m = 2, 5, 8, 11 (zero-initialised, so the P8 arm starts exactly at
# stage 1).  Works on the packed foreground target stream described in model_s2/geometry.py.

import copy

import torch
import torch.nn as nn
import torch.utils.checkpoint

from model_s2.blocks import S2BidirBlock, S2FinalBlock
from model_s2.geometry import N_REG

OUT_LAYERS = (2, 5, 8, 11)


class Stage2Renderer(nn.Module):
    def __init__(self, s1_renderer, patch, num_heads=12):
        super().__init__()
        C = s1_renderer.tgt_norm.normalized_shape[0]
        self.patch = patch
        self.hidden = C
        w1 = s1_renderer.tgt_embedder.proj.weight                          # [C, 6, 8, 8], bias-free
        self.tgt_proj = nn.Conv2d(7, C, kernel_size=patch, stride=patch, bias=False)
        with torch.no_grad():
            if patch == s1_renderer.patch_size:
                # P8: copy the 6 ray channels, new mask channel ~ N(0, 0.02) (stage-1 embedder init)
                self.tgt_proj.weight[:, :6].copy_(w1)
                nn.init.normal_(self.tgt_proj.weight[:, 6:], mean=0.0, std=0.02)
            else:
                # P4: a new 7-channel 4x4 embedder, same init as stage 1's (normal std 0.02, no bias)
                nn.init.normal_(self.tgt_proj.weight.view(C, -1), mean=0.0, std=0.02)
        self.tgt_norm = copy.deepcopy(s1_renderer.tgt_norm).requires_grad_(True)
        self.registers = nn.Parameter(s1_renderer.per_view_register_tokens.detach().clone().float())

        blocks = []
        s1_blocks = s1_renderer.renderer_core.renderer_blocks
        for i, b1 in enumerate(s1_blocks):
            b2 = S2BidirBlock(C, num_heads) if i < len(s1_blocks) - 1 else S2FinalBlock(C, num_heads)
            missing, unexpected = b2.load_state_dict(b1.state_dict(), strict=False)
            if unexpected or any(k.split(".")[0] not in b2.new_module_names() for k in missing):
                raise RuntimeError(f"block {i}: stage-1 weights do not map onto the stage-2 block "
                                   f"(missing {missing[:6]}, unexpected {unexpected[:6]})")
            b2.init_new()
            blocks.append(b2.to(w1.device))
        self.blocks = nn.ModuleList(blocks)
        self.out_layers = OUT_LAYERS
        self.out_lin = nn.ModuleList([nn.Linear(C, C) for _ in OUT_LAYERS])
        for lin in self.out_lin:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.to(w1.device)

    def forward(self, rays6, mask1, rec_rep, ctx):
        """rays6 [B, Vt, 6, H, W], mask1 [B, Vt, 1, H, W], rec_rep [BV, S, C] -> {m: Lin_m(t2(m)) [nfg, C]}"""
        lay = ctx.layout
        C = self.hidden
        x_in = torch.cat([rays6, mask1.to(rays6.dtype)], dim=2).flatten(0, 1)
        tok = self.tgt_proj(x_in).flatten(2).transpose(1, 2)                   # [BV, g*g, C]
        tok = self.tgt_norm(tok)
        g2 = tok.shape[1]
        fg = tok.reshape(-1, C)[lay.fg_bv * g2 + lay.fg_tok_flat]
        regs = self.registers.expand(lay.BV, N_REG, C).reshape(-1, C)
        x = fg.new_zeros(lay.T, C)
        x = x.index_copy(0, lay.fg_packed, fg).index_copy(0, lay.reg_packed, regs.to(fg.dtype))
        rec = rec_rep
        outs = {}
        last = len(self.blocks) - 1
        for i, blk in enumerate(self.blocks):
            if i < last:
                if self.training:
                    x, rec = torch.utils.checkpoint.checkpoint(blk, x, rec, ctx, use_reentrant=False)
                else:
                    x, rec = blk(x, rec, ctx)
            else:
                if self.training:
                    x = torch.utils.checkpoint.checkpoint(blk, x, rec, ctx, use_reentrant=False)
                else:
                    x = blk(x, rec, ctx)
            if i in self.out_layers:
                outs[i] = x
        return {m: lin(outs[m][lay.fg_packed]) for m, lin in zip(self.out_layers, self.out_lin)}
