# Stage-2 attention operators (plan 2026-09-27, section 一.9 / 二 "注意力实现").
#
#   attend_blockdiag : dense attention inside segments of a packed sequence
#                      ("fa3" = xformers memory_efficient_attention + BlockDiagonalMask with the
#                      FlashAttention-3 ops, the same kernel stage 1 uses; "ref" = explicit fp32 loop)
#   attend_masked    : exact per-token masked attention from one boolean table
#                      ("sdpa" = torch SDPA pinned to the memory-efficient kernel with an additive bias;
#                       "flex" = torch FlexAttention with a BlockMask derived from the same table;
#                       "ref"  = explicit fp32 softmax)
# Tables always give every row at least one allowed column (null key k = v = 0), so no backend ever sees
# a fully masked row. Softmax scale = 1 / sqrt(head_dim) everywhere, as in stage 1.

import math

import torch
import torch.nn.functional as F


def _up(t):
    return t.float() if t.dtype in (torch.float16, torch.bfloat16) else t


# ------------------------------------------------------------------------------------ dense segments
def _fa3_ops():
    import xformers.ops as xops
    return (xops.fmha.flash3.FwOp, xops.fmha.flash3.BwOp)


def attend_blockdiag(q, k, v, q_seqlens, kv_seqlens, backend="fa3"):
    """q [Tq, H, D], k/v [Tk, H, D] packed; segment i of q attends segment i of k/v only."""
    if backend == "fa3":
        import xformers.ops as xops
        from xformers.ops.fmha.attn_bias import BlockDiagonalMask
        bias = BlockDiagonalMask.from_seqlens(list(q_seqlens), list(kv_seqlens), device=q.device)
        return xops.memory_efficient_attention(q[None], k[None], v[None], attn_bias=bias, op=_fa3_ops())[0]
    if backend == "ref":
        out = []
        qo = ko = 0
        for nq, nk in zip(q_seqlens, kv_seqlens):
            qs, ks, vs = _up(q[qo:qo + nq]), _up(k[ko:ko + nk]), _up(v[ko:ko + nk])
            s = torch.einsum("qhd,khd->hqk", qs, ks) / math.sqrt(q.shape[-1])
            out.append(torch.einsum("hqk,khd->qhd", torch.softmax(s, -1), vs))
            qo += nq
            ko += nk
        return torch.cat(out).to(q.dtype)
    raise ValueError(backend)


# ------------------------------------------------------------------------------------ masked
class MaskTable:
    """One boolean table [B, Lq, Lk] (True = allowed) and its backend-specific encodings, built once per
    step outside activation checkpointing and shared by every layer, forward, backward and recompute."""

    def __init__(self, table, backend, dtype=torch.bfloat16):
        assert table.dtype == torch.bool and table.dim() == 3
        assert bool(table.any(-1).all()), "every row must allow at least one key"
        self.shape = tuple(table.shape)
        self.backend = backend
        self.table = table
        self.bias = None
        self.block_mask = None
        if backend == "sdpa":
            assert self.shape[2] % 8 == 0, "key length must be 8-aligned for the efficient kernel"
            self.bias = torch.zeros(self.shape, dtype=dtype, device=table.device)
            self.bias.masked_fill_(~table, float("-inf"))
            self.bias = self.bias.unsqueeze(1)                    # [B, 1, Lq, Lk], broadcast over heads
            self.table = None                                     # the bias is all the kernel needs
        elif backend == "flex":
            self.block_mask = _flex_block_mask(table)


def attend_masked(q, k, v, mask, backend=None):
    """q [B, Lq, H, D], k/v [B, Lk, H, D] -> [B, Lq, H, D]."""
    backend = backend or mask.backend
    if backend == "sdpa":
        from torch.nn.attention import SDPBackend, sdpa_kernel
        # the context sits inside the function so checkpoint recomputation re-enters it
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                               attn_mask=mask.bias.expand(-1, q.shape[2], -1, -1))
        return o.transpose(1, 2)
    if backend == "ref":
        table = mask.table if mask.table is not None else torch.isfinite(mask.bias[:, 0])
        s = torch.einsum("bqhd,bkhd->bhqk", _up(q), _up(k)) / math.sqrt(q.shape[-1])
        s = s.masked_fill(~table[:, None], float("-inf"))
        return torch.einsum("bhqk,bkhd->bqhd", torch.softmax(s, -1), _up(v)).to(q.dtype)
    if backend == "flex":
        return _flex_attend(q, k, v, mask)
    raise ValueError(backend)


# ------------------------------------------------------------------------------------ flex (G4-B)
_FLEX_BLOCK = 128
_compiled_flex = None


def _flex_block_mask(table):
    from torch.nn.attention.flex_attention import BlockMask
    B, Lq, Lk = table.shape
    assert Lq % _FLEX_BLOCK == 0 and Lk % _FLEX_BLOCK == 0, (Lq, Lk)
    nq, nk = Lq // _FLEX_BLOCK, Lk // _FLEX_BLOCK
    blk = table.view(B, nq, _FLEX_BLOCK, nk, _FLEX_BLOCK).any(4).any(2)          # [B, nq, nk]
    kv_num = blk.sum(-1).to(torch.int32).view(B, 1, nq)
    kv_idx = torch.argsort((~blk).to(torch.int8), dim=-1, stable=True).to(torch.int32).view(B, 1, nq, nk)

    def mask_mod(b, h, qi, ki):
        return table[b, qi, ki]

    return BlockMask.from_kv_blocks(kv_num, kv_idx, BLOCK_SIZE=_FLEX_BLOCK, mask_mod=mask_mod,
                                    seq_lengths=(Lq, Lk))


def _flex_attend(q, k, v, mask):
    global _compiled_flex
    from torch.nn.attention.flex_attention import flex_attention
    if _compiled_flex is None:
        # one graph per (Lq | Kr) padding bucket, train and validation batch sizes, forward and reverse call sites:
        # far above dynamo's default recompile limit of 8, past which it silently falls back to an unfused path
        import torch._dynamo
        for key in ("cache_size_limit", "recompile_limit"):
            if hasattr(torch._dynamo.config, key):
                setattr(torch._dynamo.config, key, max(64, getattr(torch._dynamo.config, key)))
        _compiled_flex = torch.compile(flex_attention, dynamic=False)
    o = _compiled_flex(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=mask.block_mask)
    return o.transpose(1, 2)
