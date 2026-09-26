import os
from typing import Optional

import torch
import torch.nn.functional as F


_FALLBACK_REASONS = set()
_FA3_LOGGED = False


try:
    from flash_attn_interface import flash_attn_func as _flash_attn_func

    _FA3_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on local CUDA extension
    _flash_attn_func = None
    _FA3_IMPORT_ERROR = exc


def _backend() -> str:
    backend = os.environ.get("RNG_ATTENTION_BACKEND", "auto").lower()
    if backend not in {"auto", "fa3", "sdpa"}:
        raise ValueError(
            "RNG_ATTENTION_BACKEND must be one of: auto, fa3, sdpa; "
            f"got {backend!r}"
        )
    return backend


def _log_fallback(reason: str) -> None:
    if reason in _FALLBACK_REASONS:
        return
    _FALLBACK_REASONS.add(reason)
    print(f"[FA3] fallback to PyTorch SDPA: {reason}")


def _log_fa3_once() -> None:
    global _FA3_LOGGED
    if _FA3_LOGGED:
        return
    _FA3_LOGGED = True
    print("[FA3] using FlashAttention-3 backend")


def _can_use_fa3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    dropout_p: float,
) -> tuple[bool, str]:
    if _flash_attn_func is None:
        return False, f"flash_attn_interface import failed: {_FA3_IMPORT_ERROR}"
    if attn_mask is not None:
        return False, "custom attention mask is not supported by FA3 wrapper"
    if dropout_p != 0.0:
        return False, f"FA3 wrapper does not support attention dropout, got {dropout_p}"
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        return False, "q/k/v are not all CUDA tensors"
    if q.dtype not in (torch.float16, torch.bfloat16):
        return False, f"FA3 requires fp16/bf16, got {q.dtype}"
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return False, f"q/k/v dtype mismatch: {q.dtype}, {k.dtype}, {v.dtype}"
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        return False, "q/k/v head counts do not match"
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        return False, "q/k/v head dimensions do not match"
    if q.shape[-1] % 8 != 0:
        return False, f"FA3 head_dim should be divisible by 8, got {q.shape[-1]}"
    return True, ""


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Run attention on [B, H, N, D] tensors with FA3 when available."""
    backend = _backend()

    if backend != "sdpa":
        ok, reason = _can_use_fa3(q, k, v, attn_mask, dropout_p)
        if ok:
            _log_fa3_once()
            q_fa3 = q.transpose(1, 2).contiguous()
            k_fa3 = k.transpose(1, 2).contiguous()
            v_fa3 = v.transpose(1, 2).contiguous()
            out = _flash_attn_func(
                q_fa3,
                k_fa3,
                v_fa3,
                softmax_scale=scale,
                causal=is_causal,
            )
            return out.transpose(1, 2).contiguous()

        if backend == "fa3":
            raise RuntimeError(f"RNG_ATTENTION_BACKEND=fa3 requested, but FA3 cannot run: {reason}")
        _log_fallback(reason)

    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


def masked_attention_1029(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask_attention: Optional[int],
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Equivalent to the old mask[:-mask_attention, -mask_attention:] = False."""
    if mask_attention is None:
        return flash_attention(q, k, v, dropout_p=dropout_p)

    target_len = int(mask_attention)
    seq_len = q.shape[2]
    if target_len <= 0 or target_len >= seq_len:
        return flash_attention(q, k, v, dropout_p=dropout_p)
    if k.shape[2] != seq_len or v.shape[2] != seq_len:
        raise ValueError("masked_attention_1029 expects self-attention q/k/v sequence lengths")

    context_len = seq_len - target_len
    context_out = flash_attention(
        q[:, :, :context_len, :],
        k[:, :, :context_len, :],
        v[:, :, :context_len, :],
        dropout_p=dropout_p,
    )
    target_out = flash_attention(
        q[:, :, context_len:, :],
        k,
        v,
        dropout_p=dropout_p,
    )
    return torch.cat([context_out, target_out], dim=2)
