"""L1 P4 sanity: alpha_mask shape correct + values in [0,1] + sensible fg ratio.

Also tests the loss.py path: exclude_bg=True with vs without alpha_mask should
give different l2_loss values (proves alpha path is active, not silently
falling back).

Usage:
    cd ~/code/RnG_feature_allignment
    python scripts/verify_render44798/verify_l1_p4_alpha.py
"""
import os
import sys
import math

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data.dataset_objaverse import ObjaverseDataset
from model.loss import LossComputer


def main():
    cfg = OmegaConf.load("configs/RnGUP_obj_448_bf16_15k_render44798.yaml")
    cfg.training.roll_augment_max_deg = 0.0
    cfg.training.num_views = 4
    ds = ObjaverseDataset(cfg)

    print(f"P4a sanity on obj: {ds.all_object_list[0][:16]}...")
    sample = ds[0]
    assert "alpha_mask" in sample, "alpha_mask missing from sample"

    am = sample["alpha_mask"]
    img = sample["image"]
    v, _, h, w = img.shape
    assert am.shape == (v, 1, h, w), f"alpha_mask shape {am.shape}, expected ({v},1,{h},{w})"
    assert am.dtype == torch.float32, f"dtype {am.dtype}, expected float32"
    assert 0.0 <= am.min().item() <= am.max().item() <= 1.0, f"value out of [0,1]: [{am.min()}, {am.max()}]"

    fg_ratios = am.mean(dim=(1, 2, 3)).tolist()
    print(f"  per-view fg ratio: " + ", ".join(f"{r:.3f}" for r in fg_ratios))
    for r in fg_ratios:
        assert 0.02 < r < 0.95, f"weird fg ratio: {r}"
    print(f"P4a PASS: alpha_mask shape OK, all fg ratios in (0.02, 0.95)")

    # P4b: test loss path with vs without alpha_mask
    cfg.training.l2_loss_weight = 1.0
    cfg.training.lpips_loss_weight = 0.0
    cfg.training.perceptual_loss_weight = 0.0
    loss = LossComputer(cfg)

    # Fake rendering: random image close to target
    target = sample["image"].unsqueeze(0)              # [1, v, 3, h, w]
    alpha_mask = sample["alpha_mask"].unsqueeze(0)     # [1, v, 1, h, w]
    rendering = target + 0.05 * torch.randn_like(target)
    rendering = rendering.clamp(0, 1)

    m_alpha = loss(rendering, target, exclude_bg=True, target_alpha_mask=alpha_mask)
    m_fallback = loss(rendering, target, exclude_bg=True, target_alpha_mask=None)
    m_full = loss(rendering, target, exclude_bg=False, target_alpha_mask=None)

    print(f"  l2 (exclude_bg + alpha_mask) = {m_alpha.l2_loss.item():.5f}")
    print(f"  l2 (exclude_bg, white-bg fb) = {m_fallback.l2_loss.item():.5f}")
    print(f"  l2 (full image, exclude_bg=False) = {m_full.l2_loss.item():.5f}")

    # alpha path and fallback should differ — proves alpha is being used
    diff = abs(m_alpha.l2_loss.item() - m_fallback.l2_loss.item())
    assert diff > 1e-6, f"alpha path and white-bg fallback gave identical loss (diff={diff:.2e}) — alpha_mask may be silently ignored"
    print(f"P4b PASS: alpha-mask l2 differs from fallback l2 (diff={diff:.5f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
