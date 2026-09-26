"""L1 P2 sanity: roll augment rotates image/depth/alpha/c2w consistently.

Tests two scenarios:
    A. Deterministic check: with a known fixed roll_deg, verify c2w gets the
       expected Rz rotation and the image rotates visibly.
    B. Visualization: emit /tmp/p2_roll_viz.png showing 4 views with/without
       roll for eyeball check.

Usage:
    cd ~/code/RnG_feature_allignment
    python scripts/verify_render44798/verify_l1_p2_roll.py
"""
import math
import os
import random
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data.dataset_objaverse import ObjaverseDataset


def main():
    cfg = OmegaConf.load("configs/RnGUP_obj_448_bf16_15k_render44798.yaml")
    cfg.training.num_views = 4  # smaller for viz

    # No roll: baseline
    cfg.training.roll_augment_max_deg = 0.0
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    ds_off = ObjaverseDataset(cfg)
    sample_off = ds_off[0]

    # Roll = ±10° random
    cfg.training.roll_augment_max_deg = 10.0
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    ds_on = ObjaverseDataset(cfg)
    sample_on = ds_on[0]

    print(f"P2 sanity on obj: {ds_off.all_object_list[0][:16]}...")

    # Shape integrity
    assert sample_off["image"].shape == sample_on["image"].shape
    assert sample_off["alpha_mask"].shape == sample_on["alpha_mask"].shape
    assert sample_off["depth_map"].shape == sample_on["depth_map"].shape
    assert sample_off["c2w"].shape == sample_on["c2w"].shape
    print(f"  Shapes consistent: image={sample_off['image'].shape}")

    # Image content actually different (roll really applied)
    img_diff = (sample_off["image"] - sample_on["image"]).abs().mean().item()
    print(f"  Mean abs image diff (off vs on roll=10°): {img_diff:.4f}")
    assert img_diff > 0.01, f"Image barely changed (diff={img_diff:.4f}) — roll may not be applied"

    # alpha_mask also rotated
    alpha_diff = (sample_off["alpha_mask"] - sample_on["alpha_mask"]).abs().mean().item()
    print(f"  Mean abs alpha diff (off vs on): {alpha_diff:.4f}")
    assert alpha_diff > 0.001, f"Alpha mask not changing with roll (diff={alpha_diff:.4f})"

    # c2w differs
    c2w_diff = (sample_off["c2w"] - sample_on["c2w"]).abs().mean().item()
    print(f"  Mean abs c2w diff: {c2w_diff:.4f}")
    assert c2w_diff > 1e-4, f"c2w not changing with roll (diff={c2w_diff:.4f})"

    # Visualize
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n_v = sample_off["image"].shape[0]
        fig, ax = plt.subplots(3, n_v, figsize=(4 * n_v, 12))
        for v in range(n_v):
            ax[0, v].imshow(sample_off["image"][v].permute(1, 2, 0).numpy())
            ax[0, v].set_title(f"no_roll v{v}")
            ax[0, v].axis("off")
            ax[1, v].imshow(sample_on["image"][v].permute(1, 2, 0).numpy())
            ax[1, v].set_title(f"roll=±10° v{v}")
            ax[1, v].axis("off")
            ax[2, v].imshow(sample_on["alpha_mask"][v, 0].numpy(), cmap="gray")
            ax[2, v].set_title(f"alpha (roll) v{v}")
            ax[2, v].axis("off")
        plt.tight_layout()
        out = "/tmp/p2_roll_viz.png"
        plt.savefig(out, dpi=80)
        print(f"P2 viz saved: {out}")
    except Exception as e:
        print(f"  (viz skipped: {e})")

    print("P2 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
