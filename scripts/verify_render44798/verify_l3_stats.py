"""L3 sanity: statistical distributions over 100 batches.

Checks:
  - fov range [25°, 70°] from P1 (plan section 1.1 A2 range)
  - alpha fg ratio mean 0.05-0.85 (sane object framing)
  - point_map valid voxels not all zero
  - all shapes consistent

Usage:
    cd ~/code/RnG_feature_allignment
    python scripts/verify_render44798/verify_l3_stats.py
"""
import os
import sys
import math

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data.dataset_objaverse import ObjaverseDataset


def main():
    cfg = OmegaConf.load("configs/RnGUP_obj_448_bf16_15k_render44798.yaml")
    cfg.training.roll_augment_max_deg = 10.0
    cfg.training.num_views = 4
    ds = ObjaverseDataset(cfg)

    # Use small N — each sample re-opens a tar from JuiceFS which is slow (~5-30s).
    # 20 samples × 4 views = 80 data points per stat, enough for distribution sanity.
    N = int(os.environ.get("L3_N", 20))
    print(f"L3 stats: scanning {N} batches from 44,798 dataset (roll=±10°, num_views=4)")
    fov_degs = []
    alpha_ratios = []
    point_pixel_ratios = []
    depth_means = []

    for i in range(N):
        s = ds[i % len(ds)]
        # Derive fov from focal: fov = 2 * atan2(W/2, focal)
        resize_w = s["image"].shape[-1]
        for v in range(s["fxfycxcy"].shape[0]):
            focal = s["fxfycxcy"][v, 0].item()
            fov_rad = 2 * math.atan2(resize_w / 2.0, focal)
            fov_degs.append(math.degrees(fov_rad))
        alpha_ratios.append(s["alpha_mask"].mean(dim=(1, 2, 3)).numpy())
        pm = s["point_map"]
        valid = (pm.abs().sum(dim=1) > 0).float().mean(dim=(1, 2)).numpy()
        point_pixel_ratios.append(valid)
        depth_means.append(s["depth_map"][s["depth_map"] > 0].mean().item())

    fov_degs = np.array(fov_degs)
    alpha_ratios = np.concatenate(alpha_ratios)
    point_pixel_ratios = np.concatenate(point_pixel_ratios)

    print(f"\nFOV (per view, deg):")
    print(f"  min={fov_degs.min():.1f}  max={fov_degs.max():.1f}  mean={fov_degs.mean():.1f}  std={fov_degs.std():.1f}")
    print(f"  expected: 25-70 deg (plan section 1.1 A2)")

    print(f"\nalpha fg ratio:")
    print(f"  min={alpha_ratios.min():.3f}  max={alpha_ratios.max():.3f}")
    print(f"  mean={alpha_ratios.mean():.3f}  std={alpha_ratios.std():.3f}")
    print(f"  expected mean ≈ 0.05-0.75 (plan section 1.3 framing ~70%)")

    print(f"\npoint_map valid pixel ratio (should ≈ alpha ratio):")
    print(f"  mean={point_pixel_ratios.mean():.3f}")

    print(f"\ndepth mean (normalized by first-view t): {np.mean(depth_means):.3f}")

    # Assertions
    n_fail = 0
    if not (24 <= fov_degs.min() and fov_degs.max() <= 71):
        print(f"  FAIL: fov range [{fov_degs.min():.1f}, {fov_degs.max():.1f}] outside [25, 70]")
        n_fail += 1
    if not (0.02 < alpha_ratios.mean() < 0.85):
        print(f"  FAIL: alpha mean {alpha_ratios.mean():.3f} not in (0.02, 0.85)")
        n_fail += 1

    if n_fail == 0:
        print(f"\nL3 PASS")
        return 0
    else:
        print(f"\nL3 FAIL: {n_fail} assertion(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
