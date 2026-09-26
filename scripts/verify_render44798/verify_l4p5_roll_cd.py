"""L4.5 follow-up: Chamfer Distance with/without rigid alignment.

Background:
  Per-view independent roll + first-view normalize  =>  world frame rotates
  globally by a function of view 0's roll. This is EXPECTED, not a bug:
  - Relative scene structure is preserved
  - The "world" coord frame in this dataset IS view 0's frame (target_first_view),
    so rolling view 0 trivially rotates everything.

This script verifies that "scene structure preserved" claim by:
  1. Raw CD between roll-off and roll-on point clouds (will be LARGE — global rotation)
  2. Aligned CD (Procrustes / Kabsch rotation) — should be SMALL (~ noise of NEAREST
     interpolation in image rotation).

Usage:
    cd ~/code/RnG_feature_allignment
    python scripts/verify_render44798/verify_l4p5_roll_cd.py
"""
import os
import random
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data.dataset_objaverse import ObjaverseDataset


def stack_obj_pointcloud(ds, obj_idx, seed):
    """Re-seed RNGs so view_selector picks the SAME indices for both configs."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    sample = ds[obj_idx]
    images = sample["image"]
    point_maps = sample["point_map"]
    alpha_masks = sample["alpha_mask"]
    v_count = images.shape[0]
    pts_all = []
    for vi in range(v_count):
        pm = point_maps[vi].permute(1, 2, 0).numpy()
        mask = (alpha_masks[vi, 0].numpy() > 0.5)
        pts = pm[mask]
        nz = np.linalg.norm(pts, axis=1) > 1e-6
        pts_all.append(pts[nz])
    return np.concatenate(pts_all, axis=0).astype(np.float32), sample["scene_name"]


def chamfer_distance(a, b, sample=20000):
    """Symmetric one-way nearest-neighbor mean distance (Chamfer L2).
    a, b: [N, 3] / [M, 3] numpy."""
    from scipy.spatial import cKDTree
    if len(a) > sample:
        a = a[np.random.choice(len(a), sample, replace=False)]
    if len(b) > sample:
        b = b[np.random.choice(len(b), sample, replace=False)]
    tree_b = cKDTree(b)
    d_ab, _ = tree_b.query(a, k=1)
    tree_a = cKDTree(a)
    d_ba, _ = tree_a.query(b, k=1)
    return 0.5 * (d_ab.mean() + d_ba.mean())


def kabsch_align(src, dst, sample=20000):
    """Find rigid R, t such that R @ src + t ≈ dst (least squares).
    Returns aligned src and (R, t).

    Uses bounding-box centroid + SVD on covariance (Kabsch).
    Assumes src/dst are roughly the same scene, just rigidly displaced.
    """
    if len(src) > sample:
        idx = np.random.choice(len(src), sample, replace=False)
        src_s = src[idx]
    else:
        src_s = src
    if len(dst) > sample:
        idx = np.random.choice(len(dst), sample, replace=False)
        dst_s = dst[idx]
    else:
        dst_s = dst

    # Center on centroids
    src_c = src_s.mean(axis=0)
    dst_c = dst_s.mean(axis=0)
    src_z = src_s - src_c
    dst_z = dst_s - dst_c

    # For Kabsch we'd need point correspondences. Instead, when the two point
    # clouds are the same scene under a global rotation, the inertia tensor
    # axes (eigenvectors of point covariance) carry that rotation. Align those.
    cov_src = src_z.T @ src_z / max(len(src_z) - 1, 1)
    cov_dst = dst_z.T @ dst_z / max(len(dst_z) - 1, 1)
    _, _, Vs = np.linalg.svd(cov_src)
    _, _, Vd = np.linalg.svd(cov_dst)
    R = Vd.T @ Vs   # rotation that maps src's frame to dst's frame
    # ensure right-handed (det = +1)
    if np.linalg.det(R) < 0:
        Vd[2, :] *= -1
        R = Vd.T @ Vs

    src_aligned = (R @ (src - src_c).T).T + dst_c
    t = dst_c - R @ src_c
    return src_aligned, R, t


def main():
    cfg = OmegaConf.load("configs/RnGUP_obj_448_bf16_15k_render44798.yaml")
    cfg.training.num_views = 8

    cfg_off = OmegaConf.merge(cfg, OmegaConf.create({"training": {"roll_augment_max_deg": 0.0}}))
    cfg_on  = OmegaConf.merge(cfg, OmegaConf.create({"training": {"roll_augment_max_deg": 10.0}}))

    ds_off = ObjaverseDataset(cfg_off)
    ds_on  = ObjaverseDataset(cfg_on)

    SEEDS = [42, 1234, 8888]   # 3 obj
    print(f"Computing CD between roll=0 and roll=10° point clouds on {len(SEEDS)} objs\n")

    print(f"{'obj':<8} {'pts_off':>8} {'pts_on':>8} {'raw_CD':>10} {'aligned_CD':>12} {'rotation_deg':>12}")
    print("-" * 70)
    for obj_i, seed in enumerate(SEEDS):
        pts_off, uid = stack_obj_pointcloud(ds_off, obj_i, seed=seed)
        pts_on, _    = stack_obj_pointcloud(ds_on,  obj_i, seed=seed)

        # Raw CD (no alignment — picks up the global rotation)
        cd_raw = chamfer_distance(pts_off, pts_on)

        # Aligned CD (rigid R, t via Kabsch on PCA axes)
        pts_off_aligned, R, t = kabsch_align(pts_off, pts_on)
        cd_aligned = chamfer_distance(pts_off_aligned, pts_on)
        # Recover rotation angle from R (axis-angle)
        cos_theta = (np.trace(R) - 1) / 2
        cos_theta = np.clip(cos_theta, -1, 1)
        rot_deg = np.degrees(np.arccos(cos_theta))

        print(f"{uid[:8]:<8} {len(pts_off):>8d} {len(pts_on):>8d} "
              f"{cd_raw:>10.4f} {cd_aligned:>12.5f} {rot_deg:>11.1f}°")

    print()
    print("Interpretation:")
    print("  - raw_CD       large  -> point clouds rigidly rotated (EXPECTED, see header comment)")
    print("  - aligned_CD   small  -> scene structure preserved after correcting for rotation")
    print("    (small = within NEAREST resampling noise, typically < 0.01)")
    print("  - rotation_deg : rigid rotation magnitude between the two clouds")
    print("                   (correlates with view 0's randomly-sampled roll, ~0-10°)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
