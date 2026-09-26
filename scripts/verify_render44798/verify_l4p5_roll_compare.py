"""L4.5 roll-vs-no-roll comparison.

Same 2 objects, same view indices, but loaded twice:
  - left  side: roll_augment_max_deg = 0    (no roll)
  - right side: roll_augment_max_deg = 10   (roll on)

Hypothesis (per plan section 18.3 P2 design):
  Because c2w is rotated BY THE SAME ANGLE as image+depth+alpha (via post-multiply
  Rz_cam(roll_rad) BEFORE first-view normalize), the back-projected world-frame
  point cloud should be IDENTICAL regardless of roll.

  If they differ visibly -> roll plumbing is broken.

Lay out (top-down on x-axis):
  obj0_roll0  obj1_roll0  |  obj0_roll10  obj1_roll10
                          GAP (3 units)

Usage:
    cd ~/code/RnG_feature_allignment
    python scripts/verify_render44798/verify_l4p5_roll_compare.py
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
    """Re-seed all RNGs so view_selector returns the same view indices.
    Returns (pts, cols) in the world frame."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    sample = ds[obj_idx]
    images = sample["image"]
    point_maps = sample["point_map"]
    alpha_masks = sample["alpha_mask"]
    v_count = images.shape[0]

    pts_all, cols_all = [], []
    for vi in range(v_count):
        pm = point_maps[vi].permute(1, 2, 0).numpy()
        rgb = (images[vi].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        mask = (alpha_masks[vi, 0].numpy() > 0.5)
        pts = pm[mask]
        cols = rgb[mask]
        # drop zero-points
        nz = np.linalg.norm(pts, axis=1) > 1e-6
        pts_all.append(pts[nz])
        cols_all.append(cols[nz])
    pts = np.concatenate(pts_all, axis=0)
    cols = np.concatenate(cols_all, axis=0)
    if len(pts) > 80_000:
        idx = np.random.choice(len(pts), 80_000, replace=False)
        pts = pts[idx]; cols = cols[idx]
    return pts, cols, sample["scene_name"], v_count


def main():
    import viser

    cfg = OmegaConf.load("configs/RnGUP_obj_448_bf16_15k_render44798.yaml")
    cfg.training.num_views = 8

    # Two configs: roll OFF and roll ON
    cfg_off = OmegaConf.merge(cfg, OmegaConf.create({"training": {"roll_augment_max_deg": 0.0}}))
    cfg_on  = OmegaConf.merge(cfg, OmegaConf.create({"training": {"roll_augment_max_deg": 10.0}}))

    ds_off = ObjaverseDataset(cfg_off)
    ds_on  = ObjaverseDataset(cfg_on)

    server = viser.ViserServer(host="0.0.0.0", port=8127)
    print(f">>> roll-vs-no-roll viser on http://localhost:8127  (forward 8127) <<<\n")

    SEEDS_PER_OBJ = [42, 1234]   # seed view_selector deterministically per obj
    SPACING = 1.5                # x-offset within a group
    GAP_BETWEEN_GROUPS = 4.0     # x-offset between roll-off and roll-on groups

    for obj_i, seed in enumerate(SEEDS_PER_OBJ):
        # ROLL OFF (left)
        pts_off, cols_off, uid_off, v_off = stack_obj_pointcloud(ds_off, obj_i, seed=seed)
        x_off = obj_i * SPACING
        pts_off_shifted = pts_off.copy(); pts_off_shifted[:, 0] += x_off

        # ROLL ON (right group)
        pts_on, cols_on, uid_on, v_on = stack_obj_pointcloud(ds_on, obj_i, seed=seed)
        x_on = obj_i * SPACING + (len(SEEDS_PER_OBJ) * SPACING + GAP_BETWEEN_GROUPS)
        pts_on_shifted = pts_on.copy(); pts_on_shifted[:, 0] += x_on

        assert uid_off == uid_on, f"obj {obj_i} uid mismatch between configs"
        uid_short = uid_off[:8]

        server.scene.add_point_cloud(
            f"/obj{obj_i}_off_{uid_short}",
            points=pts_off_shifted.astype(np.float32),
            colors=cols_off,
            point_size=0.005,
        )
        server.scene.add_label(
            f"/lbl{obj_i}_off",
            text=f"#{obj_i} roll=0  ({uid_short}, {len(pts_off)}pts)",
            position=(x_off, 0, 0.8),
        )

        server.scene.add_point_cloud(
            f"/obj{obj_i}_on_{uid_short}",
            points=pts_on_shifted.astype(np.float32),
            colors=cols_on,
            point_size=0.005,
        )
        server.scene.add_label(
            f"/lbl{obj_i}_on",
            text=f"#{obj_i} roll=±10°  ({uid_short}, {len(pts_on)}pts)",
            position=(x_on, 0, 0.8),
        )

        # Print bbox diff: if roll plumbing is right, bbox should match closely
        bbox_off = (pts_off.min(0), pts_off.max(0))
        bbox_on = (pts_on.min(0), pts_on.max(0))
        bbox_diff = np.linalg.norm(np.array(bbox_off) - np.array(bbox_on))
        print(f"  obj {obj_i} ({uid_short}): "
              f"off={len(pts_off)}pts on={len(pts_on)}pts  "
              f"bbox diff={bbox_diff:.4f}  "
              f"(small diff = roll consistent across image+c2w)")

    print(f"\nready. layout: [obj0_off, obj1_off]  GAP  [obj0_on, obj1_on]")
    print(f"Ctrl+C to exit (or kill the PID).\n")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nshutdown.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
