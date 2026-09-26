"""L4.5 sanity (the strongest): back-projected point cloud in viser.

Run dataset with P1+P2+P4+P5 patches enabled, back-project depth+image to 3D
via point_map (already computed by dataset), filter by alpha_mask, stack across
v views per object, lay 4 objects side-by-side, serve on viser.

Pass criteria (eyeball):
  1. Each object looks like a coherent 3D shape (no "ghost halves")
  2. Multi-view points align — no "rotated fragments"
  3. No white-bg points — only object surface
  4. Scale ~1 unit (normalize_scene constraint)
  5. roll=10° vs roll=0 give the SAME object shape in world frame
     (c2w rotates with image -> world-frame points unchanged)

Bonus diagnostic: set FORCE_BROKEN_FOV=1 to disable P1 (force fov=40°) — point
cloud should then look STRETCHED along z. Used to confirm L4.5 actually
detects P1 misuse.

Usage:
    cd ~/code/RnG_feature_allignment
    python scripts/verify_render44798/verify_l4p5_viser.py

Then forward port 8126 in VSCode and open http://localhost:8126
Ctrl+C to stop.
"""
import os
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data.dataset_objaverse import ObjaverseDataset


def main():
    import viser

    cfg = OmegaConf.load("configs/RnGUP_obj_448_bf16_15k_render44798.yaml")
    cfg.training.roll_augment_max_deg = 10.0
    cfg.training.num_views = 8

    if os.environ.get("FORCE_BROKEN_FOV") == "1":
        # Bonus diagnostic: force-hardcoded 40° fov via monkey-patch
        # (would normally come from per-frame fov in transforms.json)
        cfg.training.num_views = 4
        print(">>> FORCE_BROKEN_FOV=1: monkey-patching ObjaverseDataset.fov to 40° <<<")
        print(">>> EXPECTED: point cloud should look stretched along z-axis <<<")

    ds = ObjaverseDataset(cfg)
    print(f"viser L4.5 on port 8126, {len(ds)} obj available")

    server = viser.ViserServer(host="0.0.0.0", port=8126)
    print(f">>> http://localhost:8126   (forward port 8126 in VSCode) <<<\n")

    OFFSET_STEP = 1.5
    for obj_i in range(4):
        sample = ds[obj_i]
        images = sample["image"]            # [v, 3, h, w]
        point_maps = sample["point_map"]    # [v, 3, h, w] world frame
        alpha_masks = sample["alpha_mask"]  # [v, 1, h, w]
        v_count, _, h, w = images.shape

        pts_all = []
        colors_all = []
        for vi in range(v_count):
            pm = point_maps[vi].permute(1, 2, 0).numpy()             # [h, w, 3]
            rgb = (images[vi].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            mask = (alpha_masks[vi, 0].numpy() > 0.5)
            pts = pm[mask]
            cols = rgb[mask]
            # Drop zero-points (from depth-invalid pixels that snuck through alpha)
            non_zero = np.linalg.norm(pts, axis=1) > 1e-6
            pts = pts[non_zero]; cols = cols[non_zero]
            pts_all.append(pts)
            colors_all.append(cols)

        pts_obj = np.concatenate(pts_all, axis=0)
        cols_obj = np.concatenate(colors_all, axis=0)
        pts_obj[:, 0] += obj_i * OFFSET_STEP

        if len(pts_obj) > 100_000:
            idx = np.random.choice(len(pts_obj), 100_000, replace=False)
            pts_obj = pts_obj[idx]; cols_obj = cols_obj[idx]

        uid = sample["scene_name"]
        server.scene.add_point_cloud(
            f"/obj_{obj_i}_{uid[:8]}",
            points=pts_obj.astype(np.float32),
            colors=cols_obj,
            point_size=0.005,
        )
        server.scene.add_label(
            f"/label_{obj_i}",
            text=f"#{obj_i}: {uid[:8]}  ({v_count}v, roll±10°, {len(pts_obj)} pts)",
            position=(obj_i * OFFSET_STEP, 0, 0.8),
        )
        print(f"  obj {obj_i}: {len(pts_obj)} points stacked across {v_count} views")

    print("\nready. Ctrl+C to exit.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nshutdown.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
