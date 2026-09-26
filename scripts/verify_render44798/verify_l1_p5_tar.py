"""L1 P5a sanity: tar member prefix auto-detect handles both layouts.

  FluffyElephant_tar: members start with "./"
  objaverse_renders_44798: members start with "<uid>/"

Tests by loading 1 obj from 44,798 tar (uid-prefixed) and verifying it works.
If FluffyElephant_tar exists, also loads 1 from there.

Usage:
    cd ~/code/RnG_feature_allignment
    python scripts/verify_render44798/verify_l1_p5_tar.py
"""
import os
import sys

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data.dataset_objaverse import ObjaverseDataset


def try_load(cfg, label):
    print(f"  Loading from {cfg.training.tar_root_path} ({label})...")
    try:
        ds = ObjaverseDataset(cfg)
        sample = ds[0]
        print(f"    {label} PASS: image shape {sample['image'].shape}, uid {ds.all_object_list[0][:16]}...")
        return True
    except Exception as e:
        print(f"    {label} FAIL: {type(e).__name__}: {e}")
        return False


def main():
    cfg = OmegaConf.load("configs/RnGUP_obj_448_bf16_15k_render44798.yaml")
    cfg.training.roll_augment_max_deg = 0.0
    cfg.training.num_views = 4

    print("P5a sanity (tar member prefix auto-detect)")

    # 1. Our 44,798 (<uid>/<fname> layout)
    ok_44798 = try_load(cfg, "44,798 (uid-prefix)")

    # 2. FluffyElephant_tar (./<fname> layout) IF available
    fluffy_tar_root = "/home/z50057756/FluffyElephant_tar"
    fluffy_list = "data/objaverse_v1_in_lvis_25v.txt"
    if os.path.isdir(fluffy_tar_root) and os.path.exists(fluffy_list):
        cfg2 = OmegaConf.merge(cfg, OmegaConf.create({
            "training": {
                "dataset_path": fluffy_list,
                "tar_root_path": fluffy_tar_root,
                "root_path": "/home/z50057756/FluffyElephant",
                "total_frames_per_obj": 25,
                "num_views": 4,
            }
        }))
        ok_fluffy = try_load(cfg2, "FluffyElephant (./-prefix)")
    else:
        print(f"  FluffyElephant_tar not present at {fluffy_tar_root} — skipping backward compat check")
        ok_fluffy = True

    if ok_44798 and ok_fluffy:
        print("P5a PASS: both tar layouts load cleanly via auto-detect")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
