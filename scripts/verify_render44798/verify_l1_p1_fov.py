"""L1 P1 sanity: per-frame fov is read from transforms.json, not hardcoded 40°.

Usage:
    cd ~/code/RnG_feature_allignment
    python scripts/verify_render44798/verify_l1_p1_fov.py
"""
import io
import json
import math
import os
import sys
import tarfile

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data.dataset_objaverse import ObjaverseDataset


def main():
    cfg = OmegaConf.load("configs/RnGUP_obj_448_bf16_15k_render44798.yaml")
    # Roll OFF so focal computation is the only thing being tested
    cfg.training.roll_augment_max_deg = 0.0
    ds = ObjaverseDataset(cfg)

    obj0 = ds.all_object_list[0]
    print(f"P1 sanity on obj: {obj0[:16]}...")

    sample = ds[0]
    image_indices = sample["index"][:, 0].tolist()  # [v]

    # Read raw transforms.json for ground truth fov per frame.
    # Use the same scanning prefix detection as the dataset (P5a fix).
    with tarfile.open(os.path.join(ds.tar_root_path, f"{obj0}.tar")) as t:
        prefix = ""
        for m in t.getmembers():
            if m.name.startswith("./"):
                prefix = "./"
                break
            if "/" in m.name:
                prefix = m.name.split("/")[0] + "/"
                break
        transforms = json.load(t.extractfile(prefix + "transforms.json"))

    resize_w = sample["image"].shape[-1]
    print(f"  resize_w={resize_w}, {len(image_indices)} views")

    n_pass, n_fail = 0, 0
    for v, img_idx in enumerate(image_indices):
        expected_fov = transforms["frames"][img_idx].get("fov", 0.6981317007977318)
        expected_focal = (resize_w / 2) / math.tan(expected_fov / 2)
        actual_focal = sample["fxfycxcy"][v, 0].item()
        ok = abs(actual_focal - expected_focal) < 1e-3
        n_pass += ok
        n_fail += not ok
        if not ok:
            print(f"  FAIL view {v} (img_idx={img_idx}): expected focal={expected_focal:.3f} got {actual_focal:.3f}, fov={math.degrees(expected_fov):.1f}°")

    if n_fail == 0:
        print(f"P1 PASS: all {n_pass} views' focal_length matches per-frame fov from transforms.json")
        return 0
    else:
        print(f"P1 FAIL: {n_fail}/{n_pass + n_fail} views mismatched")
        return 1


if __name__ == "__main__":
    sys.exit(main())
