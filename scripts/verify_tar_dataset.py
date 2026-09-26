"""Verify loose-file and tar-backed ObjaverseDataset return bit-equal tensors.

Bypasses the random view_selector by calling preprocess_frames(object_name, list(range(25)))
directly. Compares images / fxfycxcy / intrinsics / c2ws / point_maps / depth_maps with
torch.equal for each sampled object.
"""

import argparse
import copy
import os
import random
import sys
import time

import torch
from easydict import EasyDict as edict
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_objaverse import ObjaverseDataset  # noqa: E402


def load_config(path):
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    return edict(cfg)


def tensors_match(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return False, f"dtype/shape mismatch ({a.dtype}/{a.shape} vs {b.dtype}/{b.shape})"
    if torch.equal(a, b):
        return True, None
    diff = (a - b).abs()
    return False, {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "n_diff": int((diff != 0).sum().item()),
        "n_total": int(diff.numel()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/RnGUP_obj_448_bf16_15k.yaml")
    parser.add_argument("--tar-root", default="/home/z50057756/FluffyElephant_tar")
    parser.add_argument("--num-objects", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--object-list",
        default="",
        help="optional file with explicit object names (one per line); "
        "overrides random sampling from the config dataset_path",
    )
    parser.add_argument(
        "--first-n",
        type=int,
        default=0,
        help="if >0, take the first N entries of the list instead of sampling",
    )
    parser.add_argument(
        "--use-getitem",
        action="store_true",
        help="exercise the full __getitem__ (all 9 returned fields, includes view_selector RNG) "
        "instead of preprocess_frames over all 25 views",
    )
    args = parser.parse_args()

    cfg_loose = load_config(args.config)
    # ensure loose path
    cfg_loose.training.use_tar = False

    cfg_tar = copy.deepcopy(cfg_loose)
    cfg_tar.training.use_tar = True
    cfg_tar.training.tar_root_path = args.tar_root

    ds_loose = ObjaverseDataset(cfg_loose)
    ds_tar = ObjaverseDataset(cfg_tar)
    assert ds_loose.all_object_list == ds_tar.all_object_list, "object lists differ"

    if args.object_list:
        with open(args.object_list) as f:
            objects = [l.strip() for l in f if l.strip()]
    elif args.first_n > 0:
        objects = ds_loose.all_object_list[: args.first_n]
    else:
        rng = random.Random(args.seed)
        objects = rng.sample(ds_loose.all_object_list, min(args.num_objects, len(ds_loose.all_object_list)))

    print(f"[verify_tar] comparing {len(objects)} objects", flush=True)
    print(f"[verify_tar] loose root = {ds_loose.root_path}", flush=True)
    print(f"[verify_tar] tar root   = {ds_tar.tar_root_path}", flush=True)

    n_views = ds_loose.total_frames_per_obj
    indices = list(range(n_views))
    tensor_names = ["images", "fxfycxcys", "intrinsics", "c2ws", "point_maps", "depth_maps"]

    n_ok = 0
    n_skip = 0
    failures = []
    t0 = time.time()

    for i, obj in enumerate(objects, 1):
        tar_path = os.path.join(args.tar_root, obj + ".tar")
        if not os.path.exists(tar_path):
            n_skip += 1
            failures.append((obj, "tar_missing"))
            continue
        try:
            if args.use_getitem:
                # __getitem__ samples views via random.sample inside view_selector — seed both
                # datasets identically so they see the same image_indices.
                obj_idx = ds_loose.all_object_list.index(obj)
                random.seed(args.seed + i)
                out_loose_d = ds_loose[obj_idx]
                random.seed(args.seed + i)
                out_tar_d = ds_tar[obj_idx]
                # tensor fields first, then non-tensor fields
                out_loose = [out_loose_d[k] for k in ("image", "fxfycxcy", "intrinsic", "c2w", "point_map", "depth_map", "extrinsic", "index")]
                out_tar = [out_tar_d[k] for k in ("image", "fxfycxcy", "intrinsic", "c2w", "point_map", "depth_map", "extrinsic", "index")]
                local_names = ["image", "fxfycxcy", "intrinsic", "c2w", "point_map", "depth_map", "extrinsic", "index"]
                scene_name_ok = (out_loose_d["scene_name"] == out_tar_d["scene_name"])
            else:
                out_loose = ds_loose.preprocess_frames(obj, indices)
                out_tar = ds_tar.preprocess_frames(obj, indices)
                local_names = tensor_names
                scene_name_ok = True
        except Exception as e:
            failures.append((obj, f"exception: {e!r}"))
            continue

        all_equal = scene_name_ok
        per_tensor_diff = {} if scene_name_ok else {"scene_name": "string mismatch"}
        for name, a, b in zip(local_names, out_loose, out_tar):
            ok, info = tensors_match(a, b)
            if not ok:
                all_equal = False
                per_tensor_diff[name] = info

        if all_equal:
            n_ok += 1
        else:
            failures.append((obj, per_tensor_diff))

        if i % 20 == 0 or i == len(objects):
            elapsed = time.time() - t0
            print(
                f"[verify_tar] {i}/{len(objects)}  ok={n_ok} skip={n_skip} fail={len(failures) - n_skip}  "
                f"{i / max(elapsed, 1e-6):.2f}/s",
                flush=True,
            )

    print("", flush=True)
    print(f"[verify_tar] DONE  ok={n_ok}  skip={n_skip}  fail={len(failures) - n_skip}", flush=True)
    for obj, info in failures[:5]:
        print(f"  - {obj}: {info}", flush=True)
    if len(failures) > 5:
        print(f"  ... and {len(failures) - 5} more", flush=True)

    if n_ok != len(objects):
        sys.exit(1)


if __name__ == "__main__":
    main()
