#!/usr/bin/env python
"""Select CO3D scenes whose full 25-view camera trajectory is closest to training."""

import argparse
import csv
import json
import random
import sys
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset_objaverse import ObjaverseDataset  # noqa: E402


FEATURE_KEYS = [
    "radius_cv",
    "pair_mean",
    "pair_p95",
    "nn_mean",
    "nn_p95",
    "elev_mean",
    "elev_std",
    "elev_range",
    "look_mean",
    "look_p95",
    "fov_mean_deg",
    "fov_std_deg",
]

CONTEXT = [0, 8, 16, 24]
TARGET = [idx for idx in range(25) if idx not in CONTEXT]


def read_split(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_train_frames(tar_root, object_name):
    tar_path = Path(tar_root) / f"{object_name}.tar"
    with tarfile.open(tar_path, "r:") as tf:
        member = None
        for candidate in [
            f"{object_name}/transforms.json",
            "./transforms.json",
            "transforms.json",
        ]:
            try:
                member = tf.getmember(candidate)
                break
            except KeyError:
                pass
        if member is None:
            for item in tf.getmembers():
                if item.name.endswith("transforms.json"):
                    member = item
                    break
        if member is None:
            raise FileNotFoundError(f"transforms.json not found in {tar_path}")
        with tf.extractfile(member) as f:
            return json.load(f)["frames"]


def load_loose_frames(root, scene):
    with open(Path(root) / scene / "transforms.json", "r") as f:
        return json.load(f)["frames"]


def poses_and_fovs(frames):
    poses = []
    fovs = []
    for frame in frames:
        poses.append(ObjaverseDataset.transform_pose(frame["transform_matrix"]))
        fovs.append(float(frame.get("fov", 0.6981317007977318)))
    return np.stack(poses).astype(np.float64), np.asarray(fovs, dtype=np.float64)


def angle_deg(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return np.degrees(np.arccos(np.sum(a * b, axis=-1).clip(-1.0, 1.0)))


def trajectory_features(poses, fovs):
    centers = poses[:, :3, 3]
    radii = np.linalg.norm(centers, axis=1) + 1e-12
    unit_centers = centers / radii[:, None]
    pair = np.degrees(np.arccos((unit_centers @ unit_centers.T).clip(-1.0, 1.0)))
    pair_values = pair[np.triu_indices(len(pair), 1)]
    nearest = (pair + np.eye(len(pair)) * 999.0).min(axis=1)

    rotations = poses[:, :3, :3]
    forward = rotations @ np.array([0.0, 0.0, 1.0])
    look = angle_deg(forward, -centers)
    elev = np.degrees(np.arcsin(unit_centers[:, 2].clip(-1.0, 1.0)))
    fovs_deg = np.degrees(fovs)

    return {
        "radius_cv": float(radii.std() / radii.mean()),
        "pair_mean": float(pair_values.mean()),
        "pair_p95": float(np.percentile(pair_values, 95)),
        "nn_mean": float(nearest.mean()),
        "nn_p95": float(np.percentile(nearest, 95)),
        "elev_mean": float(elev.mean()),
        "elev_std": float(elev.std()),
        "elev_range": float(elev.max() - elev.min()),
        "look_mean": float(look.mean()),
        "look_p95": float(np.percentile(look, 95)),
        "fov_mean_deg": float(fovs_deg.mean()),
        "fov_std_deg": float(fovs_deg.std()),
    }


def robust_stats(rows):
    stats = {}
    for key in FEATURE_KEYS:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        q25, q75 = np.percentile(values, [25, 75])
        scale = q75 - q25
        scale = max(scale, 0.02 if key == "radius_cv" else 1.0)
        stats[key] = {
            "median": float(np.median(values)),
            "iqr": float(q75 - q25),
            "scale": float(scale),
            "p10": float(np.percentile(values, 10)),
            "p90": float(np.percentile(values, 90)),
        }
    return stats


def feature_score(features, stats):
    z_values = []
    for key in FEATURE_KEYS:
        z = (features[key] - stats[key]["median"]) / stats[key]["scale"]
        z_values.append(np.clip(z, -5.0, 5.0))
    z_values = np.asarray(z_values, dtype=np.float64)
    return float(np.sqrt(np.mean(z_values * z_values)))


def alpha_stats(scene_dir):
    alpha_fracs = []
    bbox_fracs = []
    edge_fracs = []
    for view_idx in range(25):
        rgba = np.asarray(Image.open(Path(scene_dir) / f"{view_idx:03d}.png"))
        alpha = rgba[..., 3] > 127
        alpha_frac = float(alpha.mean())
        alpha_fracs.append(alpha_frac)
        if alpha.any():
            ys, xs = np.where(alpha)
            bbox = (
                (ys.max() - ys.min() + 1)
                * (xs.max() - xs.min() + 1)
                / alpha.size
            )
            border = np.zeros_like(alpha, dtype=bool)
            border[:3, :] = True
            border[-3:, :] = True
            border[:, :3] = True
            border[:, -3:] = True
            edge = float((alpha & border).sum() / alpha.sum())
        else:
            bbox = 0.0
            edge = 1.0
        bbox_fracs.append(float(bbox))
        edge_fracs.append(float(edge))
    return {
        "alpha_mean": float(np.mean(alpha_fracs)),
        "alpha_min": float(np.min(alpha_fracs)),
        "alpha_max": float(np.max(alpha_fracs)),
        "bbox_mean": float(np.mean(bbox_fracs)),
        "edge_frac_mean": float(np.mean(edge_fracs)),
    }


def build_training_stats(args):
    cache_path = Path(args.out_dir) / "training_trajectory_feature_stats.json"
    if cache_path.exists() and not args.recompute_train_stats:
        with open(cache_path, "r") as f:
            payload = json.load(f)
        return payload["stats"], payload

    object_names = read_split(args.train_split)
    rng = random.Random(args.seed)
    rng.shuffle(object_names)
    object_names = object_names[: args.train_sample]

    rows = []
    for object_name in object_names:
        try:
            rows.append(trajectory_features(*poses_and_fovs(load_train_frames(args.train_root, object_name))))
        except Exception as exc:
            print(f"[WARN] skip train object {object_name}: {exc}")

    stats = robust_stats(rows)
    payload = {
        "train_root": args.train_root,
        "train_split": args.train_split,
        "seed": args.seed,
        "train_sample_requested": args.train_sample,
        "train_objects_used": len(rows),
        "feature_keys": FEATURE_KEYS,
        "stats": stats,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(payload, f, indent=2)
    return stats, payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--co3d-root", default="/home/z50057756/data/co3d_teddybear_gsoformat_objdepth")
    parser.add_argument("--co3d-split", default=str(REPO_ROOT / "data/co3d_teddybear_pilot100.txt"))
    parser.add_argument("--train-root", default="/mnt/data-alpha-sg-01/team-camera/home/z50057756/data/objaverse_renders_44798")
    parser.add_argument("--train-split", default=str(REPO_ROOT / "data/objaverse_44385.txt"))
    parser.add_argument("--out-dir", default="/home/z50057756/tmp/co3d_easy_eval")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--train-sample", type=int, default=300)
    parser.add_argument("--recompute-train-stats", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats, train_payload = build_training_stats(args)
    scenes = read_split(args.co3d_split)
    rows = []
    for scene in scenes:
        features = trajectory_features(*poses_and_fovs(load_loose_frames(args.co3d_root, scene)))
        row = {
            "scene": scene,
            "score": feature_score(features, stats),
        }
        row.update(features)
        row.update(alpha_stats(Path(args.co3d_root) / scene))
        rows.append(row)
    rows.sort(key=lambda row: row["score"])

    selected = rows[: args.top_k]
    split_path = out_dir / "co3d_teddybear_traj_top30.txt"
    with open(split_path, "w") as f:
        for row in selected:
            f.write(row["scene"] + "\n")

    view_idx = {
        row["scene"]: {
            "context": CONTEXT,
            "target": TARGET,
        }
        for row in selected
    }
    view_idx_path = out_dir / "view_indices_traj_top30_uniform4_all21.json"
    with open(view_idx_path, "w") as f:
        json.dump(view_idx, f, indent=2)

    csv_path = out_dir / "trajectory_top30_scores.csv"
    fieldnames = (
        ["rank", "scene", "score"]
        + FEATURE_KEYS
        + ["alpha_mean", "alpha_min", "alpha_max", "bbox_mean", "edge_frac_mean", "selected"]
    )
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        selected_scenes = {row["scene"] for row in selected}
        for rank, row in enumerate(rows, start=1):
            record = {key: row.get(key) for key in fieldnames if key not in {"rank", "selected"}}
            record["rank"] = rank
            record["selected"] = row["scene"] in selected_scenes
            writer.writerow(record)

    summary = {
        "train_stats": {
            "path": str(out_dir / "training_trajectory_feature_stats.json"),
            "train_objects_used": train_payload["train_objects_used"],
        },
        "score_csv": str(csv_path),
        "split_file": str(split_path),
        "view_idx_file": str(view_idx_path),
        "top_k": args.top_k,
        "context": CONTEXT,
        "target_count": len(TARGET),
        "selected_scenes": [row["scene"] for row in selected],
    }
    with open(out_dir / "trajectory_top30_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
