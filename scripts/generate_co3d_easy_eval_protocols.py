#!/usr/bin/env python
"""Generate controlled CO3D-as-GSO inference protocols.

Outputs:
  - uniform 4-view interpolation protocol on all pilot scenes
  - pose-like top-K protocol scored against render44798 training camera episodes

The generated JSON files are opt-in via inference.view_idx_file_path; existing
inference behavior is unchanged when that config key is absent.
"""

import argparse
import csv
import json
import math
import os
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
    "context_pair_mean",
    "context_pair_std",
    "context_pair_min",
    "target_nn_context_mean",
    "target_nn_context_max",
    "selected_pair_mean",
    "radius_cv",
    "elev_mean",
    "elev_std",
    "elev_range",
    "look_mean",
    "look_p95",
    "fov_mean_deg",
    "fov_std_deg",
]


def read_split(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def angle_deg(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    c = np.sum(a * b, axis=-1).clip(-1.0, 1.0)
    return np.degrees(np.arccos(c))


def pair_values(mat):
    if mat.shape[0] < 2:
        return np.array([0.0], dtype=np.float64)
    return mat[np.triu_indices(mat.shape[0], 1)]


def load_loose_frames(root, scene_name):
    with open(Path(root) / scene_name / "transforms.json", "r") as f:
        return json.load(f)["frames"]


def load_tar_frames(tar_root, object_name):
    tar_path = Path(tar_root) / f"{object_name}.tar"
    with tarfile.open(tar_path, "r:") as tf:
        candidates = [
            f"{object_name}/transforms.json",
            "./transforms.json",
            "transforms.json",
        ]
        member = None
        for name in candidates:
            try:
                member = tf.getmember(name)
                break
            except KeyError:
                pass
        if member is None:
            for item in tf.getmembers():
                if item.name.endswith("/transforms.json") or item.name == "transforms.json":
                    member = item
                    break
        if member is None:
            raise FileNotFoundError(f"transforms.json not found in {tar_path}")
        with tf.extractfile(member) as f:
            return json.load(f)["frames"]


def poses_and_fovs(frames):
    poses = []
    fovs = []
    for frame in frames:
        poses.append(ObjaverseDataset.transform_pose(frame["transform_matrix"]))
        fovs.append(float(frame.get("fov", 0.6981317007977318)))
    return np.stack(poses).astype(np.float64), np.asarray(fovs, dtype=np.float64)


def episode_features(poses, fovs, context, target):
    selected = list(context) + list(target)
    centers = poses[selected, :3, 3]
    radii = np.linalg.norm(centers, axis=1) + 1e-12
    unit_centers = centers / radii[:, None]
    pair_ang = np.degrees(np.arccos((unit_centers @ unit_centers.T).clip(-1.0, 1.0)))

    context_centers = unit_centers[: len(context)]
    target_centers = unit_centers[len(context):]
    context_pair = pair_values(pair_ang[: len(context), : len(context)])
    target_to_context = np.degrees(
        np.arccos((target_centers @ context_centers.T).clip(-1.0, 1.0))
    )
    target_nn = target_to_context.min(axis=1)
    selected_pair = pair_values(pair_ang)

    rotations = poses[selected, :3, :3]
    forward = rotations @ np.array([0.0, 0.0, 1.0])
    look = angle_deg(forward, -centers)
    elev = np.degrees(np.arcsin(unit_centers[:, 2].clip(-1.0, 1.0)))
    selected_fovs = np.degrees(fovs[selected])

    return {
        "context_pair_mean": float(context_pair.mean()),
        "context_pair_std": float(context_pair.std()),
        "context_pair_min": float(context_pair.min()),
        "target_nn_context_mean": float(target_nn.mean()),
        "target_nn_context_max": float(target_nn.max()),
        "selected_pair_mean": float(selected_pair.mean()),
        "radius_cv": float(radii.std() / radii.mean()),
        "elev_mean": float(elev.mean()),
        "elev_std": float(elev.std()),
        "elev_range": float(elev.max() - elev.min()),
        "look_mean": float(look.mean()),
        "look_p95": float(np.percentile(look, 95)),
        "fov_mean_deg": float(selected_fovs.mean()),
        "fov_std_deg": float(selected_fovs.std()),
    }


def robust_stats(feature_rows):
    stats = {}
    for key in FEATURE_KEYS:
        values = np.asarray([row[key] for row in feature_rows], dtype=np.float64)
        q25, q75 = np.percentile(values, [25, 75])
        scale = q75 - q25
        if "radius" in key:
            scale = max(scale, 0.02)
        else:
            scale = max(scale, 1.0)
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


def alpha_depth_quality(scene_dir, view_idx):
    rgba = np.asarray(Image.open(scene_dir / f"{view_idx:03d}.png"))
    alpha = rgba[..., 3] > 127
    alpha_count = int(alpha.sum())
    total = int(alpha.size)
    alpha_frac = alpha_count / max(total, 1)
    if alpha_count == 0:
        edge_frac = 1.0
        bbox_touch = True
    else:
        border = np.zeros_like(alpha, dtype=bool)
        border[:3, :] = True
        border[-3:, :] = True
        border[:, :3] = True
        border[:, -3:] = True
        edge_frac = float((alpha & border).sum() / alpha_count)
        ys, xs = np.where(alpha)
        bbox_touch = (
            ys.min() <= 2 or xs.min() <= 2 or
            ys.max() >= alpha.shape[0] - 3 or xs.max() >= alpha.shape[1] - 3
        )

    depth = np.asarray(Image.open(scene_dir / f"{view_idx:03d}_depth.png"))
    depth_valid = int((depth < 65534).sum())
    return {
        "alpha_frac": float(alpha_frac),
        "edge_frac": float(edge_frac),
        "bbox_touch": bool(bbox_touch),
        "depth_valid": depth_valid,
    }


def scene_quality(scene_dir, n_views=25):
    return [alpha_depth_quality(scene_dir, idx) for idx in range(n_views)]


def is_good_view(q, min_alpha, max_edge_frac, reject_bbox_touch):
    if q["alpha_frac"] < min_alpha:
        return False
    if q["edge_frac"] > max_edge_frac:
        return False
    if reject_bbox_touch and q["bbox_touch"]:
        return False
    if q["depth_valid"] <= 0:
        return False
    return True


def build_training_distribution(args):
    cache_path = Path(args.out_dir) / "training_pose_feature_stats.json"
    if cache_path.exists() and not args.recompute_train_stats:
        with open(cache_path, "r") as f:
            cached = json.load(f)
        return cached["stats"], cached

    object_names = read_split(args.train_split)
    rng = random.Random(args.seed)
    rng.shuffle(object_names)
    object_names = object_names[: args.train_sample]

    feature_rows = []
    used_objects = 0
    for object_name in object_names:
        try:
            frames = load_tar_frames(args.train_root, object_name)
            poses, fovs = poses_and_fovs(frames)
        except Exception as exc:
            print(f"[WARN] skip train object {object_name}: {exc}")
            continue
        used_objects += 1
        n = len(frames)
        for _ in range(args.train_episodes_per_object):
            episode = rng.sample(range(n), 7)
            feature_rows.append(episode_features(poses, fovs, episode[:4], episode[4:]))

    stats = robust_stats(feature_rows)
    payload = {
        "train_root": args.train_root,
        "train_split": args.train_split,
        "seed": args.seed,
        "train_sample_requested": args.train_sample,
        "train_objects_used": used_objects,
        "episodes_per_object": args.train_episodes_per_object,
        "episode_count": len(feature_rows),
        "feature_keys": FEATURE_KEYS,
        "stats": stats,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(payload, f, indent=2)
    return stats, payload


def best_pose_like_episode(scene_name, args, stats):
    scene_dir = Path(args.co3d_root) / scene_name
    frames = load_loose_frames(args.co3d_root, scene_name)
    poses, fovs = poses_and_fovs(frames)
    quality = scene_quality(scene_dir, len(frames))

    strict_good = [
        idx for idx, q in enumerate(quality)
        if is_good_view(q, args.min_alpha, args.max_edge_frac, True)
    ]
    loose_good = [
        idx for idx, q in enumerate(quality)
        if is_good_view(q, args.min_alpha, args.max_edge_frac_relaxed, False)
    ]
    eligible = strict_good if len(strict_good) >= 7 else loose_good
    if len(eligible) < 7:
        return None

    rng = random.Random(args.seed + sum(ord(c) for c in scene_name))
    candidates = []
    # Include deterministic spread candidates in addition to random search.
    if len(eligible) >= 7:
        lin = np.linspace(0, len(eligible) - 1, 7, dtype=int).tolist()
        candidates.append([eligible[i] for i in lin])
    for _ in range(args.co3d_candidates_per_scene):
        candidates.append(rng.sample(eligible, 7))

    best = None
    for episode in candidates:
        context = episode[:4]
        target = episode[4:]
        features = episode_features(poses, fovs, context, target)
        score = feature_score(features, stats)
        row = {
            "scene": scene_name,
            "score": score,
            "context": context,
            "target": target,
            "eligible_count": len(eligible),
            "strict_good_count": len(strict_good),
            "used_relaxed_quality": len(strict_good) < 7,
        }
        row.update(features)
        if best is None or score < best["score"]:
            best = row
    return best


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


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
    parser.add_argument("--train-episodes-per-object", type=int, default=4)
    parser.add_argument("--co3d-candidates-per-scene", type=int, default=2500)
    parser.add_argument("--min-alpha", type=float, default=0.015)
    parser.add_argument("--max-edge-frac", type=float, default=0.002)
    parser.add_argument("--max-edge-frac-relaxed", type=float, default=0.03)
    parser.add_argument("--recompute-train-stats", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_names = read_split(args.co3d_split)

    interp = {}
    context = [0, 8, 16, 24]
    target = [idx for idx in range(25) if idx not in context]
    for scene_name in scene_names:
        interp[scene_name] = {"context": context, "target": target}
    interp_path = out_dir / "view_indices_interp_uniform4_all21.json"
    write_json(interp_path, interp)

    stats, train_payload = build_training_distribution(args)

    best_rows = []
    for scene_name in scene_names:
        best = best_pose_like_episode(scene_name, args, stats)
        if best is not None:
            best_rows.append(best)
        else:
            print(f"[WARN] no eligible pose-like episode for {scene_name}")
    best_rows.sort(key=lambda row: row["score"])
    top_rows = best_rows[: args.top_k]

    pose_like = {
        row["scene"]: {"context": row["context"], "target": row["target"]}
        for row in top_rows
    }
    pose_like_path = out_dir / "view_indices_pose_like_top30_4plus3.json"
    write_json(pose_like_path, pose_like)

    top_split_path = out_dir / "co3d_teddybear_pose_like_top30.txt"
    with open(top_split_path, "w") as f:
        for row in top_rows:
            f.write(row["scene"] + "\n")

    csv_path = out_dir / "pose_like_scores.csv"
    fieldnames = [
        "rank", "scene", "score", "context", "target", "eligible_count",
        "strict_good_count", "used_relaxed_quality",
    ] + FEATURE_KEYS
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(best_rows, start=1):
            record = {key: row.get(key) for key in fieldnames if key != "rank"}
            record["rank"] = rank
            record["context"] = " ".join(map(str, row["context"]))
            record["target"] = " ".join(map(str, row["target"]))
            writer.writerow(record)

    summary = {
        "out_dir": str(out_dir),
        "interp": {
            "path": str(interp_path),
            "scene_count": len(interp),
            "context": context,
            "target_count": len(target),
        },
        "pose_like": {
            "path": str(pose_like_path),
            "split_file": str(top_split_path),
            "score_csv": str(csv_path),
            "scene_count": len(pose_like),
            "top_k": args.top_k,
            "candidate_scene_count": len(best_rows),
        },
        "train_stats": {
            "path": str(out_dir / "training_pose_feature_stats.json"),
            "episode_count": train_payload["episode_count"],
            "train_objects_used": train_payload["train_objects_used"],
        },
    }
    write_json(out_dir / "protocol_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
