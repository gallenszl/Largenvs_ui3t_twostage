#!/usr/bin/env python
"""Analyze per-view pose gap vs per-view rendering performance on CO3D evals."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset_objaverse import ObjaverseDataset  # noqa: E402


DEFAULT_CONTEXT = [0, 8, 16, 24]
GAP_KEYS = ["center_gap", "rot_gap", "trans_gap", "pose_gap"]
METRIC_KEYS = ["psnr", "lpips", "ssim"]


def read_split(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def angle_deg(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return np.degrees(np.arccos(np.sum(a * b, axis=-1).clip(-1.0, 1.0)))


def rot_angle_deg(r1, r2):
    rot = r1.T @ r2
    cos_theta = (np.trace(rot) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, float(cos_theta)))))


def load_poses(data_root, scene):
    with open(Path(data_root) / scene / "transforms.json", "r") as f:
        frames = json.load(f)["frames"]
    return np.stack([
        ObjaverseDataset.transform_pose(frame["transform_matrix"])
        for frame in frames
    ]).astype(np.float64)


def alpha_fraction(data_root, scene, view_idx):
    alpha = np.asarray(Image.open(Path(data_root) / scene / f"{view_idx:03d}.png"))[..., 3]
    return float((alpha > 127).mean())


def compute_pose_rows(data_root, scenes, context):
    rows = {}
    for scene in scenes:
        poses = load_poses(data_root, scene)
        centers = poses[:, :3, 3]
        rotations = poses[:, :3, :3]
        context_centers = centers[context]
        context_rotations = rotations[context]
        radius = np.median(np.linalg.norm(context_centers, axis=1))
        for view_idx in range(len(poses)):
            if view_idx in context:
                continue
            center_gaps = angle_deg(centers[view_idx], context_centers)
            rot_gaps = np.asarray([
                rot_angle_deg(rotations[view_idx], context_rotation)
                for context_rotation in context_rotations
            ])
            trans_gaps = (
                np.linalg.norm(context_centers - centers[view_idx], axis=1)
                / (radius + 1e-12)
            )
            nearest = int(center_gaps.argmin())
            rows[(scene, view_idx)] = {
                "scene": scene,
                "view": view_idx,
                "center_gap": float(center_gaps[nearest]),
                "rot_gap": float(rot_gaps[nearest]),
                "trans_gap": float(trans_gaps[nearest]),
                "pose_gap": float(
                    center_gaps[nearest] / 90.0
                    + rot_gaps[nearest] / 90.0
                    + trans_gaps[nearest]
                ),
                "alpha_frac": alpha_fraction(data_root, scene, view_idx),
            }
    return rows


def load_per_view_metrics(run_dir, pose_rows):
    rows = []
    for sample_dir in sorted(Path(run_dir).iterdir()):
        if not sample_dir.is_dir():
            continue
        metrics_path = sample_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        scene = metrics["summary"]["scene_name"]
        for per_view in metrics["per_view"]:
            key = (scene, int(per_view["view"]))
            if key not in pose_rows:
                continue
            row = dict(pose_rows[key])
            row.update({
                "psnr": float(per_view["psnr"]),
                "lpips": float(per_view["lpips"]),
                "ssim": float(per_view["ssim"]),
            })
            rows.append(row)
    return rows


def rankdata(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    idx = 0
    while idx < len(values):
        next_idx = idx + 1
        while next_idx < len(values) and values[order[next_idx]] == values[order[idx]]:
            next_idx += 1
        ranks[order[idx:next_idx]] = (idx + next_idx - 1) / 2.0 + 1.0
        idx = next_idx
    return ranks


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def demean_by_scene(values, scenes):
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values)
    for scene in sorted(set(scenes)):
        idx = np.asarray([i for i, value in enumerate(scenes) if value == scene])
        out[idx] = values[idx] - values[idx].mean()
    return out


def residualize(values, covariates):
    values = np.asarray(values, dtype=np.float64)
    covariates = np.asarray(covariates, dtype=np.float64)
    design = np.column_stack([np.ones(len(values)), covariates])
    beta = np.linalg.lstsq(design, values, rcond=None)[0]
    return values - design @ beta


def summarize_run(rows):
    scenes = [row["scene"] for row in rows]
    alpha = np.asarray([row["alpha_frac"] for row in rows], dtype=np.float64)
    alpha_scene = demean_by_scene(alpha, scenes)

    correlations = {}
    for gap_key in GAP_KEYS:
        x = np.asarray([row[gap_key] for row in rows], dtype=np.float64)
        x_scene = demean_by_scene(x, scenes)
        x_scene_alpha = residualize(x_scene, alpha_scene[:, None])
        correlations[gap_key] = {}
        for metric_key in METRIC_KEYS:
            y = np.asarray([row[metric_key] for row in rows], dtype=np.float64)
            y_scene = demean_by_scene(y, scenes)
            y_scene_alpha = residualize(y_scene, alpha_scene[:, None])
            correlations[gap_key][metric_key] = {
                "pooled_pearson": pearson(x, y),
                "pooled_spearman": pearson(rankdata(x), rankdata(y)),
                "scene_demean_pearson": pearson(x_scene, y_scene),
                "scene_demean_spearman": pearson(rankdata(x_scene), rankdata(y_scene)),
                "scene_demean_alpha_partial_pearson": pearson(x_scene_alpha, y_scene_alpha),
            }

    center_gap = np.asarray([row["center_gap"] for row in rows], dtype=np.float64)
    quantiles = np.quantile(center_gap, [0.0, 0.25, 0.5, 0.75, 1.0])
    center_gap_bins = []
    for idx in range(4):
        lo, hi = quantiles[idx], quantiles[idx + 1]
        if idx < 3:
            selected = [row for row in rows if lo <= row["center_gap"] < hi]
        else:
            selected = [row for row in rows if lo <= row["center_gap"] <= hi]
        center_gap_bins.append({
            "bin": idx + 1,
            "lo": float(lo),
            "hi": float(hi),
            "n": len(selected),
            "center_gap_mean": float(np.mean([row["center_gap"] for row in selected])),
            "alpha_frac_mean": float(np.mean([row["alpha_frac"] for row in selected])),
            "psnr": float(np.mean([row["psnr"] for row in selected])),
            "lpips": float(np.mean([row["lpips"] for row in selected])),
            "ssim": float(np.mean([row["ssim"] for row in selected])),
        })

    return {
        "n_views": len(rows),
        "correlations": correlations,
        "center_gap_quartile_bins": center_gap_bins,
    }


def write_csv(path, rows):
    fieldnames = [
        "run", "scene", "view", "center_gap", "rot_gap", "trans_gap",
        "pose_gap", "alpha_frac", "psnr", "lpips", "ssim",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/z50057756/data/co3d_teddybear_gsoformat_objdepth")
    parser.add_argument("--split-file", default=str(REPO_ROOT / "data/co3d_teddybear_pilot100.txt"))
    parser.add_argument("--out-dir", default="/home/z50057756/tmp/co3d_easy_eval")
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("NAME", "DIR"),
        default=[
            (
                "RnG_uniform4_all21",
                str(REPO_ROOT / "experiments/evaluation/RnGUP_render44798_CO3Dteddy_objdepth_interp_uniform4_all21"),
            ),
            (
                "FA3_uniform4_all21",
                str(REPO_ROOT / "experiments/evaluation/FA3_baseline_CO3Dteddy_objdepth_interp_uniform4_all21"),
            ),
        ],
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes = read_split(args.split_file)
    pose_rows = compute_pose_rows(args.data_root, scenes, DEFAULT_CONTEXT)

    all_rows = []
    summary = {
        "context": DEFAULT_CONTEXT,
        "data_root": args.data_root,
        "split_file": args.split_file,
        "runs": {},
    }
    for run_name, run_dir in args.run:
        rows = load_per_view_metrics(run_dir, pose_rows)
        for row in rows:
            row["run"] = run_name
        all_rows.extend(rows)
        summary["runs"][run_name] = summarize_run(rows)

    write_csv(out_dir / "pose_gap_per_view_metrics.csv", all_rows)
    with open(out_dir / "pose_gap_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)

    lines = ["# Pose Gap Vs Performance\n\n"]
    lines.append("Context views: `[0, 8, 16, 24]`; targets are the other 21 views per scene.\n\n")
    lines.append("## Alpha-Controlled Scene Fixed-Effect Correlations\n")
    for run_name, run_summary in summary["runs"].items():
        lines.append(f"\n### {run_name}\n")
        lines.append("| gap | PSNR | LPIPS | SSIM |\n")
        lines.append("|---|---:|---:|---:|\n")
        for gap_key in GAP_KEYS:
            corr = run_summary["correlations"][gap_key]
            lines.append(
                f"| {gap_key} | "
                f"{corr['psnr']['scene_demean_alpha_partial_pearson']:+.3f} | "
                f"{corr['lpips']['scene_demean_alpha_partial_pearson']:+.3f} | "
                f"{corr['ssim']['scene_demean_alpha_partial_pearson']:+.3f} |\n"
            )
        lines.append("\nCenter-gap quartile bins:\n\n")
        lines.append("| bin | center gap | n | PSNR | LPIPS | SSIM | alpha |\n")
        lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
        for bin_row in run_summary["center_gap_quartile_bins"]:
            lines.append(
                f"| {bin_row['bin']} | {bin_row['lo']:.2f}-{bin_row['hi']:.2f} | "
                f"{bin_row['n']} | {bin_row['psnr']:.3f} | {bin_row['lpips']:.4f} | "
                f"{bin_row['ssim']:.4f} | {bin_row['alpha_frac_mean']:.3f} |\n"
            )
    with open(out_dir / "pose_gap_analysis.md", "w") as f:
        f.write("".join(lines))

    print(out_dir / "pose_gap_analysis.md")
    print(out_dir / "pose_gap_analysis.json")
    print(out_dir / "pose_gap_per_view_metrics.csv")


if __name__ == "__main__":
    main()
