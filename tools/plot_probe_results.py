"""Merge per-checkpoint probe CSVs and produce baseline vs REPA cos_sim plot.

Usage:
    python3 tools/plot_probe_results.py <probe_dir>

Reads probe_step{N}.csv files in <probe_dir>, merges them, and writes
- <probe_dir>/cossim_trajectory.csv  (combined long-format)
- <probe_dir>/cossim_trajectory.png  (cos_sim vs step, per layer)
"""

import csv
import glob
import os
import sys
from collections import defaultdict


def load_csvs(probe_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(probe_dir, "probe_*.csv"))):
        with open(path) as f:
            for r in csv.DictReader(f):
                rows.append(r)
    return rows


def repa_cossim_trajectory():
    """REPA's own cos_sim trajectory (from Job 72209 training logs).

    Hard-coded for the 4 probe checkpoints we evaluate, plus extras.
    Numbers come from the training stdout sampled in conversation.
    """
    return {
        # step: cos_sim_align value (REPA's own training-time alignment metric)
        2000: 0.787,
        6000: 0.739,
        8000: 0.777,
        10000: 0.754,
        15000: 0.85,  # approximate plateau, latest sampled was ~0.84-0.85
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    probe_dir = sys.argv[1]
    rows = load_csvs(probe_dir)
    if not rows:
        print(f"No probe_*.csv found in {probe_dir}")
        sys.exit(1)

    # Combined long-format CSV
    out_csv = os.path.join(probe_dir, "cossim_trajectory.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv} ({len(rows)} rows)")

    # Pretty print
    by_layer_step = defaultdict(dict)  # layer -> step -> (cos, cka)
    for r in rows:
        L = int(r["layer"])
        s = int(r["step"])
        by_layer_step[L][s] = (float(r["cos_sim_mean"]), float(r["cos_sim_std"]), float(r["cka_rank0_local"]))

    repa = repa_cossim_trajectory()
    layers = sorted(by_layer_step.keys())
    all_steps = sorted({s for d in by_layer_step.values() for s in d.keys()})

    print()
    print(f"{'Layer':>5}  {'Step':>6}  {'Baseline cos':>14}  {'Baseline CKA':>14}  {'REPA cos':>10}  {'Δ (REPA-base)':>14}")
    print("-" * 88)
    for L in layers:
        for s in all_steps:
            if s not in by_layer_step[L]:
                continue
            cos, std, cka = by_layer_step[L][s]
            repa_cos = repa.get(s)
            delta_str = f"{repa_cos - cos:+.4f}" if repa_cos is not None else "    --"
            repa_str = f"{repa_cos:.4f}" if repa_cos is not None else "  --"
            print(f"{L:>5}  {s:>6}  {cos:+.4f} ± {std:.3f}   {cka:.4f}        {repa_str:>10}  {delta_str:>14}")

    # Try optional plot via matplotlib
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        # REPA reference curve at layer 7 (the one used in training)
        if 7 in by_layer_step:
            steps_repa = sorted(repa.keys())
            ax.plot(steps_repa, [repa[s] for s in steps_repa],
                    "ko-", lw=2.5, label="REPA cos_sim_align (training log, layer 7)")

        for L in layers:
            data = sorted(by_layer_step[L].items())
            steps = [d[0] for d in data]
            cos = [d[1][0] for d in data]
            std = [d[1][1] for d in data]
            ax.errorbar(steps, cos, yerr=std, marker="o", capsize=3,
                        label=f"Baseline (no REPA), layer {L}")

        ax.axhline(0.0, color="gray", linestyle=":", lw=1)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Cosine similarity to DINOv3-ViT-L (target view, post-global features)")
        ax.set_title("Baseline endogenous DINOv3 feature similarity vs REPA's enforced alignment")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
        out_png = os.path.join(probe_dir, "cossim_trajectory.png")
        fig.tight_layout()
        fig.savefig(out_png, dpi=140)
        print(f"\nWrote {out_png}")
    except ImportError:
        print("\n[skip plot — matplotlib not available]")


if __name__ == "__main__":
    main()
