#!/usr/bin/env python
"""Aggregate sam2_refine_co3d_masks.py per-scene stats.json into a QC report.

CPU-only, login-node safe. Produces:
    <qc_dir>/summary.jsonl        one line per scene (metrics + mode flags)
    <qc_dir>/flagged_scenes.txt   scenes needing attention (needs_review/threshold fail)
    <qc_dir>/manual_review.txt    scenes that ended on fallback_old
    <qc_dir>/index.html           overlay gallery: flagged first, then random sample
Prints a percentile table to stdout.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mask_dir", default="/home/z50057756/data/co3d_sam2_masks/val")
    ap.add_argument("--qc_dir", default="/home/z50057756/data/co3d_sam2_masks/qc")
    ap.add_argument("--split_file", default=None, help="restrict to these scenes (else: all stats.json found)")
    ap.add_argument("--scene_min_median_iou", type=float, default=0.6)
    ap.add_argument("--max_flag_frac", type=float, default=0.10)
    ap.add_argument("--random_k", type=int, default=12, help="random non-flagged scenes in the gallery")
    return ap.parse_args()


def pct_row(name, vals):
    if not vals:
        return f"{name:>14}: (no data)"
    q = np.percentile(vals, [0, 5, 25, 50, 75, 95, 100])
    return f"{name:>14}: " + "  ".join(f"p{p:<3}{v:.3f}" for p, v in zip([0, 5, 25, 50, 75, 95, 100], q))


def main():
    args = parse_args()
    mask_dir, qc_dir = Path(args.mask_dir), Path(args.qc_dir)
    qc_dir.mkdir(parents=True, exist_ok=True)

    if args.split_file:
        scenes = [l.strip() for l in Path(args.split_file).read_text().splitlines() if l.strip()]
    else:
        scenes = sorted(p.parent.name for p in mask_dir.glob("*/stats.json"))

    rows, missing = [], []
    for sid in scenes:
        sp = mask_dir / sid / "stats.json"
        if not sp.exists():
            missing.append(sid)
            continue
        st = json.loads(sp.read_text())
        m = st["scene_metrics"]
        frame_ious = [v["iou"] for v in st["per_frame"].values()]
        rows.append({
            "scene": sid, "prompt_mode": st["prompt_mode"], "anchor": st.get("anchor"),
            "iou0": st.get("iou0"), "retry": st.get("retry", False),
            "fallback": st.get("fallback", False), "needs_review": st.get("needs_review", False),
            "elapsed_s": st.get("elapsed_s"), "view_num": st["view_num"],
            "frame_iou_p5": round(float(np.percentile(frame_ious, 5)), 4),
            **m,
        })

    flagged = [r for r in rows if r["needs_review"]
               or r["median_iou"] < args.scene_min_median_iou
               or r["flag_frac"] > args.max_flag_frac]
    fallbacks = [r for r in rows if r["fallback"]]
    flagged_ids = {r["scene"] for r in flagged}

    with open(qc_dir / "summary.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (qc_dir / "flagged_scenes.txt").write_text("".join(r["scene"] + "\n" for r in flagged))
    (qc_dir / "manual_review.txt").write_text("".join(r["scene"] + "\n" for r in fallbacks))

    ok = [r for r in rows if not r["fallback"]]
    print(f"scenes: {len(rows)} done, {len(missing)} missing"
          + (f" (e.g. {missing[:3]})" if missing else ""))
    print(f"prompt_mode counts: "
          + json.dumps({m: sum(r['prompt_mode'] == m for r in rows)
                        for m in sorted({r['prompt_mode'] for r in rows})}))
    print(f"retry={sum(r['retry'] for r in rows)}  fallback={len(fallbacks)}  "
          f"flagged={len(flagged)} ({100 * len(flagged) / max(1, len(rows)):.1f}%)")
    print("--- distributions over non-fallback scenes ---")
    for k in ("median_iou", "mean_iou", "min_iou", "flag_frac", "min_adj_iou",
              "area_collapse", "median_embedded_resid", "median_spill",
              "frame_iou_p5", "elapsed_s"):
        print(pct_row(k, [r[k] for r in ok if r.get(k) is not None]))

    # ------------------------------------------------------------- HTML gallery
    random.seed(0)
    sample = random.sample([r for r in rows if r["scene"] not in flagged_ids],
                           min(args.random_k, max(0, len(rows) - len(flagged))))
    gallery = [(r, True) for r in flagged] + [(r, False) for r in sample]

    def scene_row(r, is_flagged):
        sid = r["scene"]
        imgs = sorted((mask_dir / sid).glob("overlay_*.jpg"))
        rel = [str(Path("..") / mask_dir.name / sid / p.name) for p in imgs]
        cells = "".join(
            f'<td><div class="tag">{p.name.split("_")[1]}</div>'
            f'<img loading="lazy" src="{u}"></td>' for p, u in zip(imgs, rel))
        cls = "flagged" if is_flagged else ""
        meta = (f'{r["prompt_mode"]} | med {r["median_iou"]:.3f} | min {r["min_iou"]:.3f} | '
                f'flag {r["flag_frac"]:.2%} | iou0 {r["iou0"]}')
        return (f'<tr class="{cls}"><td class="sid">{sid}<br><span class="meta">{meta}</span>'
                f'</td>{cells}</tr>')

    html = f"""<!doctype html><meta charset="utf-8"><title>SAM2 CO3D mask QC</title>
<style>
body{{font-family:sans-serif;margin:16px}} table{{border-collapse:collapse}}
td{{border:1px solid #ccc;padding:4px;vertical-align:top}}
img{{max-height:260px;display:block}} .sid{{font-weight:bold;min-width:220px}}
.meta{{font-weight:normal;font-size:12px;color:#555}}
tr.flagged td{{background:#fff0f0;border-color:#e99}}
.tag{{font-size:11px;color:#777}}
</style>
<h2>SAM2 CO3D mask QC — {len(rows)} scenes, {len(flagged)} flagged (red), {len(sample)} random</h2>
<p>red contour = old PointRend mask, green contour = new SAM2 mask</p>
<table>{"".join(scene_row(r, fl) for r, fl in gallery)}</table>
"""
    (qc_dir / "index.html").write_text(html)
    print(f"\nwrote {qc_dir}/summary.jsonl, flagged_scenes.txt ({len(flagged)}), "
          f"manual_review.txt ({len(fallbacks)}), index.html "
          f"(serve: python -m http.server -d {qc_dir.parent})")


if __name__ == "__main__":
    main()
