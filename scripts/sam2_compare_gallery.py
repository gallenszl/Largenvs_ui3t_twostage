#!/usr/bin/env python
"""Side-by-side gallery: pad-to-square original vs SAM2 white-composite.

For each scene, sample a few frames; left tile = original RGB padded to a
white square (full background, like the old padding-check viz), right tile =
same frame white-composited through the SAM2 mask (object only). Lets you scan
"what SAM2 kept vs the full frame" directly. Defaults to the non-flagged
scenes ("觉得没有问题的").
"""
import argparse
import io
import json
import random
import re
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

INPUT_DIR = Path("/mnt/data-alpha-sg-02/team-camera/datasets/yuchen/co3d/webdataset/val")
MASKS = Path("/home/z50057756/data/co3d_sam2_masks/val")


def pad_square(arr, fill):
    H, W = arr.shape[:2]
    S = max(H, W)
    pt, pl = (S - H) // 2, (S - W) // 2
    if arr.ndim == 3:
        canvas = np.full((S, S, arr.shape[2]), fill, np.uint8)
    else:
        canvas = np.full((S, S), fill, np.uint8)
    canvas[pt:pt + H, pl:pl + W] = arr
    return canvas


def load_frame_rgb(tar_path, idx):
    with tarfile.open(tar_path) as outer:
        meta_name = next(n for n in outer.getnames() if n.endswith(".meta.json"))
        prefix = Path(meta_name).name[: -len(".meta.json")]
        imgs = outer.extractfile(next(n for n in outer.getnames()
                                      if n.endswith(f"{prefix}.images.tar"))).read()
    with tarfile.open(fileobj=io.BytesIO(imgs)) as tf:
        pat = re.compile(re.escape(prefix) + r"\.images_(\d+)\.jpg$")
        mm = {int(pat.search(m.name).group(1)): m for m in tf.getmembers() if pat.search(m.name)}
        data = tf.extractfile(mm[idx]).read()
    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))


def make_row(args_tuple):
    sid, view_num, n_frames, row_h = args_tuple
    try:
        idxs = np.linspace(0, view_num - 1, n_frames + 2, dtype=int)[1:-1]
        blocks = []
        for i in idxs:
            rgb = load_frame_rgb(INPUT_DIR / f"{sid}.tar", int(i))
            m = np.array(Image.open(MASKS / sid / f"{i}.png"))
            if m.shape != rgb.shape[:2]:
                m = np.array(Image.fromarray(m).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST))
            a = (m > 127)[..., None].astype(np.float32)
            comp = (rgb * a + 255.0 * (1 - a)).astype(np.uint8)
            left = pad_square(rgb, 255)
            right = pad_square(comp, 255)
            sep = np.full((left.shape[0], 4, 3), 210, np.uint8)
            blocks.append(np.concatenate([left, sep, right], axis=1))
        gap = np.full((blocks[0].shape[0], 12, 3), 255, np.uint8)
        row = blocks[0]
        for b in blocks[1:]:
            row = np.concatenate([row, gap, b], axis=1)
        scale = row_h / row.shape[0]
        out = Image.fromarray(row).resize((int(row.shape[1] * scale), row_h))
        rel = f"compare_ok/{sid}.jpg"
        out.save(Path("/home/z50057756/data/co3d_sam2_masks/qc") / rel, quality=88)
        return sid, rel
    except Exception as e:
        return sid, f"ERROR: {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split_file", default="/home/z50057756/code/RnG_feature_allignment/data/co3d_teddybear_val_1140.txt")
    ap.add_argument("--flagged", default="/home/z50057756/data/co3d_sam2_masks/qc/flagged_scenes.txt")
    ap.add_argument("--include_flagged", action="store_true", help="also render the flagged scenes")
    ap.add_argument("--sample", type=int, default=0, help="random subset of this many scenes (0 = all)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n_frames", type=int, default=3, help="frames per scene (evenly spaced)")
    ap.add_argument("--row_h", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out_html", default="/home/z50057756/data/co3d_sam2_masks/qc/compare_ok.html")
    args = ap.parse_args()

    qc = Path("/home/z50057756/data/co3d_sam2_masks/qc")
    (qc / "compare_ok").mkdir(parents=True, exist_ok=True)
    summ = {r["scene"]: r for r in (json.loads(l) for l in open(qc / "summary.jsonl"))}

    scenes = [l.strip() for l in Path(args.split_file).read_text().splitlines() if l.strip()]
    if not args.include_flagged:
        flagged = set(Path(args.flagged).read_text().split())
        scenes = [s for s in scenes if s not in flagged]
    if args.sample:
        random.seed(0)
        scenes = sorted(random.sample(scenes, min(args.sample, len(scenes))))
    if args.limit:
        scenes = scenes[:args.limit]

    tasks = [(s, summ[s]["view_num"], args.n_frames, args.row_h) for s in scenes]
    print(f"rendering {len(tasks)} scenes x {args.n_frames} frames, {args.workers} workers ...", flush=True)
    results, errors = [], []
    with ProcessPoolExecutor(args.workers) as ex:
        for k, (sid, rel) in enumerate(ex.map(make_row, tasks), 1):
            (errors if rel.startswith("ERROR") else results).append((sid, rel))
            if k % 100 == 0:
                print(f"  {k}/{len(tasks)}", flush=True)

    cards = "".join(
        f'<div class="c"><div class="h">{sid} '
        f'<span>med IoU {summ[sid]["median_iou"]:.3f} · {summ[sid]["prompt_mode"]}</span></div>'
        f'<img loading="lazy" src="{rel}"></div>'
        for sid, rel in sorted(results))
    html = f"""<!doctype html><meta charset="utf-8"><title>SAM2 compare (non-flagged)</title>
<style>body{{font-family:sans-serif;margin:14px;background:#fafafa}}
.c{{margin:0 0 18px;border:1px solid #ddd;background:#fff;border-radius:6px;overflow:hidden}}
.h{{padding:5px 9px;font-weight:bold;font-size:13px;background:#f2f2f2}}
.h span{{font-weight:normal;color:#666}} img{{display:block;max-width:100%}}</style>
<h2>SAM2 CO3D — {len(results)} non-flagged scenes</h2>
<p>each block = one frame; <b>left = original padded to square</b> (full background),
<b>right = SAM2 white-composite</b> (object only). {args.n_frames} evenly-spaced frames per scene.</p>
{cards}"""
    Path(args.out_html).write_text(html)
    print(f"done: {len(results)} ok, {len(errors)} errors -> {args.out_html}", flush=True)
    for sid, e in errors[:5]:
        print(f"  {sid}: {e}", flush=True)


if __name__ == "__main__":
    main()
