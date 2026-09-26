"""Depth arbitration for hole-screen candidates (fully automatic, no human).

A usable scene is flagged by the hole scan when it has an enclosed hole whose
interior the OLD mask calls foreground (=suspected carved-out decoration). This
stage decides each flag by DEPTH:

  see-through gap (chair slat, handle loop, wheel spokes) -> hole interior sits
  FAR BEHIND the object surface -> depth(hole) >> depth(ring) -> LEGIT, keep.
  carved decoration (bow/flower excised from the object) -> hole interior is ON
  the object surface -> depth(hole) ~= depth(ring) -> BAD, drop scene.

Ratio thresholds (user-calibrated on 28 GT examples, 0.09 separation margin):
  hole/ring depth  median >= 1.10  OR  p90 >= 1.15  -> see-through (rescue).
Conservative: a flag we cannot arbitrate (too few valid depth px) counts as
NOT-see-through -> scene dropped (favours purity, ~0.6% expected false-kill).
A scene stays usable iff EVERY flagged hole is confirmed see-through.

CO3D depth: 16-bit PNG in <prefix>.depths.tar, decoded float16->float32 COLMAP
Z-depth (reused from convert_co3d_to_gso.load_16big_png_depth); validity in
<prefix>.depth_mask_list.tar.
"""
import argparse
import io
import json
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# pin threads: many single-threaded workers beat few over-subscribed ones
# (the hole-scan step thrashed at CPULoad 374 from cv2's internal threads)
cv2.setNumThreads(1)
sys.path.insert(0, "/home/z50057756/code/RnG_feature_allignment/scripts")


def load_16big_png_depth(b):
    d_pil = Image.open(io.BytesIO(b))
    d_arr = np.array(d_pil, dtype=np.uint16)
    depth = np.frombuffer(d_arr.tobytes(), dtype=np.float16).astype(np.float32)
    return depth.reshape((d_pil.size[1], d_pil.size[0]))


def scene_depth_tars(tar_path):
    """Return (prefix, depths_tf, dmask_tf) — inner tars kept on BytesIO."""
    with tarfile.open(tar_path) as outer:
        names = outer.getnames()
        meta = next(n for n in names if n.endswith(".meta.json"))
        prefix = Path(meta).name[: -len(".meta.json")]
        dep = outer.extractfile(next(n for n in names if n.endswith(f"{prefix}.depths.tar"))).read()
        dvm = outer.extractfile(next(n for n in names if n.endswith(f"{prefix}.depth_mask_list.tar"))).read()
    return (prefix, tarfile.open(fileobj=io.BytesIO(dep)),
            tarfile.open(fileobj=io.BytesIO(dvm)))


def hole_mask_at(mask_png_path):
    m = cv2.imread(str(mask_png_path), cv2.IMREAD_GRAYSCALE)
    return None if m is None else m > 127


def enclosed_holes(mask, min_px, min_frac):
    """Enclosed background CCs (not touching border) >= size gate."""
    inv = (~mask).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, 8)
    H, W = mask.shape
    area = int(mask.sum())
    out = []
    for k in range(1, n):
        x, y, w, h, a = st[k]
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue
        if a >= min_px and (area == 0 or a / area >= min_frac):
            out.append(lab == k)
    return out


def arbitrate(task):
    """Return (sid, verdict, detail). verdict: 'keep' (all see-through) or 'drop'."""
    sid, frames, mask_root, input_dir, args = task
    try:
        mdir = Path(mask_root) / sid
        prefix, dtf, vtf = scene_depth_tars(Path(input_dir) / f"{sid}.tar")
        results = []
        for i in frames:
            m = hole_mask_at(mdir / f"{i}.png")
            if m is None:
                continue
            holes = enclosed_holes(m, args.min_px, args.min_frac)
            if not holes:
                continue
            try:
                depth = load_16big_png_depth(dtf.extractfile(f"{prefix}.depths_{i}.png").read())
                dv = np.array(Image.open(io.BytesIO(
                    vtf.extractfile(f"{prefix}.depth_mask_list_{i}.png").read()))) > 0
            except Exception:
                results.append(("no_depth", i, None)); continue
            if depth.shape != m.shape:
                depth = cv2.resize(depth, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_NEAREST)
                dv = cv2.resize(dv.astype(np.uint8), (m.shape[1], m.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            valid = dv & (depth > 0)
            for hole in holes:
                ring = (cv2.dilate(hole.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0) & m & ~hole
                hv = depth[hole & valid]
                rv = depth[ring & valid]
                if hv.size < args.min_valid or rv.size < args.min_valid:
                    results.append(("ambiguous", i, None)); continue
                ring_med = float(np.median(rv))
                if ring_med <= 0:
                    results.append(("ambiguous", i, None)); continue
                r_med = float(np.median(hv)) / ring_med
                r_p90 = float(np.percentile(hv, 90)) / ring_med
                see_through = (r_med >= args.ratio_med) or (r_p90 >= args.ratio_p90)
                results.append(("seethrough" if see_through else "decoration", i,
                                (round(r_med, 3), round(r_p90, 3))))
        dtf.close(); vtf.close()
        if not results:
            return sid, "keep", "no_hole_on_recheck"
        # keep iff EVERY arbitrated hole is see-through; any decoration/ambiguous -> drop
        labels = [r[0] for r in results]
        if all(l == "seethrough" for l in labels):
            return sid, "keep", results
        return sid, "drop", results
    except Exception as e:
        return sid, "drop", [("ERR", str(e)[:100], None)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hits_json", required=True)
    ap.add_argument("--mask_root", required=True)
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ratio_med", type=float, default=1.10)
    ap.add_argument("--ratio_p90", type=float, default=1.15)
    ap.add_argument("--min_px", type=int, default=1500)
    ap.add_argument("--min_frac", type=float, default=0.04)
    ap.add_argument("--min_valid", type=int, default=40)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    hits = json.load(open(args.hits_json))
    jobs = []
    for sid, hs in hits.items():
        frames = sorted({int(h[0]) for h in hs if isinstance(h[0], int) or str(h[0]).isdigit()})
        if frames:
            jobs.append((sid, frames, args.mask_root, args.input_dir, args))
    print(f"arbitrating {len(jobs)} hole candidates ...", flush=True)
    keep, drop = [], []
    detail = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for sid, verdict, det in ex.map(arbitrate, jobs, chunksize=8):
            detail[sid] = {"verdict": verdict, "detail": det}
            (keep if verdict == "keep" else drop).append(sid)
    out = Path(args.out_dir)
    (out / "hole_keep.txt").write_text("\n".join(sorted(keep)) + "\n")
    (out / "hole_drop.txt").write_text("\n".join(sorted(drop)) + "\n")
    json.dump(detail, open(out / "hole_verdicts.json", "w"), indent=1)
    print(f"DONE keep={len(keep)} drop={len(drop)} -> {out}/hole_drop.txt", flush=True)


if __name__ == "__main__":
    main()
