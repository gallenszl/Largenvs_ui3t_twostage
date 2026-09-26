"""Scan USABLE scenes for carved-out-decoration holes (000098 class).

Stage 1 (cheap): enclosed hole >= max(1500px, 4% mask) on any consumed frame.
Stage 2 (old-attested): >=60% of the hole is OLD-mask foreground -> the old
mask says that region is object -> carved decoration. Legit loops (cup handle)
are old-background and pass.
"""
import json
import sys
sys.path.insert(0, "/home/z50057756/code/RnG_feature_allignment/scripts")
import numpy as np
import cv2
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from sam2_eval_frame_audit import load_scene_tars, _member_map, old_mask_at, enclosed_hole


def hole_cc(mask):
    inv = (~mask).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, 8)
    H, W = mask.shape
    out = []
    for k in range(1, n):
        x, y, w, h, a = st[k]
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue
        out.append((int(a), k))
    return out, lab


def scan(task):
    sid, mask_root, input_dir = task
    try:
        mdir = Path(mask_root) / sid
        st = json.load(open(mdir / "stats.json"))
        N = st["view_num"]
        consumed = np.unique(np.linspace(0, N - 1, 25).astype(int)).tolist()
        hits = []
        ftf = None
        for i in consumed:
            m = cv2.imread(str(mdir / f"{i}.png"), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            m = m > 127
            area = int(m.sum())
            if area < 500:
                continue
            ccs, lab = hole_cc(m)
            big = [(a, k) for a, k in ccs if a >= 1500 and a / area >= 0.04]
            if not big:
                continue
            if ftf is None:
                prefix, ftf, _ = load_scene_tars(Path(input_dir) / f"{sid}.tar", False)
                mm = _member_map(ftf, prefix, "image_masks", "png")
            old = old_mask_at(ftf, prefix, mm, i)
            for a, k in big:
                hole = lab == k
                frac_old = float((hole & old).sum() / a)
                if frac_old >= 0.6:
                    hits.append((i, a, round(a / area, 4), round(frac_old, 3)))
        if ftf is not None:
            ftf.close()
        return sid, hits
    except Exception as e:
        return sid, [("ERR", str(e)[:80], 0, 0)]


if __name__ == "__main__":
    jobs = []
    VAL = "/mnt/data-alpha-sg-02/team-camera/datasets/yuchen/co3d/webdataset/val"
    TRAIN = "/mnt/data-alpha-sg-02/team-camera/datasets/yuchen/co3d/webdataset/train"
    for l in open("/home/z50057756/data/co3d_sam2_masks/qc/audit_sam3_full_rescue/usable.txt"):
        if l.strip():
            jobs.append((l.strip(), "/home/z50057756/data/co3d_sam3_masks/val", VAL))
    for l in open("/home/z50057756/data/co3d_sam2_masks/qc/audit_cat50_rescue/usable.txt"):
        if l.strip():
            sid = l.strip()
            root = "/home/z50057756/data/co3d_sam3_masks_hp/cat50_v2fb" if (
                Path("/home/z50057756/data/co3d_sam3_masks_hp/cat50_v2fb") / sid).exists() \
                else "/home/z50057756/data/co3d_sam3_masks_hp/cat50"
            jobs.append((sid, root, TRAIN))
    print(f"scanning {len(jobs)} usable scenes ...", flush=True)
    flagged = []
    with ProcessPoolExecutor(16) as ex:
        for sid, hits in ex.map(scan, jobs, chunksize=8):
            if hits:
                flagged.append((sid, hits))
                print(f"HIT {sid}: {hits[:4]}", flush=True)
    print(f"DONE flagged={len(flagged)}/{len(jobs)}")
