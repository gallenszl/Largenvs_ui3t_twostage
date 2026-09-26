#!/usr/bin/env python
"""Regenerate CO3D object masks with SAM2 video propagation.

Per scene (one CO3D webdataset outer tar = ~200-frame time-ordered video),
single pass — no detect-then-repair loop:
  1. extract inner-JPEG frames raw into a tmp dir named {i}.jpg (SAM2 needs
     int-stem JPEG dirs); load the old PointRend soft masks (arbiter/QC only).
  2. pick K=3 anchor frames spread over time (never one of the converter's
     linspace-sampled indices; old mask must be non-degenerate there).
  3. best-of-multimask at each anchor: ask SAM2's image head for its 3
     granularity candidates (box + interior positive points), VETO candidates
     that lose a part the old mask owns embedded in the object contour (an
     attached bow), keep the highest SAM2-scored survivor. Asymmetric by
     design: parts the old mask MISSES (cut-off legs) are never penalized,
     and the old mask itself never enters memory (its floods can't leak in).
     add_new_mask(winner) pins each anchor.
  4. ONE forward + reverse propagation over the whole clip, logits > 0.
  5. post-process (drop speck CCs but keep all large ones, fill tiny
     non-border holes only), QC vs old masks (advisory only); sam2_broken
     (median IoU < 0.35 = wrong object) -> denser-anchor retry -> fallback to
     old binarized masks (scene dir always complete).
  6. write {i}.png (0/255, original per-frame resolution), 3 contour-overlay
     JPEGs (anchor/worst/median), and stats.json LAST (= completion marker).

Frames are fed to SAM2 as-is (official behavior: internal anisotropic square
resize to 1024). --square_mode pad is a debug-only alternative that center-pads
frames to a square before feeding and crops the outputs back.
"""
import argparse
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import cv2
import numpy as np
import torch
from PIL import Image

OBJ_ID = 1
MIN_PROMPT_AREA = 500          # px; below this a frame cannot serve as anchor
SPECK_MIN = 64                 # px; absolute floor for keeping a CC
FRAME_IOU_FLAG = 0.5
AREA_RATIO_BAND = (0.5, 2.0)
# "embedded residual" = parts the old mask owns that the candidate lacks AND that
# sit embedded in the candidate's contour (attached decorations like a bow).
# Used to arbitrate SAM2's part/whole multimask granularity and as a QC metric.
PART_CC_MIN = 200              # px; absolute floor for a residual CC to count
PART_CC_FRAC = 0.005           # relative floor: 0.5% of candidate area
PART_CC_MAX_FRAC = 0.5         # cap: >50% of candidate is not an "attached part"
PART_EMBED_MIN = 0.3           # CC-boundary contact ratio with dilate(candidate)
PART_DILATE = 9                # px halo used for embedding/spill tests


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input_dir", default="/mnt/data-alpha-sg-02/team-camera/datasets/yuchen/co3d/webdataset/val")
    ap.add_argument("--split_file", required=True)
    ap.add_argument("--output_dir", default="/home/z50057756/data/co3d_sam2_masks/val")
    ap.add_argument("--ckpt", default="/home/z50057756/code/checkpoints/sam2/sam2.1_hiera_large.pt")
    ap.add_argument("--model_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="process at most N scenes of this shard (0 = all)")
    ap.add_argument("--scene", default=None, help="comma-separated scene ids (debug; bypasses sharding)")
    ap.add_argument("--tmp_root", default="/tmp")
    ap.add_argument("--square_mode", choices=["stretch", "pad"], default="stretch",
                    help="stretch = official SAM2 behavior (feed frames as-is); pad = debug alternative")
    ap.add_argument("--n_views", type=int, default=25,
                    help="converter linspace view count; anchors avoid these indices")
    ap.add_argument("--anchors", type=int, default=5,
                    help="number of multimask-arbitrated mask anchors per scene "
                         "(spread over the clip incl. both ends; grid mode)")
    ap.add_argument("--anchor_mode", choices=["grid", "simple", "quality"], default="grid",
                    help="grid = fixed time slots (current); simple = ONE best-scored old "
                         "mask as the sole anchor (user's minimal scheme, no arbitration); "
                         "quality = scan frames, anchor only where the multimask winner "
                         "loses nothing (resid gate), ends covered mandatorily")
    ap.add_argument("--refine_consumed", action="store_true",
                    help="after propagation, re-segment the 25 consumed frames with the "
                         "image predictor prompted by the propagated mask itself "
                         "(track-then-refine); pre-refine copies kept in <output_dir>_pre")
    ap.add_argument("--scan_frames", type=int, default=40,
                    help="quality mode: number of frames scanned for anchor candidates")
    ap.add_argument("--anchor_resid_mid", type=float, default=0.05,
                    help="quality mode: mid-clip anchors need winner resid <= this")
    ap.add_argument("--anchor_resid_end", type=float, default=0.15,
                    help="quality mode: head/tail anchors need winner resid <= this")
    ap.add_argument("--cand_resid_max", type=float, default=0.01,
                    help="veto a multimask candidate whose embedded residual vs the old "
                         "mask exceeds this (it lost an attached part, e.g. a bow)")
    ap.add_argument("--cand_old_cover_min", type=float, default=0.5,
                    help="candidate must cover at least this fraction of the (cleaned) old "
                         "mask; kills total-miss candidates that the embedded-residual "
                         "metric is blind to (its CC cap ignores 'the whole object is "
                         "missing'), and auto-shifts anchors off frames whose old mask is "
                         "locally broken (no candidate passes -> slot moves to a neighbor)")
    ap.add_argument("--cand_spill_max", type=float, default=0.0,
                    help="if >0, veto candidates whose fraction outside dilate(old,9px) "
                         "exceeds this (anti background-suction, e.g. ArUco boards); "
                         "0 disables — leg-recovery spill measured well below 0.35")
    # advisory thresholds (disagreement with the noisy old masks -> needs_review only;
    # smoke showed old masks can flood the background / go empty, so they cannot
    # auto-reject SAM2 output)
    ap.add_argument("--scene_min_median_iou", type=float, default=0.6)
    ap.add_argument("--max_flag_frac", type=float, default=0.10)
    ap.add_argument("--part_resid_thresh", type=float, default=0.03,
                    help="scene median embedded residual above this -> needs_review")
    # intrinsic (old-mask-independent) failure evidence -> retry then fallback
    ap.add_argument("--fallback_median_iou", type=float, default=0.35,
                    help="IoU-vs-old below this = likely locked onto the wrong object")
    ap.add_argument("--min_adj_iou", type=float, default=0.5,
                    help="adjacent-frame IoU of the NEW masks below this = identity jump")
    ap.add_argument("--area_collapse", type=float, default=0.15,
                    help="min/median NEW-mask area below this = tracking collapse")
    ap.add_argument("--max_empty", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


# ---------------------------------------------------------------- tar loading

def load_scene_tar(tar_path):
    with tarfile.open(tar_path) as outer:
        meta_name = next(n for n in outer.getnames() if n.endswith(".meta.json"))
        prefix = os.path.basename(meta_name)[: -len(".meta.json")]
        meta = json.load(outer.extractfile(meta_name))
        imgs_bytes = outer.extractfile(next(
            n for n in outer.getnames() if n.endswith(f"{prefix}.images.tar"))).read()
        fgms_bytes = outer.extractfile(next(
            n for n in outer.getnames() if n.endswith(f"{prefix}.image_masks.tar"))).read()
    return prefix, int(meta["view_num"]), imgs_bytes, fgms_bytes


def member_map(tf, prefix, kind, ext):
    pat = re.compile(re.escape(prefix) + r"\." + kind + r"_(\d+)\." + ext + r"$")
    out = {}
    for mem in tf.getmembers():
        mo = pat.search(mem.name)
        if mo:
            out[int(mo.group(1))] = mem
    return out


def prepare_frames(imgs_bytes, prefix, view_num, frames_dir, square_mode):
    """Write frames as {i}.jpg; return (H, W, pad_info). pad_info=None in stretch mode."""
    sizes = set()
    raw = {}
    with tarfile.open(fileobj=io.BytesIO(imgs_bytes)) as tf:
        mm = member_map(tf, prefix, "images", "jpg")
        assert sorted(mm) == list(range(view_num)), \
            f"images tar has {len(mm)} members, expected contiguous 0..{view_num - 1}"
        for i in range(view_num):
            data = tf.extractfile(mm[i]).read()
            with Image.open(io.BytesIO(data)) as im:
                sizes.add(im.size)  # (W, H); header-only read
            raw[i] = data
    assert len(sizes) == 1, f"frame sizes vary within scene: {sorted(sizes)}"
    W, H = next(iter(sizes))
    pad_info = None
    if square_mode == "stretch":
        for i, data in raw.items():
            with open(frames_dir / f"{i}.jpg", "wb") as f:
                f.write(data)
    else:  # pad (debug alternative)
        S = max(H, W)
        pl, pt = (S - W) // 2, (S - H) // 2
        pad_info = {"S": S, "pl": pl, "pt": pt, "H": H, "W": W}
        for i, data in raw.items():
            arr = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
            canvas = np.zeros((S, S, 3), np.uint8)
            canvas[pt:pt + H, pl:pl + W] = arr
            Image.fromarray(canvas).save(frames_dir / f"{i}.jpg", quality=95, subsampling=0)
    return H, W, pad_info


def load_old_masks(fgms_bytes, prefix, view_num, H, W):
    soft = []
    with tarfile.open(fileobj=io.BytesIO(fgms_bytes)) as tf:
        mm = member_map(tf, prefix, "image_masks", "png")
        assert sorted(mm) == list(range(view_num)), \
            f"masks tar has {len(mm)} members, expected contiguous 0..{view_num - 1}"
        for i in range(view_num):
            arr = np.array(Image.open(io.BytesIO(tf.extractfile(mm[i]).read())))
            assert arr.shape == (H, W), f"mask {i} shape {arr.shape} != image {(H, W)}"
            soft.append(arr)
    return soft


# --------------------------------------------------------- geometry / prompts

def iou(a, b):
    inter = int(np.logical_and(a, b).sum())
    union = int(a.sum()) + int(b.sum()) - inter
    return inter / union if union else 1.0


def clean_mask(m):
    """Drop speck CCs (< max(64px, 0.5% of largest CC)); keep every large CC."""
    m8 = m.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
    if n <= 1:
        return m.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep_min = max(SPECK_MIN, int(0.005 * int(areas.max())))
    keep = np.zeros(n, bool)
    keep[1:] = areas >= keep_min
    return keep[labels]


def edt_peak(mask):
    edt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    y, x = np.unravel_index(int(np.argmax(edt)), edt.shape)
    return float(x), float(y)


def derive_box_points(clean, H, W, max_points=4):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(clean.astype(np.uint8), connectivity=8)
    order = sorted(range(1, n), key=lambda j: -int(stats[j, cv2.CC_STAT_AREA]))
    xs0, ys0, xs1, ys1, pts = [], [], [], [], []
    for j in order:
        x, y, w, h, _ = stats[j]
        xs0.append(x); ys0.append(y); xs1.append(x + w); ys1.append(y + h)
        if len(pts) < max_points:
            pts.append(edt_peak(labels == j))
    pad = 0.02 * max(H, W)
    box = np.array([max(0.0, min(xs0) - pad), max(0.0, min(ys0) - pad),
                    min(float(W), max(xs1) + pad), min(float(H), max(ys1) + pad)], np.float32)
    return box, np.array(pts, np.float32)


def halo(mask, px=PART_DILATE):
    return cv2.dilate(mask.astype(np.uint8), np.ones((px, px), np.uint8)).astype(bool)


def embedded_resid_frac(ref, cur):
    """Fraction of `cur` area made of parts that `ref` owns, `cur` lacks, and that
    sit embedded in `cur`'s contour (attached decorations like a bow). A stool or
    floor flood in `ref` touches `cur` only along a thin line, so its boundary
    contact ratio stays below PART_EMBED_MIN and it does not count."""
    cur_area = int(cur.sum())
    if cur_area == 0:
        return 1.0 if ref.any() else 0.0
    resid = np.logical_and(ref, np.logical_not(cur))
    if not resid.any():
        return 0.0
    hal = halo(cur)
    cc_min = max(PART_CC_MIN, int(PART_CC_FRAC * cur_area))
    cc_max = int(PART_CC_MAX_FRAC * cur_area)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(resid.astype(np.uint8), connectivity=8)
    kernel3 = np.ones((3, 3), np.uint8)
    total = 0
    for j in range(1, n):
        a = int(stats[j, cv2.CC_STAT_AREA])
        if a < cc_min or a > cc_max:
            continue
        cc = (labels == j).astype(np.uint8)
        border = cv2.dilate(cc, kernel3).astype(bool) & ~cc.astype(bool)  # outer ring
        contact = float(np.logical_and(border, hal).sum()) / max(1, int(border.sum()))
        if contact >= PART_EMBED_MIN:
            total += a
    return total / cur_area


def frame_part_metrics(old_bin_i, new_i):
    """Per-frame (embedded_resid, spill) of the new mask vs the old one."""
    na = int(new_i.sum())
    if na == 0:
        return 0.0, 0.0
    resid = embedded_resid_frac(old_bin_i, new_i)
    spill = float(np.logical_and(new_i, np.logical_not(halo(old_bin_i))).sum()) / na
    return resid, spill


def pick_mask_anchors(view_num, sampled, old_bin, k):
    """k anchor slots spanning the clip INCLUDING both ends (fractions 0.06..0.94),
    shifted off sampled/degenerate frames. End coverage matters: the converter
    consumes linspace frames incl. 0 and N-1, and propagation drifts the farther
    a frame sits from its nearest anchor."""
    if k == 1:
        fracs = [0.5]
    else:
        fracs = [0.06 + 0.88 * r / (k - 1) for r in range(k)]
    anchors = []
    for fr in fracs:
        base = int(round(fr * (view_num - 1)))
        for d in sorted(range(-15, 16), key=abs):
            a = base + d
            if 0 <= a < view_num and a not in sampled and a not in anchors \
                    and old_bin[a].sum() >= MIN_PROMPT_AREA:
                anchors.append(a)
                break
    return sorted(anchors)


# ------------------------------------------------------------- postprocess/QC

def postprocess_mask(m):
    m8 = m.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
    if n <= 1:
        return m.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep_min = max(SPECK_MIN, int(0.005 * int(areas.max())))
    keep = np.zeros(n, bool)
    keep[1:] = areas >= keep_min
    out = keep[labels]
    fg = int(out.sum())
    if fg:
        H, W = out.shape
        hole_max = max(1, int(0.001 * fg))
        nb, lb, sb, _ = cv2.connectedComponentsWithStats((~out).astype(np.uint8), connectivity=4)
        for j in range(1, nb):
            x, y, w, h, a = sb[j]
            if a <= hole_max and x > 0 and y > 0 and x + w < W and y + h < H:
                out[lb == j] = True
    return out


def compute_qc(masks, old_bin, exclude):
    N = len(old_bin)
    per, prev = {}, None
    for i in range(N):
        nm, om = masks[i], old_bin[i]
        emb_resid, spill = frame_part_metrics(om, nm)
        r = {"iou": round(iou(nm, om), 4),
             "area_ratio": round(float(nm.sum()) / max(1.0, float(om.sum())), 4),
             "emb_resid": round(emb_resid, 4),
             "spill": round(spill, 4),
             "empty": bool(nm.sum() == 0)}
        if prev is not None:
            r["adj_iou"] = round(iou(nm, prev), 4)
        prev = nm
        r["flag"] = bool(r["empty"] or r["iou"] < FRAME_IOU_FLAG
                         or not (AREA_RATIO_BAND[0] <= r["area_ratio"] <= AREA_RATIO_BAND[1]))
        per[i] = r
    ev = [i for i in range(N) if i not in exclude]
    ious = [per[i]["iou"] for i in ev]
    resids = [per[i]["emb_resid"] for i in ev]
    areas = [int(masks[i].sum()) for i in ev]
    pos = [a for a in areas if a > 0]
    med_a = float(np.median(pos)) if pos else 0.0
    scene = {
        "median_iou": round(float(np.median(ious)), 4) if ious else 0.0,
        "mean_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
        "min_iou": round(float(np.min(ious)), 4) if ious else 0.0,
        "flag_frac": round(float(np.mean([per[i]["flag"] for i in ev])), 4) if ev else 1.0,
        "n_empty": int(sum(per[i]["empty"] for i in ev)),
        "min_adj_iou": round(float(min((per[i].get("adj_iou", 1.0) for i in ev), default=1.0)), 4),
        "area_collapse": round(min(areas) / med_a, 4) if med_a else 0.0,
        "median_embedded_resid": round(float(np.median(resids)), 4) if resids else 0.0,
        "p90_embedded_resid": round(float(np.percentile(resids, 90)), 4) if resids else 0.0,
        "median_spill": round(float(np.median([per[i]["spill"] for i in ev])), 4) if ev else 0.0,
    }
    return per, scene


def sam2_broken(scene, args):
    """Catastrophic identity failure only: the old masks are *mostly* right, so
    a majority of frames disagreeing means SAM2 locked onto the wrong object.
    Local signals (empty frames, adjacent-IoU dips, area collapse) fire on
    legitimate situations — object briefly out of frame, motion blur, close-ups
    — verified on pilot overlays, so they do NOT auto-reject (review only)."""
    return scene["median_iou"] < args.fallback_median_iou


def advisory_flagged(scene, args):
    """Anything a human should eyeball in the QC gallery."""
    return (scene["median_iou"] < args.scene_min_median_iou
            or scene["flag_frac"] > args.max_flag_frac
            or scene["n_empty"] > args.max_empty
            or scene["min_adj_iou"] < args.min_adj_iou
            or scene["area_collapse"] < args.area_collapse
            or scene.get("median_embedded_resid", 0.0) >= args.part_resid_thresh
            or scene.get("median_spill", 0.0) >= 0.05)


# ------------------------------------------------------------------ SAM2 glue

def to_model_pts(pts, pad):
    return pts if pad is None else pts + np.array([pad["pl"], pad["pt"]], np.float32)


def to_model_box(box, pad):
    return box if pad is None else box + np.array(
        [pad["pl"], pad["pt"], pad["pl"], pad["pt"]], np.float32)


def to_model_mask(m, pad):
    if pad is None:
        return m
    canvas = np.zeros((pad["S"], pad["S"]), bool)
    canvas[pad["pt"]:pad["pt"] + pad["H"], pad["pl"]:pad["pl"] + pad["W"]] = m
    return canvas


def to_orig_mask(m, pad):
    if pad is None:
        return m
    return m[pad["pt"]:pad["pt"] + pad["H"], pad["pl"]:pad["pl"] + pad["W"]]


def select_anchor_mask(img_predictor, frame_rgb, old_bin_model, med_old_area, args):
    """Best-of-multimask at one anchor frame (all in model/frame space).

    Ask SAM2 for its 3 candidate granularities (box + interior positive points)
    and arbitrate with the old mask:
      1. size-sanity band vs the SCENE-median old-mask area (the old masks' most
         reliable property is object scale; robust to per-frame floods): specks
         and sub-part candidates are discarded outright — the embedded-residual
         test is a decoration detector and is blind to gross under-segmentation;
      2. veto sane candidates that lose a part the old mask owns embedded in
         the object contour (an attached bow); keep the highest SAM2-scored
         survivor, else the least-losing one (min resid).
    Returns (winner|None, info); None = no sane candidate, caller shifts frame.
    The winner is SAM2 output — the old mask itself never enters memory."""
    clean = clean_mask(old_bin_model)
    if clean.sum() < MIN_PROMPT_AREA:
        clean = old_bin_model
    H, W = clean.shape
    box, pts = derive_box_points(clean, H, W)
    img_predictor.set_image(frame_rgb)
    masks, scores, _ = img_predictor.predict(
        point_coords=pts, point_labels=np.ones(len(pts), np.int32),
        box=box, multimask_output=True)
    hal_old = halo(clean)
    clean_area = max(1, int(clean.sum()))
    cands, spills, covers = [], [], []
    for m, s in zip(masks, scores):
        mb = m.astype(bool)
        a = int(mb.sum())
        spills.append(float(np.logical_and(mb, np.logical_not(hal_old)).sum()) / max(1, a))
        covers.append(float(np.logical_and(mb, clean).sum()) / clean_area)
        cands.append((embedded_resid_frac(clean, mb), float(s), mb))
    info = {"cands": [{"resid": round(r, 4), "sam2_iou": round(s, 4), "area": int(m.sum()),
                       "spill": round(sp, 4), "old_cover": round(cv, 4)}
                      for (r, s, m), sp, cv in zip(cands, spills, covers)],
            "med_old_area": int(med_old_area)}
    # calibrated on smoke: whole-object candidates cluster >=0.85x the scene-median
    # old area, half/part candidates <=0.64x — 0.7 separates them; a frame whose true
    # object is genuinely smaller just shifts the anchor to a neighboring frame
    lo, hi = 0.7 * med_old_area, 2.5 * med_old_area
    # old_cover kills total-miss candidates (e.g. a bear-sized patch of背景 cloth):
    # embedded_resid is blind to them (the whole missed object exceeds its CC cap,
    # so they score resid=0 and would auto-win the filtered tier — 000498 bug).
    # When the old mask itself is locally broken, every candidate fails this too
    # and the anchor slot shifts to a neighboring frame — the desired behavior.
    sane = [c for c, sp, cv in zip(cands, spills, covers)
            if lo <= c[2].sum() <= hi
            and cv >= args.cand_old_cover_min
            and not (args.cand_spill_max and sp > args.cand_spill_max)]
    if not sane:
        info.update(pick="no_sane")
        return None, info
    ok = [c for c in sane if c[0] <= args.cand_resid_max]
    if ok:
        resid, score, winner = max(ok, key=lambda c: c[1])
        pick = "filtered"
    else:  # nothing fully clean: take the sane candidate losing the least of old
        resid, score, winner = min(sane, key=lambda c: c[0])
        pick = "min_resid"
    info.update(picked_resid=round(float(resid), 4), picked_score=round(float(score), 4),
                pick=pick)
    return winner, info


def old_frame_sane(old_f, med_old_area):
    """Is the OLD mask locally trustworthy at this frame? Flooded/fragmented old
    frames poison every old-referenced candidate test — against a flooded old,
    an object+background candidate shows high cover, low spill AND resid~0
    (000670 anchored bear+cloth that way), so such frames must not anchor."""
    a = int(old_f.sum())
    if not (0.5 * med_old_area <= a <= 1.6 * med_old_area):
        return False
    n, _, stats, _ = cv2.connectedComponentsWithStats(old_f.astype(np.uint8), connectivity=8)
    if n <= 1:
        return False
    return float(stats[1:, cv2.CC_STAT_AREA].max()) / a >= 0.75


def run_anchor_pass(predictor, img_predictor, state, frames_dir, anchors,
                    old_bin, view_num, sampled, pad, args, anchor_log):
    """Select multimask winners per slot, consensus-filter them, then one
    bidirectional propagation.

    Slot placement additionally requires a locally-sane OLD mask (see
    old_frame_sane), shifting to neighbors otherwise. After selection, winners
    whose area strays >1.5x / <0.6x from the winner median are dropped before
    propagation: mixed object+background anchors conflict with the clean ones
    and can collapse propagation to empty between them (000670, 83 empty
    frames). Returns (masks, per, scene_m, used_anchor_frames)."""
    med_old_area = float(np.median([a for a in
                                    (int(m.sum()) for m in old_bin) if a > 0]) or 0.0)
    picked, used = [], []
    for f0 in anchors:
        placed = False
        for d in sorted(range(-15, 16), key=abs):
            f = f0 + d
            if not (0 <= f < view_num) or f in sampled or f in used \
                    or old_bin[f].sum() < MIN_PROMPT_AREA \
                    or not old_frame_sane(old_bin[f], med_old_area):
                continue
            rgb = np.array(Image.open(frames_dir / f"{f}.jpg").convert("RGB"))
            winner, info = select_anchor_mask(
                img_predictor, rgb, to_model_mask(old_bin[f], pad), med_old_area, args)
            info["frame"] = f
            anchor_log.append(info)
            if winner is None:
                continue  # no sane candidate here, shift to a neighbor
            picked.append((f, winner))
            used.append(f)
            placed = True
            break
        if not placed:
            anchor_log.append({"frame": f0, "pick": "slot_dropped"})
    if not picked:
        return None, None, None, []
    areas = np.array([int(w.sum()) for _, w in picked], dtype=np.float64)
    med_w = float(np.median(areas))
    keep = [(f, w) for (f, w), a in zip(picked, areas) if 0.6 * med_w <= a <= 1.5 * med_w]
    if len(keep) < 2:  # never propagate from a single outlier-filtered anchor
        order = sorted(zip(picked, areas), key=lambda t: abs(t[1] - med_w))
        keep = [fw for fw, _ in order[:min(2, len(order))]]
    dropped = sorted(set(f for f, _ in picked) - set(f for f, _ in keep))
    if dropped:
        anchor_log.append({"pick": "consensus_dropped", "frames": dropped,
                           "med_winner_area": int(med_w)})
    predictor.reset_state(state)
    for f, w in keep:
        predictor.add_new_mask(state, f, OBJ_ID, torch.from_numpy(w))
    final = sorted(f for f, _ in keep)
    masks = {i: postprocess_mask(m) for i, m in
             propagate_all(predictor, state, view_num, pad).items()}
    per, scene_m = compute_qc(masks, old_bin, exclude=set(final))
    return masks, per, scene_m, final


def pick_simple_anchor(old_soft, old_bin, sampled, view_num, H, W):
    """User's minimal scheme: score every frame's OLD mask for trustworthiness and
    return the single best frame — area close to the clip median, one solid piece,
    crisp soft probabilities (small PointRend ambiguity band), not touching the
    image border, near mid-clip. No SAM arbitration at all."""
    areas = [int(m.sum()) for m in old_bin]
    pos = [a for a in areas if a > 0]
    if not pos:
        return None
    med = float(np.median(pos))
    border_px = max(2, int(0.02 * min(H, W)))
    best, best_s = None, -1e18
    for i in range(view_num):
        if i in sampled or areas[i] < MIN_PROMPT_AREA:
            continue
        m = old_bin[i]
        a = float(areas[i])
        n, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        lf = float(st[1:, cv2.CC_STAT_AREA].max()) / a if n > 1 else 0.0
        ambig = float(((old_soft[i] > 25) & (old_soft[i] < 230)).sum()) / a
        ys, xs = np.nonzero(m)
        at_border = (xs.min() < border_px or ys.min() < border_px
                     or xs.max() >= W - border_px or ys.max() >= H - border_px)
        mid = 1.0 - abs(i / max(1, view_num - 1) - 0.5) * 2.0
        s = (2.0 * lf - 1.0 * ambig - 1.0 * abs(float(np.log(a / med)))
             - 3.0 * at_border + 0.3 * mid)
        if s > best_s:
            best_s, best = s, i
    return best


def run_quality_pass(predictor, img_predictor, state, frames_dir, old_bin, view_num,
                     sampled, pad, args, anchor_log):
    """Scan frames, anchor ONLY where the multimask winner loses nothing.

    Head (<0.1N) and tail (>0.9N) each get one mandatory anchor (best resid,
    gate anchor_resid_end) to kill the extrapolation tails where all our
    frag/empty failures live; mid-clip gets up to 3 bucket-best anchors gated
    at anchor_resid_mid — a bad anchor poisons memory, so slots stay EMPTY
    rather than take a lossy winner. Returns (masks, per, scene_m, used)."""
    med_old_area = float(np.median([a for a in
                                    (int(m.sum()) for m in old_bin) if a > 0]) or 0.0)
    step = max(1, view_num // max(8, args.scan_frames))
    pool = []
    for f in range(0, view_num, step):
        if f in sampled or old_bin[f].sum() < MIN_PROMPT_AREA \
                or not old_frame_sane(old_bin[f], med_old_area):
            continue
        rgb = np.array(Image.open(frames_dir / f"{f}.jpg").convert("RGB"))
        winner, info = select_anchor_mask(
            img_predictor, rgb, to_model_mask(old_bin[f], pad), med_old_area, args)
        info["frame"] = f
        anchor_log.append(info)
        if winner is None:
            continue
        pool.append((f, float(info["picked_resid"]), winner))
    if not pool:
        return None, None, None, []
    n1, n2 = 0.1 * view_num, 0.9 * view_num
    chosen = {}

    def take(cands, thr):
        ok = [p for p in cands if p[1] <= thr and p[0] not in chosen]
        if ok:
            f, _, w = min(ok, key=lambda p: p[1])
            chosen[f] = w

    take([p for p in pool if p[0] < n1], args.anchor_resid_end)
    take([p for p in pool if p[0] > n2], args.anchor_resid_end)
    mid = [p for p in pool if n1 <= p[0] <= n2]
    for b in range(3):
        lo, hi = n1 + (n2 - n1) * b / 3, n1 + (n2 - n1) * (b + 1) / 3
        take([p for p in mid if lo <= p[0] < hi], args.anchor_resid_mid)
    if len(chosen) < 2:
        anchor_log.append({"pick": "quality_insufficient",
                           "pool": len(pool), "chosen": sorted(chosen)})
        return None, None, None, []
    predictor.reset_state(state)
    for f in sorted(chosen):
        predictor.add_new_mask(state, f, OBJ_ID, torch.from_numpy(chosen[f]))
    used = sorted(chosen)
    anchor_log.append({"pick": "quality_selected", "frames": used})
    masks = {i: postprocess_mask(m) for i, m in
             propagate_all(predictor, state, view_num, pad).items()}
    per, scene_m = compute_qc(masks, old_bin, exclude=set(used))
    return masks, per, scene_m, used


def refine_consumed_frames(img_predictor, frames_dir, masks, consumed, pad, args):
    """Track-then-refine: re-segment each consumed frame with the image predictor
    prompted by the PROPAGATED mask itself (box + interior points), replacing it
    with the candidate most similar to the propagation (identity from tracking,
    boundary from a stateless single-frame decode). Empty frames borrow the
    nearest non-empty frame's mask as prompt. Returns (replacements, log)."""
    view_num = len(masks)
    reps, log = {}, []
    for i in consumed:
        src, ref = masks[i], "self"
        if src.sum() < MIN_PROMPT_AREA:
            j = None
            for d in range(1, view_num):
                for cand in (i + d, i - d):
                    if 0 <= cand < view_num and masks[cand].sum() >= MIN_PROMPT_AREA:
                        j = cand
                        break
                if j is not None:
                    break
            if j is None:
                log.append({"frame": i, "action": "kept_empty"})
                continue
            src, ref = masks[j], f"neighbor:{j}"
        prompt_m = clean_mask(src)
        pm_model = to_model_mask(prompt_m, pad)
        Hm, Wm = pm_model.shape
        box, pts = derive_box_points(pm_model, Hm, Wm)
        rgb = np.array(Image.open(frames_dir / f"{i}.jpg").convert("RGB"))
        img_predictor.set_image(rgb)
        m3, s3, _ = img_predictor.predict(
            point_coords=pts, point_labels=np.ones(len(pts), np.int32),
            box=box, multimask_output=True)
        cands = [(to_orig_mask(m.astype(bool), pad), float(s)) for m, s in zip(m3, s3)]
        ref_area = float(prompt_m.sum())
        sane = [(c, s) for c, s in cands if 0.4 * ref_area <= c.sum() <= 2.5 * ref_area]
        if not sane:
            log.append({"frame": i, "action": "kept_no_sane", "ref": ref})
            continue
        if ref == "self":
            best, score = max(sane, key=lambda t: iou(t[0], prompt_m))
            if iou(best, prompt_m) < 0.3:
                log.append({"frame": i, "action": "kept_lowiou", "ref": ref})
                continue
        else:
            best, score = max(sane, key=lambda t: t[1])
        new = postprocess_mask(best)
        log.append({"frame": i, "action": "replaced", "ref": ref,
                    "delta_iou": round(iou(new, masks[i]), 4) if masks[i].sum() else None,
                    "score": round(score, 4)})
        reps[i] = new
    return reps, log


def propagate_all(predictor, state, view_num, pad):
    masks = {}
    for fidx, _, logits in predictor.propagate_in_video(state):
        masks[fidx] = to_orig_mask((logits[0, 0] > 0.0).cpu().numpy(), pad)
    for fidx, _, logits in predictor.propagate_in_video(state, reverse=True):
        masks[fidx] = to_orig_mask((logits[0, 0] > 0.0).cpu().numpy(), pad)
    assert len(masks) == view_num, f"propagation covered {len(masks)}/{view_num} frames"
    return masks


# --------------------------------------------------------------------- output

def render_overlays(frames_dir, out_dir, selection, old_bin, masks, pad):
    for tag, i in selection.items():
        img = cv2.imread(str(frames_dir / f"{i}.jpg"))
        if img is None:
            continue
        if pad is not None:
            img = img[pad["pt"]:pad["pt"] + pad["H"], pad["pl"]:pad["pl"] + pad["W"]]
        for m, color in ((old_bin[i], (0, 0, 255)), (masks[i], (0, 255, 0))):
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cs, -1, color, 2)
        h, w = img.shape[:2]
        img = cv2.resize(img, (max(1, w // 2), max(1, h // 2)))
        cv2.imwrite(str(out_dir / f"overlay_{tag}_{i:03d}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])


def write_masks(out_dir, masks):
    def _w(item):
        i, m = item
        Image.fromarray(m.astype(np.uint8) * 255, mode="L").save(
            out_dir / f"{i}.png", optimize=True)
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(_w, masks.items()))


def sam2_commit():
    try:
        import sam2 as _s
        repo = Path(_s.__file__).resolve().parents[1]
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------- scene driver

def process_scene(predictor, img_predictor, sid, args, run_meta):
    out_dir = Path(args.output_dir) / sid
    stats_path = out_dir / "stats.json"
    if stats_path.exists():
        if not args.overwrite:
            return "skipped", None
        stats_path.unlink()  # drop the completion marker before touching PNGs
    t0 = time.time()

    prefix, view_num, imgs_bytes, fgms_bytes = load_scene_tar(Path(args.input_dir) / f"{sid}.tar")
    frames_dir = Path(tempfile.mkdtemp(prefix=f"sam2_{sid}_", dir=args.tmp_root))
    try:
        H, W, pad = prepare_frames(imgs_bytes, prefix, view_num, frames_dir, args.square_mode)
        del imgs_bytes
        old_soft = load_old_masks(fgms_bytes, prefix, view_num, H, W)
        old_bin = [s > 127 for s in old_soft]
        del fgms_bytes

        sampled = set(np.linspace(0, view_num - 1, args.n_views, dtype=int).tolist())

        anchor_log = []
        retry, fallback = False, False
        sam2_attempt = None  # best SAM2 result kept for diagnosis when falling back
        pre_refine = None    # pure-propagation snapshot when --refine_consumed
        refine_log = None
        anchors, mask_cond, used = [], set(), []
        state = predictor.init_state(video_path=str(frames_dir))
        try:
            if args.anchor_mode == "simple":
                best = pick_simple_anchor(old_soft, old_bin, sampled, view_num, H, W)
                if best is None:
                    masks = None
                else:
                    predictor.reset_state(state)
                    predictor.add_new_mask(
                        state, best, OBJ_ID,
                        torch.from_numpy(to_model_mask(clean_mask(old_bin[best]), pad)))
                    anchor_log.append({"frame": best, "pick": "simple_old_mask"})
                    masks = {i: postprocess_mask(m) for i, m in
                             propagate_all(predictor, state, view_num, pad).items()}
                    per, scene_m = compute_qc(masks, old_bin, exclude={best})
                    used = [best]
            elif args.anchor_mode == "quality":
                masks, per, scene_m, used = run_quality_pass(
                    predictor, img_predictor, state, frames_dir, old_bin, view_num,
                    sampled, pad, args, anchor_log)
            else:  # grid — previous fixed-slot behavior
                slots = pick_mask_anchors(view_num, sampled, old_bin, args.anchors)
                masks = None
                if slots:
                    masks, per, scene_m, used = run_anchor_pass(
                        predictor, img_predictor, state, frames_dir, slots,
                        old_bin, view_num, sampled, pad, args, anchor_log)
                    if masks is not None and sam2_broken(scene_m, args):
                        retry = True
                        slots5 = pick_mask_anchors(view_num, sampled, old_bin, args.anchors + 2)
                        if len(slots5) > len(used):
                            log5 = []
                            masks5, per5, scene5, used5 = run_anchor_pass(
                                predictor, img_predictor, state, frames_dir, slots5,
                                old_bin, view_num, sampled, pad, args, log5)
                            if masks5 is not None and \
                                    ((not sam2_broken(scene5, args), scene5["median_iou"],
                                      -scene5["flag_frac"])
                                     > (not sam2_broken(scene_m, args), scene_m["median_iou"],
                                        -scene_m["flag_frac"])):
                                masks, per, scene_m, used, anchor_log = \
                                    masks5, per5, scene5, used5, log5

            if masks is None:
                final_failed, fallback = True, True
                prompt_mode = f"{args.anchor_mode}_anchor_none+fallback_old"
                masks = {i: old_bin[i].copy() for i in range(view_num)}
                per, scene_m = compute_qc(masks, old_bin, exclude=set())
            else:
                anchors, mask_cond = used, set(used)
                prompt_mode = f"{args.anchor_mode}_anchor_x{len(used)}"
                final_failed = sam2_broken(scene_m, args)
                sam2_attempt = {"masks": masks, "per": per,
                                "scene": scene_m, "mask_cond": mask_cond}
                if final_failed:
                    fallback = True
                    prompt_mode += "+fallback_old"
                    masks = {i: old_bin[i].copy() for i in range(view_num)}
                    per, scene_m = compute_qc(masks, old_bin, exclude=set())
                elif args.refine_consumed:
                    consumed = sorted(sampled)
                    pre_refine = {"per": per, "scene": scene_m,
                                  "masks_consumed": {i: masks[i] for i in consumed}}
                    reps, refine_log = refine_consumed_frames(
                        img_predictor, frames_dir, masks, consumed, pad, args)
                    if reps:
                        masks = dict(masks)
                        masks.update(reps)
                        per, scene_m = compute_qc(masks, old_bin, exclude=mask_cond)
                        sam2_attempt = {"masks": masks, "per": per,
                                        "scene": scene_m, "mask_cond": mask_cond}
                    prompt_mode += "+refine"
        finally:
            predictor.reset_state(state)
            del state

        out_dir.mkdir(parents=True, exist_ok=True)
        for stale in out_dir.glob("overlay_*.jpg"):
            stale.unlink()
        # overlays always show the SAM2 attempt (post-fallback masks == old masks,
        # whose overlay would be uninformative)
        diag = sam2_attempt if (fallback and sam2_attempt) else \
            {"masks": masks, "per": per, "scene": scene_m, "mask_cond": mask_cond}
        ev = [i for i in range(view_num) if i not in diag["mask_cond"]]
        by_iou = sorted(ev, key=lambda i: diag["per"][i]["iou"])
        selection = {"worst": by_iou[0], "median": by_iou[len(by_iou) // 2]}
        selection["anchor"] = anchors[len(anchors) // 2] if anchors \
            else sorted(sampled)[len(sampled) // 2]
        render_overlays(frames_dir, out_dir, selection, old_bin, diag["masks"], pad)
        write_masks(out_dir, masks)

        stats = {
            "complete": True,
            "scene": sid, "prefix": prefix, "view_num": view_num,
            "H": H, "W": W, "square_mode": args.square_mode,
            "prompt_mode": prompt_mode,
            "anchor_mode": args.anchor_mode,
            "anchor_frames": sorted(anchors) if anchors else [],
            "anchors_mask_cond": sorted(mask_cond),
            "anchor_multimask": anchor_log,
            "refine": ({"n_replaced": sum(1 for e in refine_log if e["action"] == "replaced"),
                        "log": refine_log} if refine_log is not None else None),
            "pre_refine_scene_metrics": pre_refine["scene"] if pre_refine else None,
            "retry": retry, "fallback": fallback,
            "needs_review": bool(fallback or final_failed
                                 or advisory_flagged(diag["scene"], args)),
            "scene_metrics": scene_m,
            "sam2_attempt_metrics": diag["scene"] if fallback and sam2_attempt else None,
            "sam2_attempt_per_frame": ({str(i): diag["per"][i] for i in range(view_num)}
                                       if fallback and sam2_attempt else None),
            "overlays_from": "sam2_attempt" if fallback and sam2_attempt else "final",
            "per_frame": {str(i): per[i] for i in range(view_num)},
            "elapsed_s": round(time.time() - t0, 1),
            **run_meta,
        }
        if pre_refine is not None:
            # keep the pure-propagation stage for the refine A/B: consumed-frame
            # PNGs + a stats.json carrying pre-refine metrics (audit-compatible)
            pre_dir = Path(str(args.output_dir).rstrip("/") + "_pre") / sid
            pre_dir.mkdir(parents=True, exist_ok=True)
            write_masks(pre_dir, pre_refine["masks_consumed"])
            pre_stats = dict(stats)
            pre_stats["per_frame"] = {str(i): pre_refine["per"][i] for i in range(view_num)}
            pre_stats["scene_metrics"] = pre_refine["scene"]
            pre_stats["stage"] = "pre_refine"
            (pre_dir / "stats.json").write_text(json.dumps(pre_stats))

        tmp_stats = stats_path.with_suffix(".json.tmp")
        tmp_stats.write_text(json.dumps(stats))
        tmp_stats.rename(stats_path)  # atomic completion marker
        tag = ("FALLBACK" if fallback
               else ("REVIEW" if advisory_flagged(diag["scene"], args)
                     else ("RETRY" if retry else "ok")))
        rep = diag["scene"]
        print(f"[{sid}] {tag} mode={prompt_mode} med={rep['median_iou']} "
              f"resid={rep.get('median_embedded_resid')} flag={rep['flag_frac']} "
              f"({stats['elapsed_s']}s)", flush=True)
        return "fallback" if fallback else "done", scene_m
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)


def main():
    args = parse_args()
    scenes = [l.strip() for l in Path(args.split_file).read_text().splitlines() if l.strip()]
    if args.scene:
        want = args.scene.split(",")
        scenes = [s for s in scenes if s in want] or want
    else:
        scenes = scenes[args.shard_idx::args.num_shards]
    if args.limit:
        scenes = scenes[:args.limit]

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    predictor = build_sam2_video_predictor(args.model_cfg, args.ckpt, device="cuda")
    # image predictor wraps the SAME model (SAM2VideoPredictor IS-A SAM2Base):
    # used per anchor frame to get the 3 multimask granularity candidates
    img_predictor = SAM2ImagePredictor(predictor)
    run_meta = {"ckpt": args.ckpt, "model_cfg": args.model_cfg,
                "sam2_commit": sam2_commit(), "host": socket.gethostname(),
                "gpu": torch.cuda.get_device_name(0)}
    print(f"shard {args.shard_idx}/{args.num_shards}: {len(scenes)} scenes "
          f"on {run_meta['host']}/{run_meta['gpu']} square_mode={args.square_mode}", flush=True)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    err_log = Path(args.output_dir) / f"errors_shard{args.shard_idx}.log"
    counts = {"done": 0, "skipped": 0, "fallback": 0, "error": 0}
    for sid in scenes:
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                status, _ = process_scene(predictor, img_predictor, sid, args, run_meta)
            counts[status] += 1
        except Exception:
            counts["error"] += 1
            msg = f"[{sid}] ERROR\n{traceback.format_exc()}\n"
            print(msg, file=sys.stderr, flush=True)
            with open(err_log, "a") as f:
                f.write(msg)
    print(f"shard {args.shard_idx} summary: {counts}", flush=True)
    sys.exit(0 if counts["error"] == 0 else 2)


if __name__ == "__main__":
    main()
