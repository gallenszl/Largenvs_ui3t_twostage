#!/usr/bin/env python
"""Audit the 25 converter-consumed frames of each scene's SAM2 masks.

Downstream only ever consumes linspace(0, N-1, 25) frames per scene, so this
audits exactly those. Two signal tiers:

AUTO-BAD (old-mask-independent where possible -> scene goes to unusable.txt):
    empty   new mask empty while the old mask says an object is in frame
    frag    new mask shattered (largest CC < frag_cc_frac of area AND >= frag_min_cc big CCs)
    frag2   persistently split in two (largest CC < frag2_cc_frac AND >= 2 big CCs
            on >= frag2_min consumed frames — a 2-piece mask evades the frag rule)
    tbreak  temporal break (adj_iou vs previous frame < tbreak_adj)
    shape   solidity collapse: frame solidity drops > sol_dev below the scene's own
            median — a one-blob mask that lost a limb/part stays smooth and single-CC
            (evades empty/frag/tbreak) but its convexity dips; self-referenced, so
            constantly-concave objects (spread limbs) never trigger
    shape90 solidity collapse vs the scene P90 reference (sol_dev90) — when most of
            the video is degraded the median reference is dragged down with it;
            the healthiest frames still betray what the object should look like
    iou     catastrophic disagreement (iou < bad_iou) — well below old-mask noise
    shrink  area_ratio < bad_shrink (mask collapsed vs old)

SUSPECT (spill/resid rank scenes for HUMAN review — each has a good and a bad
cause that are numerically identical: high spill = board suction OR leg
recovery; high resid = part cut OR old-mask shadow. Never auto-verdict.)

Outputs under --out_dir:
    unusable.txt / usable.txt / suspect_ranked.txt / audit.jsonl / audit.html
audit.html renders orig|composite|contour triplets of every offending consumed
frame (unusable all, suspects ranked, random usable sample) so the filter's
accuracy can be verified in both directions.
"""
import argparse
import io
import json
import random
import re
import sys
import tarfile
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

MIN_OBJ_AREA = 500  # old-mask area below this => object plausibly out of frame


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--mask_dir", default="/home/z50057756/data/co3d_sam2_masks/val")
    ap.add_argument("--input_dir", default="/mnt/data-alpha-sg-02/team-camera/datasets/yuchen/co3d/webdataset/val")
    ap.add_argument("--split_file", default="/home/z50057756/code/RnG_feature_allignment/data/co3d_teddybear_val_1140.txt")
    ap.add_argument("--out_dir", default="/home/z50057756/data/co3d_sam2_masks/qc/audit")
    ap.add_argument("--n_views", type=int, default=25)
    ap.add_argument("--limit", type=int, default=0)
    # AUTO-BAD thresholds
    ap.add_argument("--frag_cc_frac", type=float, default=0.7)
    ap.add_argument("--frag_min_cc", type=int, default=3)
    ap.add_argument("--frag2_cc_frac", type=float, default=0.85)
    ap.add_argument("--frag2_min", type=int, default=3,
                    help="2-piece frames on at least this many consumed frames -> convict")
    ap.add_argument("--sol_dev", type=float, default=0.12,
                    help="frame solidity below (scene median - sol_dev) -> shape convict")
    ap.add_argument("--sol_dev90", type=float, default=0.19,
                    help="frame solidity below (scene P90 - sol_dev90) -> shape90 convict; "
                         "P90 restores a healthy reference when >half the video is "
                         "degraded and drags the median down with it (000156 class)")
    ap.add_argument("--shape_arm_p90", type=float, default=0.8,
                    help="arm shape/shape90/frag2 ONLY when scene P90 solidity >= this: "
                         "the rules encode a blob prior; thin/articulated objects "
                         "(chair/bicycle/wireframe) legitimately swing solidity and "
                         "split CCs with viewpoint — for them the rules are disarmed "
                         "(cross-category calibration, cat50)")
    ap.add_argument("--tbreak_adj", type=float, default=0.4)
    ap.add_argument("--bad_iou", type=float, default=0.35)
    ap.add_argument("--bad_shrink", type=float, default=0.45)
    ap.add_argument("--selfstab_adj", type=float, default=0.7,
                    help="iou/shrink convict ONLY when the new mask is also self-"
                         "unstable around that frame (adj_iou below this); a mask "
                         "that disagrees with old but tracks smoothly is treated as "
                         "old-mask drift and passes (no-human pipeline, fix 2)")
    # RESCUE (V9, opt-in; default OFF keeps the frozen ruleset byte-identical).
    # Acquits tbreak/shape/shape90 convictions ONLY with positive old-mask
    # attestation; validated by user on 30 candidates (29 kept, 1 leak closed).
    ap.add_argument("--rescue", action="store_true",
                    help="enable R0/R4-gated acquittal of tbreak/shape convictions")
    ap.add_argument("--rescue_tau", type=float, default=0.85,
                    help="R1/R2: frame-level acquittal needs iou-vs-old >= tau")
    ap.add_argument("--rescue_scene_iou", type=float, default=0.75,
                    help="R0: every non-empty consumed frame needs iou >= this")
    ap.add_argument("--rescue_ar_lo", type=float, default=0.75)
    ap.add_argument("--rescue_ar_hi", type=float, default=1.35)
    ap.add_argument("--rescue_hole_frac", type=float, default=0.04,
                    help="R4: convicted frame with enclosed hole >= this fraction "
                         "of mask area vetoes the scene (decoration-exclusion, 000098)")
    ap.add_argument("--rescue_hole_px", type=int, default=1500)
    # SUSPECT thresholds (ranking only)
    ap.add_argument("--sus_spill", type=float, default=0.25)
    ap.add_argument("--sus_ratio", type=float, default=1.25)
    ap.add_argument("--sus_resid", type=float, default=0.15)
    ap.add_argument("--usable_sample", type=int, default=20)
    ap.add_argument("--max_render_frames", type=int, default=4, help="offending frames rendered per scene")
    ap.add_argument("--max_suspect_render", type=int, default=150)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no_render", action="store_true", help="metrics + lists only (for hp compare)")
    ap.add_argument("--tag", default="", help="suffix for output filenames (hp configs)")
    return ap.parse_args()


# ------------------------------------------------------------------ tar utils

def _member_map(tf, prefix, kind, ext):
    pat = re.compile(re.escape(prefix) + r"\." + kind + r"_(\d+)\." + ext + r"$")
    out = {}
    for mem in tf.getmembers():
        mo = pat.search(mem.name)
        if mo:
            out[int(mo.group(1))] = mem
    return out


def load_scene_tars(tar_path, want_images):
    """Return (old_masks_bytes_tf, images_tf_or_None) kept open on BytesIO."""
    with tarfile.open(tar_path) as outer:
        meta_name = next(n for n in outer.getnames() if n.endswith(".meta.json"))
        prefix = Path(meta_name).name[: -len(".meta.json")]
        fgms = outer.extractfile(next(n for n in outer.getnames()
                                      if n.endswith(f"{prefix}.image_masks.tar"))).read()
        imgs = None
        if want_images:
            imgs = outer.extractfile(next(n for n in outer.getnames()
                                          if n.endswith(f"{prefix}.images.tar"))).read()
    ftf = tarfile.open(fileobj=io.BytesIO(fgms))
    itf = tarfile.open(fileobj=io.BytesIO(imgs)) if imgs else None
    return prefix, ftf, itf


def old_mask_at(ftf, prefix, mm, i):
    return np.array(Image.open(io.BytesIO(ftf.extractfile(mm[i]).read()))) > 127


# ------------------------------------------------------------------ metrics

def frag_stats(mask_bool):
    a = int(mask_bool.sum())
    if a == 0:
        return 1.0, 0
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask_bool.astype(np.uint8), connectivity=8)
    if n <= 1:
        return 1.0, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    big_min = max(64, int(0.005 * a))
    big = areas[areas >= big_min]
    return float(areas.max()) / a, int(len(big))


def solidity(mask_bool):
    pts = cv2.findNonZero(mask_bool.astype(np.uint8))
    if pts is None or len(pts) < 10:
        return None
    hull_area = cv2.contourArea(cv2.convexHull(pts))
    return float(mask_bool.sum() / hull_area) if hull_area > 0 else None


def enclosed_hole(mask_bool):
    """Largest background CC fully enclosed by the mask (px). Decoration
    carved out of the object (000098's rose) shows up here; legit see-through
    gaps are small relative to the mask."""
    inv = (~mask_bool).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    H, W = mask_bool.shape
    best = 0
    for k in range(1, n):
        x, y, w, h, a = stats[k]
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue
        best = max(best, int(a))
    return best


RESCUABLE = {"tbreak", "shape", "shape90"}


def apply_rescue(auto_bad, frames, mdir, args):
    """V9-rescue: positive-evidence acquittal. Returns (new_auto_bad, rescued).

    R0  scene gate: old attests EVERY non-empty consumed frame
        (iou >= rescue_scene_iou, ar in [ar_lo, ar_hi]); any unhealed
        disagreement anywhere disqualifies the scene (000094 class:
        correlated dark-frame failure hides behind a high-iou convicted frame).
    R4  hole veto: a convicted frame whose mask has an enclosed hole
        >= max(hole_px, hole_frac * area) is a carved-out decoration (000098).
    R1/R2  frame acquittal: tbreak/shape/shape90 dropped iff that very frame
        is old-attested (iou >= rescue_tau, non-empty).
    empty/frag/frag2/iou/shrink are never rescued."""
    nonempty = [d for d in frames.values() if not d.get("empty")]
    if len(nonempty) < 5:
        return auto_bad, {}
    if not all((d.get("iou") or 0) >= args.rescue_scene_iou
               and args.rescue_ar_lo <= (d.get("area_ratio") or 0) <= args.rescue_ar_hi
               for d in nonempty):
        return auto_bad, {}
    for i in auto_bad:
        m = np.array(Image.open(mdir / f"{i}.png")) > 127
        hole = enclosed_hole(m)
        area = int(m.sum())
        if hole >= args.rescue_hole_px and area and hole / area >= args.rescue_hole_frac:
            return auto_bad, {}
    new_bad, rescued = {}, {}
    for i, reasons in auto_bad.items():
        d = frames.get(i, {})
        attested = (d.get("iou") or 0) >= args.rescue_tau and not d.get("empty")
        keep = [x for x in reasons if x not in RESCUABLE or not attested]
        dropped = [x for x in reasons if x not in keep]
        if keep:
            new_bad[i] = keep
        if dropped:
            rescued[i] = dropped
    return new_bad, rescued


def audit_scene(task):
    sid, args_d = task
    args = argparse.Namespace(**args_d)
    try:
        mdir = Path(args.mask_dir) / sid
        st = json.loads((mdir / "stats.json").read_text())
        pf, N = st["per_frame"], st["view_num"]
        consumed = np.linspace(0, N - 1, args.n_views, dtype=int).tolist()
        auto_bad, suspects, frames = {}, {}, {}
        need_empty_check, old_ref_hits = [], {}
        sols, frag2_frames = {}, []
        for i in consumed:
            v = pf[str(i)]
            reasons, oldref, sus = [], [], []
            if v["empty"]:
                need_empty_check.append(i)  # bad only if old says object is present
            # iou/shrink compare against the OLD mask -> only trustworthy where the
            # old mask is locally sane; deferred until old sanity is known (a
            # correct new mask scores iou~0.1 against a flooded old — 000560).
            if v["iou"] < args.bad_iou and not v["empty"]:
                oldref.append("iou")
            if v["area_ratio"] < args.bad_shrink and not v["empty"]:
                oldref.append("shrink")
            if v.get("adj_iou", 1.0) < args.tbreak_adj and not v["empty"]:
                reasons.append("tbreak")
            if not v["empty"]:
                m = np.array(Image.open(mdir / f"{i}.png")) > 127
                lf, nbig = frag_stats(m)
                if lf < args.frag_cc_frac and nbig >= args.frag_min_cc:
                    reasons.append("frag")
                if lf < args.frag2_cc_frac and nbig >= 2:
                    frag2_frames.append(i)
                if m.sum() >= MIN_OBJ_AREA:
                    s = solidity(m)
                    if s is not None:
                        sols[i] = s
            if v.get("spill", 0.0) >= args.sus_spill and v["area_ratio"] > args.sus_ratio:
                sus.append("suction?")
            if v.get("emb_resid", 0.0) >= args.sus_resid:
                sus.append("partmiss?")
            if reasons:
                auto_bad[i] = reasons
            if oldref:
                old_ref_hits[i] = oldref
            if sus:
                suspects[i] = sus
            frames[i] = {k: v.get(k) for k in ("iou", "area_ratio", "emb_resid", "spill",
                                               "adj_iou", "empty")}
        # shape: solidity collapse vs the scene's own median (self-referenced,
        # old-mask-independent) — catches one-blob masks that lost a part.
        # shape90: same, against the P90 reference — the median reference dies
        # when the majority of frames are themselves degraded.
        # Both (and frag2) encode a BLOB PRIOR: armed only when the object shows
        # high convexity somewhere in the orbit (P90 >= shape_arm_p90); chairs/
        # wireframes swing solidity with viewpoint and are exempt.
        blob_armed = False
        if len(sols) >= 5:
            vals = list(sols.values())
            med_sol = float(np.median(vals))
            p90_sol = float(np.percentile(vals, 90))
            blob_armed = p90_sol >= args.shape_arm_p90
            for i, s in sols.items():
                if blob_armed and s < med_sol - args.sol_dev:
                    auto_bad.setdefault(i, []).append("shape")
                if blob_armed and s < p90_sol - args.sol_dev90:
                    auto_bad.setdefault(i, []).append("shape90")
                frames[i]["solidity"] = round(s, 3)
        # frag2: persistently split in two (2 big CCs evade the >=3-CC frag rule)
        if blob_armed and len(frag2_frames) >= args.frag2_min:
            for i in frag2_frames:
                auto_bad.setdefault(i, []).append("frag2")
        if need_empty_check or old_ref_hits:
            prefix, ftf, _ = load_scene_tars(Path(args.input_dir) / f"{sid}.tar", False)
            mm = _member_map(ftf, prefix, "image_masks", "png")
            areas = {}
            for j, mem in mm.items():
                areas[j] = int((np.array(Image.open(io.BytesIO(
                    ftf.extractfile(mem).read()))) > 127).sum())
            pos = [a for a in areas.values() if a > 0]
            med_old = float(np.median(pos)) if pos else 0.0
            def old_sane(j):
                a = areas.get(j, 0)
                if med_old <= 0 or not (0.5 * med_old <= a <= 1.6 * med_old):
                    return False
                om = old_mask_at(ftf, prefix, mm, j)
                lf, _ = frag_stats(om)
                return lf >= 0.75
            for i in need_empty_check:
                if areas.get(i, 0) >= MIN_OBJ_AREA:
                    auto_bad.setdefault(i, []).append("empty")

            def self_unstable(i):
                # instability on either transition into or out of frame i
                a_in = pf.get(str(i), {}).get("adj_iou")
                a_out = pf.get(str(i + 1), {}).get("adj_iou")
                vals = [a for a in (a_in, a_out) if a is not None]
                return bool(vals) and min(vals) < args.selfstab_adj

            for i, oldref in old_ref_hits.items():
                if not old_sane(i):
                    # old is flood/fragment here: cannot testify
                    suspects.setdefault(i, []).append("oldbad?")
                elif self_unstable(i):
                    auto_bad.setdefault(i, []).extend(oldref)
                else:
                    # disagrees with old but tracks smoothly: persistent old-mask
                    # drift (000315 class) — pass automatically (fix 2)
                    suspects.setdefault(i, []).append("olddrift?")
            ftf.close()
        rescued = {}
        if args.rescue and auto_bad and not st.get("fallback"):
            auto_bad, rescued = apply_rescue(auto_bad, frames, mdir, args)
        sus_score = sum(frames[i]["spill"] + frames[i]["emb_resid"] for i in suspects)
        return {"scene": sid, "view_num": N, "consumed": consumed,
                "rescued": {str(k): v for k, v in sorted(rescued.items())},
                "auto_bad": {str(k): v for k, v in sorted(auto_bad.items())},
                "suspects": {str(k): v for k, v in sorted(suspects.items())},
                "sus_score": round(float(sus_score), 4),
                "frames": {str(k): frames[k] for k in sorted(frames)},
                "prompt_mode": st.get("prompt_mode"), "fallback": st.get("fallback", False)}
    except Exception:
        return {"scene": sid, "error": traceback.format_exc()}


# ------------------------------------------------------------------ rendering

def render_scene(task):
    sid, frame_ids, args_d = task
    args = argparse.Namespace(**args_d)
    try:
        rdir = Path(args.out_dir) / "render"
        outs = []
        prefix, ftf, itf = load_scene_tars(Path(args.input_dir) / f"{sid}.tar", True)
        imm = _member_map(itf, prefix, "images", "jpg")
        fmm = _member_map(ftf, prefix, "image_masks", "png")
        for i in frame_ids:
            dst = rdir / f"{sid}_{i:03d}.jpg"
            if dst.exists():
                outs.append(dst.name)
                continue
            rgb = np.array(Image.open(io.BytesIO(itf.extractfile(imm[i]).read())).convert("RGB"))
            old = old_mask_at(ftf, prefix, fmm, i)
            new = np.array(Image.open(Path(args.mask_dir) / sid / f"{i}.png")) > 127
            a = new[..., None].astype(np.float32)
            comp = (rgb * a + 255.0 * (1 - a)).astype(np.uint8)
            ov = rgb[:, :, ::-1].copy()
            for m, color in ((old.astype(np.uint8), (0, 0, 255)), (new.astype(np.uint8), (0, 255, 0))):
                cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(ov, cs, -1, color, 2)
            row = np.concatenate([rgb[:, :, ::-1], comp[:, :, ::-1], ov], axis=1)
            h, w = row.shape[:2]
            row = cv2.resize(row, (max(1, w // 2), max(1, h // 2)))
            cv2.imwrite(str(dst), row, [cv2.IMWRITE_JPEG_QUALITY, 86])
            outs.append(dst.name)
        ftf.close()
        itf.close()
        return sid, outs
    except Exception as e:
        return sid, f"RENDER_ERROR: {type(e).__name__}: {e}"


# ------------------------------------------------------------------ html

def chip(txt, cls):
    return f'<span class="chip {cls}">{txt}</span>'


def scene_block(r, names):
    reasons = Counter(x for v in r["auto_bad"].values() for x in v)
    sus = Counter(x for v in r["suspects"].values() for x in v)
    chips = chip("FALLBACK_OLD", "bad") if r.get("fallback") else ""
    chips += "".join(chip(f"{k}×{n}", "bad") for k, n in reasons.most_common())
    if r.get("rescued"):
        resc = Counter(x for v in r["rescued"].values() for x in v)
        chips += "".join(chip(f"rescued:{k}×{n}", "sus") for k, n in resc.most_common())
    chips += "".join(chip(f"{k}×{n}", "sus") for k, n in sus.most_common())
    imgs = "".join(f'<div class="fr"><div class="cap">frame {n.split("_")[-1].split(".")[0]}'
                   f'</div><img loading="lazy" src="render/{n}"></div>'
                   for n in names if not str(names).startswith("RENDER_ERROR"))
    return (f'<div class="sc"><div class="hd">{r["scene"]} '
            f'<span class="meta">{r.get("prompt_mode")} · sus_score {r["sus_score"]}</span> '
            f'{chips}</div><div class="row">{imgs}</div></div>')


def main():
    args = parse_args()
    out = Path(args.out_dir)
    (out / "render").mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    scenes = [l.strip() for l in Path(args.split_file).read_text().splitlines() if l.strip()]
    if args.limit:
        scenes = scenes[:args.limit]
    args_d = vars(args)

    print(f"auditing {len(scenes)} scenes x {args.n_views} consumed frames ...", flush=True)
    rows, errors = [], []
    with ProcessPoolExecutor(args.workers) as ex:
        for k, r in enumerate(ex.map(audit_scene, [(s, args_d) for s in scenes], chunksize=8), 1):
            (errors if "error" in r else rows).append(r)
            if k % 200 == 0:
                print(f"  {k}/{len(scenes)}", flush=True)
    for e in errors[:5]:
        print("ERROR", e["scene"], e["error"].splitlines()[-1], flush=True)

    # fix 1 (no-human pipeline): fallback scenes carry OLD masks and would compare
    # old-to-old (iou==1) — they must never masquerade as usable
    unusable = [r for r in rows if r["auto_bad"] or r.get("fallback")]
    usable = [r for r in rows if not (r["auto_bad"] or r.get("fallback"))]
    suspects = sorted((r for r in usable if r["suspects"]), key=lambda r: -r["sus_score"])
    with open(out / f"audit{tag}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (out / f"unusable{tag}.txt").write_text("".join(r["scene"] + "\n" for r in unusable))
    (out / f"usable{tag}.txt").write_text("".join(r["scene"] + "\n" for r in usable))
    (out / f"suspect_ranked{tag}.txt").write_text(
        "".join(f'{r["scene"]}\t{r["sus_score"]}\t'
                f'{",".join(sorted(set(x for v in r["suspects"].values() for x in v)))}\n'
                for r in suspects))

    reason_ct = Counter(x for r in unusable for v in r["auto_bad"].values() for x in v)
    print(f"unusable: {len(unusable)}/{len(rows)}  usable: {len(usable)} "
          f"(suspect-ranked within usable: {len(suspects)})", flush=True)
    print("auto-bad frame reasons:", dict(reason_ct), flush=True)

    if args.no_render:
        return

    random.seed(0)
    pool = [r for r in usable if not r["suspects"]]
    sample = random.sample(pool, min(args.usable_sample, len(pool)))
    render_tasks = []
    for r in unusable:
        ids = sorted(int(i) for i in r["auto_bad"])[:args.max_render_frames]
        render_tasks.append((r["scene"], ids, args_d))
    for r in suspects[:args.max_suspect_render]:
        ids = sorted(int(i) for i in r["suspects"])[:args.max_render_frames]
        render_tasks.append((r["scene"], ids, args_d))
    for r in sample:
        ids = [r["consumed"][j] for j in (0, len(r["consumed"]) // 2, len(r["consumed"]) - 1)]
        render_tasks.append((r["scene"], ids, args_d))
    print(f"rendering {len(render_tasks)} scenes ...", flush=True)
    rendered = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for k, (sid, names) in enumerate(ex.map(render_scene, render_tasks, chunksize=4), 1):
            rendered[sid] = names
            if k % 100 == 0:
                print(f"  {k}/{len(render_tasks)}", flush=True)

    css = """<style>body{font-family:sans-serif;margin:14px;background:#fafafa}
.sc{margin:0 0 16px;border:1px solid #ddd;background:#fff;border-radius:6px;overflow:hidden}
.hd{padding:6px 10px;font-weight:bold;font-size:13px;background:#f1f1f1}
.meta{font-weight:normal;color:#666}.row{display:flex;flex-wrap:wrap;gap:6px;padding:6px}
.fr img{max-height:230px;display:block}.cap{font-size:11px;color:#777}
.chip{font-size:11px;padding:1px 7px;border-radius:9px;margin-left:4px;font-weight:normal}
.chip.bad{background:#ffd9d9;color:#a00}.chip.sus{background:#fff3c9;color:#875f00}
h2{margin:18px 0 8px}p.note{color:#555;font-size:13px}</style>"""
    html = [f'<!doctype html><meta charset="utf-8"><title>SAM2 consumed-frame audit</title>{css}',
            f'<h1>被消费帧审计 — {len(rows)} scenes</h1>',
            '<p class="note">每帧三联：原图 | SAM2 抠图 | 轮廓（红=旧 mask，绿=SAM2）。'
            'chips：红=AUTO-BAD 原因，黄=SUSPECT（吸板/补腿、切件/阴影——请人工裁决）。</p>',
            f'<h2>❌ unusable（自动判决，{len(unusable)} 个）</h2>']
    for r in sorted(unusable, key=lambda r: -len(r["auto_bad"])):
        html.append(scene_block(r, rendered.get(r["scene"], [])))
    html.append(f'<h2>⚠️ suspect（按分排序渲染前 {min(len(suspects), args.max_suspect_render)}'
                f'/{len(suspects)} 个，从上往下人工裁决）</h2>')
    for r in suspects[:args.max_suspect_render]:
        html.append(scene_block(r, rendered.get(r["scene"], [])))
    html.append(f'<h2>✅ usable 随机抽样对照（{len(sample)} 个）</h2>')
    for r in sample:
        html.append(scene_block(r, rendered.get(r["scene"], [])))
    (out / f"audit{tag}.html").write_text("\n".join(html))
    print(f"wrote {out}/audit{tag}.html (+ usable/unusable/suspect lists)", flush=True)


if __name__ == "__main__":
    main()
