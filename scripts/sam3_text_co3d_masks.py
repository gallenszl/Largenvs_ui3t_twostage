#!/usr/bin/env python
"""SAM3 text-concept mask generation for CO3D webdataset scenes.

Engine change vs the SAM2 pipeline: SAM3's decoupled detector+tracker re-detects
the concept EVERY frame (no pure memory extrapolation), so the drift / fragment /
empty-frame failure class disappears structurally. Per scene:

  1. extract frames (reuse SAM2 worker helpers) -> {i}.jpg dir
  2. text prompt ladder: category phrase ("teddy bear") -> generic fallback
     ("salient foreground object") when recall fails
  3. SAM3 video session: start_session -> add_prompt(text) ->
     propagate_in_video(both) -> masklets {frame: {obj_id: mask}}
  4. instance selection: the OLD mask is only a judge — pick the masklet with
     the highest median IoU vs old over ~20 sampled frames (disambiguates
     multi-instance scenes); best < --min_inst_iou -> fallback_old + review
  5. write {i}.png (original resolution), overlays, instance-selection
     visualization (instsel_*.jpg, all candidates coloured + scores, chosen
     highlighted), stats.json — output layout identical to the SAM2 pipeline,
     so audit / galleries / converter run unchanged.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import shutil
import time
import traceback
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sam2_refine_co3d_masks import (  # noqa: E402  (shared pipeline helpers)
    MIN_PROMPT_AREA, compute_qc, iou, load_old_masks, load_scene_tar,
    postprocess_mask, prepare_frames, render_overlays, write_masks)

PALETTE = [(0, 255, 0), (255, 160, 0), (0, 160, 255), (255, 0, 255),
           (0, 255, 255), (160, 0, 255), (255, 255, 0), (128, 255, 128)]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input_dir", default="/mnt/data-alpha-sg-02/team-camera/datasets/yuchen/co3d/webdataset/val")
    ap.add_argument("--split_file", required=True)
    ap.add_argument("--output_dir", default="/home/z50057756/data/co3d_sam3_masks/val")
    ap.add_argument("--ckpt", default=None, help="local checkpoint path (None = auto HF download)")
    ap.add_argument("--engine", choices=["sam3", "sam3.1"], default="sam3",
                    help="sam3.1 = multiplex tracker weights (same session API)")
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--scene", default=None, help="comma-separated scene ids (debug)")
    ap.add_argument("--tmp_root", default="/tmp")
    ap.add_argument("--n_views", type=int, default=25)
    ap.add_argument("--text", default="auto",
                    help="'auto' derives a noun phrase from the CO3D category in the "
                         "scene id (teddybear -> 'teddy bear'); or give a phrase")
    ap.add_argument("--fallback_text", default="salient foreground object")
    ap.add_argument("--ladder_low_thresh", type=float, default=0.35,
                    help="after all phrases fail, retry the first category phrase "
                         "at this permissive prob threshold (0 disables)")
    ap.add_argument("--min_inst_iou", type=float, default=0.2,
                    help="best masklet's median IoU vs old below this -> try fallback "
                         "text, then fallback_old")
    ap.add_argument("--inst_sample_frames", type=int, default=20)
    ap.add_argument("--prob_thresh", type=float, default=0.5)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


CATEGORY_PHRASES = {
    # short noun phrases (SAM3 PCS training distribution); concatenated CO3D
    # category names are split/naturalized — raw "baseballbat" is OOD for the
    # text encoder. Recall-first: the instance selector arbitrates extras.
    "apple": "apple", "backpack": "backpack", "ball": "ball", "banana": "banana",
    "baseballbat": "baseball bat", "baseballglove": "baseball glove",
    "bench": ["bench", "wooden bench", "outdoor seat"],
    "bicycle": "bicycle", "book": "book", "bottle": "bottle",
    "bowl": "bowl", "broccoli": "broccoli", "cake": "cake", "car": "car",
    "carrot": "carrot", "cellphone": "cell phone", "chair": "chair",
    "couch": ["couch", "sofa"], "cup": "cup", "donut": "donut",
    "frisbee": ["frisbee", "flying disc", "round plastic disc"],
    "hairdryer": "hair dryer", "handbag": "handbag",
    "hotdog": ["hot dog", "sausage in a bun", "sausage"],
    "hydrant": "fire hydrant", "keyboard": "computer keyboard",
    "kite": ["kite", "toy kite", "paper kite"],
    "laptop": "laptop", "microwave": "microwave oven", "motorcycle": "motorcycle",
    "mouse": "computer mouse", "orange": "orange", "parkingmeter": "parking meter",
    "pizza": "pizza", "plant": "potted plant", "remote": "remote control",
    "sandwich": ["sandwich", "slice of bread", "bread"],
    "skateboard": "skateboard",
    "stopsign": ["stop sign", "road sign"],
    "suitcase": "suitcase", "teddybear": "teddy bear", "toaster": "toaster",
    "toilet": "toilet", "toybus": "toy bus", "toyplane": "toy plane",
    "toytrain": "toy train", "toytruck": "toy truck", "tv": "television",
    "umbrella": "umbrella", "vase": "vase", "wineglass": "wine glass",
}

# generic phrases appended after category phrases (ladder v2); the first entry
# is overridable via --fallback_text
GENERIC_EXTRA = ["main object in the scene"]


def scene_phrases(sid, args):
    """Ordered phrase ladder for a scene (v2): category synonyms -> generics."""
    if args.text != "auto":
        cat_phrases = [args.text]
    else:
        # co3d_<category>_<split>-<idx>
        parts = sid.split("_")
        cat = parts[1] if len(parts) >= 3 else "object"
        v = CATEGORY_PHRASES.get(cat, cat.replace("-", " "))
        cat_phrases = list(v) if isinstance(v, (list, tuple)) else [v]
    generics = [t for t in [args.fallback_text] + GENERIC_EXTRA if t]
    out = []
    for t in cat_phrases + generics:
        if t not in out:
            out.append(t)
    return out


def run_text_attempt(predictor, frames_dir, text, prob_thresh):
    """One SAM3 session: text prompt at frame 0, propagate both ways.

    Returns {frame: {obj_id: bool mask (H, W)}}."""
    resp = predictor.handle_request(request=dict(
        type="start_session", resource_path=str(frames_dir),
        offload_video_to_cpu=True))
    session_id = resp["session_id"]
    try:
        predictor.handle_request(request=dict(
            type="add_prompt", session_id=session_id, frame_index=0,
            text=text, output_prob_thresh=prob_thresh))
        per_frame = {}
        for r in predictor.handle_stream_request(request=dict(
                type="propagate_in_video", session_id=session_id,
                propagation_direction="both", output_prob_thresh=prob_thresh)):
            out = r["outputs"]
            ids = np.asarray(out["out_obj_ids"]).tolist()
            ms = np.asarray(out["out_binary_masks"])
            per_frame[int(r["frame_index"])] = {
                int(o): ms[k].astype(bool) for k, o in enumerate(ids)}
        return per_frame
    finally:
        predictor.handle_request(request=dict(
            type="close_session", session_id=session_id))


def select_instance(per_frame, old_bin, view_num, n_sample):
    """Median IoU vs old over sampled frames, per masklet. Old = judge only."""
    eval_frames = np.linspace(0, view_num - 1, n_sample, dtype=int).tolist()
    all_ids = sorted({o for d in per_frame.values() for o in d})
    scores = {}
    H, W = old_bin[0].shape
    empty = np.zeros((H, W), bool)
    for o in all_ids:
        vals = [iou(per_frame.get(f, {}).get(o, empty), old_bin[f]) for f in eval_frames]
        scores[o] = round(float(np.median(vals)), 4)
    best = max(scores, key=scores.get) if scores else None
    return best, scores


def stitch_select(per_frame, old_bin, view_num, min_inst_iou, min_cov=0.6):
    """Instance selection for sam3.1 multiplex, whose tracker fragments ONE
    object into temporally-disjoint ids (e.g. obj0 frames 0-83, obj3 61-112,
    obj5 122-201). Each id is judged vs old ONLY on its own frames (old = judge,
    semantics unchanged); passing ids are unioned per frame into a single
    masklet. Returns (best, scores, per_frame) matching select_instance's
    contract, with per_frame collapsed to {frame: {0: mask}} on success."""
    med_own, frames_of = {}, {}
    for o in sorted({o for d in per_frame.values() for o in d}):
        fr = [f for f, d in per_frame.items() if o in d and d[o].sum() > 0]
        if not fr:
            continue
        med_own[o] = float(np.median([iou(per_frame[f][o], old_bin[f]) for f in fr]))
        frames_of[o] = fr
    scores = {f"id{o}": round(v, 4) for o, v in med_own.items()}
    keep = [o for o, v in med_own.items() if v >= min_inst_iou]
    if not keep:
        return None, scores, per_frame
    stitched = {}
    for f, d in per_frame.items():
        ms = [d[o] for o in keep if o in d and d[o].sum() > 0]
        if ms:
            u = ms[0].copy()
            for m in ms[1:]:
                u |= m
            stitched[f] = {0: u}
    if len(stitched) < min_cov * view_num:
        return None, scores, per_frame
    overall = float(np.median([iou(stitched[f][0], old_bin[f]) for f in stitched]))
    scores[0] = round(overall, 4)
    return 0, scores, stitched


def render_instsel(frames_dir, out_dir, frame, per_frame_objs, old_bin_f, chosen,
                   scores, text, tag):
    """One row: [original | white composite per masklet ...]; each tile labelled
    with its median-IoU-vs-old score, the chosen one gets a thick green border."""
    rgb = np.array(Image.open(frames_dir / f"{frame}.jpg").convert("RGB"))
    tiles = [(rgb, "original", False)]
    for o, m in sorted(per_frame_objs.items()):
        a = m[..., None].astype(np.float32)
        comp = (rgb * a + 255.0 * (1 - a)).astype(np.uint8)
        tiles.append((comp, f"obj{o}  medIoU={scores.get(o)}"
                      + ("  [CHOSEN]" if o == chosen else ""), o == chosen))
    row = []
    for t, lab, is_chosen in tiles:
        im = t[:, :, ::-1].copy()
        cv2.rectangle(im, (0, 0), (im.shape[1] - 1, im.shape[0] - 1),
                      (0, 200, 0) if is_chosen else (180, 180, 180),
                      10 if is_chosen else 2)
        cv2.putText(im, lab, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 7)
        cv2.putText(im, lab, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (0, 200, 0) if is_chosen else (255, 255, 255), 2)
        row.extend([im, np.full((im.shape[0], 8, 3), 230, np.uint8)])
    g = np.concatenate(row[:-1], axis=1)
    scale = 420.0 / g.shape[0]
    g = cv2.resize(g, (max(1, int(g.shape[1] * scale)), 420))
    cv2.putText(g, f'text="{text}"', (8, 414), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (60, 60, 60), 1)
    cv2.imwrite(str(out_dir / f"instsel_{tag}_{frame:03d}.jpg"), g,
                [cv2.IMWRITE_JPEG_QUALITY, 88])


def git_commit(repo):
    try:
        return subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def process_scene(predictor, sid, args, run_meta):
    out_dir = Path(args.output_dir) / sid
    stats_path = out_dir / "stats.json"
    if stats_path.exists():
        if not args.overwrite:
            return "skipped"
        stats_path.unlink()
    t0 = time.time()

    prefix, view_num, imgs_bytes, fgms_bytes = load_scene_tar(Path(args.input_dir) / f"{sid}.tar")
    frames_dir = Path(tempfile.mkdtemp(prefix=f"sam3_{sid}_", dir=args.tmp_root))
    try:
        H, W, pad = prepare_frames(imgs_bytes, prefix, view_num, frames_dir, "stretch")
        del imgs_bytes
        old_bin = [s > 127 for s in load_old_masks(fgms_bytes, prefix, view_num, H, W)]
        del fgms_bytes

        attempts = []  # (label, per_frame, best_obj, scores)
        ladder = [(t, args.prob_thresh) for t in scene_phrases(sid, args)]
        if args.ladder_low_thresh > 0:
            # last resort: first category phrase again at a permissive threshold;
            # junk instances are arbitrated away by IoU-vs-old selection + audit
            ladder.append((ladder[0][0], args.ladder_low_thresh))
        text0, per_frame, best, scores = None, {}, None, {}
        for text, pth in ladder:
            label = text if pth == args.prob_thresh else f"{text}@{pth}"
            per_t = run_text_attempt(predictor, frames_dir, text, pth)
            if args.engine == "sam3.1":
                # multiplex fragments identities -> judge per-id segments, stitch
                best_t, scores_t, per_t = stitch_select(
                    per_t, old_bin, view_num, args.min_inst_iou)
            else:
                best_t, scores_t = select_instance(per_t, old_bin, view_num,
                                                   args.inst_sample_frames)
            attempts.append((label, per_t, best_t, scores_t))
            if best_t is not None and (best is None or scores_t[best_t] > scores[best]):
                text0, per_frame, best, scores = label, per_t, best_t, scores_t
            if best is not None and scores[best] >= args.min_inst_iou:
                break  # success: normal scenes stop at the first rung, v1-identical

        fallback = best is None or scores[best] < args.min_inst_iou
        empty = np.zeros((H, W), bool)
        if fallback:
            prompt_mode = "sam3_text+fallback_old"
            masks = {i: old_bin[i].copy() for i in range(view_num)}
        else:
            prompt_mode = f"sam3_text_x{len(scores)}inst"
            masks = {i: postprocess_mask(per_frame.get(i, {}).get(best, empty))
                     for i in range(view_num)}
        per, scene_m = compute_qc(masks, old_bin, exclude=set())

        out_dir.mkdir(parents=True, exist_ok=True)
        for stale in list(out_dir.glob("overlay_*.jpg")) + list(out_dir.glob("instsel_*.jpg")):
            stale.unlink()
        # instance-selection visualization whenever the judge actually had a job
        instsel = bool(len(scores) >= 2 or len(attempts) > 1 or fallback)
        if instsel and per_frame:
            frames_with_most = sorted(per_frame, key=lambda f: -len(per_frame[f]))
            rep = frames_with_most[0]
            render_instsel(frames_dir, out_dir, rep, per_frame[rep], old_bin[rep],
                           best, scores, text0, "rep")
        ev = list(range(view_num))
        by_iou = sorted(ev, key=lambda i: per[i]["iou"])
        selection = {"worst": by_iou[0], "median": by_iou[len(by_iou) // 2],
                     "anchor": by_iou[-1]}
        render_overlays(frames_dir, out_dir, selection, old_bin, masks, pad)
        write_masks(out_dir, masks)

        stats = {
            "complete": True, "engine": args.engine,
            "scene": sid, "prefix": prefix, "view_num": view_num,
            "H": H, "W": W, "square_mode": "stretch",
            "prompt_mode": prompt_mode, "anchor_mode": "sam3_text",
            "text_used": text0, "texts_tried": [a[0] for a in attempts],
            "n_masklets": len(scores), "inst_scores": scores,
            "chosen_obj": best, "anchor_frames": [], "anchors_mask_cond": [],
            "retry": len(attempts) > 1, "fallback": fallback,
            "needs_review": bool(fallback),
            "scene_metrics": scene_m,
            "sam2_attempt_metrics": None, "overlays_from": "final",
            "per_frame": {str(i): per[i] for i in range(view_num)},
            "elapsed_s": round(time.time() - t0, 1),
            **run_meta,
        }
        tmp_stats = stats_path.with_suffix(".json.tmp")
        tmp_stats.write_text(json.dumps(stats))
        tmp_stats.rename(stats_path)
        tag = "FALLBACK" if fallback else ("RETRY" if len(attempts) > 1 else "ok")
        print(f"[{sid}] {tag} text='{text0}' inst={len(scores)} best={best} "
              f"best_iou={None if best is None else scores[best]} "
              f"med={scene_m['median_iou']} resid={scene_m.get('median_embedded_resid')} "
              f"({stats['elapsed_s']}s)", flush=True)
        return "fallback" if fallback else "done"
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

    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if args.engine == "sam3.1":
        # multiplex tracker (improved memory), same session/request API.
        # use_fa3=False: FlashAttention-3 is not installed in this env.
        from sam3.model_builder import build_sam3_multiplex_video_predictor
        predictor = build_sam3_multiplex_video_predictor(use_fa3=False)
        # base start_session unconditionally forwards offload_state_to_cpu,
        # which the multiplex init_state() does not accept
        _orig_init_state = predictor.model.init_state
        def _init_state_compat(*a, **kw):
            kw.pop("offload_state_to_cpu", None)
            return _orig_init_state(*a, **kw)
        predictor.model.init_state = _init_state_compat
    else:
        from sam3.model_builder import build_sam3_video_predictor
        predictor = build_sam3_video_predictor(checkpoint_path=args.ckpt)
    run_meta = {"ckpt": args.ckpt or f"hf:facebook/{args.engine}",
                "sam3_commit": git_commit("/home/z50057756/code/sam3"),
                "host": socket.gethostname(),
                "gpu": torch.cuda.get_device_name(0)}
    print(f"shard {args.shard_idx}/{args.num_shards}: {len(scenes)} scenes on "
          f"{run_meta['host']}/{run_meta['gpu']}", flush=True)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    err_log = Path(args.output_dir) / f"errors_shard{args.shard_idx}.log"
    counts = {"done": 0, "skipped": 0, "fallback": 0, "error": 0}
    for sid in scenes:
        try:
            counts[process_scene(predictor, sid, args, run_meta)] += 1
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
