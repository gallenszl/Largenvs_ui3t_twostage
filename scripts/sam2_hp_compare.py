#!/usr/bin/env python
"""Compare SAM2 mask configs on the hp subset, frame-by-frame, for human judging.

Runs the consumed-frame audit metrics (imported from sam2_eval_frame_audit) on
each config's mask dir over the same scene list, then renders per offending
frame a side-by-side row [orig | <cfg0> | <cfg1> | ...] of white composites so
the user can score configs against each other. Outputs:
    <out_dir>/hp_audit_<name>.jsonl      per-config audit rows
    <out_dir>/hp_compare.html            summary table + per-scene rows
"""
import argparse
import io
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sam2_eval_frame_audit as audit


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hp_split", default="/home/z50057756/data/co3d_sam2_masks/qc/audit/hp200.txt")
    ap.add_argument("--input_dir", default="/mnt/data-alpha-sg-02/team-camera/datasets/yuchen/co3d/webdataset/val")
    ap.add_argument("--out_dir", default="/home/z50057756/data/co3d_sam2_masks/qc/audit")
    ap.add_argument("--configs", nargs="+", required=True,
                    help="name=mask_dir pairs, first is baseline, e.g. "
                         "c0=/home/z50057756/data/co3d_sam2_masks/val")
    ap.add_argument("--n_views", type=int, default=25)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max_frames", type=int, default=4)
    return ap.parse_args()


def audit_args_dict(mask_dir, input_dir, n_views):
    d = dict(mask_dir=str(mask_dir), input_dir=str(input_dir), n_views=n_views,
             frag_cc_frac=0.7, frag_min_cc=3, tbreak_adj=0.4,
             bad_iou=0.35, bad_shrink=0.45,
             sus_spill=0.25, sus_ratio=1.25, sus_resid=0.15)
    return d


def render_row(task):
    sid, frame_ids, cfg_dirs, input_dir, out_dir = task
    try:
        prefix, ftf, itf = audit.load_scene_tars(Path(input_dir) / f"{sid}.tar", True)
        imm = audit._member_map(itf, prefix, "images", "jpg")
        outs = []
        for i in frame_ids:
            dst = Path(out_dir) / "hp_render" / f"{sid}_{i:03d}.jpg"
            if dst.exists():
                outs.append(dst.name)
                continue
            rgb = np.array(Image.open(io.BytesIO(itf.extractfile(imm[i]).read())).convert("RGB"))
            tiles = [rgb]
            for _, mdir in cfg_dirs:
                p = Path(mdir) / sid / f"{i}.png"
                if p.exists():
                    m = (np.array(Image.open(p)) > 127)[..., None].astype(np.float32)
                    tiles.append((rgb * m + 255.0 * (1 - m)).astype(np.uint8))
                else:
                    tiles.append(np.full_like(rgb, 200))
            row = np.concatenate(tiles, axis=1)[:, :, ::-1]
            h, w = row.shape[:2]
            row = cv2.resize(row, (max(1, w // 2), max(1, h // 2)))
            cv2.imwrite(str(dst), row, [cv2.IMWRITE_JPEG_QUALITY, 86])
            outs.append(dst.name)
        ftf.close()
        itf.close()
        return sid, outs
    except Exception as e:
        return sid, f"RENDER_ERROR: {type(e).__name__}: {e}"


def main():
    args = parse_args()
    out = Path(args.out_dir)
    (out / "hp_render").mkdir(parents=True, exist_ok=True)
    cfgs = [tuple(c.split("=", 1)) for c in args.configs]
    scenes = [l.strip() for l in Path(args.hp_split).read_text().splitlines() if l.strip()]
    print(f"{len(scenes)} scenes x {len(cfgs)} configs", flush=True)

    results = {}  # name -> {scene: row}
    for name, mdir in cfgs:
        ad = audit_args_dict(mdir, args.input_dir, args.n_views)
        rows = {}
        with ProcessPoolExecutor(args.workers) as ex:
            for r in ex.map(audit.audit_scene, [(s, ad) for s in scenes], chunksize=8):
                rows[r["scene"]] = r
        n_err = sum(1 for r in rows.values() if "error" in r)
        results[name] = rows
        with open(out / f"hp_audit_{name}.jsonl", "w") as f:
            for s in scenes:
                f.write(json.dumps(rows[s]) + "\n")
        print(f"  [{name}] audited ({n_err} errors)", flush=True)

    # ---------------- summary table
    summ = []
    for name, _ in cfgs:
        rows = results[name]
        ok_rows = [r for r in rows.values() if "error" not in r]
        n_unus = sum(1 for r in ok_rows if r["auto_bad"])
        n_badf = sum(len(r["auto_bad"]) for r in ok_rows)
        reasons = Counter(x for r in ok_rows for v in r["auto_bad"].values() for x in v)
        n_sus = sum(1 for r in ok_rows if r["suspects"] and not r["auto_bad"])
        summ.append((name, n_unus, n_badf, dict(reasons), n_sus,
                     round(float(np.mean([r["sus_score"] for r in ok_rows])), 3)))
        print(f"[{name}] unusable={n_unus}/{len(ok_rows)} bad_frames={n_badf} "
              f"reasons={dict(reasons)} suspect_only={n_sus}", flush=True)

    # ---------------- render union of offending frames
    tasks = []
    frame_sel = {}
    for sid in scenes:
        union = set()
        for name, _ in cfgs:
            r = results[name].get(sid, {})
            union |= {int(i) for i in r.get("auto_bad", {})}
            union |= {int(i) for i in r.get("suspects", {})}
        if union:
            frame_sel[sid] = sorted(union)[:args.max_frames]
            tasks.append((sid, frame_sel[sid], cfgs, args.input_dir, str(out)))
    print(f"rendering {len(tasks)} scenes ...", flush=True)
    rendered = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for k, (sid, names) in enumerate(ex.map(render_row, tasks, chunksize=4), 1):
            rendered[sid] = names
            if k % 50 == 0:
                print(f"  {k}/{len(tasks)}", flush=True)

    # ---------------- html
    css = """<style>body{font-family:sans-serif;margin:14px;background:#fafafa}
table.sum{border-collapse:collapse;margin-bottom:14px}
table.sum td,table.sum th{border:1px solid #ccc;padding:4px 10px;font-size:13px}
.sc{margin:0 0 16px;border:1px solid #ddd;background:#fff;border-radius:6px;overflow:hidden}
.hd{padding:6px 10px;font-weight:bold;font-size:13px;background:#f1f1f1}
.meta{font-weight:normal;color:#666}.fr{padding:6px}
.fr img{max-width:100%;display:block}.cap{font-size:12px;color:#555;margin:2px 0}
.chip{font-size:11px;padding:1px 7px;border-radius:9px;margin-left:4px}
.chip.bad{background:#ffd9d9;color:#a00}.chip.sus{background:#fff3c9;color:#875f00}
.chip.okc{background:#dcf5dc;color:#1a7a1a}</style>"""
    names = [n for n, _ in cfgs]
    head = "".join(f"<th>{n}</th>" for n in names)
    body = ""
    for name, n_unus, n_badf, reasons, n_sus, ms in summ:
        body += (f"<tr><td><b>{name}</b></td><td>{n_unus}</td><td>{n_badf}</td>"
                 f"<td>{reasons}</td><td>{n_sus}</td><td>{ms}</td></tr>")
    html = [f'<!doctype html><meta charset="utf-8"><title>SAM2 hp compare</title>{css}',
            f'<h1>超参对比 — {len(scenes)} scenes × {len(names)} configs</h1>',
            '<p>每帧一行大图：<b>[原图 | ' + " | ".join(names) + ']</b>（白底抠图）。'
            'chips 按配置列出该帧判决：红=AUTO-BAD，黄=SUSPECT，绿=clean。</p>',
            '<table class="sum"><tr><th>config</th><th>unusable scenes</th><th>bad frames</th>'
            '<th>reasons</th><th>suspect-only</th><th>mean sus_score</th></tr>'
            f'{body}</table>']
    for sid in scenes:
        if sid not in frame_sel:
            continue
        blocks = ""
        for i, fname in zip(frame_sel[sid], rendered.get(sid, [])):
            if str(fname).startswith("RENDER_ERROR"):
                continue
            chips = ""
            for name, _ in cfgs:
                r = results[name].get(sid, {})
                if str(i) in r.get("auto_bad", {}):
                    chips += f'<span class="chip bad">{name}:{"+".join(r["auto_bad"][str(i)])}</span>'
                elif str(i) in r.get("suspects", {}):
                    chips += f'<span class="chip sus">{name}:{"+".join(r["suspects"][str(i)])}</span>'
                else:
                    chips += f'<span class="chip okc">{name}:ok</span>'
            blocks += (f'<div class="fr"><div class="cap">frame {i} {chips}</div>'
                       f'<img loading="lazy" src="hp_render/{fname}"></div>')
        html.append(f'<div class="sc"><div class="hd">{sid}</div>{blocks}</div>')
    (out / "hp_compare.html").write_text("\n".join(html))
    print(f"wrote {out}/hp_compare.html", flush=True)


if __name__ == "__main__":
    main()
