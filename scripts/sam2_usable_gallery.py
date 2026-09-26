#!/usr/bin/env python
"""Gallery for a scene list against an arbitrary mask dir: per scene render
K evenly-spaced CONSUMED frames as [orig | white-composite] pairs. Used to
eyeball the true usable-rate of an audit's usable list for one config."""
import argparse
import io
import json
import re
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--list", dest="list_file", required=True, help="scene ids, one per line")
    ap.add_argument("--input_dir", default="/mnt/data-alpha-sg-02/team-camera/datasets/yuchen/co3d/webdataset/val")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--title", default="usable gallery")
    ap.add_argument("--n_frames", type=int, default=3)
    ap.add_argument("--n_views", type=int, default=25)
    ap.add_argument("--row_h", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    return ap.parse_args()


def pad_square(arr, fill=255):
    H, W = arr.shape[:2]
    S = max(H, W)
    pt, pl = (S - H) // 2, (S - W) // 2
    canvas = np.full((S, S, 3), fill, np.uint8)
    canvas[pt:pt + H, pl:pl + W] = arr
    return canvas


def make_row(task):
    sid, args_d = task
    args = argparse.Namespace(**args_d)
    try:
        st = json.loads((Path(args.mask_dir) / sid / "stats.json").read_text())
        N = st["view_num"]
        consumed = np.linspace(0, N - 1, args.n_views, dtype=int)
        idxs = [int(consumed[j]) for j in
                np.linspace(0, len(consumed) - 1, args.n_frames, dtype=int)]
        with tarfile.open(Path(args.input_dir) / f"{sid}.tar") as outer:
            meta = next(n for n in outer.getnames() if n.endswith(".meta.json"))
            pfx = Path(meta).name[: -len(".meta.json")]
            imgs = outer.extractfile(next(n for n in outer.getnames()
                                          if n.endswith(f"{pfx}.images.tar"))).read()
        tf = tarfile.open(fileobj=io.BytesIO(imgs))
        pat = re.compile(re.escape(pfx) + r"\.images_(\d+)\.jpg$")
        mm = {int(pat.search(m.name).group(1)): m for m in tf.getmembers() if pat.search(m.name)}
        blocks = []
        for i in idxs:
            rgb = np.array(Image.open(io.BytesIO(tf.extractfile(mm[i]).read())).convert("RGB"))
            m = np.array(Image.open(Path(args.mask_dir) / sid / f"{i}.png")) > 127
            a = m[..., None].astype(np.float32)
            comp = (rgb * a + 255.0 * (1 - a)).astype(np.uint8)
            sep = np.full((rgb.shape[0], 4, 3), 210, np.uint8)
            blocks.append(np.concatenate([pad_square(rgb), np.full((pad_square(rgb).shape[0], 4, 3), 210, np.uint8), pad_square(comp)], axis=1))
        gap = np.full((blocks[0].shape[0], 12, 3), 255, np.uint8)
        row = blocks[0]
        for b in blocks[1:]:
            row = np.concatenate([row, gap, b], axis=1)
        scale = args.row_h / row.shape[0]
        out = Image.fromarray(row).resize((int(row.shape[1] * scale), args.row_h))
        rel = f"rows/{sid}.jpg"
        out.save(Path(args.out_dir) / rel, quality=88)
        med = st.get("scene_metrics", {}).get("median_iou")
        return sid, rel, med, st.get("prompt_mode")
    except Exception as e:
        return sid, f"ERROR: {type(e).__name__}: {e}", None, None


def main():
    args = parse_args()
    out = Path(args.out_dir)
    (out / "rows").mkdir(parents=True, exist_ok=True)
    scenes = [l.strip() for l in Path(args.list_file).read_text().splitlines() if l.strip()]
    args_d = vars(args)
    print(f"rendering {len(scenes)} scenes x {args.n_frames} consumed frames ...", flush=True)
    rows, errs = [], []
    with ProcessPoolExecutor(args.workers) as ex:
        for sid, rel, med, mode in ex.map(make_row, [(s, args_d) for s in scenes], chunksize=4):
            (errs if str(rel).startswith("ERROR") else rows).append((sid, rel, med, mode))
    cards = "".join(
        f'<div class="c"><div class="h">{i+1}. {sid} <span>med IoU {med} · {mode}</span>'
        f'<label style="float:right"><input type="checkbox" class="bad" data-sid="{sid}"> 判坏</label></div>'
        f'<img loading="lazy" src="{rel}"></div>'
        for i, (sid, rel, med, mode) in enumerate(rows))
    store_key = f"badmarks:{out.name}"
    html = f"""<!doctype html><meta charset="utf-8"><title>{args.title}</title>
<style>body{{font-family:sans-serif;margin:14px;background:#fafafa}}
.c{{margin:0 0 18px;border:1px solid #ddd;background:#fff;border-radius:6px;overflow:hidden}}
.h{{padding:5px 9px;font-weight:bold;font-size:13px;background:#f2f2f2}}
.h span{{font-weight:normal;color:#666}} img{{display:block;max-width:100%}}
#tally{{position:fixed;top:8px;right:12px;background:#333;color:#fff;padding:6px 12px;
border-radius:8px;font-size:14px;z-index:9;cursor:pointer}}
#exp{{position:fixed;top:44px;right:12px;width:340px;height:180px;z-index:9;display:none;
font-size:12px;font-family:monospace}}</style>
<div id="tally">判坏 0 / {len(rows)}</div>
<textarea id="exp" readonly></textarea>
<h2>{args.title} — {len(rows)} scenes</h2>
<p>每块 = 一个被消费帧：<b>左原图 | 右抠图</b>，每行 3 帧。发现坏的勾"判坏"，右上角实时计数
→ 可用率 = 1 − 判坏/总数。勾选自动保存在浏览器（刷新不丢）；<b>点右上角计数框</b>弹出/收起
已勾 ID 清单，全选复制即可发我。</p>
{cards}
<script>
const KEY='{store_key}';
const saved=new Set(JSON.parse(localStorage.getItem(KEY)||'[]'));
document.querySelectorAll('.bad').forEach(cb=>{{if(saved.has(cb.dataset.sid))cb.checked=true;}});
function sync(){{
  const ids=[...document.querySelectorAll('.bad:checked')].map(cb=>cb.dataset.sid);
  localStorage.setItem(KEY,JSON.stringify(ids));
  document.getElementById('tally').textContent='判坏 '+ids.length+' / {len(rows)}';
  document.getElementById('exp').value=ids.join('\\n');
}}
document.addEventListener('change',e=>{{if(e.target.classList.contains('bad'))sync();}});
document.getElementById('tally').addEventListener('click',()=>{{
  const t=document.getElementById('exp');
  t.style.display=t.style.display==='block'?'none':'block';t.select();}});
sync();
</script>"""
    (out / "index.html").write_text(html)
    print(f"done: {len(rows)} ok, {len(errs)} errors -> {out}/index.html", flush=True)


if __name__ == "__main__":
    main()
