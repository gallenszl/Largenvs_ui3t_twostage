"""Render the hole-scan hits: [orig | cutout(hole visible) ] per hit frame."""
import io
import re
import sys
import tarfile
from pathlib import Path
import numpy as np
import cv2

LOG = "/home/z50057756/code/RnG_feature_allignment/slurm_logs/hole_scan-85833.out"
OUT = Path("/home/z50057756/data/co3d_sam2_masks/qc/hole_scan_gallery")
(OUT / "rows").mkdir(parents=True, exist_ok=True)
VAL = Path("/mnt/data-alpha-sg-02/team-camera/datasets/yuchen/co3d/webdataset/val")
TRAIN = Path("/mnt/data-alpha-sg-02/team-camera/datasets/yuchen/co3d/webdataset/train")

hits = []
for line in open(LOG):
    m = re.match(r"HIT (\S+): \[(.*)\]", line.strip())
    if not m: continue
    sid = m.group(1)
    frames = sorted({int(x) for x in re.findall(r"\((\d+),", m.group(2))})[:3]
    hits.append((sid, frames))

def frame_jpg(sid, idx):
    tar = (VAL if "teddybear_val" in sid else TRAIN) / f"{sid}.tar"
    with tarfile.open(tar) as outer:
        inner = next(n for n in outer.getnames() if n.endswith("images.tar"))
        with tarfile.open(fileobj=io.BytesIO(outer.extractfile(inner).read())) as it:
            mem = next(m for m in it.getmembers() if m.name.endswith(f"images_{idx}.jpg"))
            buf = np.frombuffer(it.extractfile(mem).read(), np.uint8)
            return cv2.imdecode(buf, cv2.IMREAD_COLOR)

cards = []
for sid, frames in hits:
    if "teddybear_val" in sid:
        mroot = Path("/home/z50057756/data/co3d_sam3_masks/val")
    else:
        v2 = Path("/home/z50057756/data/co3d_sam3_masks_hp/cat50_v2fb") / sid
        mroot = v2.parent if v2.exists() else Path("/home/z50057756/data/co3d_sam3_masks_hp/cat50")
    panels = []
    for f in frames:
        img = frame_jpg(sid, f)
        mask = cv2.imread(str(mroot / sid / f"{f}.png"), cv2.IMREAD_GRAYSCALE) > 127
        cut = np.full_like(img, 255)
        cut[mask] = img[mask]
        pair = np.hstack([img, cut])
        h = 360
        pair = cv2.resize(pair, (int(pair.shape[1] * h / pair.shape[0]), h))
        panels.append(pair)
    row = np.hstack(panels)
    name = f"{sid}.jpg"
    cv2.imwrite(str(OUT / "rows" / name), row, [cv2.IMWRITE_JPEG_QUALITY, 88])
    cards.append(f'<div class="c"><div class="h">{sid} <span>hit frames {frames}</span>'
                 f'<label style="float:right"><input type="checkbox" class="bad" data-sid="{sid}"> 判坏</label></div>'
                 f'<img loading="lazy" src="rows/{name}"></div>')
    print("row", sid, flush=True)

html = f"""<!doctype html><meta charset="utf-8"><title>hole scan</title>
<style>body{{font-family:sans-serif;margin:14px}}.c{{margin:0 0 16px;border:1px solid #ddd;border-radius:6px;overflow:hidden}}
.h{{padding:5px 9px;font-weight:bold;font-size:13px;background:#f2f2f2}}.h span{{font-weight:normal;color:#666}}
img{{display:block;max-width:100%}}#tally{{position:fixed;top:8px;right:12px;background:#333;color:#fff;
padding:6px 12px;border-radius:8px;z-index:9;cursor:pointer}}
#exp{{position:fixed;top:44px;right:12px;width:320px;height:160px;display:none;z-index:9;font-family:monospace;font-size:12px}}</style>
<div id="tally">判坏 0 / {len(cards)}</div><textarea id="exp" readonly></textarea>
<h2>usable 里的封闭洞命中 ({len(cards)} 场景) — 每块=[原图|白底抠图]，洞=抠图内的白窟窿</h2>
<p>判别：洞里是<b>装饰物</b>（花/蝴蝶结被挖掉）=坏；洞是<b>透过缝隙看到的背景</b>（椅档/车架/提手）=好（旧mask把缝隙填了才被扫中）。勾选自动保存，点右上角计数框导出。</p>
{"".join(cards)}
<script>const KEY='badmarks:hole_scan';
const saved=new Set(JSON.parse(localStorage.getItem(KEY)||'[]'));
document.querySelectorAll('.bad').forEach(cb=>{{if(saved.has(cb.dataset.sid))cb.checked=true;}});
function sync(){{const ids=[...document.querySelectorAll('.bad:checked')].map(cb=>cb.dataset.sid);
localStorage.setItem(KEY,JSON.stringify(ids));
document.getElementById('tally').textContent='判坏 '+ids.length+' / {len(cards)}';
document.getElementById('exp').value=ids.join('\\n');}}
document.addEventListener('change',e=>{{if(e.target.classList.contains('bad'))sync();}});
document.getElementById('tally').addEventListener('click',()=>{{const t=document.getElementById('exp');
t.style.display=t.style.display==='block'?'none':'block';t.select();}});sync();</script>"""
(OUT / "index.html").write_text(html)
print(f"DONE {len(cards)} rows -> {OUT}/index.html")
