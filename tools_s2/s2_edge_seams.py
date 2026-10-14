# Seams along the refined / unrefined token boundary (user 09-28, fg_dense 14k, GSO 000018 Ecoforms_Plant_Container_QP6CORAL
# posed view 0: "looking at this picture the result got worse, the edges are all problems").
# Stage 2 re-renders exactly the target tokens whose 8x8 patch holds >= 1 GT-mask pixel (model_s2/geometry.build_layout:
# max-pool of alpha > 0.5, no dilation); every other token keeps its stage-1 features (only the shared colour head is
# retrained).  Where stage 1 painted the object beyond that token cover, the stage-1 colour survives in whole 8x8 blocks
# next to refined tokens that now paint white -> a staircase along the token grid.  Per target view, from the validation
# PNGs of a stage-2 run (step 0 = the frozen stage 1, G9-checked) and the loader's GT mask:
#   R / O      refined tokens (the rule above) / the other tokens
#   painted    a pixel is painted if some colour channel is more than TAU below white (1 - min_c x_c > TAU)
#   ghost      painted prediction pixels inside O (object colour in tokens stage 2 never re-renders; GT is white there)
#   seam       a token edge where the image breaks at the tile border: with J_b = the colour jump across the border
#              (8 pixel pairs, mean over pairs and RGB of |x(p) - x(q)|) and J_in = the larger of the two jumps one
#              pixel inside either tile (same average), seam if  J_b(pred) - J_in(pred) > TAU  (the border jump stands
#              out from the local gradient: a soft edge that merely crosses the border is not a seam)  and
#              J_b(pred) - J_b(GT) > TAU  (the GT does not jump there).  Counted per edge class: R-O (refined next to
#              unrefined), R-R, O-O; the R-O count at step 0 (stage 1 does not know R) and the R-R / O-O counts are
#              controls.  (A first version used J_b(pred) - J_b(GT) > TAU alone; it counted the steep part of stage
#              1's soft edge -- about 0.16 per pixel for a dark object -- as a seam wherever it crossed a tile border.)
#   paired     per view, R-O seams at a step minus at step 0; split by the stage-1 overreach = stage-1 painted pixels
#              in O (>= OVER px: stage 1 drew the object at least OVER/64 tiles' worth outside the refined tiles)
#   err_O      share of the view's squared error that sits in O
# Proposal B (stage 2 without the GT target mask) needs a token cover built from stage 1 alone; for the tokens holding a
# painted stage-1 pixel, dilated by k = 0, 1, 2 tokens (3x3): share of GT-mask pixels left outside (never re-rendered),
# views with any GT pixel outside, token count vs the GT cover R.
# --ovl_dir: also write o<uid>_ovl.png per entry (RGBA, one row of the Vt target views): inner boundary of R in blue,
# GT contour in green (read by tools_s2/s2_web_assets.py --ovl_dir).
#   python tools_s2/s2_edge_seams.py --config <cfg> --exp_dir <ckpt dir> --out x.json --panel_dir <dir>
#                                   [--panel_steps 0,14000,26000] [--focus posed:000018:0] [--panels 5] [--ovl_dir <dir>]
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
MODES = ("posed", "unposed")
P = 8
TAU = 0.1
OVER = 256
KS = (0, 1, 2)
BLUE = np.array([30, 144, 255], dtype=np.uint8)
GREEN = np.array([0, 200, 83], dtype=np.uint8)


def halves(path):
    a = np.asarray(Image.open(path).convert("RGB")).astype(np.float64) / 255.0
    h = a.shape[0] // 2
    return a[:h], a[h:]


def token_cover(px):
    """px [H, W] bool -> tokens [g, g] holding >= 1 True pixel (= build_layout's max-pool)"""
    g = px.shape[0] // P
    return px.reshape(g, P, g, P).any(axis=(1, 3))


def up(tok):
    return np.repeat(np.repeat(tok, P, 0), P, 1)


def edge_jumps(img):
    """img [H, W, 3] -> per token edge (horizontal neighbours [g, g-1], vertical neighbours [g-1, g]): (J_b, J_in) =
    the mean colour jump across the tile border and the larger of the two mean jumps one pixel inside either tile"""
    g = img.shape[0] // P
    c = np.arange(1, g) * P - 1                                   # last pixel of the left / upper tile

    def jh(i, j):                                                 # |x[:, i] - x[:, j]|, mean over RGB and the 8 rows
        return np.abs(img[:, i] - img[:, j]).mean(-1).reshape(g, P, g - 1).mean(1)

    def jv(i, j):
        return np.abs(img[i] - img[j]).mean(-1).reshape(g - 1, g, P).mean(2)
    return ((jh(c, c + 1), np.maximum(jh(c - 1, c), jh(c + 1, c + 2))),
            (jv(c, c + 1), np.maximum(jv(c - 1, c), jv(c + 1, c + 2))))


def edge_classes(tok):
    """tok [g, g] bool (R) -> {R-O, R-R, O-O} masks over the horizontal and vertical token edges"""
    out = {}
    for key, (a, b) in (("h", (tok[:, :-1], tok[:, 1:])), ("v", (tok[:-1], tok[1:]))):
        out[key] = {"RO": a != b, "RR": a & b, "OO": ~a & ~b}
    return out


def painted(img):
    return (1.0 - img.min(-1)) > TAU


def view_stats(gt, pred, m, gt_jumps=None, cls=None):
    """one view: gt / pred [H, W, 3] in [0, 1], m [H, W] GT mask -> seams per edge class, ghost pixels, err_O share"""
    tok = token_cover(m)
    R = up(tok)
    if cls is None:
        cls = edge_classes(tok)
    if gt_jumps is None:
        gt_jumps = edge_jumps(gt)
    pj = edge_jumps(pred)
    sm = {k: ((pj[i][0] - pj[i][1]) > TAU) & ((pj[i][0] - gt_jumps[i][0]) > TAU) for i, k in enumerate(("h", "v"))}
    seams = {c: int(sum(int((sm[k] & cls[k][c]).sum()) for k in ("h", "v"))) for c in ("RO", "RR", "OO")}
    se = ((pred - gt) ** 2).mean(-1)
    return dict(seams=seams, ghost=int((painted(pred) & ~R).sum()), se_O=float(se[~R].sum()), se=float(se.sum()))


def overlay_rgba(m):
    """m [H, W] GT mask -> RGBA overlay: inner boundary of the refined-token cover R (blue), GT contour (green)"""
    R = up(token_cover(m))
    out = np.zeros(m.shape + (4,), dtype=np.uint8)
    for sel, col in ((R, BLUE), (m, GREEN)):
        p = np.pad(sel, 1, mode="edge")
        inner = sel & ~(p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:])
        out[inner, :3] = col
        out[inner, 3] = 255
    return out


def load_masks(config, exp):
    """GT target masks per validation entry, exactly as the loader builds them: {uid: [Vt, H, W] bool}.  Alignment
    check against what the evaluation wrote: same object name as <uid>/metrics.json and the loader's target images equal
    the GT half of <uid>/gt_vs_pred.png (step 0, both modes) up to PNG quantisation."""
    cwd = os.getcwd()
    os.chdir(REPO)
    from data.dataset_gso_ours import GSODataset_ours
    from tools_s2.s2_bench import load_cfg
    cfg = load_cfg(config, 1)
    ds = GSODataset_ours(cfg)
    os.chdir(cwd)
    nin = int(cfg.training.val_dataset_cfgs.training.num_input_views)
    masks, worst = {}, 0.0
    for i in range(len(ds)):
        uid = f"{i:06d}"
        it = ds[i]
        masks[uid] = it["alpha_mask"][nin:, 0].numpy() > 0.5
        img = it["image"][nin:].permute(2, 0, 3, 1).numpy()                   # [H, Vt, W, 3]
        img = img.reshape(img.shape[0], -1, 3)
        for mode in MODES:
            d = f"{exp}/eval_iter_{0:08d}_{mode}/{uid}"
            name = json.load(open(f"{d}/metrics.json"))["summary"]["scene_name"]
            assert name == it["scene_name"], (uid, mode, name, it["scene_name"])
            gt = halves(f"{d}/gt_vs_pred.png")[0]
            worst = max(worst, float(np.abs(gt - img).max()))
    print(f"[seam] loader vs written GT: names match for {len(masks)} entries x {len(MODES)} modes, "
          f"max |loader image - PNG GT| = {worst * 255:.2f} grey levels", flush=True)
    assert worst <= 2.01 / 255, worst                                       # misaligned views differ by 100s of levels
    return masks


def panel(rows_imgs, masks_v, labels, scale=2):
    """a 2-row panel: row 1 the images, row 2 the same with the overlay; labels above the columns"""
    ovl = overlay_rgba(masks_v)
    tiles = []
    for img in rows_imgs:
        a = (img * 255).round().astype(np.uint8)
        b = a.copy()
        sel = ovl[..., 3] > 0
        b[sel] = ovl[sel, :3]
        tiles.append((a, b))
    H = rows_imgs[0].shape[0] * scale
    gap, head = 6, 18
    W = len(tiles) * (H + gap) - gap
    canvas = Image.new("RGB", (W, head + 2 * H + gap), (128, 128, 128))
    d = ImageDraw.Draw(canvas)
    for j, ((a, b), lab) in enumerate(zip(tiles, labels)):
        x = j * (H + gap)
        d.text((x + 4, 3), lab, fill=(255, 255, 255))
        canvas.paste(Image.fromarray(a).resize((H, H), Image.NEAREST), (x, head))
        canvas.paste(Image.fromarray(b).resize((H, H), Image.NEAREST), (x, head + H + gap))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--panel_dir", required=True)
    ap.add_argument("--panel_steps", default="0,14000,26000")
    ap.add_argument("--focus", default="posed:000018:0")
    ap.add_argument("--panels", type=int, default=5)
    ap.add_argument("--ovl_dir", default=None)
    args = ap.parse_args()
    exp = os.path.abspath(args.exp_dir)
    os.makedirs(args.panel_dir, exist_ok=True)
    steps = sorted({int(re.search(r"eval_iter_(\d+)_", p).group(1)) for p in glob.glob(f"{exp}/eval_iter_*_posed")
                    if os.path.isdir(p.replace("_posed", "_unposed"))})
    steps = [s for s in steps if s % 2000 == 0]
    pst = [int(s) for s in args.panel_steps.split(",")]
    assert all(s in steps for s in pst), (pst, steps)
    masks = load_masks(args.config, exp)
    if args.ovl_dir:
        os.makedirs(args.ovl_dir, exist_ok=True)
        for uid, M in masks.items():
            Image.fromarray(np.concatenate([overlay_rgba(m) for m in M], axis=1), "RGBA").save(
                f"{args.ovl_dir}/o{uid}_ovl.png", optimize=True)
        print(f"[seam] overlays -> {args.ovl_dir}", flush=True)
    print(f"[seam] steps {steps}, entries {len(masks)}, TAU {TAU}", flush=True)
    recs = []
    for mode in MODES:
        for uid, M in sorted(masks.items()):
            d0 = f"{exp}/eval_iter_{0:08d}_{mode}/{uid}"
            if not os.path.isdir(d0):
                continue
            name = json.load(open(f"{d0}/metrics.json"))["summary"]["scene_name"]
            gt, pr0 = halves(f"{d0}/gt_vs_pred.png")
            preds = {0: pr0}
            for s in steps[1:]:
                preds[s] = halves(f"{exp}/eval_iter_{s:08d}_{mode}/{uid}/gt_vs_pred.png")[1]
            H = gt.shape[0]
            for v in range(gt.shape[1] // H):
                sl = slice(v * H, (v + 1) * H)
                m, g = M[v], gt[:, sl]
                tok = token_cover(m)
                cls = edge_classes(tok)
                gj = edge_jumps(g)
                r = dict(mode=mode, uid=uid, name=name, view=v, gt_px=int(m.sum()), R_tokens=int(tok.sum()),
                         edges={c: int(cls["h"][c].sum() + cls["v"][c].sum()) for c in ("RO", "RR", "OO")},
                         gt_ghost=int((painted(g) & ~up(tok)).sum()),
                         steps={str(s): view_stats(g, preds[s][:, sl], m, gj, cls) for s in steps})
                p0 = token_cover(painted(preds[0][:, sl]))
                cov = {}
                for k in KS:
                    pk = ndimage.binary_dilation(p0, structure=np.ones((3, 3), bool), iterations=k) if k else p0
                    cov[str(k)] = dict(tokens=int(pk.sum()), miss_px=int((m & ~up(pk)).sum()),
                                       miss_tokens=int((tok & ~pk).sum()))
                r["B_cover"] = cov
                recs.append(r)
        print(f"[seam] {mode}: {sum(1 for r in recs if r['mode'] == mode)} views", flush=True)
    # ---- aggregates
    summ = {}
    for mode in MODES:
        rs = [r for r in recs if r["mode"] == mode]
        nv = len(rs)
        S = {}
        for s in steps:
            k = str(s)
            ro = np.array([r["steps"][k]["seams"]["RO"] for r in rs])
            gh = np.array([r["steps"][k]["ghost"] for r in rs])
            S[k] = dict(
                seam_rate={c: sum(r["steps"][k]["seams"][c] for r in rs) / max(1, sum(r["edges"][c] for r in rs))
                           for c in ("RO", "RR", "OO")},
                seams_RO_per_view=float(ro.mean()), views_seams_RO_ge1=float((ro >= 1).mean()),
                views_seams_RO_ge5=float((ro >= 5).mean()), seams_RO_max=int(ro.max()),
                ghost_px_per_view=float(gh.mean()), views_ghost_ge64=float((gh >= 64).mean()),
                err_O_share=sum(r["steps"][k]["se_O"] for r in rs) / sum(r["steps"][k]["se"] for r in rs),
                mse=sum(r["steps"][k]["se"] for r in rs) / (nv * 256 * 256),
                mse_O=sum(r["steps"][k]["se_O"] for r in rs) / (nv * 256 * 256))
        g0 = np.array([r["steps"]["0"]["ghost"] for r in rs])
        over = g0 >= OVER
        PR = {}
        for s in steps[1:]:
            d = np.array([r["steps"][str(s)]["seams"]["RO"] - r["steps"]["0"]["seams"]["RO"] for r in rs])
            gs = np.array([r["steps"][str(s)]["ghost"] for r in rs])
            PR[str(s)] = dict(mean_delta=float(d.mean()), more3=float((d >= 3).mean()), fewer3=float((d <= -3).mean()),
                              over=dict(share=float(over.mean()), mean_delta=float(d[over].mean()) if over.any() else None,
                                        more3=float((d[over] >= 3).mean()) if over.any() else None,
                                        ghost_step0=float(g0[over].mean()) if over.any() else None,
                                        ghost_step=float(gs[over].mean()) if over.any() else None),
                              rest=dict(share=float((~over).mean()), mean_delta=float(d[~over].mean()),
                                        more3=float((d[~over] >= 3).mean()), ghost_step0=float(g0[~over].mean()),
                                        ghost_step=float(gs[~over].mean())))
        gt_px = sum(r["gt_px"] for r in rs)
        B = {str(k): dict(miss_px_share=sum(r["B_cover"][str(k)]["miss_px"] for r in rs) / gt_px,
                          views_any_miss=float(np.mean([r["B_cover"][str(k)]["miss_px"] > 0 for r in rs])),
                          views_miss_ge64=float(np.mean([r["B_cover"][str(k)]["miss_px"] >= 64 for r in rs])),
                          tokens_per_view=float(np.mean([r["B_cover"][str(k)]["tokens"] for r in rs])))
             for k in KS}
        summ[mode] = dict(views=nv, R_tokens_per_view=float(np.mean([r["R_tokens"] for r in rs])),
                          edges_per_view={c: float(np.mean([r["edges"][c] for r in rs])) for c in ("RO", "RR", "OO")},
                          gt_ghost_px_per_view=float(np.mean([r["gt_ghost"] for r in rs])), steps=S, paired=PR,
                          B_cover=B)
        print(f"[seam] ===== {mode}: {nv} views; refined tokens (GT cover R) per view {summ[mode]['R_tokens_per_view']:.1f} "
              f"of 1024; token edges per view R-O {summ[mode]['edges_per_view']['RO']:.1f} / R-R "
              f"{summ[mode]['edges_per_view']['RR']:.1f} / O-O {summ[mode]['edges_per_view']['OO']:.1f}; GT painted px in O "
              f"per view {summ[mode]['gt_ghost_px_per_view']:.2f}", flush=True)
        print("[seam]   step   seam rate R-O / R-R / O-O   R-O seams/view  views>=1  views>=5  max | ghost px/view  "
              "views>=64 | err share in O  MSE  MSE_O", flush=True)
        for s in steps:
            x = S[str(s)]
            print(f"[seam]   {s:6d}   {x['seam_rate']['RO']:.4f} / {x['seam_rate']['RR']:.4f} / {x['seam_rate']['OO']:.4f}"
                  f"   {x['seams_RO_per_view']:8.2f}      {100 * x['views_seams_RO_ge1']:5.1f}%   "
                  f"{100 * x['views_seams_RO_ge5']:5.1f}%  {x['seams_RO_max']:3d} | {x['ghost_px_per_view']:8.1f}   "
                  f"{100 * x['views_ghost_ge64']:5.1f}%  | {100 * x['err_O_share']:5.1f}%   {x['mse']:.5f}  {x['mse_O']:.5f}",
                  flush=True)
        print(f"[seam]   paired per view (R-O seams at step - at step 0); stage-1 overreach = stage-1 painted px in O >= "
              f"{OVER}: {100 * over.mean():.1f}% of views", flush=True)
        print("[seam]   step   mean delta  >= +3 / <= -3 | overreach views: mean delta  >= +3  ghost step0 -> step | "
              "other views: mean delta  >= +3  ghost step0 -> step", flush=True)
        for s in steps[1:]:
            x = PR[str(s)]
            o, t = x["over"], x["rest"]
            print(f"[seam]   {s:6d}   {x['mean_delta']:+7.2f}   {100 * x['more3']:5.1f}% / {100 * x['fewer3']:5.1f}% | "
                  + (f"{o['mean_delta']:+7.2f}  {100 * o['more3']:5.1f}%  {o['ghost_step0']:7.1f} -> {o['ghost_step']:7.1f}"
                     if o["mean_delta"] is not None else "n/a")
                  + f" | {t['mean_delta']:+7.2f}  {100 * t['more3']:5.1f}%  {t['ghost_step0']:6.1f} -> {t['ghost_step']:6.1f}",
                  flush=True)
        print("[seam]   proposal-B cover (stage-1 painted tokens dilated by k): k | GT px left outside | views with any / "
              ">= 64 GT px outside | tokens per view (GT cover R: "
              f"{summ[mode]['R_tokens_per_view']:.1f})", flush=True)
        for k in KS:
            b = B[str(k)]
            print(f"[seam]   k={k}  {100 * b['miss_px_share']:6.2f}%  {100 * b['views_any_miss']:5.1f}% / "
                  f"{100 * b['views_miss_ge64']:5.1f}%  {b['tokens_per_view']:7.1f}", flush=True)
    # ---- panels: the focus view, the views with the most R-O seams at the last panel step, and views at the median
    last = str(pst[-1])
    fm, fu, fv = args.focus.split(":")
    pick = [r for r in recs if r["mode"] == fm and r["uid"] == fu and r["view"] == int(fv)]
    order = sorted(recs, key=lambda r: -r["steps"][last]["seams"]["RO"])
    names = {r["name"] for r in pick}
    for r in order:
        if len(pick) >= 1 + args.panels:
            break
        if r["name"] not in names:
            pick.append(r)
            names.add(r["name"])
    med = sorted(recs, key=lambda r: r["steps"][last]["seams"]["RO"])
    mid = len(med) // 2
    for r in med[mid:] + med[:mid][::-1]:
        if len(pick) >= 4 + args.panels:
            break
        if r["name"] not in names:
            pick.append(r)
            names.add(r["name"])
    for j, r in enumerate(pick):
        imgs, labs = [], []
        for s in pst:
            gt, pr = halves(f"{exp}/eval_iter_{s:08d}_{r['mode']}/{r['uid']}/gt_vs_pred.png")
            H = gt.shape[0]
            sl = slice(r["view"] * H, (r["view"] + 1) * H)
            if not imgs:
                imgs.append(gt[:, sl])
                labs.append("GT")
            imgs.append(pr[:, sl])
            x = r["steps"][str(s)]
            labs.append(f"step {s}: R-O seams {x['seams']['RO']}, ghost {x['ghost']} px")
        fn = f"{args.panel_dir}/p{j:02d}_{r['mode']}_{r['uid']}_v{r['view']}.png"
        panel(imgs, masks[r["uid"]][r["view"]], labs).save(fn)
        print(f"[seam] panel {fn}: {r['name']} R tokens {r['R_tokens']}, R-O edges {r['edges']['RO']}, "
              + ", ".join(f"step {s} seams {r['steps'][str(s)]['seams']} ghost {r['steps'][str(s)]['ghost']}" for s in pst),
              flush=True)
    json.dump(dict(tau=TAU, steps=steps, summary=summ, views=recs), open(args.out, "w"))
    print(f"[seam] DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
