# What do the GT silhouettes look like in each training / evaluation set?  (user 09-28: "is the GT wrong, what is the
# distribution at the GT contour", for the RnG_MOE and the RnG training data)
# Sources (random objects, random views, raw 512 px RGBA):
#   all287k      RnG_MOE / this repo's training data (Objaverse, per-object tars, webp)
#   fluffy       RnG's training data (FluffyElephant, per-object tars, png)
#   obj40v       RnG's second training set of the mix arms (objaverse_rendering_set, png)
#   gso_v1       RnG_MOE's evaluation set (gso_sim2real_25v, loose png)
#   gso_rv       RnG's evaluation set (gso_render_rv, loose png)
# Per image:
#   alpha edge width at 512 px: partially transparent pixels (5 < a < 250) per silhouette pixel
#   compositing check: colour / nearby opaque colour against alpha at partial pixels (straight ~ slope 0 / intercept 1,
#     premultiplied ~ slope 1 / intercept 0)
#   the GT the model sees (the loader exactly: white composite rgb*a + 255*(1-a), LANCZOS to 256, alpha NEAREST > 127):
#     normalised luminance t(d) = (L(d) - L_in) / (L_bg - L_in) per signed distance d to the hard mask (px at 256,
#     < 0 inside), L_in = mean luminance at d in [-5, -3], L_bg = 1 (white); views with 1 - L_in < 0.2 skipped;
#     10 %-90 % transition width; mean luminance of the far background (d > 8)
# Plus the stage-1 prediction on gso_v1 (step 0 of a stage-2 run's validation) with the same t(d), for comparison.
#   python tools_s2/s2_gt_edge_stats.py --exp_dir <stage-2 ckpt dir> --config <cfg> --out x.json
import argparse
import glob
import io
import json
import os
import random
import sys
import tarfile

import numpy as np
from PIL import Image
from scipy import ndimage

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
D = np.arange(-4, 9)


def lum(x):
    return x @ np.array([0.299, 0.587, 0.114])


def comp_stats(rgba):
    a = rgba[..., 3].astype(np.float64) / 255.0
    rgb = rgba[..., :3].astype(np.float64) / 255.0
    op = (a > 0.999).astype(np.float64)
    k = np.ones((7, 7))
    ref = np.stack([ndimage.convolve(rgb[..., c] * op, k, mode="nearest") for c in range(3)], -1)
    den = ndimage.convolve(op, k, mode="nearest")[..., None]
    ref = ref / np.maximum(den, 1e-9)
    sel = (a > 0.05) & (a < 0.95) & (den[..., 0] > 0) & (lum(ref) > 0.15)
    if sel.sum() < 20:
        return None
    r = lum(rgb[sel]) / np.maximum(lum(ref[sel]), 1e-6)
    slope, icpt = np.polyfit(a[sel], r, 1)
    return float(slope), float(icpt)


def profile_from(img256, mask256):
    """img [H, W, 3] in [0, 1], mask bool -> (t(d) sums, counts) or None"""
    if not mask256.any() or mask256.all():
        return None
    sd = np.where(mask256, -ndimage.distance_transform_edt(mask256), ndimage.distance_transform_edt(~mask256))
    L = lum(img256)
    ring = (sd >= -5) & (sd <= -3)
    if ring.sum() < 20:
        return None
    L_in = L[ring].mean()
    if 1.0 - L_in < 0.2:
        return None
    t = (L - L_in) / (1.0 - L_in)
    r = np.rint(sd).astype(int)
    s = np.array([t[r == d].sum() for d in D])
    n = np.array([(r == d).sum() for d in D])
    far = L[sd > 8].mean() if (sd > 8).any() else np.nan
    return s, n, far


def analyse(rgba):
    a = rgba[..., 3]
    hard = a > 127
    if not hard.any():
        return None
    boundary = hard & ~ndimage.binary_erosion(hard)
    partial = (a > 5) & (a < 250)
    cs = comp_stats(rgba)
    af = (a.astype(np.float32) / 255.0)[..., None]
    rgb = (rgba[..., :3].astype(np.float32) * af + 255.0 * (1.0 - af)).astype(np.uint8)
    H = 256
    img = np.asarray(Image.fromarray(rgb).resize((H, H), Image.LANCZOS)).astype(np.float64) / 255.0
    m = np.asarray(Image.fromarray(a).resize((H, H), Image.NEAREST)) > 127
    pr = profile_from(img, m)
    return dict(size=list(rgba.shape[:2]), width=float(partial.sum() / max(boundary.sum(), 1)), comp=cs, prof=pr,
                touches_border=bool(hard[0].any() or hard[-1].any() or hard[:, 0].any() or hard[:, -1].any()))


def summarise(rows):
    ok = [r for r in rows if r]
    comps = [r["comp"] for r in ok if r["comp"]]
    profs = [r["prof"] for r in ok if r["prof"]]
    s = sum(p[0] for p in profs)
    n = sum(p[1] for p in profs)
    t = (s / np.maximum(n, 1)).tolist()
    out = dict(images=len(ok), sizes=sorted({tuple(r["size"]) for r in ok}),
               alpha_edge_width_px512_median=float(np.median([r["width"] for r in ok])) if ok else None,
               comp_slope_median=float(np.median([c[0] for c in comps])) if comps else None,
               comp_intercept_median=float(np.median([c[1] for c in comps])) if comps else None,
               profile_views=len(profs), d=D.tolist(), t=t,
               far_bg_lum_mean=float(np.nanmean([p[2] for p in profs])) if profs else None,
               far_bg_lum_min=float(np.nanmin([p[2] for p in profs])) if profs else None,
               touches_border_frac=float(np.mean([r["touches_border"] for r in ok])) if ok else None)
    out["transition_10_90_px256"] = transition(D, np.array(t))
    return out


def transition(d, t):
    def cross(level):
        for i in range(len(d) - 1):
            if t[i] < level <= t[i + 1]:
                return d[i] + (level - t[i]) / (t[i + 1] - t[i])
        return None
    a, b = cross(0.1), cross(0.9)
    return None if a is None or b is None else float(b - a)


def from_tars(paths, n_views, rng, member_filter):
    rows = []
    for p in paths:
        try:
            with tarfile.open(p) as t:
                mem = [x for x in t.getmembers() if member_filter(x.name)]
                for x in rng.sample(mem, min(n_views, len(mem))):
                    rows.append(analyse(np.asarray(Image.open(io.BytesIO(t.extractfile(x).read())).convert("RGBA"))))
        except Exception as e:                                                          # noqa: BLE001
            print(f"[gt] {p}: {type(e).__name__}: {e}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_obj", type=int, default=40)
    ap.add_argument("--n_views", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.chdir(REPO)
    rng = random.Random(0)
    res = {}
    img_filter = lambda nm: nm.endswith((".webp", ".png")) and "depth" not in nm and "normal" not in nm  # noqa: E731
    # all287k
    names = [l.strip() for l in open("data/objaverse_all287k.txt")]
    res["all287k"] = summarise(from_tars([f"/projects/u6ga/zhelun/RnG_all287k_tars/{n}.tar" for n in rng.sample(names, args.n_obj)],
                                         args.n_views, rng, img_filter))
    print(f"[gt] all287k: {json.dumps(res['all287k'])}", flush=True)
    for key, root in (("fluffy", "/projects/u6ga/zhelun/datasets/FluffyElephant_tar"),
                      ("obj40v", "/projects/u6ga/zhelun/datasets/objaverse_rendering_set_tar")):
        shards = sorted(e.path for e in os.scandir(root) if e.is_dir())
        tars = []
        for sh in rng.sample(shards, min(len(shards), args.n_obj)):
            ts = [e.path for e in os.scandir(sh) if e.name.endswith(".tar")]
            if ts:
                tars.append(rng.choice(ts))
        res[key] = summarise(from_tars(tars[:args.n_obj], args.n_views, rng, img_filter))
        print(f"[gt] {key}: {json.dumps(res[key])}", flush=True)
    for key, root in (("gso_v1", "/projects/u6ga/zhelun/RnG_samples_eval/gso_sim2real_25v"),
                      ("gso_rv", "/scratch/u6ga/zhelun.u6ga/RnG_fa3_experiments/gso_render_rv_loose")):
        objs = sorted(os.listdir(root))
        rows = []
        for o in rng.sample(objs, args.n_obj):
            fs = sorted(glob.glob(f"{root}/{o}/[0-9][0-9][0-9].png"))
            for f in rng.sample(fs, min(args.n_views, len(fs))):
                rows.append(analyse(np.asarray(Image.open(f).convert("RGBA"))))
        res[key] = summarise(rows)
        print(f"[gt] {key}: {json.dumps(res[key])}", flush=True)
    # stage-1 prediction (step 0) on gso_v1, same normalisation, loader's hard mask
    from data.dataset_gso_ours import GSODataset_ours
    from tools_s2.s2_bench import load_cfg
    cfg = load_cfg(args.config, 1)
    ds = GSODataset_ours(cfg)
    nin = int(cfg.training.val_dataset_cfgs.training.num_input_views)
    for mode in ("posed", "unposed"):
        acc_s, acc_n, gs, gn = 0, 0, 0, 0
        for i in range(len(ds)):
            uid = f"{i:06d}"
            p = f"{args.exp_dir}/eval_iter_{0:08d}_{mode}/{uid}/gt_vs_pred.png"
            if not os.path.exists(p):
                continue
            item = ds[i]
            alpha = item["alpha_mask"][nin:, 0].numpy() > 0.5
            a = np.asarray(Image.open(p).convert("RGB")).astype(np.float64) / 255.0
            H = a.shape[0] // 2
            for v in range(a.shape[1] // H):
                for part, tgt in ((a[H:], "pred"), (a[:H], "gt")):
                    pr = profile_from(part[:, v * H:(v + 1) * H], alpha[v])
                    if pr:
                        if tgt == "pred":
                            acc_s = acc_s + pr[0]; acc_n = acc_n + pr[1]
                        else:
                            gs = gs + pr[0]; gn = gn + pr[1]
        t_pred = (acc_s / np.maximum(acc_n, 1)).tolist()
        t_gt = (gs / np.maximum(gn, 1)).tolist()
        res[f"stage1_pred_gso_v1_{mode}"] = dict(d=D.tolist(), t=t_pred, transition_10_90_px256=transition(D, np.array(t_pred)))
        res[f"gso_v1_eval_gt_{mode}"] = dict(d=D.tolist(), t=t_gt, transition_10_90_px256=transition(D, np.array(t_gt)))
        print(f"[gt] stage-1 prediction on gso_v1 ({mode}): {json.dumps(res[f'stage1_pred_gso_v1_{mode}'])}", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("[gt] ===== summary: normalised luminance t(d) (0 = object colour, 1 = white background), d = px at 256 "
          "from the GT silhouette (< 0 inside)", flush=True)
    print("[gt] source                 imgs  a-edge@512  comp slope/icpt  far-bg L   10-90% width  t(" +
          ",".join(f"{d:+d}" for d in D) + ")", flush=True)
    for k, r in res.items():
        if "images" in r:
            print(f"[gt] {k:22s} {r['images']:4d}  {r['alpha_edge_width_px512_median']:9.2f}  "
                  f"{r['comp_slope_median']:+.2f}/{r['comp_intercept_median']:.2f}      {r['far_bg_lum_mean']:.4f}"
                  f"(min {r['far_bg_lum_min']:.3f})  {r['transition_10_90_px256']}  " + " ".join(f"{x:.3f}" for x in r["t"]), flush=True)
        else:
            print(f"[gt] {k:22s}                                                 {r['transition_10_90_px256']}  "
                  + " ".join(f"{x:.3f}" for x in r["t"]), flush=True)
    print(f"[gt] DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
