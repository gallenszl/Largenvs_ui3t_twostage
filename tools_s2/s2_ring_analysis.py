# Where is the stage-1 error, and where does the ring around the object come from?  (user 09-28)
# Uses the training validations of one stage-2 run (step 0 = the frozen stage 1, G9-checked; a later step = stage 2),
# the GT alpha from the same loader the validation used (data/dataset_gso_ours.py), and the raw 512 px renders.
#   A. error vs signed distance to the GT silhouette (px at 256; < 0 inside the object): pixel share, MSE, share of the
#      total squared error, for step 0 and the later step, posed and unposed
#   B. sub-pixel misalignment: per view, phase correlation (1/20 px) of prediction vs GT; the shifts' distribution and
#      the MSE per band after moving the prediction by its shift (if the ring were a misalignment, it would vanish)
#   C. edge profile: mean luminance of GT and prediction per signed-distance bin (blur = flatter slope, shift = offset
#      crossing, fringe = a dip / bump that only one of them has)
#   D. GT compositing check on the raw 512 px RGBA: at edge pixels (0.05 < alpha < 0.95), colour / nearby opaque
#      colour against alpha.  Straight alpha (what the loader assumes when it composites rgb*a + 255*(1-a)): slope ~0,
#      intercept ~1.  Premultiplied colours: slope ~1, intercept ~0 (the loader would then darken every edge).  Done
#      for the GSO v1 renders and for training renders (Objaverse tars, webp) for comparison.
#   python tools_s2/s2_ring_analysis.py --config configs/S2P8fgdense_...yaml --exp_dir <ckpt dir> --step 14000 --out x.json
import argparse
import glob
import io
import json
import os
import sys
import tarfile

import numpy as np
from PIL import Image
from scipy import ndimage

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
BANDS = [(-1e9, -8), (-8, -4), (-4, -2), (-2, 0), (0, 2), (2, 4), (4, 8), (8, 1e9)]
BAND_NAMES = ["inside <-8", "inside -8..-4", "inside -4..-2", "inside -2..0", "outside 0..2", "outside 2..4",
              "outside 4..8", "outside >8"]
PROF = np.arange(-6, 7)                                     # profile bins: rounded signed distance


def halves(path):
    a = np.asarray(Image.open(path).convert("RGB")).astype(np.float64) / 255.0
    h = a.shape[0] // 2
    return a[:h], a[h:]


def signed_distance(m):
    """m bool [H, W] (True = object) -> distance (px) to the nearest pixel of the other class, negative inside"""
    return np.where(m, -ndimage.distance_transform_edt(m), ndimage.distance_transform_edt(~m))


def lum(x):
    return x @ np.array([0.299, 0.587, 0.114])


def edge_stats(rgba):
    """slope / intercept of (colour / nearby opaque colour) against alpha at partially transparent pixels"""
    a = rgba[..., 3].astype(np.float64) / 255.0
    rgb = rgba[..., :3].astype(np.float64) / 255.0
    op = (a > 0.999).astype(np.float64)
    k = np.ones((7, 7))
    num = np.stack([ndimage.convolve(rgb[..., c] * op, k, mode="nearest") for c in range(3)], -1)
    den = ndimage.convolve(op, k, mode="nearest")[..., None]
    ref = num / np.maximum(den, 1e-9)
    sel = (a > 0.05) & (a < 0.95) & (den[..., 0] > 0) & (lum(ref) > 0.15)
    if sel.sum() < 20:
        return None
    r = (lum(rgb[sel]) / np.maximum(lum(ref[sel]), 1e-6))
    x = a[sel]
    slope, icpt = np.polyfit(x, r, 1)
    return dict(n=int(sel.sum()), slope=float(slope), intercept=float(icpt), mean_ratio=float(r.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--n_train_objects", type=int, default=24)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.chdir(REPO)
    from data.dataset_gso_ours import GSODataset_ours
    from tools_s2.s2_bench import load_cfg
    from skimage.registration import phase_cross_correlation
    cfg = load_cfg(args.config, 1)
    ds = GSODataset_ours(cfg)
    nin = int(cfg.training.val_dataset_cfgs.training.num_input_views)
    steps = [0, args.step]
    acc = {m: {s: dict(cnt=np.zeros(len(BANDS)), se=np.zeros(len(BANDS)), se_sh=np.zeros(len(BANDS)),
                       prof_gt=np.zeros(len(PROF)), prof_pr=np.zeros(len(PROF)), prof_n=np.zeros(len(PROF)),
                       shifts=[]) for s in steps} for m in ("posed", "unposed")}
    gt_check = 0.0
    n_obj = 0
    for i in range(len(ds)):
        uid = f"{i:06d}"
        if not os.path.isdir(f"{args.exp_dir}/eval_iter_{0:08d}_posed/{uid}"):
            continue
        item = ds[i]
        name = item["scene_name"]
        mj = json.load(open(f"{args.exp_dir}/eval_iter_{0:08d}_posed/{uid}/metrics.json"))
        assert mj["summary"]["scene_name"] == name, (uid, name, mj["summary"]["scene_name"])
        alpha = item["alpha_mask"][nin:, 0].numpy() > 0.5                              # [Vt, H, W]
        tgt = item["image"][nin:].permute(0, 2, 3, 1).numpy()                          # [Vt, H, W, 3] in [0, 1]
        n_obj += 1
        for m in ("posed", "unposed"):
            for s in steps:
                gt, pred = halves(f"{args.exp_dir}/eval_iter_{s:08d}_{m}/{uid}/gt_vs_pred.png")
                H = gt.shape[0]
                Vt = gt.shape[1] // H
                A = acc[m][s]
                for v in range(Vt):
                    g = gt[:, v * H:(v + 1) * H]
                    p = pred[:, v * H:(v + 1) * H]
                    gt_check = max(gt_check, float(np.abs(g - tgt[v]).max()))
                    sd = signed_distance(alpha[v])
                    se = ((p - g) ** 2).mean(-1)
                    shift, _, _ = phase_cross_correlation(lum(g), lum(p), upsample_factor=20)
                    A["shifts"].append([float(shift[0]), float(shift[1])])
                    ps = np.stack([ndimage.shift(p[..., c], shift, order=3, mode="nearest") for c in range(3)], -1)
                    se_sh = ((ps - g) ** 2).mean(-1)
                    for b, (lo, hi) in enumerate(BANDS):
                        sel = (sd >= lo) & (sd < hi) if lo < 0 else (sd > lo) & (sd <= hi)
                        A["cnt"][b] += sel.sum()
                        A["se"][b] += se[sel].sum()
                        A["se_sh"][b] += se_sh[sel].sum()
                    r = np.rint(sd).astype(int)
                    lg, lp = lum(g), lum(p)
                    for j, d in enumerate(PROF):
                        sel = r == d
                        A["prof_gt"][j] += lg[sel].sum()
                        A["prof_pr"][j] += lp[sel].sum()
                        A["prof_n"][j] += sel.sum()
        print(f"[ring] {uid} {name} done", flush=True)
    res = dict(step=args.step, n_objects=n_obj, gt_png_vs_loader_maxabs=gt_check, bands=BAND_NAMES, modes={})
    for m in ("posed", "unposed"):
        res["modes"][m] = {}
        for s in steps:
            A = acc[m][s]
            sh = np.array(A["shifts"])
            tot = A["se"].sum()
            res["modes"][m][str(s)] = dict(
                pixel_share=(A["cnt"] / A["cnt"].sum()).tolist(), mse=(A["se"] / A["cnt"]).tolist(),
                error_share=(A["se"] / tot).tolist(), mse_after_shift=(A["se_sh"] / A["cnt"]).tolist(),
                total_mse=float(tot / A["cnt"].sum()), total_mse_after_shift=float(A["se_sh"].sum() / A["cnt"].sum()),
                shift_mean=sh.mean(0).tolist(), shift_std=sh.std(0).tolist(),
                shift_abs_median=np.median(np.abs(sh), 0).tolist(),
                profile_d=PROF.tolist(), profile_gt=(A["prof_gt"] / A["prof_n"]).tolist(),
                profile_pred=(A["prof_pr"] / A["prof_n"]).tolist())
    # D. compositing check on raw renders
    gso_root = cfg.training.val_dataset_cfgs.root_dir
    gso = []
    for name in ds.all_object_list[:24]:
        for f in sorted(glob.glob(f"{gso_root}/{name}/0[0-9][0-9].png"))[:3]:
            st = edge_stats(np.asarray(Image.open(f).convert("RGBA")))
            if st:
                gso.append(st)
    tr = []
    tar_root = cfg.training.tar_root_path
    names = [l.strip() for l in open(cfg.training.dataset_path)][:: max(1, 287562 // args.n_train_objects)][:args.n_train_objects]
    for nm in names:
        try:
            with tarfile.open(f"{tar_root}/{nm}.tar") as t:
                mem = [x for x in t.getmembers() if x.name.endswith((".webp", ".png")) and "depth" not in x.name][:3]
                for x in mem:
                    st = edge_stats(np.asarray(Image.open(io.BytesIO(t.extractfile(x).read())).convert("RGBA")))
                    if st:
                        tr.append(st)
        except Exception as e:                                                          # noqa: BLE001
            print(f"[ring] train tar {nm}: {type(e).__name__}: {e}", flush=True)

    def summ(lst):
        if not lst:
            return None
        return dict(images=len(lst), slope_median=float(np.median([x["slope"] for x in lst])),
                    intercept_median=float(np.median([x["intercept"] for x in lst])),
                    mean_ratio_median=float(np.median([x["mean_ratio"] for x in lst])))
    res["compositing"] = dict(gso_v1=summ(gso), train_objaverse=summ(tr))
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"[ring] GT png vs loader max |diff| {gt_check:.4f} (1/255 = {1 / 255:.4f})", flush=True)
    for m in ("posed", "unposed"):
        for s in steps:
            r = res["modes"][m][str(s)]
            print(f"[ring] ===== {m} step {s}: total MSE {r['total_mse']:.5f} -> after per-view shift "
                  f"{r['total_mse_after_shift']:.5f}; shift mean (dy, dx) {np.round(r['shift_mean'], 3).tolist()} "
                  f"std {np.round(r['shift_std'], 3).tolist()} median|.| {np.round(r['shift_abs_median'], 3).tolist()}",
                  flush=True)
            for b, nm in enumerate(BAND_NAMES):
                print(f"[ring]   {nm:15s} pixels {100 * r['pixel_share'][b]:5.1f}% | MSE {r['mse'][b]:.5f} | share of error "
                      f"{100 * r['error_share'][b]:5.1f}% | MSE after shift {r['mse_after_shift'][b]:.5f}", flush=True)
            print("[ring]   profile d: " + " ".join(f"{d:+d}" for d in PROF), flush=True)
            print("[ring]   GT lum   : " + " ".join(f"{x:.3f}" for x in r["profile_gt"]), flush=True)
            print("[ring]   pred lum : " + " ".join(f"{x:.3f}" for x in r["profile_pred"]), flush=True)
    print(f"[ring] compositing (colour / nearby opaque colour vs alpha; straight ~ slope 0 intercept 1, premultiplied ~ "
          f"slope 1 intercept 0): {json.dumps(res['compositing'])}", flush=True)
    print(f"[ring] DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
