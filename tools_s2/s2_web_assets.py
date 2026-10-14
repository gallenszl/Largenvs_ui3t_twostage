# Assets for the fg_dense-vs-stage-1 comparison page (user 09-28): from the training validations of one stage-2 run
# (eval_iter_<step>_{posed,unposed}/<uid>/gt_vs_pred.png, metrics*.json), every 2k step against step 0 (= the frozen
# stage 1 for block 8, G9-checked).  Writes, into --out:
#   data.json            per checkpoint averages (all objects), per object / step / mode metrics incl. per view
#   o<uid>_base.jpg           row 0: the 10 GT target views; row 1: the 4 input views        (tile px per view)
#   o<uid>_<mode>_pred.jpg    one row per step: the prediction
#   o<uid>_<mode>_err.jpg     one row per step: |pred - GT| error map (mean over RGB, x4, inferno colours)
#   o<uid>_<mode>_chg.jpg     one row per step > 0: |pred - pred_step0| change map (same colours)
# Tiles are the native 256 px when --tile 256 (no resampling); otherwise every view is resized on its own (resizing the
# whole 10-view strip at once let LANCZOS mix 2-3 px of the neighbouring view into each tile edge).
# Objects with image sprites: by the posed LPIPS change at --select_step (default: the latest step) vs step 0, the --n
# most improved, the --n around the median and the --n least improved (distinct names).  Sanity check: every GT half is
# pixel-identical across steps and modes (confirms the layout: top half GT, bottom half prediction).
# --seams_json (tools_s2/s2_edge_seams.py --out): per view and step, the refined/unrefined token-edge seam count and the
# ghost pixels are appended to the view metrics ([psnr, lpips, fg_psnr, seams_RO, ghost_px]) and --n_seam more objects
# (the most posed R-O seams at --select_step, distinct names) get sprites; --ovl_dir: copy o<uid>_ovl.png of the shown
# objects (s2_edge_seams.py --ovl_dir) next to the sprites.
#   python tools_s2/s2_web_assets.py --exp_dir <ckpt dir> --out <dir> [--n 8] [--tile 256] [--select_step 14000]
#                                    [--seams_json x.json --n_seam 6] [--ovl_dir <dir>]
import argparse
import glob
import json
import os
import re
import shutil

import numpy as np
from PIL import Image

MODES = ("posed", "unposed")
INFERNO = np.array([[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21],
                    [250, 193, 39], [252, 255, 164]], dtype=np.float64)


def cmap(x):
    t = np.clip(x, 0.0, 1.0) * (len(INFERNO) - 1)
    i = np.clip(np.floor(t).astype(int), 0, len(INFERNO) - 2)
    f = (t - i)[..., None]
    return (INFERNO[i] * (1 - f) + INFERNO[i + 1] * f).round().astype(np.uint8)


def halves(path):
    a = np.asarray(Image.open(path).convert("RGB"))
    h = a.shape[0] // 2
    return a[:h], a[h:]


def resize(a, tile, n_views):
    """strip [h, n_views * h, 3] -> [tile, n_views * tile, 3], every view resized on its own (no cross-view bleed)"""
    h = a.shape[0]
    if h == tile:
        return a
    return np.concatenate([np.asarray(Image.fromarray(a[:, v * h:(v + 1) * h]).resize((tile, tile), Image.LANCZOS))
                           for v in range(n_views)], axis=1)


def read_avg(d):
    out = {}
    try:
        v = [float(x) for x in open(f"{d}/average_metrics.txt").readline().split(":")[1].split(",")]
        out.update(psnr=v[0], lpips=v[1], ssim=v[2], fg_psnr=v[3])
    except Exception:                                                                    # noqa: BLE001
        pass
    try:
        out["abs_rel"] = float(open(f"{d}/average_metrics_depth.txt").read().splitlines()[2].split(",")[1])
    except Exception:                                                                    # noqa: BLE001
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--select_step", type=int, default=None)
    ap.add_argument("--quality", type=int, default=82)
    ap.add_argument("--seams_json", default=None)
    ap.add_argument("--n_seam", type=int, default=6)
    ap.add_argument("--ovl_dir", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    steps = sorted({int(re.search(r"eval_iter_(\d+)_", p).group(1)) for p in glob.glob(f"{args.exp_dir}/eval_iter_*_posed")
                    if os.path.isdir(p.replace("_posed", "_unposed"))})
    steps = [s for s in steps if s % 2000 == 0]
    last = args.select_step if args.select_step is not None else steps[-1]
    assert last in steps, (last, steps)
    uids = sorted(os.path.basename(p) for p in glob.glob(f"{args.exp_dir}/eval_iter_{0:08d}_posed/[0-9]*"))
    print(f"[web] steps {steps}, objects {len(uids)}", flush=True)
    data = dict(exp=os.path.basename(args.exp_dir.rstrip("/")), steps=steps, modes=list(MODES), tile=args.tile,
                avg={m: {str(s): read_avg(f"{args.exp_dir}/eval_iter_{s:08d}_{m}") for s in steps} for m in MODES},
                objects=[], layout="per_kind", select_step=last)
    for uid in uids:
        o = dict(uid=uid, metrics={})
        for m in MODES:
            o["metrics"][m] = {}
            for s in steps:
                d = f"{args.exp_dir}/eval_iter_{s:08d}_{m}/{uid}"
                mj = json.load(open(f"{d}/metrics.json"))
                o["name"] = mj["summary"]["scene_name"]
                dep = json.load(open(f"{d}/metrics_depth.json"))["summary"]
                o["metrics"][m][str(s)] = dict({k: float(mj["summary"][k]) for k in ("psnr", "lpips", "ssim", "fg_psnr")},
                                               abs_rel=float(dep["abs_rel"]),
                                               views=[[round(float(v["psnr"]), 3), round(float(v["lpips"]), 5),
                                                       round(float(v["fg_psnr"]), 3)] for v in mj["per_view"]])
        data["objects"].append(o)
    seam = None
    if args.seams_json:
        sj = json.load(open(args.seams_json))
        seam = {(r["mode"], r["uid"], r["view"]): r for r in sj["views"]}
        for o in data["objects"]:
            for m in MODES:
                for s in steps:
                    for v, row in enumerate(o["metrics"][m][str(s)]["views"]):
                        x = seam[(m, o["uid"], v)]["steps"][str(s)]
                        row += [x["seams"]["RO"], x["ghost"]]
        data["seams"] = dict(tau=sj["tau"], summary=sj["summary"])
    # selection: posed LPIPS change latest - step 0, distinct names
    seen, cand = set(), []
    for o in data["objects"]:
        if o["name"] in seen:
            continue
        seen.add(o["name"])
        cand.append((o["metrics"]["posed"][str(last)]["lpips"] - o["metrics"]["posed"]["0"]["lpips"], o["uid"]))
    cand.sort()
    n = args.n
    mid = len(cand) // 2
    groups = {"most improved": cand[:n], "around the median": cand[mid - n // 2: mid - n // 2 + n],
              "least improved": cand[-n:]}
    shown = []
    for g, lst in groups.items():
        for dl, uid in lst:
            if uid not in [u for u, _ in shown]:
                shown.append((uid, g))
    if seam is not None:
        byu = {o["uid"]: o for o in data["objects"]}
        names = {byu[u]["name"] for u, _ in shown}
        tot = sorted(((-sum(v[3] for v in o["metrics"]["posed"][str(last)]["views"]), o["uid"]) for o in data["objects"]))
        k = 0
        for t, uid in tot:
            if k >= args.n_seam:
                break
            if byu[uid]["name"] not in names:
                shown.append((uid, "most seams"))
                names.add(byu[uid]["name"])
                k += 1
    data["shown"] = [dict(uid=u, group=g) for u, g in shown]
    data["selection_rule"] = (f"posed LPIPS change step {last} - step 0: the {n} most improved, the {n} around the median, "
                              f"the {n} least improved (distinct object names)"
                              + (f"; plus the {args.n_seam} with the most posed R-O seams at step {last}" if seam else ""))
    gt_max_diff = 0
    for uid, g in shown:
        gt_ref = None
        rows_pred, rows_err, rows_chg = {m: [] for m in MODES}, {m: [] for m in MODES}, {m: [] for m in MODES}
        for m in MODES:
            p0 = None
            for s in steps:
                gt, pred = halves(f"{args.exp_dir}/eval_iter_{s:08d}_{m}/{uid}/gt_vs_pred.png")
                if gt_ref is None:
                    gt_ref = gt
                gt_max_diff = max(gt_max_diff, int(np.abs(gt.astype(int) - gt_ref.astype(int)).max()))
                nv = gt.shape[1] // gt.shape[0]
                err = np.abs(pred.astype(np.float64) - gt.astype(np.float64)).mean(-1) / 255.0 * 4.0
                rows_pred[m].append(resize(pred, args.tile, nv))
                rows_err[m].append(resize(cmap(err), args.tile, nv))
                if p0 is None:
                    p0 = pred
                else:
                    chg = np.abs(pred.astype(np.float64) - p0.astype(np.float64)).mean(-1) / 255.0 * 4.0
                    rows_chg[m].append(resize(cmap(chg), args.tile, nv))
            for kind, rows in (("pred", rows_pred[m]), ("err", rows_err[m]), ("chg", rows_chg[m])):
                Image.fromarray(np.concatenate(rows, axis=0)).save(f"{args.out}/o{uid}_{m}_{kind}.jpg",
                                                                   quality=args.quality, optimize=True)
        inp = np.asarray(Image.open(f"{args.exp_dir}/eval_iter_{0:08d}_posed/{uid}/input.png").convert("RGB"))
        ni = inp.shape[1] // inp.shape[0]
        base = np.zeros((2 * args.tile, args.tile * nv, 3), dtype=np.uint8) + 255
        base[:args.tile] = resize(gt_ref, args.tile, nv)
        base[args.tile:, :args.tile * ni] = resize(inp, args.tile, ni)
        Image.fromarray(base).save(f"{args.out}/o{uid}_base.jpg", quality=args.quality, optimize=True)
        if args.ovl_dir:
            shutil.copyfile(f"{args.ovl_dir}/o{uid}_ovl.png", f"{args.out}/o{uid}_ovl.png")
        print(f"[web] {uid} {g}: sprites written", flush=True)
    data["n_views"] = int(nv)
    data["n_inputs"] = int(ni)
    data["gt_identical_across_steps_and_modes"] = gt_max_diff == 0
    data["overlay"] = bool(args.ovl_dir)
    json.dump(data, open(f"{args.out}/data.json", "w"), separators=(",", ":"))
    sz = sum(os.path.getsize(f"{args.out}/{f}") for f in os.listdir(args.out))
    print(f"[web] GT pixel-identical across steps / modes: {gt_max_diff == 0} (max diff {gt_max_diff}); "
          f"{len(os.listdir(args.out))} files, {sz / 2**20:.1f} MiB -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
