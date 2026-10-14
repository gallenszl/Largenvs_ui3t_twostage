# Reproduction check of two evaluation directories of the same objects (the G6 / G9 step-0 "identical?" question;
# README §5.4 names it `s2_compare.py --gate`, which was never implemented there): per metric the largest
# per-object |A - B| (metrics.json / metrics_depth.json / metrics_pose.json summaries), the number of views whose
# LPIPS differs, and how many of the saved PNGs (render and depth panels) are byte-identical.
#   python tools_s2/s2_pairdiff.py <dirA> <dirB> [--json out.json]
import argparse
import filecmp
import json
import os

RGB = ("psnr", "lpips", "ssim", "fg_psnr")
DEPTH = ("abs_rel",)
PNGS = ("gt_vs_pred.png", "gt_vs_pred_depth.png")


def pairdiff(A, B):
    objs = sorted(d for d in os.listdir(B) if d.isdigit() and os.path.isdir(os.path.join(A, d)))
    mx, worst, lp_views, png = {}, {}, 0, {p: 0 for p in PNGS}
    for o in objs:
        for fn, keys in (("metrics.json", RGB), ("metrics_depth.json", DEPTH)):
            a = json.load(open(f"{A}/{o}/{fn}"))["summary"]
            b = json.load(open(f"{B}/{o}/{fn}"))["summary"]
            for k in keys:
                d = abs(float(a[k]) - float(b[k]))
                if d >= mx.get(k, -1.0):
                    mx[k], worst[k] = d, (o, float(a[k]), float(b[k]))
        a = json.load(open(f"{A}/{o}/metrics.json"))["per_view"]
        b = json.load(open(f"{B}/{o}/metrics.json"))["per_view"]
        lp_views += sum(va["lpips"] != vb["lpips"] for va, vb in zip(a, b))
        pa = json.load(open(f"{A}/{o}/metrics_pose.json"))["summary"]
        pb = json.load(open(f"{B}/{o}/metrics_pose.json"))["summary"]
        for k, v in pa.items():
            if isinstance(v, (int, float)):
                mx["pose." + k] = max(mx.get("pose." + k, 0.0), abs(float(v) - float(pb[k])))
        for p in PNGS:
            png[p] += filecmp.cmp(f"{A}/{o}/{p}", f"{B}/{o}/{p}", shallow=False)
    return dict(n_objects=len(objs), max_abs_diff=mx, worst=worst, lpips_views_differing=lp_views,
                png_identical={p: f"{n}/{len(objs)}" for p, n in png.items()},
                identical=all(v == 0 for v in mx.values()) and all(n == len(objs) for n in png.values()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dirA")
    ap.add_argument("dirB")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    r = pairdiff(args.dirA, args.dirB)
    print(f"{args.dirA}\n  vs {args.dirB}\n  objects {r['n_objects']}  identical {r['identical']}  "
          f"LPIPS views differing {r['lpips_views_differing']}  PNG identical {r['png_identical']}")
    for k, v in r["max_abs_diff"].items():
        w = r["worst"].get(k)
        print(f"  max|diff| {k:14s} {v:.3e}" + (f"   obj {w[0]}  A {w[1]:.6f}  B {w[2]:.6f}" if w and v > 0 else ""))
    if args.json:
        json.dump(r, open(args.json, "w"), indent=1)
