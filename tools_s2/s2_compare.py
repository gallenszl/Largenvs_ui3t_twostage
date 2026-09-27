# Paired comparison of two evaluation directories (plan 2026-09-27, section 四): per-object metrics of A and
# B (same objects, same views), mean difference A - B and a 95 % interval from 1000 bootstrap resamples over
# objects.  Main metric LPIPS (signal line 0.003); PSNR (0.6 dB) / FG-PSNR (0.5 dB) secondary.
#   python tools_s2/s2_compare.py <dirA> <dirB> [--out result.json] [--boot 1000]
# Also compares two stored evaluations for the G6 reproduction gate:  --gate  (max |per-object diff| checks).

import argparse
import glob
import json
import math
import os

import numpy as np

MAIN = ("lpips", "psnr", "fg_psnr", "ssim")
REG = ("silhouette", "unseen", "seen", "texture", "depth_edge")


def load_dir(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "[0-9]" * 6, "metrics.json"))):
        od = os.path.dirname(f)
        m = json.load(open(f))
        s = m["summary"]
        row = {k: float(s[k]) for k in MAIN if k in s}
        fd = os.path.join(od, "metrics_depth.json")
        if os.path.exists(fd):
            ds = json.load(open(fd))["summary"]
            if ds.get("abs_rel") is not None:
                row["abs_rel"] = float(ds["abs_rel"])
        fr = os.path.join(od, "regions.json")
        if os.path.exists(fr):
            views = json.load(open(fr))
            for r in REG:
                ps, lp, ar = [], [], []
                for v in views:
                    e = v[r]
                    if e["n"] > 0:
                        mse = e["se"] / (3.0 * e["n"])
                        ps.append(-10.0 * math.log10(max(mse, 1e-12)))
                        lp.append(e["lpips"] / e["n"])
                        if "absrel" in e:
                            ar.append(e["absrel"] / e["n"])
                if ps:
                    row[f"{r}_psnr"] = float(np.mean(ps))
                    row[f"{r}_lpips"] = float(np.mean(lp))
                if ar:
                    row[f"{r}_absrel"] = float(np.mean(ar))
        out[s.get("scene_name", os.path.basename(od))] = row
    return out


def compare(A, B, n_boot=1000, seed=0):
    objs = sorted(set(A) & set(B))
    keys = sorted({k for o in objs for k in A[o]} & {k for o in objs for k in B[o]})
    rng = np.random.default_rng(seed)
    res = {"n_objects": len(objs)}
    for k in keys:
        pairs = [(A[o][k], B[o][k]) for o in objs if k in A[o] and k in B[o]]
        if not pairs:
            continue
        a = np.array([p[0] for p in pairs])
        b = np.array([p[1] for p in pairs])
        d = a - b
        idx = rng.integers(0, len(d), size=(n_boot, len(d)))
        boot = d[idx].mean(1)
        res[k] = dict(n=len(d), A=float(a.mean()), B=float(b.mean()), diff=float(d.mean()),
                      ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                      max_abs_diff=float(np.abs(d).max()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirA")
    ap.add_argument("dirB")
    ap.add_argument("--out")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    A, B = load_dir(args.dirA), load_dir(args.dirB)
    res = compare(A, B, args.boot)
    print(f"A = {args.dirA}\nB = {args.dirB}\nobjects paired: {res['n_objects']}")
    print(f"{'metric':22s} {'A':>10s} {'B':>10s} {'A-B':>10s}  95% CI             max|diff|")
    for k, v in res.items():
        if k == "n_objects":
            continue
        print(f"{k:22s} {v['A']:10.5f} {v['B']:10.5f} {v['diff']:+10.5f}  [{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}]  {v['max_abs_diff']:.5f}")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
