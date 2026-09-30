"""Paired comparison of two full GSO evaluations of stage-1 checkpoints (scripts_s2/uni3t_fulleval_gso.sbatch).

  python3 tools_s2/uni3t_eval_compare.py --root <eval root> --a <tag A> --b <tag B> \
      [--ref <tag> <tag> ...] [--modes posed unposed] [--n_boot 1000] [--json out.json]
  python3 tools_s2/uni3t_eval_compare.py --a <eval dir A> --b <eval dir B> [--ref <dir> ...]   (e.g. eval_iter_<step>)

Reads the per-object tables each evaluation writes (<tag>_<mode>_novel/summary.csv, summary_depth.csv,
summary_pose.csv), aligns objects by index, and reports for PSNR, FG-PSNR, LPIPS, abs_rel, Racc_5, Tacc_5,
Auc_30:
  * A, B and B - A;
  * a 95 % interval of B - A from a paired bootstrap over objects (objects resampled with replacement,
    both arms take the same resample; this is the object-sampling spread only, not checkpoint-to-checkpoint);
  * when --ref tags are given (e.g. one training run at several steps), the max - min of each metric over
    them: the checkpoint-to-checkpoint spread to compare B - A against.
Racc_5 / Tacc_5 / Auc_30 follow utils/metric_utils.py exactly: they are OBJECT-level -- each object's rError /
tError is the mean over its view pairs; Racc_5 = % of objects with rError < 5 deg, Tacc_5 = % with tError < 5 deg,
Auc_30 = mean over integer thresholds 1..30 deg of the fraction of objects with max(rError, tError) below it.
Every aggregate is also checked against the evaluation's own average_metrics*.txt (parser check).
"""
import argparse
import json
import os

import numpy as np

METRICS = [  # name, higher is better
    ("psnr", True), ("fg_psnr", True), ("lpips", False), ("abs_rel", False),
    ("Racc_5", True), ("Tacc_5", True), ("Auc_30", True),
]


def read_table(path):
    rows = {}
    with open(path) as f:
        header = f.readline().strip().split(",")
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("average"):
                continue
            parts = ln.split(",")
            rows[parts[0]] = {k: float(v) for k, v in zip(header[1:], parts[1:])}
    return rows


def load_eval(root, tag, mode):
    d = tag if os.path.isdir(tag) else os.path.join(root, f"{tag}_{mode}_novel")  # a directory may be given directly
    img, dep, pose = (read_table(os.path.join(d, f)) for f in ("summary.csv", "summary_depth.csv", "summary_pose.csv"))
    keys = sorted(set(img) & set(dep) & set(pose))
    if not (len(keys) == len(img) == len(dep) == len(pose)):
        raise ValueError(f"{d}: object sets differ ({len(img)} / {len(dep)} / {len(pose)})")
    cols = {
        "psnr": np.array([img[k]["psnr"] for k in keys]),
        "fg_psnr": np.array([img[k]["fg_psnr"] for k in keys]),
        "lpips": np.array([img[k]["lpips"] for k in keys]),
        "abs_rel": np.array([dep[k]["abs_rel"] for k in keys]),
        "rError": np.array([pose[k]["rError"] for k in keys]),
        "tError": np.array([pose[k]["tError"] for k in keys]),
    }
    return d, keys, cols


def auc_30(r, t):
    m = np.maximum(r, t)
    hist, _ = np.histogram(m, bins=np.arange(31))
    return float(np.mean(np.cumsum(hist / float(len(m))))) * 100.0


def aggregate(cols, idx=None):
    c = cols if idx is None else {k: v[idx] for k, v in cols.items()}
    return {
        "psnr": float(c["psnr"].mean()), "fg_psnr": float(c["fg_psnr"].mean()), "lpips": float(c["lpips"].mean()),
        "abs_rel": float(c["abs_rel"].mean()),
        "Racc_5": float(np.mean(c["rError"] < 5) * 100), "Tacc_5": float(np.mean(c["tError"] < 5) * 100),
        "Auc_30": auc_30(c["rError"], c["tError"]),
    }


def check_against_files(d, agg):
    avg = [float(x) for x in open(os.path.join(d, "average_metrics.txt")).readline().split(":", 1)[1].split(",")]
    dep = [float(x) for x in open(os.path.join(d, "average_metrics_depth.txt")).read().strip().splitlines()[2].split(",")]
    pose = {}
    for ln in open(os.path.join(d, "average_metrics_pose.txt")):
        if ":" in ln and ln.split(":")[0] in ("Racc_5", "Tacc_5", "Auc_30"):
            pose[ln.split(":")[0]] = float(ln.split(":")[1].strip().rstrip("%"))
    ref = {"psnr": avg[0], "lpips": avg[1], "fg_psnr": avg[3], "abs_rel": dep[1], **pose}
    for k, v in ref.items():
        tol = 1e-4 if k in ("Racc_5", "Tacc_5", "Auc_30") else 1e-9 * max(1.0, abs(v))
        if abs(agg[k] - v) > tol:
            raise ValueError(f"{d}: recomputed {k}={agg[k]} but the evaluation wrote {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="", help="eval root; not needed when --a/--b/--ref are directories")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--ref", nargs="*", default=[])
    ap.add_argument("--modes", nargs="+", default=["posed", "unposed"])
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json")
    a = ap.parse_args()
    if os.path.isdir(a.a):  # directories given directly (e.g. training-internal eval_iter_<step>): one pass
        a.modes = ["dirs"]
    out = {}
    for mode in a.modes:
        da, ka, ca = load_eval(a.root, a.a, mode)
        db, kb, cb = load_eval(a.root, a.b, mode)
        if ka != kb:
            raise ValueError(f"{mode}: the two evaluations cover different objects")
        agg_a, agg_b = aggregate(ca), aggregate(cb)
        check_against_files(da, agg_a)
        check_against_files(db, agg_b)
        rng = np.random.default_rng(a.seed)
        n = len(ka)
        boots = {m: [] for m, _ in METRICS}
        for _ in range(a.n_boot):
            idx = rng.integers(0, n, n)
            ga, gb = aggregate(ca, idx), aggregate(cb, idx)
            for m, _ in METRICS:
                boots[m].append(gb[m] - ga[m])
        refs = {}
        for t in a.ref:
            dr, kr, cr = load_eval(a.root, t, mode)
            if kr != ka:
                raise ValueError(f"{mode}: reference {t} covers different objects")
            refs[t] = aggregate(cr)
            check_against_files(dr, refs[t])
        res = {"n_objects": n, "a": agg_a, "b": agg_b, "diff": {}, "ci95": {}, "ref": refs, "ref_spread": {}}
        print(f"\n== {mode}: B = {a.b}  minus  A = {a.a}  ({n} objects, paired bootstrap {a.n_boot}x)")
        hdr = f"{'metric':8s} {'A':>9s} {'B':>9s} {'B-A':>9s} {'95% CI of B-A':>22s}"
        if refs:
            hdr += f" {'ref spread':>11s}  " + "  ".join(f"{t[-12:]:>12s}" for t in refs)
        print(hdr)
        for m, hib in METRICS:
            dlt = agg_b[m] - agg_a[m]
            lo, hi = np.percentile(boots[m], [2.5, 97.5])
            res["diff"][m], res["ci95"][m] = dlt, [float(lo), float(hi)]
            line = f"{m:8s} {agg_a[m]:9.4f} {agg_b[m]:9.4f} {dlt:+9.4f} [{lo:+9.4f}, {hi:+9.4f}]"
            if refs:
                vals = [refs[t][m] for t in refs]
                res["ref_spread"][m] = float(max(vals) - min(vals))
                line += f" {max(vals) - min(vals):11.4f}  " + "  ".join(f"{v:12.4f}" for v in vals)
            print(line)
        out[mode] = res
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
