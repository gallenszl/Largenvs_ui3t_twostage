# Stage-2 training gates (plan G9).
#   step0  <val_root> <s1ref_root>
#          the training's step-0 validation (eval_iter_00000000_{posed,unposed}) against the stock stage-1
#          inference.py on the same set (s1ref_subset64_{posed,unposed}); per object, G6a lines:
#          |dPSNR| <= 0.02 dB, |dLPIPS| <= 2e-4, |dabs_rel| <= 1e-4 (also reports SSIM / FG-PSNR)
#   safety <val_root> <step> <log> [<log> ...] [--world 4]
#          the 4k safety line: skipped steps (non-finite loss) < 1 % of steps, render_std stable (no step below
#          0.5x its median), validation LPIPS at <step> not worse than step 0 by more than 0.003, both modes
import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s2_compare import compare, load_dir  # noqa: E402

MODES = ("posed", "unposed")
LINES = {"psnr": 0.02, "lpips": 2e-4, "abs_rel": 1e-4}


def step0(val_root, ref_root):
    ok = True
    for mode in MODES:
        a = load_dir(os.path.join(val_root, f"eval_iter_00000000_{mode}"))
        b = load_dir(os.path.join(ref_root, f"s1ref_subset64_{mode}"))
        res = compare(a, b, n_boot=10)
        print(f"[g9] step 0 {mode}: {res['n_objects']} objects paired (train {len(a)}, reference {len(b)})")
        if res["n_objects"] == 0 or res["n_objects"] != len(b):
            print(f"[g9]   FAIL: object sets differ")
            ok = False
        for k in ("psnr", "lpips", "ssim", "fg_psnr", "abs_rel"):
            if k not in res:
                continue
            v = res[k]
            line = LINES.get(k)
            flag = "" if line is None else ("ok" if v["max_abs_diff"] <= line else "FAIL")
            print(f"[g9]   {k:8s} train {v['A']:.5f} ref {v['B']:.5f} mean diff {v['diff']:+.2e} "
                  f"max|diff| {v['max_abs_diff']:.2e} {('line ' + format(line, 'g')) if line else ''} {flag}")
            if flag == "FAIL":
                ok = False
    print(f"[g9] step-0 gate {'PASS' if ok else 'FAIL'}")
    return ok


def safety(val_root, step, logs, world):
    ok = True
    txt = "\n".join(open(p, errors="replace").read() for p in logs)
    steps = [int(s) for s in re.findall(r"Forwad step:\s+(\d+)", txt)]
    skips = txt.count("NaN or Inf loss detected on at least one rank") / max(world, 1)
    n = max(steps) if steps else 0
    frac = skips / max(n, 1)
    print(f"[g9] steps logged up to {n}; skipped (non-finite loss) {skips:.0f} = {100 * frac:.3f} % (line < 1 %)")
    ok &= frac < 0.01
    rs = np.array([float(x) for x in re.findall(r"render_std: ([0-9.eE+-]+)", txt)])
    if len(rs):
        med = float(np.median(rs))
        print(f"[g9] render_std: {len(rs)} readings, median {med:.4f}, min {rs.min():.4f}, max {rs.max():.4f}, "
              f"last {rs[-1]:.4f} (line: none below 0.5 x median)")
        ok &= bool(rs.min() >= 0.5 * med)
    for mode in MODES:
        a = load_dir(os.path.join(val_root, f"eval_iter_00000000_{mode}"))
        b = load_dir(os.path.join(val_root, f"eval_iter_{step:08d}_{mode}"))
        res = compare(b, a, n_boot=1000)
        if "lpips" not in res:
            print(f"[g9] {mode}: no LPIPS at step {step}")
            ok = False
            continue
        v = res["lpips"]
        worse = v["diff"] > 0.003
        print(f"[g9] {mode}: LPIPS step {step} {v['A']:.5f} vs step 0 {v['B']:.5f}, diff {v['diff']:+.5f} "
              f"95% CI [{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}] (line: not worse by > 0.003) "
              f"{'FAIL' if worse else 'ok'}")
        for k in ("psnr", "fg_psnr", "abs_rel"):
            if k in res:
                print(f"[g9]   {k:8s} step {step} {res[k]['A']:.5f} vs step 0 {res[k]['B']:.5f} ({res[k]['diff']:+.4f})")
        ok &= not worse
    print(f"[g9] safety gate at {step}: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["step0", "safety"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--world", type=int, default=4)
    a = ap.parse_args()
    if a.what == "step0":
        ok = step0(a.args[0], a.args[1])
    else:
        ok = safety(a.args[0], int(a.args[1]), a.args[2:], a.world)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
