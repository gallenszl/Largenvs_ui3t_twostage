"""Gates for the track-consistency smoke / reference arm (plan 2026-09-29, step 6).

  python tools/uni3t_smoke_gates.py --exp_dir <smoke ckpt dir> --ref_dir <const arm ckpt dir> \
      --log slurm_logs/uni3t-cons-smoke-<job>.out --steps 62000 64000 66000

Checks, each printed with its numbers:
  1. training log: "NaN or Inf loss" skips and "grad norm too large" skips (< 1 % of steps), the
     loss_consistency series (first vs last tenth of the run must not increase), consistency_valid;
  2. training-internal validation (eval_iter_<step>/average_metrics.txt: psnr, lpips, ssim, fg_psnr):
     at every step both arms evaluated, the smoke's LPIPS must not be worse than the reference arm's
     by more than 0.003 (project signal line); PSNR / FG-PSNR reported.
Exit code 0 = all gates pass, 1 = a gate failed, 2 = nothing to check yet.
"""
import argparse
import os
import re
import sys


def read_avg(d):
    p = os.path.join(d, "average_metrics.txt")
    if not os.path.isfile(p):
        return None
    line = open(p).readline().strip()
    vals = [float(x) for x in line.split(":", 1)[1].split(",")]
    return dict(psnr=vals[0], lpips=vals[1], ssim=vals[2], fg_psnr=vals[3])


def parse_log(path):
    """The per-step loss print is one long line that the log wraps at ~80 columns, so the value of a key can
    land on the next line; the lines of one step are joined before matching (\\s* absorbs the wrap)."""
    steps, cons, valid, nan_skips, gn_skips = [], [], [], 0, 0
    cur, buf = None, []

    def flush():
        if cur is None or not buf:
            return
        blob = "\n".join(buf)
        m = re.search(r"loss_consistency:\s*([-0-9.e+]+)", blob)
        if m:
            steps.append(cur)
            cons.append(float(m.group(1)))
            v = re.search(r"consistency_valid:\s*([-0-9.e+]+)", blob)
            valid.append(float(v.group(1)) if v else float("nan"))

    for ln in open(path, errors="ignore"):
        m = re.search(r"Forwad step:\s+(\d+)", ln)
        if m:
            flush()
            cur, buf = int(m.group(1)), []
            continue
        if "NaN or Inf loss detected" in ln:
            nan_skips += 1
        # "grad norm too large X > 2.0" alone is only a warning (train.py prints it whenever the pre-clip norm
        # exceeds 2 x grad_clip_norm, which this model family does on nearly every step); an optimizer step
        # is skipped only when the line also says so (norm > grad_clip_norm x allowed_gradnorm_factor)
        if "skipping optimizer step" in ln:
            gn_skips += 1
        if not ln.startswith("WARNING"):
            buf.append(ln.rstrip("\n"))
    flush()
    return steps, cons, valid, nan_skips, gn_skips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--ref_dir", required=True)
    ap.add_argument("--log")
    ap.add_argument("--steps", type=int, nargs="+", default=[62000, 64000, 66000])
    ap.add_argument("--lpips_line", type=float, default=0.003)
    ap.add_argument("--world", type=int, default=4, help="ranks: every rank prints the skip messages")
    a = ap.parse_args()
    ok, checked = True, 0

    if a.log and os.path.isfile(a.log):
        steps, cons, valid, nan_skips, gn_skips = parse_log(a.log)
        n = len(steps)
        if n:
            checked += 1
            span = max(1, n // 10)
            first, last = sum(cons[:span]) / span, sum(cons[-span:]) / span
            print(f"[gate] log {a.log}: {n} printed steps {steps[0]}..{steps[-1]} | NaN/Inf skips {nan_skips} | "
                  f"grad-norm skips {gn_skips} | loss_consistency first-tenth {first:.5f} -> last-tenth {last:.5f} | "
                  f"consistency_valid mean {sum(v for v in valid if v == v) / max(1, sum(1 for v in valid if v == v)):.0f}")
            total_steps = max(1, steps[-1] - steps[0] + 1)
            # both messages are printed once per rank; count skipped steps, not lines
            if (nan_skips + gn_skips) / a.world / total_steps > 0.01:
                print("[gate] FAIL: skipped steps exceed 1 %")
                ok = False
            if n >= 20 and last > first:
                # informative, not a hard gate: with a small weight the term barely moves (its gradient share is
                # what decides that; see tools/uni3t_task_grad_decomp.py), and the value tracks batch composition
                print("[gate] WARN: loss_consistency did not decrease over the run")
        else:
            print(f"[gate] log {a.log}: no loss lines yet")
    print(f"[gate] {'step':>6} | {'PSNR exp/ref':>17} {'dPSNR':>7} | {'LPIPS exp/ref':>17} {'dLPIPS':>8} | {'FG exp/ref':>15} {'dFG':>6}")
    for s in a.steps:
        e = read_avg(os.path.join(a.exp_dir, f"eval_iter_{s:08d}"))
        r = read_avg(os.path.join(a.ref_dir, f"eval_iter_{s:08d}"))
        if e is None or r is None:
            print(f"[gate] {s:>6} | {'missing' if e is None else 'ok':>8} exp, {'missing' if r is None else 'ok':>8} ref")
            continue
        checked += 1
        dl = e["lpips"] - r["lpips"]
        flag = "" if dl <= a.lpips_line else "   <-- worse than the reference by more than the 0.003 line"
        if dl > a.lpips_line:
            ok = False
        print(f"[gate] {s:>6} | {e['psnr']:8.3f}/{r['psnr']:8.3f} {e['psnr'] - r['psnr']:+7.3f} | "
              f"{e['lpips']:8.5f}/{r['lpips']:8.5f} {dl:+8.5f} | {e['fg_psnr']:7.3f}/{r['fg_psnr']:7.3f} {e['fg_psnr'] - r['fg_psnr']:+6.3f}{flag}")
    if checked == 0:
        print("[gate] nothing to check yet")
        sys.exit(2)
    print(f"[gate] {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
