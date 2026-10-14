# Where does the stage-1 edge halo sit relative to the 8x8 token patches, and what does stage 2 change there?
# (user 09-28: "one token is 8x8; stage 2 only refines the foreground, the background is untouched, so how can it fix
# the ring?")  Stage 2 refines every target token whose 8x8 patch holds at least one GT-foreground pixel
# (model_s2/geometry.build_layout: max-pool of the mask); the colour head (per-token linear, copied from stage 1) is
# trainable, so even untouched tokens can change colour.  Per pixel class, from the validation PNGs of a stage-2 run:
#   fg            GT foreground pixels
#   bg_mixed      background pixels inside a patch that holds foreground (a refined token)
#   bg_pure_near  background pixels in a pure-background patch within 8 px of the silhouette (untouched token)
#   bg_pure_far   the other pure-background pixels
# pixel share, MSE and share of the total error at step 0 (= stage 1) and at the later step, plus how far the later
# step's output moved from step 0 inside each class (mean |step - step0| and the share of pixels that moved at all).
#   python tools_s2/s2_patch_split.py --config configs/S2P8fgdense_...yaml --exp_dir <ckpt dir> --step 14000 --out x.json
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
CLASSES = ("fg", "bg_mixed", "bg_pure_near", "bg_pure_far")


def halves(path):
    a = np.asarray(Image.open(path).convert("RGB")).astype(np.float64) / 255.0
    h = a.shape[0] // 2
    return a[:h], a[h:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.chdir(REPO)
    from data.dataset_gso_ours import GSODataset_ours
    from tools_s2.s2_bench import load_cfg
    cfg = load_cfg(args.config, 1)
    ds = GSODataset_ours(cfg)
    nin = int(cfg.training.val_dataset_cfgs.training.num_input_views)
    P = args.patch
    res = {}
    for mode in ("posed", "unposed"):
        acc = {c: dict(n=0, se0=0.0, se1=0.0, moved_sum=0.0, moved_n=0) for c in CLASSES}
        for i in range(len(ds)):
            uid = f"{i:06d}"
            p0 = f"{args.exp_dir}/eval_iter_{0:08d}_{mode}/{uid}/gt_vs_pred.png"
            if not os.path.exists(p0):
                continue
            alpha = ds[i]["alpha_mask"][nin:, 0].numpy() > 0.5
            gt, pr0 = halves(p0)
            _, pr1 = halves(f"{args.exp_dir}/eval_iter_{args.step:08d}_{mode}/{uid}/gt_vs_pred.png")
            H = gt.shape[0]
            for v in range(gt.shape[1] // H):
                m = alpha[v]
                sl = slice(v * H, (v + 1) * H)
                g, a0, a1 = gt[:, sl], pr0[:, sl], pr1[:, sl]
                patch_fg = m.reshape(H // P, P, H // P, P).any(axis=(1, 3))
                mixed = np.repeat(np.repeat(patch_fg, P, 0), P, 1)
                dist_out = ndimage.distance_transform_edt(~m)
                cls = {"fg": m, "bg_mixed": mixed & ~m, "bg_pure_near": ~mixed & (dist_out <= 8),
                       "bg_pure_far": ~mixed & (dist_out > 8)}
                se0 = ((a0 - g) ** 2).mean(-1)
                se1 = ((a1 - g) ** 2).mean(-1)
                mv = np.abs(a1 - a0).max(-1)
                for c, sel in cls.items():
                    A = acc[c]
                    A["n"] += int(sel.sum())
                    A["se0"] += float(se0[sel].sum())
                    A["se1"] += float(se1[sel].sum())
                    A["moved_sum"] += float(mv[sel].sum())
                    A["moved_n"] += int((mv[sel] > 1.5 / 255).sum())
        tot_n = sum(A["n"] for A in acc.values())
        tot0 = sum(A["se0"] for A in acc.values())
        tot1 = sum(A["se1"] for A in acc.values())
        res[mode] = {c: dict(pixel_share=A["n"] / tot_n, mse_step0=A["se0"] / A["n"], mse_step=A["se1"] / A["n"],
                             err_share_step0=A["se0"] / tot0, err_share_step=A["se1"] / tot1,
                             mean_abs_change=A["moved_sum"] / A["n"], share_pixels_changed=A["moved_n"] / A["n"])
                     for c, A in acc.items()}
        print(f"[patch] ===== {mode}: class | pixels | MSE step 0 -> step {args.step} | share of total error step 0 -> "
              f"{args.step} | mean |step - step0| (0-1) | pixels changed by > 1 grey level", flush=True)
        for c in CLASSES:
            r = res[mode][c]
            print(f"[patch]   {c:13s} {100 * r['pixel_share']:5.1f}% | {r['mse_step0']:.5f} -> {r['mse_step']:.5f} | "
                  f"{100 * r['err_share_step0']:5.1f}% -> {100 * r['err_share_step']:5.1f}% | {r['mean_abs_change']:.4f} | "
                  f"{100 * r['share_pixels_changed']:5.1f}%", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"[patch] DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
