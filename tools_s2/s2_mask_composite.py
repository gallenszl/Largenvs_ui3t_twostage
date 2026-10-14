# How much of stage 2's gain is the GT target silhouette it is given?  (user 09-28: "is stage 1 underestimated?")
# Give stage 1 the same information after the fact: composite its prediction onto the white background with the GT
# alpha, pred' = pred * a + 1 * (1 - a), i.e. keep its colours inside the silhouette and white outside.  Same for the
# stage-2 prediction.  Two alphas: the loader's hard mask (nearest, what stage 2 is fed) and a soft alpha (the raw
# 512 px alpha box-averaged to 256, matching the anti-aliased GT edge).  Metrics per view (PSNR, LPIPS-VGG as the
# evaluation computes them, foreground PSNR), averaged over views then over the 64 entries, plus MSE by
# silhouette band.  Predictions and GT are the 8-bit validation PNGs of the run.
#   python tools_s2/s2_mask_composite.py --config configs/S2P8fgdense_...yaml --exp_dir <ckpt dir> --step 14000 --out x.json
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def halves(path):
    a = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    h = a.shape[0] // 2
    return a[:h], a[h:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.chdir(REPO)
    import lpips
    from data.dataset_gso_ours import GSODataset_ours
    from tools_s2.s2_bench import load_cfg
    cfg = load_cfg(args.config, 1)
    ds = GSODataset_ours(cfg)
    nin = int(cfg.training.val_dataset_cfgs.training.num_input_views)
    root = cfg.training.val_dataset_cfgs.root_dir
    lp = lpips.LPIPS(net="vgg").cuda().eval()
    variants = ("raw", "hard", "soft")
    steps = (0, args.step)
    rows = {m: {s: {v: dict(psnr=[], lpips=[], fg_psnr=[]) for v in variants} for s in steps} for m in ("posed", "unposed")}
    band = {m: {s: {v: dict(ring_se=0.0, ring_n=0, in_se=0.0, in_n=0) for v in variants} for s in steps}
            for m in ("posed", "unposed")}
    for i in range(len(ds)):
        uid = f"{i:06d}"
        if not os.path.isdir(f"{args.exp_dir}/eval_iter_{0:08d}_posed/{uid}"):
            continue
        item = ds[i]
        name = item["scene_name"]
        hard = item["alpha_mask"][nin:, 0].numpy().astype(np.float32)                   # [Vt, H, W]
        H = hard.shape[-1]
        tgt_idx = ds.view_selector(i)[nin:]
        soft = np.stack([np.asarray(Image.open(f"{root}/{name}/{j:03d}.png").getchannel("A").resize((H, H), Image.BOX),
                                    dtype=np.float32) / 255.0 for j in tgt_idx])
        for m in ("posed", "unposed"):
            for s in steps:
                gt, pred = halves(f"{args.exp_dir}/eval_iter_{s:08d}_{m}/{uid}/gt_vs_pred.png")
                Vt = gt.shape[1] // H
                gts = np.stack([gt[:, v * H:(v + 1) * H] for v in range(Vt)])
                prs = np.stack([pred[:, v * H:(v + 1) * H] for v in range(Vt)])
                outs = dict(raw=prs, hard=prs * hard[..., None] + (1 - hard[..., None]),
                            soft=prs * soft[..., None] + (1 - soft[..., None]))
                g_t = torch.from_numpy(gts).permute(0, 3, 1, 2).cuda() * 2 - 1
                for vname, o in outs.items():
                    se = ((o - gts) ** 2).mean(-1)                                           # [Vt, H, W]
                    R = rows[m][s][vname]
                    R["psnr"] += (-10 * np.log10(np.maximum(se.mean((1, 2)), 1e-12))).tolist()
                    fgm = hard > 0.5
                    R["fg_psnr"] += [float(-10 * np.log10(max(se[v][fgm[v]].mean(), 1e-12))) for v in range(Vt) if fgm[v].any()]
                    with torch.no_grad():
                        R["lpips"] += lp(torch.from_numpy(o).permute(0, 3, 1, 2).cuda() * 2 - 1, g_t).flatten().tolist()
                    for v in range(Vt):
                        m_ = hard[v] > 0.5
                        sd = np.where(m_, -ndimage.distance_transform_edt(m_), ndimage.distance_transform_edt(~m_))
                        ring = np.abs(sd) <= 4
                        inner = sd < -4
                        B = band[m][s][vname]
                        B["ring_se"] += float(se[v][ring].sum()); B["ring_n"] += int(ring.sum())
                        B["in_se"] += float(se[v][inner].sum()); B["in_n"] += int(inner.sum())
        print(f"[mask] {uid} {name} done", flush=True)
    res = dict(step=args.step, modes={})
    for m in ("posed", "unposed"):
        res["modes"][m] = {}
        for s in steps:
            res["modes"][m][str(s)] = {v: dict(psnr=float(np.mean(rows[m][s][v]["psnr"])), lpips=float(np.mean(rows[m][s][v]["lpips"])),
                                               fg_psnr=float(np.mean(rows[m][s][v]["fg_psnr"])),
                                               ring_mse=band[m][s][v]["ring_se"] / band[m][s][v]["ring_n"],
                                               interior_mse=band[m][s][v]["in_se"] / band[m][s][v]["in_n"])
                                       for v in variants}
    json.dump(res, open(args.out, "w"), indent=1)
    for m in ("posed", "unposed"):
        print(f"[mask] ===== {m}   (PSNR over views, LPIPS-VGG, fg PSNR, MSE within 4 px of the silhouette, MSE deeper "
              f"than 4 px inside)", flush=True)
        for s in steps:
            for v in variants:
                r = res["modes"][m][str(s)][v]
                print(f"[mask]   step {s:5d} {v:4s}: PSNR {r['psnr']:.3f} | LPIPS {r['lpips']:.5f} | fg PSNR {r['fg_psnr']:.3f} | "
                      f"ring MSE {r['ring_mse']:.5f} | interior MSE {r['interior_mse']:.5f}", flush=True)
    print(f"[mask] DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
