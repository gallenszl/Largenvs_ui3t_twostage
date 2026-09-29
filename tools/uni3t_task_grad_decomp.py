"""Per-task gradient decomposition of the three-task model on REAL training batches.

Question (user 09-25): in the layers the tasks share -- above all the renderer target stream,
which the point head reads at blocks [2, 5, 8, 11] -- is the point-map gradient larger than the
RGB gradient, and do the two point the same way?

Per batch, one forward+backward per task with identical RNG (python / numpy / torch reseeded,
so the camera-token drop and the target-view draw are the same in every pass):
    rgb    = l2_w * l2 + lpips_w * lpips + perc_w * perc       (LossComputer.forward)
    camera = weight_camera * loss_camera                        (MultiTaskLossComputer)
    point  = total - rgb - camera                               (= weight_point * point loss;
                                                                 the rgb/camera nodes get
                                                                 gradient +1 - 1 = 0 exactly)
plus one pass on the total for the first 3 batches, to check g_total == g_rgb+g_point+g_camera.
The total loss value must be identical in every pass of a batch (same forward); a mismatch is
printed as a WARNING and counted.

Per parameter group: squared norms, cross-task dot products, the fraction of weights where
|g_point| > |g_rgb|, and each task's cosine between consecutive batches (how consistent the
gradient is from batch to batch). The same batches are reused for every checkpoint.

    python tools/uni3t_task_grad_decomp.py <config> <n_batches> <ckpt> [<ckpt> ...]
"""
import importlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict as edict
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.lagernvs_wrapper import LagerNVSInRnG  # noqa: E402

TASKS = ("rgb", "point", "camera")          # + "consistency" when training.weight_consistency > 0 (set in main)
PAIRS = (("rgb", "point"), ("rgb", "camera"), ("point", "camera"))
REC_SUBS = ("cross_attn_rec", "mlp_rec", "norm1_rec", "norm2_rec")
BLK = "model.renderer.renderer_core.renderer_blocks."
OUT_DIR = Path("/home/z50057756/tmp/uni3t_taskgrad")


def groups_of(name):
    """Coarse group + (for renderer blocks) a per-block group."""
    if name.startswith("point_head."):
        return ["point_head"]
    if name.startswith("camera_head."):
        return ["camera_head"]
    if name.startswith("model.renderer.final_layer."):
        return ["rgb_head"]
    if name.startswith(BLK):
        rest = name[len(BLK):]
        blk, sub = int(rest.split(".")[0]), rest.split(".")[1]
        if sub in REC_SUBS:
            return ["renderer_rec", f"rec_blk{blk:02d}"]
        return ["renderer_target", f"target_blk{blk:02d}"]
    if name.startswith("model.renderer."):          # tgt_embedder, tgt_norm, registers
        return ["renderer_target", "target_in"]
    if name.startswith("model.reconstructor.vggt.aggregator.patch_embed."):
        return ["vggt_patch_embed"]
    if name.startswith("model.reconstructor.vggt."):
        return ["vggt"]
    if name.startswith("model.reconstructor."):     # geo_feature_connector, camera_mlp
        return ["connector"]
    return ["other"]


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)      # also seeds every CUDA device


def task_parts(cfg, lm):
    tr = cfg.training
    rgb = (tr.l2_loss_weight * lm.l2_loss + tr.lpips_loss_weight * lm.lpips_loss
           + tr.perceptual_loss_weight * lm.perceptual_loss)
    cam = tr.weight_camera * lm.loss_camera
    tot = lm.loss
    parts = {"rgb": rgb, "camera": cam, "total": tot}
    if "consistency" in TASKS:
        # track-consistency term (model/track_consistency.py); point = the rest, as before
        parts["consistency"] = float(tr.get("weight_consistency", 0.0) or 0.0) * lm["loss_consistency"]
        parts["point"] = tot - rgb - cam - parts["consistency"]
    else:
        parts["point"] = tot - rgb - cam
    return parts


def main():
    global TASKS, PAIRS
    cfg_path, n_batches, ckpts = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    cfg = edict(OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True))
    if float(cfg.training.get("weight_consistency", 0.0) or 0.0) > 0.0:
        TASKS = TASKS + ("consistency",)
        PAIRS = PAIRS + (("rgb", "consistency"), ("point", "consistency"))
        print(f"[td] track-consistency task on: weight_consistency={cfg.training.weight_consistency} "
              f"consistency={dict(cfg.training.get('consistency', {}) or {})}", flush=True)
    cfg.ddp_info = edict(global_rank=0, world_size=1, local_rank=0, device="cuda:0",
                         is_main_process=True, seed=int(cfg.training.get("seed", 777)))
    dev = "cuda"
    print(f"[td] config {cfg_path} | weights l2={cfg.training.l2_loss_weight} "
          f"lpips={cfg.training.lpips_loss_weight} perc={cfg.training.perceptual_loss_weight} "
          f"camera={cfg.training.weight_camera} point={cfg.training.weight_point} "
          f"| cam_cond_zero_p={cfg.training.cam_cond_zero_p}", flush=True)

    # the SAME batches for every checkpoint
    module, cls = cfg.training.dataset_name.rsplit(".", 1)
    ds = importlib.import_module(module).__dict__[cls](cfg)
    gen = torch.Generator().manual_seed(1234)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.training.batch_size_per_gpu, shuffle=True,
                                     num_workers=8, drop_last=True, generator=gen)
    it = iter(dl)
    batches = [next(it) for _ in range(n_batches)]
    del it, dl
    print(f"[td] {n_batches} batches of {cfg.training.batch_size_per_gpu} from {cls} ({len(ds)} items)", flush=True)

    model = LagerNVSInRnG(cfg).to(dev)
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    gmap = [groups_of(n) for n, _ in params]
    gnames = sorted({g for gs in gmap for g in gs})
    other = [n for (n, _), gs in zip(params, gmap) if gs == ["other"]]
    assert not other, f"unassigned trainable params: {other[:5]}"
    numel = {g: 0 for g in gnames}
    for (_, p), gs in zip(params, gmap):
        for g in gs:
            numel[g] += p.numel()
    print("[td] groups: " + ", ".join(f"{g}={numel[g]/1e6:.1f}M" for g in gnames
                                      if not g.startswith(("target_blk", "rec_blk"))), flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for ckpt in ckpts:
        t0 = time.time()
        sd = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=False)
        step = sd.get("fwdbwd_pass_step")
        missing, unexpected = model.load_state_dict(sd["model"], strict=False)
        assert not missing and not unexpected, (missing[:5], unexpected[:5])
        del sd
        model.train()        # training conditions: camera-token drop, grad checkpointing
        print(f"\n[td] ===== ckpt step {step}: {ckpt} (load {time.time()-t0:.0f}s) =====", flush=True)

        rows, prev = [], None
        n_mismatch = 0
        for i, cb in enumerate(batches):
            b = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in cb.items()}
            passes = TASKS + (("total",) if i < 3 else ())
            grads, lossvals, comps = {}, {}, {}
            for task in passes:
                model.zero_grad(set_to_none=True)
                seed_all(10_000 + i)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(b, exclude_bg=False)       # all ckpts are past exclude_bg_until (9999)
                parts = task_parts(cfg, out.loss_metrics)
                lossvals[task] = float(parts["total"])
                if task == "rgb":
                    lm = out.loss_metrics
                    comps = {k: float(lm[k]) for k in ("l2_loss", "perceptual_loss", "loss_camera",
                                                       "loss_conf_point", "loss_reg_point", "loss_grad_point",
                                                       "loss_consistency", "consistency_valid") if k in lm}
                    comps.update({f"part_{k}": float(v) for k, v in parts.items()})
                parts[task].backward()
                grads[task] = [p.grad.detach().float().clone() if p.grad is not None else None
                               for _, p in params]
                del out, parts
            ref = lossvals["rgb"]
            same = all(v == ref for v in lossvals.values())
            if not same:
                n_mismatch += 1
                print(f"[td] WARNING b{i:02d} total loss differs across passes: {lossvals}", flush=True)

            acc = {g: {"sq": {t: 0.0 for t in TASKS}, "dot": {f"{a}|{c}": 0.0 for a, c in PAIRS},
                       "dom": 0, "prevdot": {t: 0.0 for t in TASKS}, "resid": 0.0, "totsq": 0.0}
                   for g in gnames}
            for j, gs in enumerate(gmap):
                gtask = {t: grads[t][j] for t in TASKS}
                z = next((x for x in gtask.values() if x is not None), None)
                if z is None:            # no task reaches this tensor in this batch
                    continue
                gtask = {t: (x if x is not None else torch.zeros_like(z)) for t, x in gtask.items()}
                vals = {"sq": {t: gtask[t].pow(2).sum().item() for t in TASKS},
                        "dot": {f"{a}|{c}": (gtask[a] * gtask[c]).sum().item() for a, c in PAIRS},
                        "dom": (gtask["point"].abs() > gtask["rgb"].abs()).sum().item()}
                if "total" in grads:
                    gt = grads["total"][j] if grads["total"][j] is not None else torch.zeros_like(z)
                    vals["resid"] = (gt - sum(gtask.values())).pow(2).sum().item()
                    vals["totsq"] = gt.pow(2).sum().item()
                if prev is not None:
                    vals["prevdot"] = {t: (gtask[t] * prev[t][j].float()).sum().item() if prev[t][j] is not None
                                       else 0.0 for t in TASKS}
                for g in gs:
                    a = acc[g]
                    for t in TASKS:
                        a["sq"][t] += vals["sq"][t]
                        if prev is not None:
                            a["prevdot"][t] += vals["prevdot"][t]
                    for k in a["dot"]:
                        a["dot"][k] += vals["dot"][k]
                    a["dom"] += vals["dom"]
                    a["resid"] += vals.get("resid", 0.0)
                    a["totsq"] += vals.get("totsq", 0.0)
            new_prev = {t: [x.to(torch.bfloat16) if x is not None else None for x in grads[t]] for t in TASKS}
            prev_sq = rows[-1]["acc"] if rows else None
            row = {"i": i, "loss_same_all_passes": same, "loss_values": lossvals, "components": comps, "acc": acc}
            if prev_sq is not None:
                for g in gnames:
                    for t in TASKS:
                        den = (acc[g]["sq"][t] * prev_sq[g]["sq"][t]) ** 0.5
                        acc[g].setdefault("cos_prev", {})[t] = acc[g]["prevdot"][t] / den if den > 0 else float("nan")
            rows.append(row)
            prev = new_prev
            del grads
            tg = acc["renderer_target"]
            nr, npnt = tg["sq"]["rgb"] ** 0.5, tg["sq"]["point"] ** 0.5
            cos_rp = tg["dot"]["rgb|point"] / max(nr * npnt, 1e-30)
            if "consistency" in TASKS:
                ncs = tg["sq"]["consistency"] ** 0.5
                add_cons = (f" |g_cons|={ncs:.4f} cons/point={ncs / max(npnt, 1e-30):.3f} "
                            f"cos(p,c)={tg['dot']['point|consistency'] / max(npnt * ncs, 1e-30):+.3f}")
            else:
                add_cons = ""
            add = (f" | additivity resid/total (all params) = "
                   f"{(sum(acc[g]['resid'] for g in gnames if not g.startswith(('target_blk','rec_blk','target_in')))/max(sum(acc[g]['totsq'] for g in gnames if not g.startswith(('target_blk','rec_blk','target_in'))),1e-30))**0.5:.2e}"
                   if i < 3 else "")
            print(f"[td] step {step} b{i:02d} same_loss={same} | target stream |g_rgb|={nr:.4f} |g_point|={npnt:.4f} "
                  f"cos={cos_rp:+.3f}{add_cons}{add} | {time.time()-t0:.0f}s", flush=True)

        # ---- summary over batches ----
        def med(xs):
            xs = sorted(x for x in xs if x == x)
            n = len(xs)
            return float("nan") if n == 0 else (xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2]))
        summ = {}
        for g in gnames:
            s = {t: [r["acc"][g]["sq"][t] for r in rows] for t in TASKS}
            d = {k: [r["acc"][g]["dot"][k] for r in rows] for k in rows[0]["acc"][g]["dot"]}
            e = {t: sum(s[t]) / len(rows) for t in TASKS}
            etot = sum(e.values())
            summ[g] = {
                "numel": numel[g],
                "norm_median": {t: med([x ** 0.5 for x in s[t]]) for t in TASKS},
                "energy_share": {t: e[t] / etot if etot > 0 else float("nan") for t in TASKS},
                "cos_median": {k: med([d[k][r] / max((s[k.split('|')[0]][r] * s[k.split('|')[1]][r]) ** 0.5, 1e-30)
                                       for r in range(len(rows))]) for k in d},
                "point_bigger_frac_median": med([r["acc"][g]["dom"] / numel[g] for r in rows]),
                "cos_prev_median": {t: med([r["acc"][g].get("cos_prev", {}).get(t, float("nan")) for r in rows])
                                    for t in TASKS},
            }
        additivity = (sum(r["acc"][g]["resid"] for r in rows[:3] for g in gnames
                          if not g.startswith(("target_blk", "rec_blk", "target_in")))
                      / max(sum(r["acc"][g]["totsq"] for r in rows[:3] for g in gnames
                                if not g.startswith(("target_blk", "rec_blk", "target_in"))), 1e-30)) ** 0.5
        print(f"\n[td] ===== step {step}: medians over {len(rows)} batches | loss identical across passes in "
              f"{len(rows)-n_mismatch}/{len(rows)} batches | additivity ||g_tot-sum||/||g_tot|| = {additivity:.2e} =====")
        print(f"[td] {'group':<17}{'params':>8} | {'|g_rgb|':>9}{'|g_point|':>10}{'|g_cam|':>9} | "
              f"{'E_rgb':>6}{'E_pt':>6}{'E_cam':>6} | {'cos(r,p)':>9}{'cos(r,c)':>9} | {'p>r':>5} | "
              f"{'cons_rgb':>9}{'cons_pt':>8}{'cons_cam':>9}")
        order = ["vggt_patch_embed", "vggt", "connector", "renderer_target", "renderer_rec", "rgb_head",
                 "point_head", "camera_head", "target_in"] + [g for g in gnames if g.startswith("target_blk")] \
            + [g for g in gnames if g.startswith("rec_blk")]
        for g in order:
            if g not in summ:
                continue
            x = summ[g]
            print(f"[td] {g:<17}{x['numel']/1e6:>7.1f}M | {x['norm_median']['rgb']:>9.4f}{x['norm_median']['point']:>10.4f}"
                  f"{x['norm_median']['camera']:>9.4f} | {100*x['energy_share']['rgb']:>5.1f}%{100*x['energy_share']['point']:>5.1f}%"
                  f"{100*x['energy_share']['camera']:>5.1f}% | {x['cos_median']['rgb|point']:>+9.3f}{x['cos_median']['rgb|camera']:>+9.3f}"
                  f" | {100*x['point_bigger_frac_median']:>4.0f}% | {x['cos_prev_median']['rgb']:>+9.3f}"
                  f"{x['cos_prev_median']['point']:>+8.3f}{x['cos_prev_median']['camera']:>+9.3f}", flush=True)
        if "consistency" in TASKS:
            print(f"[td] {'group':<17} | {'|g_cons|':>9} {'E_cons':>7} {'cons/point':>11} {'cos(p,c)':>9} {'cos(r,c)':>9} {'cons_prev':>9}")
            for g in order:
                if g not in summ:
                    continue
                x = summ[g]
                print(f"[td] {g:<17} | {x['norm_median']['consistency']:>9.4f} {100*x['energy_share']['consistency']:>6.1f}% "
                      f"{x['norm_median']['consistency'] / max(x['norm_median']['point'], 1e-30):>11.3f} "
                      f"{x['cos_median']['point|consistency']:>+9.3f} {x['cos_median']['rgb|consistency']:>+9.3f} "
                      f"{x['cos_prev_median']['consistency']:>+9.3f}", flush=True)
        comp_med = {k: med([r["components"][k] for r in rows]) for k in rows[0]["components"]}
        print(f"[td] loss components (median): " + " ".join(f"{k}={v:.4f}" for k, v in comp_med.items()))
        print(f"[td] peak mem {torch.cuda.max_memory_allocated()/2**30:.1f} GiB | {time.time()-t0:.0f}s", flush=True)
        out_f = OUT_DIR / f"taskgrad_step{step}.json"
        json.dump({"config": cfg_path, "ckpt": ckpt, "step": step, "n_batches": len(rows),
                   "n_loss_mismatch": n_mismatch, "additivity": additivity, "summary": summ,
                   "components_median": comp_med,
                   "rows": [{k: v for k, v in r.items() if k != "acc"} for r in rows]},
                  open(out_f, "w"), indent=1)
        print(f"[td] wrote {out_f}", flush=True)
    print("[td] DONE", flush=True)


if __name__ == "__main__":
    main()
