# Did the block-8 stage-2 renderer (r1) learn anything, or did its residual stay ~0?  (user 09-29, before any new
# experiment).  Read-only.  Per checkpoint of one config:
#  weights   how far each trainable group moved from where training started
#            out_lin    the Lin_m output layers, initialised at 0: RMS of the weights = RMS displacement
#            copied     renderer blocks (without inj_*), tgt_norm + registers, the 6 ray channels of tgt_proj, colour
#                       head, point head: initialised as copies of stage 1, so W - (stage-1 copy) is the exact
#                       displacement (the fresh model built from the config holds that copy)
#            new        inj_x / inj_rec and the mask channel of tgt_proj: random init (training's seed is not
#                       reproduced here), so only their RMS is reported
#            yardsticks sum of lr over the steps taken (the largest displacement per element Adam can make: every
#                       update ~lr, all of the same sign) and lr * sqrt(steps) (updates of random sign)
#  function  on the first --n_scenes GSO v1 validation entries (10 target views each, posed):
#            residual size per output layer m: RMS(Lin_m(t2)) / RMS(T1_m) over the refined tokens
#            renders of {stage-1 heads, trained heads} x {residual off, on}: PSNR / LPIPS (the evaluation's
#            functions) against GT, and the mean |change| against the stage-1 render, so a change splits into
#            "heads retrained" and "residual"
#  checks    (stage-1 heads, residual off) must equal the stage-1 render; (trained heads, residual on) must equal
#            the model's own forward
#   python tools_s2/s2_residual_audit.py --config <cfg> --ckpts a.pt,b.pt [--n_scenes 16] --out x.json
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
AMP = dict(device_type="cuda", dtype=torch.bfloat16)


def lr_sum(step, lr=3.5e-5, warmup=1000, anneal_from=21000, total=26000):
    """sum over update steps 1..step of the constant_anneal schedule (linear warmup, constant, 1-sqrt cooldown)"""
    s = np.arange(1, step + 1, dtype=np.float64)
    f = np.minimum(1.0, s / warmup)
    cool = s > anneal_from
    f[cool] = 1.0 - np.sqrt((s[cool] - anneal_from) / (total - anneal_from))
    return float(lr * f.sum())


def group_of(name):
    if name.startswith("renderer.out_lin"):
        return "out_lin"
    if ".inj_x." in name or ".inj_rec." in name:
        return "inj (new)"
    if name.startswith("renderer.blocks"):
        return "blocks (copied)"
    if name.startswith("renderer.tgt_norm") or name.startswith("renderer.registers"):
        return "tgt_norm+registers (copied)"
    if name.startswith("color_head"):
        return "colour head (copied)"
    if name.startswith("point_head"):
        return "point head (copied)"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpts", required=True)
    ap.add_argument("--n_scenes", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(28900 + int(os.environ.get("SLURM_JOB_ID", "3")) % 400))
        dist.init_process_group("gloo", rank=0, world_size=1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.chdir(REPO)
    from model_s2.geometry import N_REG
    from model_s2.heads_s2 import assemble_tokens, render_color, render_points
    from model_s2.stage2_wrapper import Stage2LagerNVS
    from tools_s2.s2_bench import load_cfg
    from tools_s2.s2_infer_bench import val_scenes
    from utils.metric_utils import compute_lpips, compute_psnr

    cfg = load_cfg(args.config, 1)
    name = str(cfg.exp_name)
    model = Stage2LagerNVS(cfg).cuda().eval()
    model.val_cam_cond_zero_p = 0.0
    init = {n: p.detach().float().clone() for n, p in model.named_parameters() if p.requires_grad}
    s1 = model.stage1.model
    s1_color, s1_point = s1.model.renderer.final_layer, s1.point_head
    scenes = val_scenes(cfg, args.n_scenes)
    res_all = dict(config=args.config, exp=name, n_scenes=len(scenes), ckpts={})

    def functional(tag):
        rows = []
        for b in scenes:
            with torch.no_grad(), torch.autocast(**AMP):
                inp, tgt, images, rays, cam, posed, vin = model.stage1.prepare(b, True, False, True, False, 0.0)
                p1 = model.stage1.pass1(images, rays, cam, vin)
                ctx, mask_in, _, _ = model.make_ctx(p1, inp, tgt, posed, vin)
                trays = rays[:, vin:]
                res = model.renderer(trays, mask_in.to(trays.dtype), p1["rec_rep"], ctx)
                full = model(b, target_has_input=False, is_valid=True).render
            lay = ctx.layout
            B, Vt = tgt.image.shape[:2]
            H = W = model.img
            r = {}
            for m, rm in res.items():
                T1 = p1["T1"][m]
                N1, C = T1.shape[1], T1.shape[2]
                t1 = T1.reshape(-1, C)[lay.fg_bv * N1 + N_REG + lay.fg_tok_flat].float()
                rr = rm.float()
                r[f"res_rms_m{m}"] = float(rr.square().mean().sqrt())
                r[f"res_over_T1_m{m}"] = float(rr.square().mean().sqrt() / t1.square().mean().sqrt())
            gt = tgt.image.float().reshape(B * Vt, 3, H, W)
            s1_img = p1["render"].float()
            out = {}
            for hname, ch, ph in (("s1heads", s1_color, s1_point), ("trained", model.color_head, model.point_head)):
                for rname, sc in (("res_off", 0.0), ("res_on", 1.0)):
                    with torch.no_grad(), torch.autocast(**AMP):
                        Ts = {m: assemble_tokens(p1["T1"][m], res[m] * sc, lay, 8, 8) for m in res}
                        img = render_color(ch, Ts[11], 8, B, Vt, H, W, True)
                        pts, _ = render_points(ph, Ts, trays, B, N_REG)
                    img = img.float()
                    pr = img.reshape(B * Vt, 3, H, W)
                    d = (img - s1_img).abs()
                    out[f"{hname}_{rname}"] = dict(
                        psnr=float(compute_psnr(gt, pr).mean()), lpips=float(compute_lpips(gt, pr).mean()),
                        mean_abs_change_levels=float(d.mean() * 255), share_px_changed_gt10=float((d.max(2).values > 10 / 255).float().mean()),
                        pts_rel_change=float((pts.float() - p1["points"].float()).abs().mean() / p1["points"].float().abs().mean()),
                        img=img if (hname, rname) in (("s1heads", "res_off"), ("trained", "res_on")) else None)
            r["check_s1_exact"] = float((out["s1heads_res_off"]["img"] - s1_img).abs().max())
            r["check_full_forward"] = float((out["trained_res_on"]["img"] - full.float()).abs().max())
            for k in out:
                out[k].pop("img")
            r["renders"] = out
            rows.append(r)
            del p1, res, ctx, full
            torch.cuda.empty_cache()
        agg = {k: float(np.mean([x[k] for x in rows])) for k in rows[0] if k != "renders" and not k.startswith("check")}
        agg["check_s1_exact_max"] = max(x["check_s1_exact"] for x in rows)
        agg["check_full_forward_max"] = max(x["check_full_forward"] for x in rows)
        agg["renders"] = {v: {k: float(np.mean([x["renders"][v][k] for x in rows])) for k in rows[0]["renders"][v]}
                          for v in rows[0]["renders"]}
        print(f"[audit] {name} {tag}: residual RMS / stage-1 token RMS: "
              + ", ".join(f"m{m} {agg[f'res_over_T1_m{m}']:.4f}" for m in (2, 5, 8, 11))
              + f" | checks: stage-1 path max|diff| {agg['check_s1_exact_max']:.2e}, own forward max|diff| "
              f"{agg['check_full_forward_max']:.2e}", flush=True)
        for v, d in agg["renders"].items():
            print(f"[audit]   {v:16s} PSNR {d['psnr']:.3f}  LPIPS {d['lpips']:.5f}  mean |change vs stage 1| "
                  f"{d['mean_abs_change_levels']:.3f} levels, px changed > 10 levels {100 * d['share_px_changed_gt10']:.3f}%, "
                  f"points rel change {d['pts_rel_change']:.5f}", flush=True)
        return agg

    for ck in ["init"] + args.ckpts.split(","):
        if ck == "init":
            step = 0
        else:
            sd = torch.load(ck, map_location="cpu", weights_only=True, mmap=True)
            step = int(sd.get("fwdbwd_pass_step", -1))
            missing, unexpected = model.load_state_dict(sd["model"], strict=False)
            bad = [k for k in missing if not k.startswith("loss_computer.")]
            assert not bad and not unexpected, (bad[:5], list(unexpected)[:5])
            del sd
        cur = {n: p.detach().float() for n, p in model.named_parameters() if p.requires_grad}
        acc = {}
        for n, w0 in init.items():
            w = cur[n]
            parts = []
            if n == "renderer.tgt_proj.weight":
                parts = [("tgt_proj rays (copied)", w[:, :6], w0[:, :6], True), ("tgt_proj mask (new)", w[:, 6:], w0[:, 6:], False)]
            else:
                g = group_of(n)
                if g is None:
                    raise RuntimeError(f"unclassified trainable parameter {n}")
                known = g != "inj (new)"
                parts = [(g, w, w0, known)]
            for g, a, a0, known in parts:
                s = acc.setdefault(g, dict(n=0, d2=0.0, w02=0.0, w2=0.0, dmax=0.0, known=known))
                s["n"] += a.numel()
                s["w2"] += float(a.double().square().sum())
                s["w02"] += float(a0.double().square().sum())
                if known:
                    dd = (a - a0).double()
                    s["d2"] += float(dd.square().sum())
                    s["dmax"] = max(s["dmax"], float(dd.abs().max()))
        L = lr_sum(step)
        rw = 3.5e-5 * math.sqrt(max(step, 1))
        wstats = {}
        print(f"[audit] ===== {name} step {step}: yardsticks sum(lr) {L:.4f}, lr*sqrt(steps) {rw:.5f} (per element)", flush=True)
        for g, s in acc.items():
            rms_w = math.sqrt(s["w2"] / s["n"])
            if s["known"]:
                rms_d = math.sqrt(s["d2"] / s["n"])
                rel = math.sqrt(s["d2"] / s["w02"]) if s["w02"] > 0 else float("nan")
                wstats[g] = dict(numel=s["n"], rms_disp=rms_d, rel_disp=rel, max_disp=s["dmax"], rms_w=rms_w,
                                 rms_disp_over_sum_lr=rms_d / L if L > 0 else None)
                print(f"[audit]   {g:28s} n {s['n']:>10d} | RMS displacement {rms_d:.3e} (= {rms_d / L if L else 0:.3f} x sum(lr), "
                      f"{rms_d / rw:.2f} x lr*sqrt(steps)) | relative to init {rel:.4f} | max {s['dmax']:.3e}", flush=True)
            else:
                wstats[g] = dict(numel=s["n"], rms_w=rms_w, rms_w_init_fresh=math.sqrt(s["w02"] / s["n"]))
                print(f"[audit]   {g:28s} n {s['n']:>10d} | RMS {rms_w:.3e} (a fresh random init has {math.sqrt(s['w02'] / s['n']):.3e}; "
                      f"displacement not measurable)", flush=True)
        res_all["ckpts"][str(step)] = dict(ckpt=ck, sum_lr=L, weights=wstats, function=functional(f"step {step}"))
        json.dump(res_all, open(args.out, "w"), indent=1)
    print(f"[audit] DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
