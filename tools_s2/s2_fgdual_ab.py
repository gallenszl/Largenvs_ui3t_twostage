# Is the 09-28 fg_dual window fix (model_s2/blocks._scatter_pad) the same computation as the code that trained
# steps 0-3000 of S2P8fgdual (gather from a shared zero row)?  On the real training checkpoint and real training
# batches, forward + backward three times: new, old, new again.  Compared: loss, render, points and every trainable
# gradient.  new-vs-new measures the run-to-run nondeterminism already present (atomic accumulation in attention
# backward); old-vs-new must not exceed it.
#   python tools_s2/s2_fgdual_ab.py --config configs/S2P8fgdual_...yaml --ckpt <ckpt_...pt> --out x.json
import argparse
import json
import os
import random
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def old_fg_window(q, k, v, win, key, ctx, direction):
    """the pre-fix computation: pad by gathering from a shared zero row, select the output rows by indexing"""
    from model_s2.blocks import attend_masked
    BV, Lq, Lk = win["BV"], win[key + "_Lq"], win[key + "_Lk"]

    def src_of(dst, n_src, rows, sub=None):
        s = torch.full((rows,), n_src, dtype=torch.long, device=dst.device)          # n_src: the zero row
        s[dst] = torch.arange(dst.numel(), device=dst.device) if sub is None else sub
        return s
    qs = src_of(win[key + "_q_dst"], q.shape[0], BV * Lq).view(BV, Lq)
    kv_src = win[key + "_kv_src"]
    ks = src_of(win[key + "_kv_dst"], k.shape[0], BV * Lk, kv_src).view(BV, Lk)
    qz = torch.cat([q, q.new_zeros(1, *q.shape[1:])])
    kz = torch.cat([k, k.new_zeros(1, *k.shape[1:])])
    vz = torch.cat([v, v.new_zeros(1, *v.shape[1:])])
    o = attend_masked(qz[qs], kz[ks], vz[ks], win[key + "_mask"], ctx.masked_backend)
    return o.reshape(-1, *o.shape[2:])[win[key + "_out"]]


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n_batches", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(28200 + int(os.environ.get("SLURM_JOB_ID", "3")) % 400))
        dist.init_process_group("gloo", rank=0, world_size=1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.chdir(REPO)
    import model_s2.blocks as Bk
    from model_s2.stage2_wrapper import Stage2LagerNVS
    from tools_s2.s2_bench import cached_batches, load_cfg
    cfg = load_cfg(args.config, args.batch)
    model = Stage2LagerNVS(cfg).cuda()
    model.load_ckpt(args.ckpt)
    model.train()
    batches = cached_batches(cfg, args.n_batches)
    new_fn = Bk.fg_window

    def run(b, fn):
        Bk.fg_window = fn
        try:
            seed_all(4321)
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(b)
            out.loss_metrics.loss.backward()
            g = {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.grad is not None}
            return dict(loss=float(out.loss_metrics.loss), render=out.render.detach().float().clone(),
                        points=out.points.detach().float().clone()), g
        finally:
            Bk.fg_window = new_fn

    def diff(a, ga, b, gb):
        keys = sorted(ga)
        gmax = max(float((ga[k] - gb[k]).abs().max()) for k in keys)
        num = sum(float((ga[k].double() - gb[k].double()).norm()) ** 2 for k in keys) ** 0.5
        den = sum(float(gb[k].double().norm()) ** 2 for k in keys) ** 0.5
        return dict(loss_absdiff=abs(a["loss"] - b["loss"]), render_bitwise=bool(torch.equal(a["render"], b["render"])),
                    render_maxabs=float((a["render"] - b["render"]).abs().max()),
                    points_bitwise=bool(torch.equal(a["points"], b["points"])),
                    grads_bitwise=sum(bool(torch.equal(ga[k], gb[k])) for k in keys), grads_total=len(keys),
                    grad_max_absdiff=gmax, grad_global_rel=num / den, same_keys=set(ga) == set(gb))
    res = dict(ckpt=args.ckpt, batch=args.batch, rows=[])
    for i, b in enumerate(batches):
        n1, g1 = run(b, new_fn)
        o, go = run(b, old_fg_window)
        n2, g2 = run(b, new_fn)
        r = dict(batch=i, loss_new=n1["loss"], loss_old=o["loss"], new_vs_new=diff(n2, g2, n1, g1),
                 old_vs_new=diff(o, go, n1, g1))
        res["rows"].append(r)
        print(f"[ab] batch {i}: loss new {n1['loss']:.8f} old {o['loss']:.8f}", flush=True)
        for k in ("new_vs_new", "old_vs_new"):
            print(f"[ab]   {k}: {json.dumps(r[k])}", flush=True)
        del n1, g1, o, go, n2, g2
        torch.cuda.empty_cache()
    ok = all(r["old_vs_new"]["render_bitwise"] and r["old_vs_new"]["same_keys"]
             and r["old_vs_new"]["grad_global_rel"] <= max(2.0 * r["new_vs_new"]["grad_global_rel"], 1e-12)
             for r in res["rows"])
    res["verdict"] = "EQUIVALENT" if ok else "DIFFERENT"
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"[ab] verdict: forward bitwise equal and old-vs-new gradient difference within 2x the new-vs-new "
          f"run-to-run difference -> {res['verdict']}", flush=True)


if __name__ == "__main__":
    main()
