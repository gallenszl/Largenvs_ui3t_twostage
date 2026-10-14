# Is the stage-2 sparse attention the same function as full attention with -inf on the disallowed pairs?
# (user 09-27).  Stage-2 attention call sites, every block (model_s2/blocks.py):
#   self  target self-attention               attend_blockdiag  FA3 over packed per-view segments
#   comp  target -> scene block summaries     attend_blockdiag  FA3, each view's segment vs its own summaries
#   sel   target -> scene tokens (fwd table)  attend_masked     FlexAttention, BlockMask from the boolean table
#   rev   scene -> target tokens (rev table)  attend_masked     FlexAttention (bidirectional blocks only)
# fg_dense / fg_dual structures (model.stage2.structure) instead of comp / sel / rev:
#   glob_t2s / glob_s2t  foreground-global cross-attention, both directions   attend_blockdiag  FA3
#   win_t2s / win_s2t    fg_dual's geometric-window branch, both directions    attend_masked     FlexAttention
# Reference for every site: ONE dense score matrix over the site's complete key set (self / comp: the whole packed
# sequence of all views, pairs from different views masked), masked_fill(-inf) where the pair is not allowed,
# softmax, times V -- plain einsum, built here from the raw boolean table / the raw segment lengths; it uses neither
# the MaskTable encodings, nor the BlockMask, nor xformers' BlockDiagonalMask.  WHICH pairs are allowed (the table)
# is the specification both sides share and is not under test here (tests/test_s2_geometry checks the tables
# against a per-pixel brute force, and the pack <-> pad round trip).
#
# Part A, operator level, on the real q / k / v / tables of one eval forward (real data, smoke checkpoint):
#   prod16  production kernel on the bf16 inputs the model feeds it vs the fp64 dense-masked reference
#           (output, dq, dk, dv; one fixed random upstream gradient)
#   sdpa16  yardstick: torch SDPA over the same full key set with the same boolean mask, bf16 -- what "full
#           attention + -inf" itself costs in bf16; prod16 is judged against it
#   fp32    sel / rev: FlexAttention on fp32 inputs (TF32 off) vs fp64 dense -- the semantics with bf16 taken out
#   nomask  negative control: fp64 dense attention with NO mask vs the fp64 masked reference (must be far off,
#           otherwise the comparison could not fail)
#   null    rows whose only allowed key is an all-zero key/value row: production output must be exactly 0
# Part B, end to end: every stage-2 attention call replaced (dense32 = fp32 dense-masked einsum, sdpa16 = bf16
#   SDPA with the boolean mask) vs production: renders, points, loss and every trainable gradient (one training
#   batch, train mode, forward + backward) and validation renders (eval).  dense32 is the reference.
# Pre-registered pass rules:
#   A1  fp32 rel-L2 <= 1e-5 at every sel / rev call
#   A2  prod16 rel-L2 (output and each gradient) <= 2 x sdpa16 rel-L2, at every call
#   A3  nomask rel-L2 >= 0.05 at every call
#   A4  null rows: max |prod16 output| == 0
#   B1  renders, points, global gradient: rel-L2(prod, dense32) <= 2 x rel-L2(sdpa16, dense32)
#   B0  (precondition) the frozen stage-1 render is bitwise identical in every run
#   python tools_s2/s2_dense_equiv.py --config configs/S2P8_...yaml --s2_ckpt <ckpt> --out x.json
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import model_s2.blocks as Bk  # noqa: E402

ORIG = dict(masked=Bk.attend_masked, blockdiag=Bk.attend_blockdiag, t2s=Bk.target_to_scene,
            s2t=Bk.scene_to_target, fg_global=Bk.fg_global, fg_window=Bk.fg_window)
ST = dict(mode="prod", site=None, blk=-1, check=False, label="")
RECS = []


def rel(a, b):
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


def maxabs(a, b):
    return float((a.detach().double() - b.detach().double()).abs().max())


def seg_allowed(qlens, klens, dev):
    """pair (i, j) of the packed sequences allowed iff query i and key j belong to the same segment -> [1, Tq, Tk]"""
    sq = torch.repeat_interleave(torch.arange(len(qlens), device=dev), torch.as_tensor(list(qlens), device=dev))
    sk = torch.repeat_interleave(torch.arange(len(klens), device=dev), torch.as_tensor(list(klens), device=dev))
    return (sq[:, None] == sk[None, :])[None]


def dense_ref(q, k, v, allowed, dout=None, use_mask=True, cd=torch.float64):
    """q [B, Lq, H, D], k / v [B, Lk, H, D], allowed bool [B, Lq, Lk]; one head at a time.
    -> output [B, Lq, H, D] in cd, and (dq, dk, dv) in cd for the upstream gradient dout (or None)."""
    B, Lq, H, D = q.shape
    scale = 1.0 / math.sqrt(D)
    o = torch.empty(B, Lq, H, D, dtype=cd, device=q.device)
    g = [torch.empty(t.shape, dtype=cd, device=q.device) for t in (q, k, v)] if dout is not None else None
    for h in range(H):
        with torch.enable_grad() if dout is not None else torch.no_grad():
            qh, kh, vh = [t[:, :, h].detach().to(cd).requires_grad_(dout is not None) for t in (q, k, v)]
            s = torch.einsum("bqd,bkd->bqk", qh, kh) * scale
            if use_mask:
                s = s.masked_fill(~allowed, float("-inf"))
            oh = torch.einsum("bqk,bkd->bqd", torch.softmax(s, -1), vh)
        if dout is not None:
            gq, gk, gv = torch.autograd.grad(oh, (qh, kh, vh), dout[:, :, h].to(cd))
            g[0][:, :, h], g[1][:, :, h], g[2][:, :, h] = gq, gk, gv
        o[:, :, h] = oh.detach()
    return o, g


def dense_diff(q, k, v, allowed, cd=torch.float32):
    """differentiable dense-masked attention for the end-to-end run, one head at a time, output in q.dtype"""
    B, Lq, H, D = q.shape
    scale = 1.0 / math.sqrt(D)
    outs = []
    with torch.autocast("cuda", enabled=False):
        for h in range(H):
            s = torch.einsum("bqd,bkd->bqk", q[:, :, h].to(cd), k[:, :, h].to(cd)) * scale
            s = s.masked_fill(~allowed, float("-inf"))
            outs.append(torch.einsum("bqk,bkd->bqd", torch.softmax(s, -1), v[:, :, h].to(cd)))
    return torch.stack(outs, 2).to(q.dtype)


def sdpa_masked(q, k, v, allowed):
    """torch SDPA over the full key set, boolean mask (True = allowed), in q's dtype"""
    H = q.shape[2]
    with torch.autocast("cuda", enabled=False):
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.to(q.dtype).transpose(1, 2),
                                           v.to(q.dtype).transpose(1, 2),
                                           attn_mask=allowed[:, None].expand(-1, H, -1, -1))
    return o.transpose(1, 2)


def check_call(kind, site, q, k, v, allowed, prod_fn, fp32_fn):
    t0 = time.time()
    torch.manual_seed(1000 + len(RECS))
    dout = torch.randn(q.shape, device=q.device, dtype=torch.float32).to(q.dtype)
    rec = dict(label=ST["label"], kind=kind, site=site, block=ST["blk"], q_shape=list(q.shape),
               k_shape=list(k.shape), dtype=str(q.dtype), allowed_frac=float(allowed.float().mean()))
    with torch.autocast("cuda", enabled=False):
        ref_o, ref_g = dense_ref(q, k, v, allowed, dout)
        nm_o, _ = dense_ref(q, k, v, allowed, None, use_mask=False)
        with torch.enable_grad():
            qp, kp, vp = [t.detach().clone().requires_grad_(True) for t in (q, k, v)]
            po = prod_fn(qp, kp, vp)
            po.backward(dout)
            qy, ky, vy = [t.detach().clone().requires_grad_(True) for t in (q, k, v)]
            yo = sdpa_masked(qy, ky, vy, allowed)
            yo.backward(dout)
        with torch.no_grad():
            f32 = fp32_fn(q.float(), k.float(), v.float()) if fp32_fn is not None else None
    for name, o_, gs in (("prod16", po, (qp.grad, kp.grad, vp.grad)), ("sdpa16", yo, (qy.grad, ky.grad, vy.grad))):
        rec[name] = dict(out=rel(o_, ref_o), out_maxabs=maxabs(o_, ref_o), dq=rel(gs[0], ref_g[0]),
                         dk=rel(gs[1], ref_g[1]), dv=rel(gs[2], ref_g[2]),
                         finite=bool(torch.isfinite(o_).all()) and all(bool(torch.isfinite(x).all()) for x in gs))
    rec["nomask"] = rel(nm_o, ref_o)
    if f32 is not None:
        rec["fp32"] = dict(out=rel(f32, ref_o), out_maxabs=maxabs(f32, ref_o))
    # rows whose single allowed key has an all-zero key and value (the null key / padding): output must be 0
    cnt = allowed.sum(-1)
    col = allowed.half().argmax(-1)
    zero_kv = (k.detach().abs().amax((-1, -2)) == 0) & (v.detach().abs().amax((-1, -2)) == 0)
    null = (cnt == 1) & torch.gather(zero_kv, 1, col)
    rec["null_rows"] = int(null.sum())
    rec["null_max_out"] = float(po.detach().abs().amax((-1, -2))[null].max()) if rec["null_rows"] else 0.0
    rec["null_max_ref"] = float(ref_o.abs().amax((-1, -2))[null].max()) if rec["null_rows"] else 0.0
    if kind == "masked":
        B_, Lq, Lk = allowed.shape
        tiles = allowed.view(B_, Lq // 128, 128, Lk // 128, 128).any(4).any(2)
        rec["flex_tiles_computed_frac"] = float(tiles.float().mean())
    rec["sec"] = time.time() - t0
    RECS.append(rec)
    p, y = rec["prod16"], rec["sdpa16"]
    print(f"[deq] A {rec['label']} blk {rec['block']:2d} {site:8s} q{rec['q_shape']} k{rec['k_shape']} "
          f"allowed {rec['allowed_frac']:.4f} | out prod {p['out']:.2e} sdpa {y['out']:.2e} | dq {p['dq']:.2e}/"
          f"{y['dq']:.2e} dk {p['dk']:.2e}/{y['dk']:.2e} dv {p['dv']:.2e}/{y['dv']:.2e} | "
          f"fp32 {rec.get('fp32', {}).get('out', float('nan')):.2e} | nomask {rec['nomask']:.3f} | "
          f"null {rec['null_rows']} max {rec['null_max_out']:.1e} | {rec['sec']:.0f}s", flush=True)


# ------------------------------------------------------------------------------------ patched call sites
def w_t2s(*a, **k):
    ST["site"] = "t2s"
    try:
        return ORIG["t2s"](*a, **k)
    finally:
        ST["site"] = None


def w_s2t(*a, **k):
    ST["site"] = "s2t"
    try:
        return ORIG["s2t"](*a, **k)
    finally:
        ST["site"] = None


def w_fg_global(q, k, v, q_seqlens, kv_seqlens, ctx, direction):
    ST["site"] = "glob_" + direction
    try:
        return ORIG["fg_global"](q, k, v, q_seqlens, kv_seqlens, ctx, direction)
    finally:
        ST["site"] = None


def w_fg_window(q, k, v, win, key, ctx, direction):
    ST["site"] = "win_" + direction
    try:
        return ORIG["fg_window"](q, k, v, win, key, ctx, direction)
    finally:
        ST["site"] = None


def w_blockdiag(q, k, v, q_seqlens, kv_seqlens, backend="fa3"):
    if ST["site"] is None:
        ST["blk"] += 1                              # target self-attention opens every block
        site = "self"
    elif ST["site"] == "t2s":
        site = "comp"
    else:
        site = ST["site"]                           # glob_t2s / glob_s2t
    mode = ST["mode"]
    if mode == "prod":
        o = ORIG["blockdiag"](q, k, v, q_seqlens, kv_seqlens, backend)
        if ST["check"]:
            check_call("blockdiag", site, q[None], k[None], v[None], seg_allowed(q_seqlens, kv_seqlens, q.device),
                       lambda a, b, c: ORIG["blockdiag"](a[0], b[0], c[0], q_seqlens, kv_seqlens, backend)[None],
                       None)
        return o
    allowed = seg_allowed(q_seqlens, kv_seqlens, q.device)
    if mode == "dense32":
        return dense_diff(q[None], k[None], v[None], allowed)[0]
    if mode == "sdpa16":
        return sdpa_masked(q[None], k[None], v[None], allowed)[0]
    raise ValueError(mode)


def w_masked(q, k, v, mask, backend=None):
    site = {"t2s": "sel", "s2t": "rev"}.get(ST["site"], ST["site"])       # win_t2s / win_s2t for fg_dual
    mode = ST["mode"]
    if mode == "prod":
        o = ORIG["masked"](q, k, v, mask, backend)
        if ST["check"]:
            assert mask.table is not None, "needs attn_backend flex (keeps the boolean table)"
            check_call("masked", site, q, k, v, mask.table,
                       lambda a, b, c: ORIG["masked"](a, b, c, mask, backend),
                       lambda a, b, c: ORIG["masked"](a, b, c, mask, "flex"))
        return o
    if mode == "dense32":
        return dense_diff(q, k, v, mask.table)
    if mode == "sdpa16":
        return sdpa_masked(q, k, v, mask.table)
    raise ValueError(mode)


def install():
    Bk.target_to_scene, Bk.scene_to_target = w_t2s, w_s2t
    Bk.fg_global, Bk.fg_window = w_fg_global, w_fg_window
    Bk.attend_blockdiag, Bk.attend_masked = w_blockdiag, w_masked


# ------------------------------------------------------------------------------------ runs
def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def forward(model, b, train):
    if train:
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(b)
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return model(b, target_has_input=False, is_valid=True)


def part_a(model, b, label, is_val):
    ST.update(mode="prod", check=True, blk=-1, label=label)
    n0 = len(RECS)
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        if is_val:
            model(b, target_has_input=False, is_valid=True)
        else:
            model(b)
    ST["check"] = False
    return RECS[n0:]


def e2e(model, b, train, mode):
    ST.update(mode=mode, check=False, blk=-1)
    seed_all(4321)
    model.zero_grad(set_to_none=True)
    out = forward(model, b, train)
    grads = None
    if train:
        out.loss_metrics.loss.backward()
        grads = {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.grad is not None}
    ST["mode"] = "prod"
    r = dict(render=out.render.detach().float().clone(), points=out.points.detach().float().clone(),
             render_s1=out.render_s1.detach().float().clone(), loss=float(out.loss_metrics.loss))
    return r, grads


def psnr(a, b):
    mse = float(((a.double() - b.double()) ** 2).mean())
    return float("inf") if mse == 0 else -10.0 * math.log10(mse)


def cmp_runs(r, ref, g=None, gref=None):
    d = dict(render=rel(r["render"], ref["render"]), render_maxabs=maxabs(r["render"], ref["render"]),
             render_psnr_db=psnr(r["render"], ref["render"]), points=rel(r["points"], ref["points"]),
             loss_absdiff=abs(r["loss"] - ref["loss"]), s1_identical=bool(torch.equal(r["render_s1"], ref["render_s1"])))
    if gref is not None:
        keys = [n for n in gref if float(gref[n].norm()) > 0]
        num = math.sqrt(sum(float((g[n].double() - gref[n].double()).norm()) ** 2 for n in keys))
        den = math.sqrt(sum(float(gref[n].double().norm()) ** 2 for n in keys))
        per = sorted(((rel(g[n], gref[n]), n) for n in keys), reverse=True)
        d.update(grad_global=num / den, grad_median=float(np.median([p for p, _ in per])), grad_max=per[0][0],
                 grad_worst=[(n, p) for p, n in per[:5]], grad_tensors=len(keys),
                 grad_missing=sorted(set(gref) ^ set(g)))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--s2_ckpt", required=True)
    ap.add_argument("--n_val", type=int, default=2)
    ap.add_argument("--train_batch", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(28900 + int(os.environ.get("SLURM_JOB_ID", "5")) % 400))
        dist.init_process_group("gloo", rank=0, world_size=1)
    torch.backends.cuda.matmul.allow_tf32 = False       # fp32 legs must be true fp32 (FlexAttention reads this)
    torch.backends.cudnn.allow_tf32 = False
    os.chdir(REPO)
    from model_s2.stage2_wrapper import Stage2LagerNVS
    from tools_s2.s2_bench import cached_batches, load_cfg
    from tools_s2.s2_infer_bench import val_scenes
    cfg = load_cfg(args.config, args.train_batch)
    model = Stage2LagerNVS(cfg).cuda()
    model.load_ckpt(args.s2_ckpt)
    model.val_cam_cond_zero_p = 0.0
    assert model.masked_backend == "flex" and model.dense_backend == "fa3", (model.masked_backend, model.dense_backend)
    model.stage1.model.process_val_data.config.training.num_target_views = 10
    vals = val_scenes(cfg, args.n_val)
    trains = cached_batches(cfg, 1)
    install()
    res = dict(config=args.config, s2_ckpt=args.s2_ckpt, train_batch=args.train_batch, n_val=args.n_val)

    # ---- Part A
    recs = part_a(model, vals[0], "val0", True) + part_a(model, trains[0], "train0", False)
    res["A_calls"] = recs
    rules = {}
    masked = [r for r in recs if r["kind"] == "masked"]
    rules["A1_fp32_le_1e-5"] = all(r["fp32"]["out"] <= 1e-5 for r in masked)
    rules["A2_prod_le_2x_sdpa"] = all(r["prod16"][q] <= 2.0 * r["sdpa16"][q] for r in recs
                                      for q in ("out", "dq", "dk", "dv"))
    rules["A3_nomask_ge_0.05"] = all(r["nomask"] >= 0.05 for r in recs)
    rules["A4_null_rows_exact_0"] = all(r["null_max_out"] == 0.0 for r in recs)
    rules["A_finite"] = all(r["prod16"]["finite"] for r in recs)
    worst = {}
    for q in ("out", "dq", "dk", "dv"):
        r = max(recs, key=lambda r_: r_["prod16"][q] / max(r_["sdpa16"][q], 1e-300))
        worst[q] = dict(ratio=r["prod16"][q] / max(r["sdpa16"][q], 1e-300), prod16=r["prod16"][q],
                        sdpa16=r["sdpa16"][q], where=f"{r['label']} blk {r['block']} {r['site']}")
    res["A_worst_ratio"] = worst
    print("[deq] A worst prod16 / sdpa16 ratio per quantity:", json.dumps(worst), flush=True)

    # ---- Part B
    res["B"] = {}
    for i, b in enumerate(vals):
        runs = {m: e2e(model, b, False, m)[0] for m in ("dense32", "prod", "sdpa16")}
        res["B"][f"val{i}"] = {m: cmp_runs(runs[m], runs["dense32"]) for m in ("prod", "sdpa16")}
        print(f"[deq] B val{i}:", json.dumps(res["B"][f"val{i}"]), flush=True)
        del runs
    runs, grads = {}, {}
    for m in ("dense32", "prod", "sdpa16"):
        runs[m], grads[m] = e2e(model, trains[0], True, m)
    res["B"]["train0"] = {m: cmp_runs(runs[m], runs["dense32"], grads[m], grads["dense32"]) for m in ("prod", "sdpa16")}
    print("[deq] B train0:", json.dumps(res["B"]["train0"]), flush=True)
    rng = runs["dense32"]["render"]
    res["B"]["render_range_dense32"] = [float(rng.min()), float(rng.max())]
    del runs, grads
    ok_b, ok_s1 = True, True
    for key, d in res["B"].items():
        if not isinstance(d, dict) or "prod" not in d:
            continue
        p, y = d["prod"], d["sdpa16"]
        ok_s1 &= p["s1_identical"] and y["s1_identical"]
        for q in ("render", "points") + (("grad_global",) if "grad_global" in p else ()):
            ok_b &= p[q] <= 2.0 * y[q]
    rules["B0_stage1_identical"] = ok_s1
    rules["B1_prod_le_2x_sdpa"] = ok_b
    res["rules"] = rules
    for k_, v_ in rules.items():
        print(f"[deq] RULE {k_}: {'PASS' if v_ else 'FAIL'}", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"[deq] {'ALL PASS' if all(rules.values()) else 'SOME FAIL'} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
