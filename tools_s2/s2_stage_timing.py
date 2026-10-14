# Where does inference time go, stage by stage?  (user 09-27: reconstruction = DINOv2 + aggregator, r0 = stage-1
# renderer, r1 = stage-2 renderer; the goal is r1 as cheap as possible and at least cheaper than r0.)
# One process, one GPU, batch 1, 1 and 10 target views, block-8 config with a stage-2 checkpoint (non-zero residual).
# Segments = the stage-2 forward split as tools_s2/s2_graph_bench.Pipe, plus the reconstruction alone:
#   prep      data preparation (rays, camera tokens)                                        eager only
#   recon     VGGT reconstructor = DINOv2 + aggregator on the 4 input views
#   s1pass    stage-1 pass 1 = recon + r0 + camera head + point head  (stage-1 heads = s1pass - recon - r0)
#   r0        stage-1 renderer on the target views
#   rerender  stage-1 pass 2 = r0 on the 4 input cameras + point head (needed only by window / visibility tables)
#   tables    layout + forward / reverse mask tables + x_prior                              eager only
#   r1        stage-2 renderer
#   s2heads   stage-2 colour / point heads
# Modes (device sync before and after each timed call; median of --reps after --warm calls):
#   E_prod    eager, fp32 weights, repository code as is.  Contains host-device syncs: RoPE reads
#             int(positions.max()) at every attention call, the aggregator builds zeros on the CPU and copies them,
#             the block-diagonal mask is built with a host -> device copy at every call
#   E_nosync  eager, fp32 weights, those three removed in-process (same values; tools_s2/s2_graph_bench.py patches)
#   E_bf16    eager, the Linear / Conv / RMSNorm weights stored in bf16 (exactly what autocast casts them to at every
#             call, so the maths is unchanged) + no syncs
#   G_fp32    CUDA Graph replay, fp32 weights (the per-call weight casts are recorded inside the graph)
#   G_bf16    CUDA Graph replay, bf16 weights  = the segment's pure GPU time
# Decomposition: sync cost = E_prod - E_nosync; weight casts, host + GPU = E_nosync - E_bf16; weight casts, GPU only
#   = G_fp32 - G_bf16; launch / host overhead = E_bf16 - G_bf16; GPU compute = G_bf16.
# Profiled (torch.profiler, one call per segment and scene) in E_prod and E_bf16: kernels counted op by op as in
#   tools_s2/s2_launch_profile.py (comparable with the H200 job 137900), kernels launched by aten::copy_, host reads /
#   syncs / pageable copies.  GPU time is always the graph replay, never a profiler sum.
# r1 candidates (current + the timing-only prototypes fg_dense / fg_static / fg_dual of tools_s2/s2_fgdense_whatif.py)
#   with bf16 weights: eager and, when a sync check passes, CUDA Graph replay.  The prototypes' per-scene foreground
#   packing (and fg_dual's window tables) is computed once per scene and reused, as mask tables are for r1 current.
#   python tools_s2/s2_stage_timing.py --config configs/S2P8_...yaml --s2_ckpt <ckpt> --out x.json
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from tools_s2.s2_graph_bench import (Pipe, capture, patch_aggregator_zeros, patch_rope_max_position,  # noqa: E402
                                     to_bf16)

SEGS = ("prep", "recon", "s1pass", "r0", "rerender", "tables", "r1", "s2heads")
GRAPHABLE = ("recon", "s1pass", "r0", "rerender", "r1", "s2heads")
SYNC_NAMES = ("aten::_local_scalar_dense", "cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaMemcpy")


class P(Pipe):
    def recon(self):
        ed = self.m.stage1.model.model
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return ed.reconstructor(self.images[:, :self.vin], self.cam[:, :self.vin], return_tokens=True)

    def build(self):
        self.prep()
        self.p1 = self.g1()
        self.c2w_in, self.K_in = self.glue1()
        self.P_in2 = self.g2()
        self.ctx, self.mask1 = self.glue2()
        self.res = self.g3()
        self.g4()
        return self

    def fns(self):
        return dict(recon=self.recon, s1pass=self.g1, r0=self.g1r, rerender=self.g2, tables=self.glue2, r1=self.g3,
                    s2heads=self.g4)


def timed(fn, warm, reps):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts))


def profiled(fn):
    from torch.profiler import ProfilerActivity, profile, record_function
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function("SEG"):
            fn()
        torch.cuda.synchronize()
    ev = prof.events()
    dev = [e for e in ev if e.device_type == torch.autograd.DeviceType.CUDA]
    out = dict(pageable_h2d=sum(1 for e in dev if "Pageable" in e.name and "HtoD" in e.name),
               syncs={n: sum(1 for e in ev if e.name == n) for n in SYNC_NAMES})
    # kernels grouped by the operator that launched them: walk from the segment range, an op's own kernels only
    # (the counting of tools_s2/s2_launch_profile.py, so the counts compare with the H200 job 137900; its summed
    # kernel time is known to miss kernels -- about half of the CUDA Graph replay time -- so GPU time is taken from
    # the graph replay, not from here; job 6907304 also showed that summing the raw device events double-counts)
    seg = [e for e in ev if e.name == "SEG"]
    by = {}
    stack = list(seg[0].cpu_children) if seg else []
    while stack:
        e = stack.pop()
        ks = getattr(e, "kernels", [])
        if ks:
            b = by.setdefault(e.name, [0, 0.0])
            b[0] += len(ks)
            b[1] += sum(k.duration for k in ks) / 1000.0
        stack.extend(e.cpu_children)
    out["kernels"] = sum(v[0] for v in by.values())
    out["kernel_ms_walk"] = sum(v[1] for v in by.values())
    out["copy_kernels"] = by.get("aten::copy_", [0, 0.0])[0]
    out["copy_ms"] = by.get("aten::copy_", [0, 0.0])[1]
    out["top_ops_by_count"] = sorted(([k, v[0], round(v[1], 3)] for k, v in by.items()), key=lambda r: -r[1])[:10]
    out["host_ms"] = seg[0].time_range.elapsed_us() / 1000.0 if seg else None
    return out


def mean_dicts(rows):
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--s2_ckpt", required=True)
    ap.add_argument("--n_scenes", type=int, default=4)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--vts", default="1,10")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(28100 + int(os.environ.get("SLURM_JOB_ID", "3")) % 400))
        dist.init_process_group("gloo", rank=0, world_size=1)
    torch.backends.cuda.matmul.allow_tf32 = True          # production inference setting (config use_tf32 true)
    torch.backends.cudnn.allow_tf32 = True
    os.chdir(REPO)
    import xformers.ops as xops
    from easydict import EasyDict as edict
    from xformers.ops.fmha.attn_bias import BlockDiagonalMask
    import model_s2.blocks as Bk
    import tools_s2.s2_fgdense_whatif as W
    from model_s2.masked_attention import _fa3_ops
    from model_s2.stage2_wrapper import Stage2LagerNVS
    from tools_s2.s2_bench import load_cfg
    from tools_s2.s2_infer_bench import val_scenes
    from tools_s2.s2_renderer_debug import unpatch

    cfg = load_cfg(args.config, 1)
    model = Stage2LagerNVS(cfg).cuda().eval()
    model.load_ckpt(args.s2_ckpt)
    model.val_cam_cond_zero_p = 0.0
    zero = torch.zeros((), device="cuda")
    model.loss_computer.forward = lambda *a, **k: edict(loss=zero)
    for blk in model.renderer.blocks:                 # timing-only reverse gate of the fg_dual prototype
        if isinstance(blk, Bk.S2BidirBlock):
            blk._rgate_tmp = torch.nn.Linear(768, 1536, bias=False).cuda()
            torch.nn.init.trunc_normal_(blk._rgate_tmp.weight, std=0.02)
    nin = int(cfg.training.val_dataset_cfgs.training.num_input_views)
    scenes = val_scenes(cfg, args.n_scenes)
    pvd = model.stage1.model.process_val_data.config.training
    vts = [int(x) for x in args.vts.split(",")]
    # block 4's stage-2 heads copy a parent-index table host -> device at every call (heads_s2.assemble_tokens),
    # which a CUDA Graph cannot record (job 6907304); they stay eager there
    graphable = GRAPHABLE if model.patch == model.s1_patch else tuple(s for s in GRAPHABLE if s != "s2heads")
    res = dict(config=args.config, s2_ckpt=args.s2_ckpt, n_scenes=len(scenes), warm=args.warm, reps=args.reps,
               gpu=torch.cuda.get_device_name(0), graphable=list(graphable), modes={})

    def whole(b):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return model(b, target_has_input=False, is_valid=True)

    def eager_mode(mode, prof):
        out = {}
        for vt in vts:
            pvd.num_target_views = vt
            rows, prows, wall = [], [], []
            for b in scenes:
                pp = P(model, b)
                t = dict(prep=timed(pp.prep, args.warm, args.reps))
                p = P(model, b).build()
                for name, fn in p.fns().items():
                    t[name] = timed(fn, args.warm, args.reps)
                rows.append(t)
                wall.append(timed(lambda: whole(b), args.warm, args.reps))
                if prof:
                    prows.append({name: profiled(fn) for name, fn in p.fns().items()})
                del p, pp
                torch.cuda.empty_cache()
            o = dict(ms=mean_dicts(rows), whole_forward_ms=float(np.mean(wall)))
            if prof:
                o["profile"] = {s: dict(mean_dicts([r[s] for r in prows]),
                                        syncs={n: float(np.mean([r[s]["syncs"][n] for r in prows])) for n in SYNC_NAMES},
                                        top_ops_by_count=prows[0][s]["top_ops_by_count"]) for s in prows[0]}
            out[f"vt{vt}"] = o
            print(f"[stage] {mode} vt={vt}: whole forward {o['whole_forward_ms']:.1f} ms | " +
                  ", ".join(f"{k} {v:.1f}" for k, v in o["ms"].items()), flush=True)
            if prof:
                for s, d in o["profile"].items():
                    print(f"[stage]   {mode} vt={vt} {s:8s} kernels {d['kernels']:6.0f} ({d['kernel_ms_walk']:6.1f} ms, op-walk) | "
                          f"copy_ {d['copy_kernels']:5.0f} ({d['copy_ms']:5.1f} ms) | host reads/syncs "
                          f"{json.dumps({k: round(v) for k, v in d['syncs'].items() if v})} | pageable H2D "
                          f"{d['pageable_h2d']:.0f}", flush=True)
        res["modes"][mode] = out
        json.dump(res, open(args.out, "w"), indent=1)

    def graph_mode(mode):
        out = {}
        for vt in vts:
            pvd.num_target_views = vt
            rows = []
            for b in scenes:
                p = P(model, b).build()
                t = {}
                for name, fn in p.fns().items():
                    if name not in graphable:
                        continue
                    g, _ = capture(fn)
                    t[name] = timed(g.replay, args.warm, args.reps)
                    del g
                rows.append(t)
                del p
                torch.cuda.empty_cache()
            out[f"vt{vt}"] = dict(ms=mean_dicts(rows))
            print(f"[stage] {mode} vt={vt}: " + ", ".join(f"{k} {v:.1f}" for k, v in out[f'vt{vt}']['ms'].items()),
                  flush=True)
        res["modes"][mode] = out
        json.dump(res, open(args.out, "w"), indent=1)

    def sync_free(fn):
        """one eager call under set_sync_debug_mode('error'); True, or the first host-device sync it hit"""
        fn()
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            fn()
            return True
        except Exception as e:                                                         # noqa: BLE001
            return f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
        finally:
            torch.cuda.set_sync_debug_mode(0)
            torch.cuda.synchronize()

    def candidates():
        """r1 prototypes, bf16 weights: eager, and CUDA Graph replay when the call is sync-free.  The per-scene
        foreground packing (and fg_dual's window tables) are computed once per scene and reused -- they are the
        prototype's stand-in for inputs a real implementation builds outside the renderer, like the mask tables."""
        orig_pack, orig_win = W.scene_fg_pack, W.fg_window_tables
        memo = {}

        def pack_memo(alpha_in, BV, S, g=37):
            key = ("pack", id(alpha_in), BV, S)
            if key not in memo:
                memo[key] = orig_pack(alpha_in, BV, S, g)
            return memo[key]

        def win_memo(ctx, idx, seq, BV, S):
            key = ("win", id(ctx), BV, S)
            if key not in memo:
                memo[key] = orig_win(ctx, idx, seq, BV, S)
            return memo[key]
        W.scene_fg_pack, W.fg_window_tables = pack_memo, win_memo
        out = {}
        try:
            for vt in vts:
                pvd.num_target_views = vt
                for var in ("current", "fg_dense", "fg_static", "fg_dual"):
                    rows, prows, checks = [], [], []
                    for b in scenes:
                        p = P(model, b).build()
                        vps = []
                        if var != "current":
                            W.apply(vps, var)
                        memo.clear()
                        try:
                            W.G["alpha_in"] = b["alpha_mask"][:, :nin].float().view(1, nin, 1, 256, 256)
                            W.G["scene_fg_share"] = []
                            t = dict(eager=timed(p.g3, args.warm, args.reps))
                            if var != "current" and not W.G["scene_fg_share"]:
                                raise RuntimeError(f"{var} did not run its patched renderer")
                            prows.append(profiled(p.g3))
                            ok = sync_free(p.g3)
                            checks.append(ok)
                            if ok is True:
                                g, _ = capture(p.g3)
                                t["graph"] = timed(g.replay, args.warm, args.reps)
                                del g
                            rows.append(t)
                        finally:
                            unpatch(vps)
                        del p
                        torch.cuda.empty_cache()
                    o = dict(eager_ms=float(np.mean([r["eager"] for r in rows])),
                             graph_ms=(float(np.mean([r["graph"] for r in rows]))
                                       if all("graph" in r for r in rows) else None),
                             kernels=float(np.mean([r["kernels"] for r in prows])),
                             copy_kernels=float(np.mean([r["copy_kernels"] for r in prows])),
                             sync_check=[c if c is True else str(c) for c in checks])
                    out[f"vt{vt}_{var}"] = o
                    g_ = "  n/a " if o["graph_ms"] is None else f"{o['graph_ms']:6.1f}"
                    print(f"[stage] r1 {var:9s} vt={vt:2d}: eager {o['eager_ms']:6.1f} ms | graph {g_} ms | "
                          f"kernels {o['kernels']:5.0f} (copy_ {o['copy_kernels']:4.0f}) | sync-free "
                          f"{sum(c is True for c in checks)}/{len(checks)}", flush=True)
        finally:
            W.scene_fg_pack, W.fg_window_tables = orig_pack, orig_win
        res["r1_candidates_bf16"] = out
        json.dump(res, open(args.out, "w"), indent=1)

    # 1. production code, fp32 weights
    eager_mode("E_prod", prof=True)
    # 2. remove the three host syncs (same values)
    cache = {}

    def attend_blockdiag_cached(q, k, v, q_seqlens, kv_seqlens, backend="fa3"):
        key = (tuple(q_seqlens), tuple(kv_seqlens))
        if key not in cache:
            cache[key] = BlockDiagonalMask.from_seqlens(list(q_seqlens), list(kv_seqlens), device=q.device)
        return xops.memory_efficient_attention(q[None], k[None], v[None], attn_bias=cache[key], op=_fa3_ops())[0]
    Bk.attend_blockdiag = attend_blockdiag_cached
    patch_aggregator_zeros()
    patch_rope_max_position()
    eager_mode("E_nosync", prof=False)
    graph_mode("G_fp32")
    # 3. bf16 weights
    n = to_bf16(model)
    res["bf16_converted_modules"] = n
    eager_mode("E_bf16", prof=True)
    graph_mode("G_bf16")
    candidates()

    # decomposition table
    dec = {}
    M = res["modes"]
    for vt in vts:
        k = f"vt{vt}"
        rows = {}
        for s in SEGS:
            e_prod, e_ns, e16 = M["E_prod"][k]["ms"][s], M["E_nosync"][k]["ms"][s], M["E_bf16"][k]["ms"][s]
            g32 = M["G_fp32"][k]["ms"].get(s)
            g16 = M["G_bf16"][k]["ms"].get(s)
            rows[s] = dict(E_prod=e_prod, sync=e_prod - e_ns, casts=e_ns - e16,
                           casts_gpu=(g32 - g16) if g16 is not None else None,
                           launch=(e16 - g16) if g16 is not None else None, gpu=g16,
                           kernels_prod=M["E_prod"][k]["profile"][s]["kernels"] if s in M["E_prod"][k]["profile"] else None,
                           copy_kernels_prod=M["E_prod"][k]["profile"][s]["copy_kernels"] if s in M["E_prod"][k]["profile"] else None,
                           kernels_bf16=M["E_bf16"][k]["profile"][s]["kernels"] if s in M["E_bf16"][k]["profile"] else None,
                           kernel_ms_bf16=M["E_bf16"][k]["profile"][s]["kernel_ms_walk"] if s in M["E_bf16"][k]["profile"] else None)
        dec[k] = rows
        print(f"[stage] ===== decomposition vt={vt} (ms): segment | E_prod = sync + casts + launch + GPU | "
              f"kernels prod (copy_) -> bf16 | profiled kernel ms bf16", flush=True)
        for s, r in rows.items():
            f = lambda x: "   -  " if x is None else f"{x:6.1f}"  # noqa: E731
            print(f"[stage] {s:8s} | {r['E_prod']:6.1f} = {f(r['sync'])} + {f(r['casts'])} + {f(r['launch'])} + "
                  f"{f(r['gpu'])} | casts GPU {f(r['casts_gpu'])} | kernels {r['kernels_prod'] or 0:6.0f} "
                  f"({r['copy_kernels_prod'] or 0:5.0f}) -> {r['kernels_bf16'] or 0:6.0f} | {f(r['kernel_ms_bf16'])}",
                  flush=True)
    res["decomposition"] = dec
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"[stage] DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
