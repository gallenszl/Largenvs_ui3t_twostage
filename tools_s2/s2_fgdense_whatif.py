# Timing-only what-if (no quality claim): the stage-2 renderer with "full attention restricted to the foreground on
# both sides" instead of block summaries + geometric windows.  Patched inside this process only.
#   fg_dense    target foreground tokens (+ registers) <-> scene foreground tokens (input-view alpha at the 37 x 37
#               token grid, dilated by one token), dense FA3 attention in both directions, per-target-view scene
#               copies updated every block as now; no summaries, no windows, no mask tables needed
#   fg_static   same, but the scene side is not updated and is shared by all target views of a scene (target ->
#               scene attention only, scene keys / values computed once per scene per block)
# compared with the current implementation, for 1 and 10 target views (batch 1) and the training shape (batch 8,
# 6 target views, forward only).  Also reports the foreground share of the scene tokens.
#   python tools_s2/s2_fgdense_whatif.py --config configs/S2P8_...yaml --out x.json
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from tools_s2.s2_renderer_debug import coarse, collect, fwd, gpu_busy, patch, unpatch  # noqa: E402

G = {}


def scene_fg_pack(alpha_in, BV, S, g=37):
    """alpha_in [B, Vin, 1, H, W] -> packed scene foreground indices for every target view and segment lengths."""
    B, Vin = alpha_in.shape[:2]
    a = F.adaptive_max_pool2d(alpha_in.flatten(0, 1).float(), g) > 0.5            # [B*Vin, 1, g, g]
    a = F.max_pool2d(a.float(), 3, 1, 1) > 0                                         # one-token dilation
    fg = a.view(B, Vin * g * g)
    assert fg.shape[1] == S, (fg.shape, S)
    Vt = BV // B
    fgv = fg.repeat_interleave(Vt, 0)                                                # [BV, S]
    base = torch.arange(BV, device=fg.device).view(BV, 1) * S
    idx = (base + torch.arange(S, device=fg.device).view(1, S))[fgv]                 # per target view, in order
    seq = [int(v) for v in fgv.sum(1).tolist()]
    idx1 = (torch.arange(B, device=fg.device).view(B, 1) * S + torch.arange(S, device=fg.device).view(1, S))[fg]
    seq1 = [int(v) for v in fg.sum(1).tolist()]
    return idx, seq, idx1, seq1, float(fg.float().mean())


def dense_cross(attn, qin, kvin, q_seq, kv_seq):
    import model_s2.blocks as Bk
    H = attn.num_heads
    q = attn.q_norm(Bk._heads(attn.q_proj(qin), H))
    k = attn.k_norm(Bk._heads(attn.k_proj(kvin), H))
    v = Bk._heads(attn.v_proj(kvin), H)
    o = Bk.attend_blockdiag(q, k, v, q_seq, kv_seq, "fa3")
    return attn.attn_fc_dropout(attn.proj(o.flatten(-2)))


def bidir_fg(self, x, rec, ctx):
    import model_s2.blocks as Bk
    x = x + self.inj_x(ctx.x_prior)
    rec = rec + self.inj_rec(ctx.rec_prior_p)
    x = x + Bk.self_attention_packed(self.self_attn, self.norm1_x(x), ctx)
    xn = self.norm2_x(x)
    rn = self.norm1_rec(rec)
    dx = dense_cross(self.cross_attn_x, xn, rn, ctx.q_seqlens, ctx.s_seq)
    drec = dense_cross(self.cross_attn_rec, rn, xn, ctx.s_seq, ctx.q_seqlens)
    x = x + dx
    rec = rec + drec
    x = x + self.mlp_x(self.norm3_x(x))
    rec = rec + self.mlp_rec(self.norm2_rec(rec))
    return x, rec


def final_fg(self, x, rec, ctx):
    import model_s2.blocks as Bk
    x = x + self.inj_x(ctx.x_prior)
    rec = rec + self.inj_rec(ctx.rec_prior_p)
    x = x + Bk.self_attention_packed(self.self_attn, self.norm1(x), ctx)
    xn = self.norm2(x)
    rn = self.norm2_kv(rec)
    x = x + dense_cross(self.cross_attn, xn, rn, ctx.q_seqlens, ctx.s_seq)
    x = x + self.mlp(self.norm_ffn(x))
    return x


def bidir_fg_static(self, x, rec, ctx):
    import model_s2.blocks as Bk
    x = x + self.inj_x(ctx.x_prior)
    x = x + Bk.self_attention_packed(self.self_attn, self.norm1_x(x), ctx)
    xn = self.norm2_x(x)
    rn = self.norm1_rec(rec)                                  # one scene copy per scene, never updated
    x = x + dense_cross(self.cross_attn_x, xn, rn, ctx.q_seq_per_scene, ctx.s_seq1)
    x = x + self.mlp_x(self.norm3_x(x))
    return x, rec


def final_fg_static(self, x, rec, ctx):
    import model_s2.blocks as Bk
    x = x + self.inj_x(ctx.x_prior)
    x = x + Bk.self_attention_packed(self.self_attn, self.norm1(x), ctx)
    xn = self.norm2(x)
    rn = self.norm2_kv(rec)
    x = x + dense_cross(self.cross_attn, xn, rn, ctx.q_seq_per_scene, ctx.s_seq1)
    x = x + self.mlp(self.norm_ffn(x))
    return x


def renderer_fg(self, rays6, mask1, rec_rep, ctx):
    from model_s2.geometry import N_REG
    lay = ctx.layout
    C = self.hidden
    x_in = torch.cat([rays6, mask1.to(rays6.dtype)], dim=2).flatten(0, 1)
    tok = self.tgt_norm(self.tgt_proj(x_in).flatten(2).transpose(1, 2))
    g2 = tok.shape[1]
    fgt = tok.reshape(-1, C)[lay.fg_bv * g2 + lay.fg_tok_flat]
    regs = self.registers.expand(lay.BV, N_REG, C).reshape(-1, C)
    x = fgt.new_zeros(lay.T, C).index_copy(0, lay.fg_packed, fgt).index_copy(0, lay.reg_packed, regs.to(fgt.dtype))
    BV, S = rec_rep.shape[:2]
    idx, seq, idx1, seq1, share = scene_fg_pack(G["alpha_in"], BV, S)
    G["scene_fg_share"].append(share)
    static = G["mode"] == "fg_static"
    if static:
        B = len(seq1)
        Vt = BV // B
        rec = rec_rep.reshape(-1, C)[idx1 * 1 if Vt == 1 else _first_copy(idx1, S, Vt)]
        ctx.s_seq1 = seq1
        qs = ctx.q_seqlens
        ctx.q_seq_per_scene = [sum(qs[b * Vt:(b + 1) * Vt]) for b in range(B)]
    else:
        rec = rec_rep.reshape(-1, C)[idx]
        ctx.s_seq = seq
        ctx.rec_prior_p = ctx.rec_prior.reshape(-1, C)[idx]
    outs = {}
    last = len(self.blocks) - 1
    for i, blk in enumerate(self.blocks):
        if i < last:
            x, rec = blk(x, rec, ctx)
        else:
            x = blk(x, rec, ctx)
        if i in self.out_layers:
            outs[i] = x
    return {m: lin(outs[m][lay.fg_packed]) for m, lin in zip(self.out_layers, self.out_lin)}


def _first_copy(idx1, S, Vt):
    """indices into rec_rep (BV*S rows) of the first target view's copy of every scene: row b*Vt*S + j."""
    b = idx1 // S
    j = idx1 % S
    return b * Vt * S + j


def apply(ps, mode):
    import model_s2.blocks as Bk
    import model_s2.renderer_s2 as R
    G["mode"] = mode
    patch(ps, R.Stage2Renderer, "forward", renderer_fg)
    if mode == "fg_dense":
        patch(ps, Bk.S2BidirBlock, "forward", bidir_fg)
        patch(ps, Bk.S2FinalBlock, "forward", final_fg)
    else:
        patch(ps, Bk.S2BidirBlock, "forward", bidir_fg_static)
        patch(ps, Bk.S2FinalBlock, "forward", final_fg_static)


def run_set(model, batches, is_valid, vt_label, res):
    for var in ("stage1_only", "current", "fg_dense", "fg_static"):
        vps = []
        model.enabled = var != "stage1_only"
        G["scene_fg_share"] = []
        if var in ("fg_dense", "fg_static"):
            apply(vps, var)
        try:
            def one(b):
                G["alpha_in"] = b["alpha_in"]
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    model(b["batch"], target_has_input=False, is_valid=is_valid)
            for _ in range(2):
                for b in batches:
                    one(b)
            torch.cuda.synchronize()
            walls = []
            for _ in range(3):
                for b in batches:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    one(b)
                    torch.cuda.synchronize()
                    walls.append((time.perf_counter() - t0) * 1000.0)
            ps, state = [], {"p2": False}
            coarse(ps, model, state)
            try:
                for b in batches:
                    one(b)
                parts = collect(len(batches))
            finally:
                unpatch(ps)
            G["alpha_in"] = batches[0]["alpha_in"]
            from torch.profiler import ProfilerActivity, profile
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                t0 = time.perf_counter()
                one(batches[0])
                torch.cuda.synchronize()
                wall = (time.perf_counter() - t0) * 1000.0
            kern = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
            busy = dict(wall_ms=wall, kernel_ms=sum(e.time_range.elapsed_us() for e in kern) / 1000.0,
                        n_kernels=len(kern))
            r = dict(wall_median_ms=float(np.median(walls)), parts_ms=parts, gpu_busy=busy,
                     scene_fg_share=float(np.mean(G["scene_fg_share"])) if G["scene_fg_share"] else None)
        except Exception as e:                                                         # noqa: BLE001
            import traceback
            traceback.print_exc()
            r = dict(error=f"{type(e).__name__}: {e}")
        finally:
            unpatch(vps)
        res[f"{vt_label}_{var}"] = r
        if "error" in r:
            print(f"[fg] {vt_label} {var}: ERROR {r['error']}", flush=True)
            continue
        b_ = r["gpu_busy"]
        print(f"[fg] {vt_label} {var}: wall median {r['wall_median_ms']:.1f} ms | GPU busy "
              f"{100 * b_['kernel_ms'] / b_['wall_ms']:.0f} % ({b_['n_kernels']} kernels) | scene fg share "
              f"{r['scene_fg_share']} | " + ", ".join(f"{k} {v:.1f}" for k, v in sorted(r['parts_ms'].items())),
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_scenes", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(28700 + int(os.environ.get("SLURM_JOB_ID", "3")) % 400))
        dist.init_process_group("gloo", rank=0, world_size=1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.chdir(REPO)
    from easydict import EasyDict as edict
    from model_s2.stage2_wrapper import Stage2LagerNVS
    from tools_s2.s2_bench import cached_batches, load_cfg
    from tools_s2.s2_infer_bench import val_scenes
    cfg = load_cfg(args.config, 8)
    model = Stage2LagerNVS(cfg).cuda().eval()
    model.val_cam_cond_zero_p = 0.0
    zero = torch.zeros((), device="cuda")
    model.loss_computer.forward = lambda *a, **k: edict(loss=zero)
    nin = int(cfg.training.val_dataset_cfgs.training.num_input_views)
    scenes = [dict(batch=b, alpha_in=b["alpha_mask"][:, :nin].float().view(1, nin, 1, 256, 256))
              for b in val_scenes(cfg, args.n_scenes)]
    ntr_in = int(cfg.training.num_input_views)
    train_b = [dict(batch=b, alpha_in=b["alpha_mask"][:, :ntr_in].float().reshape(b["alpha_mask"].shape[0], ntr_in,
                                                                                    1, 256, 256))
               for b in cached_batches(cfg, 2)]
    pvd = model.stage1.model.process_val_data.config.training
    res = {}
    for vt in (1, 10):
        pvd.num_target_views = vt
        run_set(model, scenes, True, f"infer_vt{vt}", res)
    run_set(model, train_b, False, "train_b8", res)
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"[fg] DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
