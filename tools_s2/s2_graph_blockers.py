# Find what prevents CUDA Graph capture: every segment of the batch-1 inference path is run eagerly (after warmup)
# under torch.cuda.set_sync_debug_mode("error"), which raises at the first operation that synchronises the host
# with the GPU (device -> host reads, pageable host -> device copies) and prints where it happens.
#   python tools_s2/s2_graph_blockers.py --config configs/S2P8_...yaml
import argparse
import os
import sys
import traceback

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(28100 + int(os.environ.get("SLURM_JOB_ID", "3")) % 400))
        dist.init_process_group("gloo", rank=0, world_size=1)
    os.chdir(REPO)
    import xformers.ops as xops
    from easydict import EasyDict as edict
    from xformers.ops.fmha.attn_bias import BlockDiagonalMask
    import model_s2.blocks as Bk
    from model_s2.masked_attention import _fa3_ops
    from model_s2.stage2_wrapper import Stage2LagerNVS
    from tools_s2.s2_bench import load_cfg
    from tools_s2.s2_graph_bench import Pipe
    from tools_s2.s2_infer_bench import val_scenes
    cache = {}

    def attend_blockdiag_cached(q, k, v, q_seqlens, kv_seqlens, backend="fa3"):
        key = (tuple(q_seqlens), tuple(kv_seqlens))
        if key not in cache:
            cache[key] = BlockDiagonalMask.from_seqlens(list(q_seqlens), list(kv_seqlens), device=q.device)
        return xops.memory_efficient_attention(q[None], k[None], v[None], attn_bias=cache[key], op=_fa3_ops())[0]
    Bk.attend_blockdiag = attend_blockdiag_cached
    cfg = load_cfg(args.config, 1)
    model = Stage2LagerNVS(cfg).cuda().eval()
    model.val_cam_cond_zero_p = 0.0
    zero = torch.zeros((), device="cuda")
    model.loss_computer.forward = lambda *a, **k: edict(loss=zero)
    b = val_scenes(cfg, 1)[0]
    model.stage1.model.process_val_data.config.training.num_target_views = 1
    p = Pipe(model, b)
    p.prep()
    for _ in range(2):                                   # warmup: caches (RoPE positions, masks, flex kernels)
        p.p1 = p.g1()
        p.c2w_in, p.K_in = p.glue1()
        p.P_in2 = p.g2()
        p.ctx, p.mask1 = p.glue2()
        p.res = p.g3()
        p.g4()
        p.g1r()
    torch.cuda.synchronize()
    for name in ("g1", "g1r", "g2", "g3", "g4"):
        torch.cuda.set_sync_debug_mode("error")
        try:
            getattr(p, name)()
            torch.cuda.set_sync_debug_mode(0)
            torch.cuda.synchronize()
            print(f"[blockers] {name}: no host-device synchronisation", flush=True)
        except Exception as e:                                                         # noqa: BLE001
            torch.cuda.set_sync_debug_mode(0)
            tb = traceback.format_exc().strip().splitlines()
            frames = [ln.strip() for ln in tb if ln.strip().startswith("File ") and "site-packages/torch/" not in ln]
            print(f"[blockers] {name}: {type(e).__name__}: {str(e).splitlines()[0][:200]}", flush=True)
            for ln in frames[-8:]:
                print(f"[blockers]     {ln}", flush=True)
            torch.cuda.synchronize()
    print("[blockers] DONE", flush=True)


if __name__ == "__main__":
    main()
