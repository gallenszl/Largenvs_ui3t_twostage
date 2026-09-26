"""Measure per-group gradient norms of the three-task model on REAL training batches.

Decides the `clip_group_prefixes` list for the uni3t clipgrp arm with data instead of
a guess: which head inflates the global norm, and by how much the shared trunk is
being compressed under the historical single global clip.

    sbatch ... python tools/uni3t_gradnorm_groups.py <config> <ckpt.pt> [n_batches]
"""
import json
import os
import sys
import time
from pathlib import Path

import torch
from easydict import EasyDict as edict
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import importlib  # noqa: E402
from model.lagernvs_wrapper import LagerNVSInRnG  # noqa: E402
from utils.training_utils import build_clip_groups  # noqa: E402

PREFIXES = ["camera_head.", "point_head."]   # -> groups camera_head. / point_head. / rest


def group_norms(clip_groups):
    """Pre-clip L2 norm of each group's gradient, computed directly (grads untouched)."""
    out = {}
    for g, ps in clip_groups:
        sq = torch.zeros((), device="cuda", dtype=torch.float32)
        for p in ps:
            if p.grad is not None:
                sq += p.grad.float().pow(2).sum()
        out[g] = sq.sqrt().item()
    return out


def main():
    cfg_path, ckpt_path = sys.argv[1], sys.argv[2]
    n_batches = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    cfg = edict(OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True))
    cfg.ddp_info = edict(global_rank=0, world_size=1, local_rank=0, device="cuda:0",
                         is_main_process=True, seed=int(cfg.training.get("seed", 777)))
    torch.manual_seed(cfg.ddp_info.seed)
    device = "cuda"

    print(f"[gn] config: {cfg_path}")
    print(f"[gn] ckpt  : {ckpt_path}")
    print(f"[gn] weight_camera={cfg.training.weight_camera} weight_point={cfg.training.weight_point} "
          f"cam_cond_zero_p={cfg.training.cam_cond_zero_p} grad_clip_norm={cfg.training.grad_clip_norm}")

    model = LagerNVSInRnG(cfg).to(device)
    sd = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=False)
    step = sd.get("fwdbwd_pass_step")
    missing, unexpected = model.load_state_dict(sd["model"], strict=False)
    print(f"[gn] loaded step {step}: missing={len(missing)} unexpected={len(unexpected)}")
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    del sd
    model.train()   # same conditions as training: dropout of cond cameras, grad ckpt on

    optimized = {n: p for n, p in model.named_parameters() if p.requires_grad}
    groups = build_clip_groups(optimized, PREFIXES)
    print("[gn] groups: " + ", ".join(f"{g}={sum(q.numel() for q in ps)/1e6:.1f}M" for g, ps in groups))

    module, class_name = cfg.training.dataset_name.rsplit(".", 1)
    Dataset = importlib.import_module(module).__dict__[class_name]
    ds = Dataset(cfg)
    gen = torch.Generator().manual_seed(1234)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.training.batch_size_per_gpu, shuffle=True,
                                     num_workers=4, drop_last=True, generator=gen)
    print(f"[gn] dataset {class_name}: {len(ds)} items, batch {cfg.training.batch_size_per_gpu}, "
          f"{n_batches} batches")

    # step 16000+ is past exclude_bg_until (9999) in this arm, so exclude_bg=False
    exclude_bg = False
    rows = []
    t0 = time.time()
    it = iter(dl)
    for i in range(n_batches):
        batch = next(it)
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch, exclude_bg=exclude_bg)
        loss = out.loss_metrics.loss
        loss.backward()
        gn = group_norms(groups)
        total = sum(v * v for v in gn.values()) ** 0.5
        row = {"i": i, "loss": float(loss), **{f"norm/{g}": v for g, v in gn.items()}, "norm/global": total}
        # what the two clipping schemes do to the shared trunk ('rest')
        clip = float(cfg.training.grad_clip_norm)
        row["trunk_coef/global_clip"] = min(1.0, clip / max(total, 1e-12))
        row["trunk_coef/grouped_clip"] = min(1.0, clip / max(gn["rest"], 1e-12))
        rows.append(row)
        print(f"[gn] b{i:02d} loss={row['loss']:.4f} | " +
              " ".join(f"{g}={gn[g]:.3f}" for g in gn) +
              f" | global={total:.3f} | trunk coef global={row['trunk_coef/global_clip']:.3f} "
              f"grouped={row['trunk_coef/grouped_clip']:.3f}", flush=True)

    def med(k):
        v = sorted(r[k] for r in rows); n = len(v)
        return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])
    keys = [k for k in rows[0] if k != "i"]
    summary = {k: med(k) for k in keys}
    summary["n_batches"] = len(rows)
    summary["step"] = step
    print("\n[gn] ===== medians over %d real batches (step %s) =====" % (len(rows), step))
    for g, _ in groups:
        share = summary[f"norm/{g}"] ** 2 / max(summary["norm/global"] ** 2, 1e-12)
        print(f"[gn]   ||g_{g:<13}|| = {summary[f'norm/{g}']:8.3f}   share of ||g||^2 = {100*share:5.1f}%")
    print(f"[gn]   ||g_global||        = {summary['norm/global']:8.3f}")
    print(f"[gn]   trunk update coef:  global clip = {summary['trunk_coef/global_clip']:.3f}   "
          f"grouped clip = {summary['trunk_coef/grouped_clip']:.3f}   "
          f"ratio = {summary['trunk_coef/grouped_clip']/max(summary['trunk_coef/global_clip'],1e-12):.2f}x")
    print(f"[gn]   peak mem {torch.cuda.max_memory_allocated()/2**30:.1f} GiB, {time.time()-t0:.0f}s")

    out_dir = Path("/home/z50057756/tmp/uni3t_gradnorm"); out_dir.mkdir(parents=True, exist_ok=True)
    out_f = out_dir / f"gradnorm_groups_step{step}.json"
    json.dump({"config": cfg_path, "ckpt": ckpt_path, "prefixes": PREFIXES,
               "rows": rows, "median": summary}, open(out_f, "w"), indent=1)
    print(f"[gn] wrote {out_f}")
    print("[gn] DONE")


if __name__ == "__main__":
    main()
