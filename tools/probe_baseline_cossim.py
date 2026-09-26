"""Probe baseline checkpoint features vs DINOv3 features (Exp 2).

For one baseline checkpoint, runs forward on the GSO val subset, extracts
aggregator intermediate features at multiple layers, and computes per-token
cosine similarity + CKA against DINOv3-ViT-L target features.

This diagnoses whether the vanilla baseline (no REPA) endogenously learns
DINOv3-like features over training. Compared against REPA's logged
cos_sim_align trajectory, this determines whether REPA's plateau is caused
by baseline catch-up (redundancy) or by something else.

Run via torchrun with DDP across 4 GPUs on a single checkpoint.
"""

import argparse
import csv
import importlib
import math
import os
import sys
from typing import Dict, List

# Make project root importable (setup.py lives at root, this script in tools/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from einops import rearrange

# Project modules
from setup import init_config, init_distributed
from model.vggt.stage1.encoders.dinov3 import DINOv3TargetEncoder

amp_dtype_mapping = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "tf32": torch.float32,
}


def parse_extra_args():
    """Parse extra CLI args after init_config. init_config consumes its own args."""
    # We piggy-back on init_config's argparse-like override mechanism via env vars.
    # Probe-specific args come from env to avoid conflicts.
    ckpt_path = os.environ.get("PROBE_CKPT_PATH", "")
    layers_str = os.environ.get("PROBE_LAYERS", "3,7,12,17,22")
    out_csv = os.environ.get("PROBE_OUT_CSV", "")
    label = os.environ.get("PROBE_LABEL", "")
    max_objects = int(os.environ.get("PROBE_MAX_OBJECTS", "0"))  # 0 = all
    if not ckpt_path or not out_csv:
        raise RuntimeError(
            "Probe requires env vars PROBE_CKPT_PATH and PROBE_OUT_CSV"
        )
    layers = [int(s) for s in layers_str.split(",") if s.strip()]
    return ckpt_path, layers, out_csv, label, max_objects


def apply_dtype(model, config):
    if not config.training.get("use_bf16", False):
        return model
    model = model.to(amp_dtype_mapping["bf16"])
    model.camera_head.to(torch.float32)
    model.point_head.to(torch.float32)
    model.rgb_head.to(torch.float32)
    model.loss_computer.to(torch.float32)
    return model


@torch.no_grad()
def probe_forward(model, data_batch, layers_to_probe):
    """Run aggregator only + extract intermediate features at target view.

    Mirrors the first half of RnG.forward(is_valid=True) without RGB / point /
    camera heads or loss computation.

    Returns:
        features: dict layer_idx -> [B*V_out, C=embed_dim, H, W] (fp32)
        target_image: [B, V_out, 3, H, W] for DINOv3 input
    """
    raw_model = model.module if hasattr(model, "module") else model

    process_data = raw_model.process_val_data
    input, target = process_data(
        data_batch,
        has_target_image=True,
        target_has_input=False,
        compute_rays=True,
    )

    input_pose_cond = raw_model.get_posed_input(ray_o=input.ray_o, ray_d=input.ray_d)
    target_pose_cond = raw_model.get_posed_input(ray_o=target.ray_o, ray_d=target.ray_d)
    pose_cond = torch.cat([input_pose_cond, target_pose_cond], dim=1)
    pose_tokens = raw_model.pose_tokenizer(pose_cond)

    aggregated_tokens_list, patch_start_idx = raw_model.aggregator(
        input.image, pose_tokens, posed_input=not raw_model.config.unposed
    )
    aggregated_tokens_list = [t.float() for t in aggregated_tokens_list]

    features = {}
    for L in layers_to_probe:
        # [B*V_out, V_in+1, P, 2C] -> [B*V_out, 1, N_patch, C] (post-global half)
        feat = aggregated_tokens_list[L][:, -1:, patch_start_idx:, raw_model.embed_dim:]
        BV, _, N, C = feat.shape
        H = W = int(math.sqrt(N))
        feat = feat.reshape(BV, H, W, C).permute(0, 3, 1, 2).contiguous()
        features[L] = feat  # [BV, C, H, W]

    return features, target.image


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA between feature matrices X and Y.

    Inputs:
      X: [N, D1]
      Y: [N, D2]
    Returns:
      scalar similarity in [0, 1]
    """
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    XtY_norm_sq = (X.t() @ Y).norm() ** 2
    XtX_norm = (X.t() @ X).norm()
    YtY_norm = (Y.t() @ Y).norm()
    denom = XtX_norm * YtY_norm + 1e-12
    return (XtY_norm_sq / denom).item()


def main():
    ckpt_path, layers_to_probe, out_csv, label, max_objects = parse_extra_args()

    # Load config (consumes CLI args via init_config)
    config = init_config()
    os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))

    ddp_info = init_distributed(seed=777)
    dist.barrier()

    torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
    torch.backends.cudnn.allow_tf32 = config.training.use_tf32

    # Build val dataset (same as inference.py path)
    dataset_name = config.training.get("val_dataset_name")
    module, class_name = dataset_name.rsplit(".", 1)
    Dataset = importlib.import_module(module).__dict__[class_name]
    dataset = Dataset(config)

    if max_objects > 0 and ddp_info.is_main_process:
        print(f"[probe] full val dataset = {len(dataset)} objects; capping to {max_objects}")

    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.training.num_workers,
        prefetch_factor=config.training.prefetch_factor,
        persistent_workers=True,
        pin_memory=False,
        drop_last=False,
        sampler=sampler,
    )

    # Build model
    module, class_name = config.model.class_name.rsplit(".", 1)
    LVSM = importlib.import_module(module).__dict__[class_name]
    model = LVSM(config).to(ddp_info.device)
    model = apply_dtype(model, config)
    model = DDP(model, device_ids=[ddp_info.local_rank])

    # Force-load this specific ckpt path (NOT the directory the config points at)
    if ddp_info.is_main_process:
        print(f"[probe] loading ckpt: {ckpt_path}")
    model.module.load_ckpt(ckpt_path)
    model.eval()

    # Build DINOv3 target encoder (frozen, bf16, eval-locked)
    align_cfg = config.training.feature_alignment
    dinov3 = DINOv3TargetEncoder(
        weights_path=align_cfg.target_encoder_weights,
        hub_dir=align_cfg.target_encoder_hub_dir,
        resolution=int(align_cfg.get("target_resolution", 512)),
    ).to(ddp_info.device)
    dinov3.to(torch.bfloat16)
    dinov3.eval()

    if ddp_info.is_main_process:
        print(f"[probe] DINOv3 grid = {dinov3.grid}x{dinov3.grid}, layers_to_probe = {layers_to_probe}")

    # Accumulators per layer: list of (mean_cos_per_sample, std_cos_per_sample)
    # plus all-tokens flattened for global CKA.
    per_layer_cos = {L: [] for L in layers_to_probe}  # list of per-sample mean cos
    cka_pred_tokens = {L: [] for L in layers_to_probe}  # for CKA accumulation
    cka_tgt_tokens = {L: [] for L in layers_to_probe}
    n_objects_seen = 0

    sampler.set_epoch(0)
    with torch.no_grad(), torch.autocast(
        enabled=config.training.use_amp,
        device_type="cuda",
        dtype=amp_dtype_mapping[config.training.amp_dtype],
    ):
        for batch_idx, batch in enumerate(dataloader):
            if max_objects > 0 and n_objects_seen >= max_objects:
                break

            batch = {
                k: v.to(ddp_info.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            features, target_image = probe_forward(model, batch, layers_to_probe)
            # features[L]: [BV, C, H, W] fp32 from aggregator
            # target_image: [B, V_out, 3, H, W] bf16

            # DINOv3 features (bf16 internally, fp32 output)
            target_img_flat = rearrange(target_image, "b v c h w -> (b v) c h w")
            z_tgt = dinov3(target_img_flat).float()  # [BV, H_t*W_t, C]
            BV, Nt, Ct = z_tgt.shape
            Ht = Wt = dinov3.grid  # 32 for resolution=512

            for L in layers_to_probe:
                feat = features[L]  # [BV, C, H, W]
                _, C, H, W = feat.shape

                # Resize agg feat to match DINOv3 grid if necessary
                if (H, W) != (Ht, Wt):
                    feat = F.interpolate(
                        feat, size=(Ht, Wt), mode="bilinear", align_corners=False
                    )
                z_pred = feat.flatten(2).transpose(1, 2)  # [BV, Nt, C]

                # Per-token cosine similarity
                z_pred_n = F.normalize(z_pred, dim=-1)
                z_tgt_n = F.normalize(z_tgt, dim=-1)
                cos = (z_pred_n * z_tgt_n).sum(dim=-1)  # [BV, Nt]
                # mean over (V_out, tokens) per object/batch
                per_layer_cos[L].append(cos.mean().item())

                # Save flattened features for global CKA at the end
                cka_pred_tokens[L].append(z_pred.reshape(-1, C).cpu())
                cka_tgt_tokens[L].append(z_tgt.reshape(-1, Ct).cpu())

            n_objects_seen += target_image.shape[0]

            if ddp_info.is_main_process and batch_idx % 10 == 0:
                print(
                    f"[probe] batch {batch_idx}, seen={n_objects_seen}, "
                    f"sample cos@L7={per_layer_cos[7][-1] if 7 in per_layer_cos else 'n/a':.4f}"
                    if 7 in per_layer_cos and per_layer_cos[7]
                    else f"[probe] batch {batch_idx}, seen={n_objects_seen}"
                )

    dist.barrier()

    # Gather per-rank stats to rank 0 (cos_sim only — keep it simple, CKA on rank 0)
    rank_cos = {
        L: torch.tensor(per_layer_cos[L], device=ddp_info.device, dtype=torch.float32)
        for L in layers_to_probe
    }

    gathered = {}
    for L in layers_to_probe:
        local = rank_cos[L]
        local_size = torch.tensor([local.numel()], device=ddp_info.device, dtype=torch.long)
        sizes = [torch.zeros_like(local_size) for _ in range(ddp_info.world_size)]
        dist.all_gather(sizes, local_size)
        max_size = max(s.item() for s in sizes)
        # pad
        padded = torch.zeros(max_size, device=ddp_info.device, dtype=torch.float32)
        padded[: local.numel()] = local
        out_buf = [torch.zeros_like(padded) for _ in range(ddp_info.world_size)]
        dist.all_gather(out_buf, padded)
        all_vals = []
        for buf, sz in zip(out_buf, sizes):
            all_vals.append(buf[: sz.item()].cpu())
        gathered[L] = torch.cat(all_vals).numpy()

    if not ddp_info.is_main_process:
        dist.barrier()
        return

    # Rank 0: compute CKA from concatenated rank-local features
    # (CKA is approximate since each rank only has its shard; for diagnostic
    # purposes we report rank-0-local CKA — adequate for trajectory shape.)
    import numpy as np

    # Parse step from ckpt path
    base = os.path.basename(ckpt_path)
    step_str = "".join(c for c in base if c.isdigit())
    try:
        step = int(step_str)
    except ValueError:
        step = -1

    # Write CSV
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "label", "step", "ckpt", "layer",
                "cos_sim_mean", "cos_sim_std", "n_samples",
                "cka_rank0_local",
            ])
        for L in layers_to_probe:
            vals = gathered[L]
            cos_mean = float(np.mean(vals))
            cos_std = float(np.std(vals))
            n = int(len(vals))

            # CKA from rank-0-local cached features
            X = torch.cat(cka_pred_tokens[L], dim=0).float()  # rank-0 only
            Y = torch.cat(cka_tgt_tokens[L], dim=0).float()
            cka_val = linear_cka(X, Y)

            writer.writerow([
                label, step, ckpt_path, L,
                f"{cos_mean:.6f}", f"{cos_std:.6f}", n,
                f"{cka_val:.6f}",
            ])
            print(
                f"[probe] step={step} L={L:>2}  "
                f"cos={cos_mean:+.4f} ± {cos_std:.4f}  CKA={cka_val:.4f}  (n={n})"
            )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
