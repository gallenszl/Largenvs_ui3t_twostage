import torch
from torch import Tensor
from jaxtyping import Float
from einops import reduce, rearrange
from skimage.metrics import structural_similarity
import functools
import os
from PIL import Image, ImageFilter
from utils import data_utils
import numpy as np
from easydict import EasyDict as edict
import json
from rich import print
import cv2
from model.vggt.utils.pose_enc import pose_encoding_to_extri_intri
import math


import warnings
# Suppress warnings for LPIPS loss loading
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated since 0.13")
warnings.filterwarnings("ignore", category=UserWarning, message="Arguments other than a weight enum.*")

@torch.no_grad()
def compute_psnr(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, "batch"]:
    """
    Compute Peak Signal-to-Noise Ratio between ground truth and predicted images.
    
    Args:
        ground_truth: Images with shape [batch, channel, height, width], values in [0, 1]
        predicted: Images with shape [batch, channel, height, width], values in [0, 1]
        
    Returns:
        PSNR values for each image in the batch
    """
    ground_truth = torch.clamp(ground_truth, 0, 1)
    predicted = torch.clamp(predicted, 0, 1)
    mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
    return -10 * torch.log10(mse) 



@functools.lru_cache(maxsize=None)
def get_lpips_model(net_type="vgg", device="cuda"):
    from lpips import LPIPS
    return LPIPS(net=net_type).to(device)

@torch.no_grad()
def compute_lpips(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
    normalize: bool = True,
) -> Float[Tensor, "batch"]:
    """
    Compute Learned Perceptual Image Patch Similarity between images.
    
    Args:
        ground_truth: Images with shape [batch, channel, height, width]
        predicted: Images with shape [batch, channel, height, width]
        The value range is [0, 1] when we have set the normalize flag to True.
        It will be [-1, 1] when the normalize flag is set to False.
    Returns:
        LPIPS values for each image in the batch (lower is better)
    """

    _lpips_fn = get_lpips_model(device=predicted.device)
    batch_size = 10  # Process in batches to save memory
    values = [
        _lpips_fn(
            ground_truth[i : i + batch_size],
            predicted[i : i + batch_size],
            normalize=normalize,
        )
        for i in range(0, ground_truth.shape[0], batch_size)
    ]
    return torch.cat(values, dim=0).squeeze()



@torch.no_grad()
def compute_ssim(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    """
    Compute Structural Similarity Index between images.
    
    Args:
        ground_truth: Images with shape [batch, channel, height, width], values in [0, 1]
        predicted: Images with shape [batch, channel, height, width], values in [0, 1]
        
    Returns:
        SSIM values for each image in the batch (higher is better)
    """
    ssim_values= []
    
    for gt, pred in zip(ground_truth, predicted):
        # Move to CPU and convert to numpy
        gt_np = gt.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        
        # Calculate SSIM
        ssim = structural_similarity(
            gt_np,
            pred_np,
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        ssim_values.append(ssim)
    
    # Convert back to tensor on the same device as input
    return torch.tensor(ssim_values, dtype=predicted.dtype, device=predicted.device)



# --- Foreground texture metrics (Wave0 E2) ---
# Ported from tmp/p0_texture_diagnostics/gso_targetinput_texture_metrics.py
# (texture_for_pair + PIL MinFilter erode), kept numerically identical so new
# numbers stay comparable with the 2026-07 texture diagnostics anchors
# (GSO t=i hf_ratio 0.4005 / CO3D t=i 0.3167). fg_ssim is new (masked skimage
# SSIM map). All helpers take numpy HWC float [0,1] images + HW bool mask.

def _fg_gray(img):
    return img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114


def _fg_lap_abs(g):
    p = np.pad(g, ((1, 1), (1, 1)), mode="edge")
    return np.abs(p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4.0 * p[1:-1, 1:-1])


def _fg_erode(mask):
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    mask_img = mask_img.filter(ImageFilter.MinFilter(3))
    return np.asarray(mask_img) > 0


def compute_fg_texture_metrics(gt, pred, mask):
    """Foreground texture metrics for one view.

    Args:
        gt, pred: numpy HWC float in [0, 1]
        mask: numpy HW bool, already eroded
    Returns dict with fg_psnr / hf_ratio / rgb_std_ratio / chroma_ratio /
    fg_ssim / mask_frac; NaN where the mask is empty or a ratio is undefined.
    """
    out = {
        "fg_psnr": float("nan"),
        "hf_ratio": float("nan"),
        "rgb_std_ratio": float("nan"),
        "chroma_ratio": float("nan"),
        "fg_ssim": float("nan"),
        "mask_frac": float(mask.mean()),
    }
    if mask.sum() == 0:
        return out
    diff = gt[mask] - pred[mask]
    mse = float(np.mean(diff * diff))
    out["fg_psnr"] = float("inf") if mse <= 0 else 20.0 * math.log10(1.0 / math.sqrt(mse))
    gt_hf = float(_fg_lap_abs(_fg_gray(gt))[mask].mean())
    pred_hf = float(_fg_lap_abs(_fg_gray(pred))[mask].mean())
    out["hf_ratio"] = pred_hf / gt_hf if gt_hf > 1e-12 else float("nan")
    gt_std = float(gt[mask].std())
    pred_std = float(pred[mask].std())
    out["rgb_std_ratio"] = pred_std / gt_std if gt_std > 1e-12 else float("nan")
    gt_chroma = float(np.std(gt[mask], axis=1).mean())
    pred_chroma = float(np.std(pred[mask], axis=1).mean())
    # 1e-3 floor (not 1e-12): near-gray GT objects otherwise blow the ratio up
    # to ~1e5 and poison the finite-mean aggregation (seen on GSO roll10).
    out["chroma_ratio"] = pred_chroma / gt_chroma if gt_chroma > 1e-3 else float("nan")
    _, ssim_map = structural_similarity(
        gt, pred, win_size=11, gaussian_weights=True, channel_axis=2,
        data_range=1.0, full=True,
    )
    out["fg_ssim"] = float(ssim_map[mask].mean())
    return out


FG_METRIC_KEYS = ("fg_psnr", "hf_ratio", "rgb_std_ratio", "chroma_ratio", "fg_ssim", "mask_frac")


@torch.no_grad()
def _export_results(
    result: edict,
    out_dir: str,
    compute_metrics: bool = False
):
    """
    Save results including images and optional metrics and videos.
    
    Args:
        result: EasyDict containing input, target, and rendered images, and optionally video frames
        out_dir: Directory to save the evaluation results
        compute_metrics: Whether to compute and save metrics
    """
    os.makedirs(out_dir, exist_ok=True)
    
    input_data, target_data = result.input, result.target
    
    for batch_idx in range(input_data.image.size(0)):
        uid = input_data.index[batch_idx, 0, -1].item()
        scene_name = input_data.scene_name[batch_idx]
        sample_dir = os.path.join(out_dir, f"{uid:06d}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Get target view indices
        target_indices = target_data.index[batch_idx, :, 0].cpu().numpy()
        
        # Save images
        _save_images(result, batch_idx, sample_dir)
        
        # Compute and save metrics if requested
        if compute_metrics:
            _save_metrics(
                target_data.image[batch_idx],
                result.render[batch_idx],
                target_indices,
                sample_dir,
                scene_name
            )
        
        # Save video if available
        if hasattr(result, "video_rendering"):
            _save_video(result.video_rendering[batch_idx], sample_dir)


def pts_to_depth(result):
    pts = result.points
    pose = result.target.c2w
    w2c = torch.inverse(pose)
    b,v,c,h,w = pts.shape
    ones = torch.ones((b,v,1,h,w), device=pts.device, dtype=pts.dtype)
    pts = torch.cat([pts, ones], dim=2)

    # w2c: b v 4 4
    # pts: b v 4 h w
    pts_cam = torch.einsum('bvij,bvjhw->bvihw', w2c, pts)
    depth = pts_cam[:, :, 2, :, :] / pts_cam[:, :, 3, :, :]
    result.depth = depth
    
def export_results(
    result: edict,
    out_dir: str, 
    compute_metrics: bool = False
):
    os.makedirs(out_dir, exist_ok=True)
    
    input_data, target_data = result.input, result.target
    
    for batch_idx in range(input_data.image.size(0)):
        uid = input_data.index[batch_idx, 0, -1].item()
        scene_name = input_data.scene_name[batch_idx]
        sample_dir = os.path.join(out_dir, f"{uid:06d}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Get target view indices
        target_indices = target_data.index[batch_idx, :, 0].cpu().numpy()

        pts_to_depth(result)
        
        # Save images
        _save_images(result, batch_idx, sample_dir)

        _save_depths(result, batch_idx, sample_dir)
        
        # Compute and save metrics if requested
        if compute_metrics:
            # Wave0 E2: dataset-provided target alpha rides along in result.target
            # (ProcessData.fetch_views gathers every batch key); None otherwise.
            target_alpha_all = getattr(target_data, "alpha_mask", None)
            _save_metrics(
                target_data.image[batch_idx],
                result.render[batch_idx],
                target_indices,
                sample_dir,
                scene_name,
                target_alpha=None if target_alpha_all is None else target_alpha_all[batch_idx],
            )

            _save_metrics_depth(
                target_data.depth_map[batch_idx],
                result.depth[batch_idx],
                target_indices,
                sample_dir,
                scene_name
            )

            # print(result.camera[-1].shape, input_data.c2w.shape, batch_idx)
            v_target = target_indices.shape[0]

            _save_metrics_pose(
                input_data.c2w[batch_idx],
                result.camera[-1][batch_idx*v_target],
                target_indices,
                sample_dir,
                scene_name
            )


def visualize_intermediate_results(out_dir, result):
    os.makedirs(out_dir, exist_ok=True)

    input, target = result.input, result.target

    if result.render is not None:
            
        target_image = target.image
        rendered_image = result.render
        b, v, _, h, w = rendered_image.size()
        rendered_image = rendered_image.reshape(b * v, -1, h, w)
        target_image = target_image.reshape(b * v, -1, h, w)
        if hasattr(result, 'points'):
            points_est = result.points
            points_gt  = target.point_map
            points_est = points_est.reshape(b * v, -1, h, w)
            points_gt  = points_gt.reshape(b * v, -1, h, w)
            
            points_est = points_est / 2 + 0.5
            points_gt  = points_gt  / 2 + 0.5

            visualized_image = torch.cat((target_image, rendered_image, points_est, points_gt), dim=3).detach().cpu()
            visualized_image = rearrange(visualized_image, "(b v) c h (m w) -> (b m h) (v w) c", v=v, m=4)
        else:
            visualized_image = torch.cat((target_image, rendered_image), dim=3).detach().cpu()
            visualized_image = rearrange(visualized_image, "(b v) c h (m w) -> (b h) (v m w) c", v=v, m=2)
        visualized_image = (visualized_image.float().numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
        
        uids = [target.index[b, 0, -1].item() for b in range(target.index.size(0))]

        uid_based_filename = f"{uids[0]:08}_{uids[-1]:08}"
        Image.fromarray(visualized_image).save(
            os.path.join(out_dir, f"supervision_{uid_based_filename}.jpg")
        )
        with open(os.path.join(out_dir, f"uids.txt"), "w") as f:
            uids = "_".join([f"{uid:08}" for uid in uids])
            f.write(uids)

    input_uids = [input.index[b, 0, -1].item() for b in range(input.index.size(0))]
    input_uid_based_filename = f"{input_uids[0]:08}_{input_uids[-1]:08}"
    
    # Create a grid of input images
    b, v, c, h, w = input.image.size()
    input_images = input.image.reshape(b * v, c, h, w).detach().float().cpu()
    input_grid = rearrange(input_images, "(b v) c h w -> (b h) (v w) c", v=v)
    input_grid = (input_grid.numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
    
    # Save the input image grid
    Image.fromarray(input_grid).save(
        os.path.join(out_dir, f"input_{input_uid_based_filename}.jpg")
    )


def _save_images(result, batch_idx, out_dir):
    """Save visualization images."""
    # Save input image
    input_img = result.input.image[batch_idx]
    input_img = rearrange(input_img, "v c h w -> h (v w) c")
    input_img = (input_img.cpu().numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
    Image.fromarray(input_img).save(os.path.join(out_dir, "input.png"))

    # Save GT vs prediction side-by-side
    comparison = torch.cat(
        (result.target.image[batch_idx], result.render[batch_idx]), 
        dim=2
    ).detach().cpu()
    comparison = rearrange(comparison, "v c h w -> h (v w) c")
    comparison = (comparison.numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
    Image.fromarray(comparison).save(os.path.join(out_dir, "gt_vs_pred.png"))
    

def _save_depths(result, batch_idx, out_dir):
    """Save visualization images."""

    # 获取GT和pred
    gt = result.target.depth_map[batch_idx].detach().cpu()  # v h w
    pred = result.depth[batch_idx].detach().cpu()           # v h w

    # 对齐背景区域：GT为0的位置，pred也设为0
    mask = gt > 0
    pred = pred.float()
    pred = pred * mask.float()
    pred = np.maximum(pred, 1e-6)


    # 归一化并应用颜色映射
    gt_np = gt.numpy()
    pred_np = pred.numpy()

    # 使用GT的全局min/max范围来归一化GT和pred，确保颜色一致
    global_min = gt_np.min()
    global_max = gt_np.max()

    def _normalize_apply_colormap(depth, d_min, d_max):
        v, h, w = depth.shape
        colored_list = []
        for i in range(v):
            d = depth[i]
            d_normalized = np.clip((d - d_min) / (d_max - d_min) * 255, 0, 255).astype(np.uint8)
            d_colored = cv2.applyColorMap(d_normalized, cv2.COLORMAP_JET)
            colored_list.append(d_colored)
        return np.stack(colored_list, axis=0)  # v h w c

    gt_colored = _normalize_apply_colormap(gt_np, global_min, global_max)   # v h w c
    pred_colored = _normalize_apply_colormap(pred_np, global_min, global_max) # v h w c

    # 纵向拼接：第一行GT，第二行pred
    # 先将v个视图按宽度拼接，然后纵向合并GT和pred
    gt_row = np.concatenate([gt_colored[i] for i in range(gt_colored.shape[0])], axis=1)  # h (v*w) c
    pred_row = np.concatenate([pred_colored[i] for i in range(pred_colored.shape[0])], axis=1)  # h (v*w) c
    comparison = np.concatenate([gt_row, pred_row], axis=0)  # 2*h (v*w) c
    Image.fromarray(comparison).save(os.path.join(out_dir, "gt_vs_pred_depth.png"))


def _save_metrics(target, prediction, view_indices, out_dir, scene_name, target_alpha=None):
    target = target.to(torch.float32)
    prediction = prediction.to(torch.float32)
    psnr_values = compute_psnr(target, prediction)
    lpips_values = compute_lpips(target, prediction)
    ssim_values = compute_ssim(target, prediction)

    if lpips_values.ndim == 0:
        lpips_values = lpips_values.unsqueeze(0)

    # Wave0 E2: per-view foreground texture metrics when a real alpha mask is
    # available (target_alpha: [v, 1, h, w], values in {0, 1}). Datasets that
    # don't emit alpha_mask keep the exact legacy metrics.json layout.
    fg_rows = None
    if target_alpha is not None:
        fg_rows = []
        gt_np = target.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        pred_np = prediction.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        alpha_np = target_alpha.to(torch.float32).cpu().numpy()
        for i in range(gt_np.shape[0]):
            mask = _fg_erode(alpha_np[i, 0] > 0.5)
            fg_rows.append(compute_fg_texture_metrics(gt_np[i], pred_np[i], mask))

    metrics = {
        "summary": {
            "scene_name": scene_name,
            "psnr": float(psnr_values.mean()),
            "lpips": float(lpips_values.mean()),
            "ssim": float(ssim_values.mean())
        },
        "per_view": []
    }

    if fg_rows is not None:
        for key in FG_METRIC_KEYS:
            vals = np.array([row[key] for row in fg_rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                metrics["summary"][key] = float(vals.mean())

    for i, view_idx in enumerate(view_indices):
        entry = {
            "view": int(view_idx), "psnr": float(psnr_values[i]), "lpips": float(lpips_values[i]), "ssim": float(ssim_values[i])
        }
        if fg_rows is not None:
            entry.update(fg_rows[i])
        metrics["per_view"].append(entry)

    # Save metrics to a single JSON file
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def _save_metrics_depth(target, prediction, view_indices, out_dir, scene_name):
    # print(target.shape)
    # print(prediction.shape)
    
    target = target.to(torch.float32).cpu().numpy()
    prediction = prediction.to(torch.float32).cpu().numpy()

    metric_names = ["silog", "abs_rel", "log10", "rms", "sq_rel", "log_rms", "d1", "d2", "d3"]
    metric_lists = {name: [] for name in metric_names}
    per_view = []
    warnings = []

    for b_idx in range(target.shape[0]):
        gt = target[b_idx]
        mask = gt > 0
        pred = prediction[b_idx]
        valid_depth_pixels = int(mask.sum())

        if valid_depth_pixels == 0:
            view_idx = int(view_indices[b_idx])
            warning = (
                f"WARNING: no valid GT depth pixels; skipping depth metrics "
                f"for scene={scene_name}, view={view_idx}"
            )
            print(warning)
            warnings.append(warning)
            per_view.append({
                "view": view_idx,
                "skipped": True,
                "valid_depth_pixels": valid_depth_pixels,
                **{name: None for name in metric_names},
            })
            continue

        gt = gt[mask]
        pred = pred[mask]
        pred = np.maximum(pred, 1e-6)  # Ensure pred is always positive for log computation

        values = compute_errors(gt, pred)
        if not np.all(np.isfinite(values)):
            view_idx = int(view_indices[b_idx])
            warning = (
                f"WARNING: non-finite depth metric; skipping depth metrics "
                f"for scene={scene_name}, view={view_idx}"
            )
            print(warning)
            warnings.append(warning)
            per_view.append({
                "view": view_idx,
                "skipped": True,
                "valid_depth_pixels": valid_depth_pixels,
                **{name: None for name in metric_names},
            })
            continue

        view_metrics = {
            "view": int(view_indices[b_idx]),
            "skipped": False,
            "valid_depth_pixels": valid_depth_pixels,
        }
        for name, value in zip(metric_names, values):
            value = float(value)
            metric_lists[name].append(value)
            view_metrics[name] = value
        per_view.append(view_metrics)

    summary = {"scene_name": scene_name}
    for name in metric_names:
        if metric_lists[name]:
            summary[name] = float(np.mean(metric_lists[name]))
        else:
            summary[name] = None

    metrics = {
        "summary": summary,
        "per_view": per_view,
        "warnings": warnings,
        "valid_view_count": sum(1 for item in per_view if not item["skipped"]),
        "skipped_view_count": sum(1 for item in per_view if item["skipped"]),
    }
    
    # Save metrics to a single JSON file
    with open(os.path.join(out_dir, "metrics_depth.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog_var = np.mean(err ** 2) - np.mean(err) ** 2
    silog = np.sqrt(max(silog_var, 0.0)) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]


def _save_metrics_pose(target, prediction, view_indices, out_dir, scene_name):
    target = target.to(torch.float32).cpu()
    prediction = prediction.to(torch.float32).cpu()
    pred_ext, intrinsic = pose_encoding_to_extri_intri(prediction.unsqueeze(0), (256,256))

    ext = torch.eye(4).unsqueeze(0).repeat(prediction.shape[0], 1,1)
    pred_ext = pred_ext.squeeze(0)
    ext[:,:3,:] = pred_ext
    pred_pose = torch.inverse(ext)

    '''
    pred_pose = pred_pose.float()
    t1 = pred_pose[0, :3, 3].clone()
    target_pos = torch.tensor([0, 0, -1], dtype=torch.float32, device=pred_pose.device)

    T = torch.eye(4, dtype=torch.float32, device=pred_pose.device)
    T[:3, 3] = target_pos - t1
    pred_pose = T @ pred_pose
    target_first_view = torch.tensor(
        [[1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 0, 0, 1]], dtype=torch.float32, device=pred_pose.device
    )
    c2w_first = pred_pose[0].clone()

    norm_first_view_t = torch.norm(c2w_first[:3, 3])
    c2w_first[:3, 3] /= norm_first_view_t
    inv_first_view = torch.inverse(c2w_first)

    c2w_first_gt = target[0].clone()
    norm_first_view_t_gt = torch.norm(c2w_first_gt[:3, 3])
    c2w_first_gt[:3, 3] /= norm_first_view_t_gt
    inv_first_view_gt = torch.inverse(c2w_first_gt)

    for i in range(pred_pose.shape[0]):
        # pred_pose[i, :3, 3] /= norm_first_view_t
        pred_pose[i] = target_first_view @ inv_first_view @ pred_pose[i]

        # target[i, :3, 3] /= norm_first_view_t_gt
        target[i] = target_first_view @ inv_first_view_gt @ target[i]
    '''
    # print(scene_name)
    # print(target)
    # print(pred)
    # print('----')

    rErrors, tErrors = compute_pose_error(pred_pose, target)
    metrics = {
        "summary":{
            "scene_name": scene_name,
            "rError": float(rErrors.mean()),
            "tError": float(tErrors.mean()),
        },
        "pose": {
            "pred_ext": pred_ext[:,:3].numpy().tolist(),
            "gt_ext": torch.inverse(target)[:,:3].numpy().tolist(),
            "pred_pose": pred_pose.numpy().tolist(),
            "gt_pose": target.numpy().tolist(),
        },
        "per_pair": []
    }

    for i, pair_idx in enumerate(range(rErrors.shape[0])):
        metrics["per_pair"].append({
            "pair": int(pair_idx),
            "rError": float(rErrors[i]),
            "tError": float(tErrors[i])
        })

    with open(os.path.join(out_dir, "metrics_pose.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def _compute_pose_error(preds, gts):
    assert len(preds.shape) == 3, 'pred cam must be n*4*4'
    assert len(gts.shape) == 3, 'gt cam must be n*4*4'

    n_cam = preds.shape[0]
    pair_idx_i1, pair_idx_i2 = batched_all_pairs(1, n_cam)
    
    preds = torch.inverse(preds[pair_idx_i1]).bmm(preds[pair_idx_i2])
    gts   = torch.inverse(gts[pair_idx_i1]).bmm(gts[pair_idx_i2])
    
    r_errors = []
    t_errors = []
    for pred, gt in zip(preds, gts):
        # pred, gt: [4,4]
        R_pred = pred[:3,:3]
        t_pred = pred[:3,3]
        R_gt = gt[:3,:3]
        t_gt = gt[:3,3]
        R_diff = R_pred @ R_gt.T
        cos_theta = (np.trace(R_diff) - 1) / 2
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        t_error = np.linalg.norm(t_pred - t_gt)
        r_errors.append(np.degrees(theta)) 
        t_errors.append(t_error)
    r_errors = np.array(r_errors)
    t_errors = np.array(t_errors)
    
    # return r_errors, t_errors
    return torch.from_numpy(r_errors), torch.from_numpy(t_errors)


def compute_pose_error(preds, gts):
    # pred_cameras = PerspectiveCameras(R=preds[:,:3,:3], T=preds[:,:3,3])
    # gt_cameras = PerspectiveCameras(R=gts[:,:3,:3], T=gts[:,:3,3])
    # gt_se3 = gt_cameras.get_world_to_view_transform().get_matrix()
    # pred_se3 = pred_cameras.get_world_to_view_transform().get_matrix()

    gt_se3 = torch.eye(4).unsqueeze(0).repeat(gts.shape[0],1,1).contiguous()
    pred_se3 = torch.eye(4).unsqueeze(0).repeat(preds.shape[0],1,1).contiguous()

    gt_se3[:,:3,:3] = gts[:,:3,:3]
    gt_se3[:, 3,:3] = gts[:,:3, 3]
    pred_se3[:,:3,:3] = preds[:,:3,:3]
    pred_se3[:, 3,:3] = preds[:,:3, 3]

    n_cam = preds.shape[0]
    pair_idx_i1, pair_idx_i2 = batched_all_pairs(1, n_cam)

    relative_pose_gt = closed_form_inverse(gt_se3[pair_idx_i1]).bmm(gt_se3[pair_idx_i2])
    relative_pose_pred = closed_form_inverse(pred_se3[pair_idx_i1]).bmm(pred_se3[pair_idx_i2])

    rel_rangle_deg = rotation_angle(relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3])
    rel_tangle_deg = translation_angle(relative_pose_gt[:, 3, :3], relative_pose_pred[:, 3, :3])

    return rel_rangle_deg, rel_tangle_deg

def closed_form_inverse(se3):
    R = se3[:, :3, :3]
    T = se3[:, 3:, :3]

    # Compute the transpose of the rotation
    R_transposed = R.transpose(1, 2)

    # Compute the left part of the inverse transformation
    left_bottom = -T.bmm(R_transposed)
    left_combined = torch.cat((R_transposed, left_bottom), dim=1)

    # Keep the right-most column as it is
    right_col = se3[:, :, 3:].detach().clone()
    inverted_matrix = torch.cat((left_combined, right_col), dim=-1)

    return inverted_matrix


def rotation_angle(rot_gt, rot_pred, batch_size=None):
    # rot_gt, rot_pred (B, 3, 3)
    rel_angle_cos = so3_relative_angle(rot_gt, rot_pred, eps=1e-4)
    rel_rangle_deg = rel_angle_cos * 180 / np.pi

    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)

    return rel_rangle_deg


def translation_angle(tvec_gt, tvec_pred, batch_size=None):
    # tvec_gt, tvec_pred (B, 3,)
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi

    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

    return rel_tangle_deg


def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e6):
    """Normalize the translation vectors and compute the angle between them."""
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t


def so3_relative_angle(
    R1: torch.Tensor,
    R2: torch.Tensor,
    cos_angle: bool = False,
    cos_bound: float = 1e-4,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Calculates the relative angle (in radians) between pairs of
    rotation matrices `R1` and `R2` with `angle = acos(0.5 * (Trace(R1 R2^T)-1))`

    .. note::
        This corresponds to a geodesic distance on the 3D manifold of rotation
        matrices.

    Args:
        R1: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        R2: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        cos_angle: If==True return cosine of the relative angle rather than
            the angle itself. This can avoid the unstable calculation of `acos`.
        cos_bound: Clamps the cosine of the relative rotation angle to
            [-1 + cos_bound, 1 - cos_bound] to avoid non-finite outputs/gradients
            of the `acos` call. Note that the non-finite outputs/gradients
            are returned when the angle is requested (i.e. `cos_angle==False`)
            and the rotation angle is close to 0 or π.
        eps: Tolerance for the valid trace check of the relative rotation matrix
            in `so3_rotation_angle`.
    Returns:
        Corresponding rotation angles of shape `(minibatch,)`.
        If `cos_angle==True`, returns the cosine of the angles.

    Raises:
        ValueError if `R1` or `R2` is of incorrect shape.
        ValueError if `R1` or `R2` has an unexpected trace.
    """
    R12 = torch.bmm(R1, R2.permute(0, 2, 1))
    return so3_rotation_angle(R12, cos_angle=cos_angle, cos_bound=cos_bound, eps=eps)


def so3_rotation_angle(
    R: torch.Tensor,
    eps: float = 1e-4,
    cos_angle: bool = False,
    cos_bound: float = 1e-4,
) -> torch.Tensor:
    """
    Calculates angles (in radians) of a batch of rotation matrices `R` with
    `angle = acos(0.5 * (Trace(R)-1))`. The trace of the
    input matrices is checked to be in the valid range `[-1-eps,3+eps]`.
    The `eps` argument is a small constant that allows for small errors
    caused by limited machine precision.

    Args:
        R: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        eps: Tolerance for the valid trace check.
        cos_angle: If==True return cosine of the rotation angles rather than
            the angle itself. This can avoid the unstable
            calculation of `acos`.
        cos_bound: Clamps the cosine of the rotation angle to
            [-1 + cos_bound, 1 - cos_bound] to avoid non-finite outputs/gradients
            of the `acos` call. Note that the non-finite outputs/gradients
            are returned when the angle is requested (i.e. `cos_angle==False`)
            and the rotation angle is close to 0 or π.

    Returns:
        Corresponding rotation angles of shape `(minibatch,)`.
        If `cos_angle==True`, returns the cosine of the angles.

    Raises:
        ValueError if `R` is of incorrect shape.
        ValueError if `R` has an unexpected trace.
    """

    N, dim1, dim2 = R.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    rot_trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    if ((rot_trace < -1.0 - eps) + (rot_trace > 3.0 + eps)).any():
        raise ValueError("A matrix has trace outside valid range [-1-eps,3+eps].")

    # phi ... rotation angle
    phi_cos = (rot_trace - 1.0) * 0.5

    if cos_angle:
        return phi_cos
    else:
        if cos_bound > 0.0:
            bound = 1.0 - cos_bound
            return acos_linear_extrapolation(phi_cos, (-bound, bound))
        else:
            return torch.acos(phi_cos)


DEFAULT_ACOS_BOUND: float = 1.0 - 1e-4


def acos_linear_extrapolation(
    x: torch.Tensor,
    bounds = (-DEFAULT_ACOS_BOUND, DEFAULT_ACOS_BOUND),
) -> torch.Tensor:
    """
    Implements `arccos(x)` which is linearly extrapolated outside `x`'s original
    domain of `(-1, 1)`. This allows for stable backpropagation in case `x`
    is not guaranteed to be strictly within `(-1, 1)`.

    More specifically::

        bounds=(lower_bound, upper_bound)
        if lower_bound <= x <= upper_bound:
            acos_linear_extrapolation(x) = acos(x)
        elif x <= lower_bound: # 1st order Taylor approximation
            acos_linear_extrapolation(x)
                = acos(lower_bound) + dacos/dx(lower_bound) * (x - lower_bound)
        else:  # x >= upper_bound
            acos_linear_extrapolation(x)
                = acos(upper_bound) + dacos/dx(upper_bound) * (x - upper_bound)

    Args:
        x: Input `Tensor`.
        bounds: A float 2-tuple defining the region for the
            linear extrapolation of `acos`.
            The first/second element of `bound`
            describes the lower/upper bound that defines the lower/upper
            extrapolation region, i.e. the region where
            `x <= bound[0]`/`bound[1] <= x`.
            Note that all elements of `bound` have to be within (-1, 1).
    Returns:
        acos_linear_extrapolation: `Tensor` containing the extrapolated `arccos(x)`.
    """

    lower_bound, upper_bound = bounds

    if lower_bound > upper_bound:
        raise ValueError("lower bound has to be smaller or equal to upper bound.")

    if lower_bound <= -1.0 or upper_bound >= 1.0:
        raise ValueError("Both lower bound and upper bound have to be within (-1, 1).")

    # init an empty tensor and define the domain sets
    acos_extrap = torch.empty_like(x)
    x_upper = x >= upper_bound
    x_lower = x <= lower_bound
    x_mid = (~x_upper) & (~x_lower)

    # acos calculation for upper_bound < x < lower_bound
    acos_extrap[x_mid] = torch.acos(x[x_mid])
    # the linear extrapolation for x >= upper_bound
    acos_extrap[x_upper] = _acos_linear_approximation(x[x_upper], upper_bound)
    # the linear extrapolation for x <= lower_bound
    acos_extrap[x_lower] = _acos_linear_approximation(x[x_lower], lower_bound)

    return acos_extrap


def _acos_linear_approximation(x: torch.Tensor, x0: float) -> torch.Tensor:
    """
    Calculates the 1st order Taylor expansion of `arccos(x)` around `x0`.
    """
    return (x - x0) * _dacos_dx(x0) + math.acos(x0)


def _dacos_dx(x: float) -> float:
    """
    Calculates the derivative of `arccos(x)` w.r.t. `x`.
    """
    return (-1.0) / math.sqrt(1.0 - x * x)


def calculate_auc_np(r_error, t_error, max_threshold=30):
    """
    Calculate the Area Under the Curve (AUC) for the given error arrays.

    :param r_error: numpy array representing R error values (Degree).
    :param t_error: numpy array representing T error values (Degree).
    :param max_threshold: maximum threshold value for binning the histogram.
    :return: cumulative sum of normalized histogram of maximum error values.
    """

    # Concatenate the error arrays along a new axis
    error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)

    # Compute the maximum error value for each pair
    max_errors = np.max(error_matrix, axis=1)

    # Define histogram bins
    bins = np.arange(max_threshold + 1)

    # Calculate histogram of maximum error values
    histogram, _ = np.histogram(max_errors, bins=bins)

    # Normalize the histogram
    num_pairs = float(len(max_errors))
    normalized_histogram = histogram.astype(float) / num_pairs

    # Compute and return the cumulative sum of the normalized histogram
    return np.mean(np.cumsum(normalized_histogram))


def calculate_auc(r_error, t_error, max_threshold=30):
    """
    Calculate the Area Under the Curve (AUC) for the given error arrays using PyTorch.

    :param r_error: torch.Tensor representing R error values (Degree).
    :param t_error: torch.Tensor representing T error values (Degree).
    :param max_threshold: maximum threshold value for binning the histogram.
    :return: cumulative sum of normalized histogram of maximum error values.
    """

    # Concatenate the error tensors along a new axis
    error_matrix = torch.stack((r_error, t_error), dim=1)

    # Compute the maximum error value for each pair
    max_errors, _ = torch.max(error_matrix, dim=1)

    # Define histogram bins
    bins = torch.arange(max_threshold + 1)

    # Calculate histogram of maximum error values
    histogram = torch.histc(max_errors, bins=max_threshold + 1, min=0, max=max_threshold)

    # Normalize the histogram
    num_pairs = float(max_errors.size(0))
    normalized_histogram = histogram / num_pairs

    # Compute and return the cumulative sum of the normalized histogram
    return torch.cumsum(normalized_histogram, dim=0).mean()


def batched_all_pairs(B, N):
    # B, N = se3.shape[:2]
    i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    i1, i2 = [(i[None] + torch.arange(B)[:, None] * N).reshape(-1) for i in [i1_, i2_]]

    return i1, i2


def _save_video(frames, out_dir):
    """
    Save video from rendered frames.
    Input frames should be in [v, c, h, w] format.
    """
    frames = np.ascontiguousarray(np.array(frames.to(torch.float32)))
    frames = rearrange(frames, "v c h w -> v h w c")
    data_utils.create_video_from_frames(
        frames, 
        f"{out_dir}/rendered_video.mp4", 
        framerate=30
    )


def summarize_evaluation(evaluation_folder, ret_dict=False):
    # Find and sort all valid subfolders
    subfolders = sorted(
        [
            os.path.join(evaluation_folder, dirname)
            for dirname in os.listdir(evaluation_folder)
            if os.path.isdir(os.path.join(evaluation_folder, dirname))
        ],
        key=lambda x: int(os.path.basename(x)) if os.path.basename(x).isdigit() else os.path.basename(x)
    )

    metrics = {}
    valid_subfolders = []
    
    for subfolder in subfolders:
        json_path = os.path.join(subfolder, "metrics.json")
        if not os.path.exists(json_path):
            print(f"!!! Metrics file not found in {subfolder}, skipping...")
            continue
            
        valid_subfolders.append(subfolder)
        
        with open(json_path, "r") as f:
            try:
                data = json.load(f)
                # Extract summary metrics
                for metric_name, metric_value in data["summary"].items():
                    if metric_name == "scene_name":
                        continue
                    metrics.setdefault(metric_name, []).append(metric_value)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error reading metrics from {json_path}: {e}")

    if not valid_subfolders:
        print(f"No valid metrics files found in {evaluation_folder}")
        return

    csv_file = os.path.join(evaluation_folder, "summary.csv")
    with open(csv_file, "w") as f:
        header = ["Index"] + list(metrics.keys())
        f.write(",".join(header) + "\n")
        
        for i, subfolder in enumerate(valid_subfolders):
            basename = os.path.basename(subfolder)
            values = [str(metric_values[i]) for metric_values in metrics.values()]
            f.write(f"{basename},{','.join(values)}\n")
        
        f.write("\n")
        
        averages = [str(sum(values) / len(values)) for values in metrics.values()]
        f.write(f"average,{','.join(averages)}\n")
    
    print(f"Summary written to {csv_file}")
    print(f"Average: {','.join(averages)}")

    # export average metrics to a text file
    with open(os.path.join(evaluation_folder, "average_metrics.txt"), "w") as f:
        f.write(f"Average: {','.join(averages)}\n")

    if ret_dict:
        avg_metric_dict = {k:v for k,v in zip(metrics.keys(), averages)}
        return avg_metric_dict


def summarize_evaluation_depth(evaluation_folder, ret_dict=False):
    # Find and sort all valid subfolders
    subfolders = sorted(
        [
            os.path.join(evaluation_folder, dirname)
            for dirname in os.listdir(evaluation_folder)
            if os.path.isdir(os.path.join(evaluation_folder, dirname))
        ],
        key=lambda x: int(os.path.basename(x)) if os.path.basename(x).isdigit() else os.path.basename(x)
    )

    metrics = {}
    valid_subfolders = []
    
    for subfolder in subfolders:
        json_path = os.path.join(subfolder, "metrics_depth.json")
        if not os.path.exists(json_path):
            print(f"!!! Metrics file not found in {subfolder}, skipping...")
            continue
            
        valid_subfolders.append(subfolder)
        
        with open(json_path, "r") as f:
            try:
                data = json.load(f)
                # Extract summary metrics
                for metric_name, metric_value in data["summary"].items():
                    if metric_name == "scene_name":
                        continue
                    metrics.setdefault(metric_name, []).append(metric_value)
                    if metric_value is None or not np.isfinite(float(metric_value)):
                        print(
                            f"WARNING: non-finite summary depth metric "
                            f"{metric_name} in {json_path}; excluding from average"
                        )
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error reading metrics from {json_path}: {e}")

    if not valid_subfolders:
        print(f"No valid metrics files found in {evaluation_folder}")
        return

    csv_file = os.path.join(evaluation_folder, "summary_depth.csv")
    with open(csv_file, "w") as f:
        header = ["Index"] + list(metrics.keys())
        f.write(",".join(header) + "\n")
        
        for i, subfolder in enumerate(valid_subfolders):
            basename = os.path.basename(subfolder)
            values = [str(metric_values[i]) for metric_values in metrics.values()]
            f.write(f"{basename},{','.join(values)}\n")
        
        f.write("\n")
        
        averages = []
        for values in metrics.values():
            finite_values = [
                float(v) for v in values
                if v is not None and np.isfinite(float(v))
            ]
            if finite_values:
                averages.append(str(sum(finite_values) / len(finite_values)))
            else:
                averages.append("nan")
        f.write(f"average,{','.join(averages)}\n")
    
    print(f"Summary written to {csv_file}")
    print(f"Average: {','.join(averages)}")

    # export average metrics to a text file
    with open(os.path.join(evaluation_folder, "average_metrics_depth.txt"), "w") as f:
        f.write(f"Average:\n")
        f.write(f"{','.join(metrics.keys())}\n")
        f.write(f"{','.join(averages)}\n")

    if ret_dict:
        avg_metric_dict = {k:v for k,v in zip(metrics.keys(), averages)}
        return avg_metric_dict


def summarize_evaluation_pose(evaluation_folder, ret_dict=False):
    # Find and sort all valid subfolders
    subfolders = sorted(
        [
            os.path.join(evaluation_folder, dirname)
            for dirname in os.listdir(evaluation_folder)
            if os.path.isdir(os.path.join(evaluation_folder, dirname))
        ],
        key=lambda x: int(os.path.basename(x)) if os.path.basename(x).isdigit() else os.path.basename(x)
    )

    metrics = {}
    valid_subfolders = []
    
    for subfolder in subfolders:
        json_path = os.path.join(subfolder, "metrics_pose.json")
        if not os.path.exists(json_path):
            print(f"!!! Metrics file not found in {subfolder}, skipping...")
            continue
            
        valid_subfolders.append(subfolder)
        
        with open(json_path, "r") as f:
            try:
                data = json.load(f)
                # Extract summary metrics
                for metric_name, metric_value in data["summary"].items():
                    if metric_name == "scene_name":
                        continue
                    metrics.setdefault(metric_name, []).append(metric_value)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error reading metrics from {json_path}: {e}")

    if not valid_subfolders:
        print(f"No valid metrics files found in {evaluation_folder}")
        return

    csv_file = os.path.join(evaluation_folder, "summary_pose.csv")
    with open(csv_file, "w") as f:
        header = ["Index"] + list(metrics.keys())
        f.write(",".join(header) + "\n")
        
        for i, subfolder in enumerate(valid_subfolders):
            basename = os.path.basename(subfolder)
            values = [str(metric_values[i]) for metric_values in metrics.values()]
            f.write(f"{basename},{','.join(values)}\n")
        
        f.write("\n")
        
        averages = [str(sum(values) / len(values)) for values in metrics.values()]
        f.write(f"average,{','.join(averages)}\n")
        
    rError = np.array(metrics['rError'])
    tError = np.array(metrics['tError'])

    Racc_5 = np.mean(rError < 5) * 100
    Racc_15 = np.mean(rError < 15) * 100
    Racc_30 = np.mean(rError < 30) * 100

    Tacc_5 = np.mean(tError < 5) * 100
    Tacc_15 = np.mean(tError < 15) * 100
    Tacc_30 = np.mean(tError < 30) * 100

    Auc_30 = calculate_auc_np(rError, tError, max_threshold=30) * 100

    print(f"Summary written to {csv_file}")
    print(f"Average: {','.join(averages)}")
    print(f"Racc_5: {Racc_5:.2f}%")
    print(f"Racc_15: {Racc_15:.2f}%")
    print(f"Racc_30: {Racc_30:.2f}%")
    print(f"Tacc_5: {Tacc_5:.2f}%")
    print(f"Tacc_15: {Tacc_15:.2f}%")
    print(f"Tacc_30: {Tacc_30:.2f}%")
    print(f"Auc_30: {Auc_30:.2f}%")

    # export average metrics to a text file
    with open(os.path.join(evaluation_folder, "average_metrics_pose.txt"), "w") as f:
        f.write(f"Average: {','.join(averages)}\n")
        f.write(f"Racc_5: {Racc_5:.5f}%\n")
        f.write(f"Racc_15: {Racc_15:.5f}%\n")
        f.write(f"Racc_30: {Racc_30:.5f}%\n")
        f.write(f"Tacc_5: {Tacc_5:.5f}%\n")
        f.write(f"Tacc_15: {Tacc_15:.5f}%\n")
        f.write(f"Tacc_30: {Tacc_30:.5f}%\n")
        f.write(f"Auc_30: {Auc_30:.5f}%\n")

    if ret_dict:
        avg_metric_dict = {k:v for k,v in zip(metrics.keys(), averages)}
        # UNI3T: Auc_30 was computed above but only printed / written to the txt file,
        # so callers asking for ret_dict could never see it.
        avg_metric_dict["Auc_30"] = Auc_30
        return avg_metric_dict
