# Stage-2 region metrics (plan 2026-09-27, section 一.10): per target view, from ground truth only
#   silhouette : pixels within 4 px of the GT alpha boundary (both sides)
#   unseen     : foreground pixels no input view sees (GT depth, 1 % depth test, pixel-centre convention)
#   seen       : foreground pixels at least one input view sees (outside the silhouette band)
#   texture    : foreground pixels outside the silhouette band whose GT image-gradient magnitude is in the
#                top 20 % of that view
#   depth_edge : foreground pixels outside the band with a > 5 % relative depth jump to a fg 8-neighbour
# For each region: summed squared error / pixel count (-> PSNR), summed spatial-LPIPS (VGG, same net as the
# main metric) / pixel count, and for depth_edge the abs_rel of the predicted depth.

import functools
import json
import os

import torch
import torch.nn.functional as F

from model_s2.geometry import inside_image, nearest_pixel, project

BAND = 4
TEX_TOP = 0.20
DEPTH_EDGE = 0.05
VIS_TAU = 0.01
REGIONS = ("silhouette", "unseen", "seen", "texture", "depth_edge")


@functools.lru_cache(maxsize=None)
def _spatial_lpips(device):
    from lpips import LPIPS
    return LPIPS(net="vgg", spatial=True).to(device).eval()


def _dilate(m, r):
    return F.max_pool2d(m.float()[:, None], 2 * r + 1, 1, r)[:, 0] > 0


def region_masks(alpha_t, depth_t, c2w_t, K_t, depth_in, alpha_in, c2w_in, K_in, img_t):
    """one object: alpha_t [Vt, H, W], depth_t [Vt, H, W], img_t [Vt, 3, H, W]; inputs [Vi, ...]"""
    Vt, H, W = alpha_t.shape
    dev = alpha_t.device
    fg = (alpha_t > 0.5) & (depth_t > 0)
    band = _dilate(alpha_t > 0.5, BAND) & _dilate(alpha_t <= 0.5, BAND)
    # GT visibility of each target pixel in the input views
    y, x = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float64),
                          torch.arange(W, device=dev, dtype=torch.float64), indexing="ij")
    d = depth_t.double()
    Kt = K_t.double()
    pc = torch.stack([(x[None] + 0.5 - Kt[:, 2, None, None]) / Kt[:, 0, None, None] * d,
                      (y[None] + 0.5 - Kt[:, 3, None, None]) / Kt[:, 1, None, None] * d, d], -1)   # [Vt,H,W,3]
    R, t = c2w_t[:, :3, :3].double(), c2w_t[:, :3, 3].double()
    X = torch.einsum("vij,vhwj->vhwi", R, pc) + t[:, None, None, :]
    Xf = X.reshape(Vt, 1, H * W, 3)
    u, v, z = project(Xf, c2w_in.double()[None], K_in.double()[None])                          # [Vt, Vi, HW]
    ins = inside_image(u, v, z, H, W)
    vi, ui = nearest_pixel(u, v, H, W)
    Vi = depth_in.shape[0]
    Din = depth_in.double().reshape(1, Vi, H * W).expand(Vt, Vi, H * W)
    Ain = (alpha_in.reshape(1, Vi, H * W) > 0.5).expand(Vt, Vi, H * W)
    Dk = torch.gather(Din, 2, vi * W + ui)
    Ak = torch.gather(Ain, 2, vi * W + ui)
    vis = (ins & Ak & (Dk > 0) & ((z - Dk).abs() <= VIS_TAU * Dk)).any(1).reshape(Vt, H, W)
    # texture: GT gradient magnitude
    g = img_t.float().mean(1, keepdim=True)
    gx = F.pad(g[..., :, 1:] - g[..., :, :-1], (0, 1))
    gy = F.pad(g[..., 1:, :] - g[..., :-1, :], (0, 0, 0, 1))
    gm = (gx.square() + gy.square()).sqrt()[:, 0]
    inner = fg & ~band
    tex = torch.zeros_like(inner)
    for i in range(Vt):
        vals = gm[i][inner[i]]
        if vals.numel() > 0:
            thr = torch.quantile(vals, 1.0 - TEX_TOP)
            tex[i] = inner[i] & (gm[i] >= thr)
    # depth edges
    dp = F.pad(d[:, None], (1, 1, 1, 1), mode="replicate")[:, 0]
    fp = F.pad(fg[:, None].float(), (1, 1, 1, 1))[:, 0] > 0
    mx = torch.zeros_like(d)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            dn = dp[:, 1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
            fn = fp[:, 1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
            rel = torch.where(fg & fn, (d - dn).abs() / d.clamp_min(1e-9), torch.zeros_like(d))
            mx = torch.maximum(mx, rel)
    dedge = inner & (mx > DEPTH_EDGE)
    return dict(silhouette=band, unseen=fg & ~vis, seen=inner & vis, texture=tex, depth_edge=dedge)


@torch.no_grad()
def region_metrics(result, batch_idx=0):
    """per-view region statistics of one object of an export-ready result edict (render, points, target, input)."""
    inp, tgt = result.input, result.target
    b = batch_idx
    img_gt = tgt.image[b].float().clamp(0, 1)
    pred = result.render[b].float().clamp(0, 1)
    Vt, _, H, W = img_gt.shape
    masks = region_masks(tgt.alpha_mask[b, :, 0].float(), tgt.depth_map[b].float().reshape(Vt, H, W),
                         tgt.c2w[b].float(), tgt.fxfycxcy[b].float(),
                         inp.depth_map[b].float().reshape(-1, H, W), inp.alpha_mask[b, :, 0].float(),
                         inp.c2w[b].float(), inp.fxfycxcy[b].float(), img_gt)
    se = (img_gt - pred).square().sum(1)                                           # [Vt, H, W]
    lp = _spatial_lpips(pred.device)(img_gt, pred, normalize=True)[:, 0].float()  # [Vt, H, W]
    # predicted depth (z of predicted points in the target camera)
    P = result.points[b].float().permute(0, 2, 3, 1)                               # [Vt, H, W, 3]
    R, t = tgt.c2w[b, :, :3, :3].float(), tgt.c2w[b, :, :3, 3].float()
    zpred = torch.einsum("vhwi,vi->vhw", P - t[:, None, None, :], R[:, :, 2])
    dgt = tgt.depth_map[b].float().reshape(Vt, H, W)
    out = []
    for i in range(Vt):
        row = {}
        for name in REGIONS:
            m = masks[name][i]
            n = int(m.sum())
            row[name] = dict(n=n, se=float(se[i][m].sum()) if n else 0.0, lpips=float(lp[i][m].sum()) if n else 0.0)
            if name == "depth_edge" and n:
                row[name]["absrel"] = float(((zpred[i][m] - dgt[i][m]).abs() / dgt[i][m]).sum())
        out.append(row)
    return out


def save_region_metrics(result, out_dir):
    for b in range(result.input.image.size(0)):
        uid = result.input.index[b, 0, -1].item()
        d = os.path.join(out_dir, f"{uid:06d}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "regions.json"), "w") as f:
            json.dump(region_metrics(result, b), f)
