"""Track-consistency loss for the uni3t stage-1 training, after VGGT-Omega's open-source
`training/losses/base_loss/consistency.py` + `training/data/track_util.py`, restricted to the target
views (the views the point head predicts).

For every ordered pair of target views (a, b) and every foreground pixel u of a:
  * the GT point of u (the loader's point_map, i.e. the same target the point loss regresses to) is
    projected into b with b's GT camera -> continuous (u_b, v_b) (pixel i covers [i, i+1)) and depth z;
  * b's GT depth D is read at the containing pixel and, optionally, at a 4-neighbour set; the pixel
    whose depth is closest to z is kept; the pair is visible when D > 0, the pixel is `border` pixels
    inside the image, and |z - D| < tau * D + abs_tol and |z - D| < tau * z + abs_tol (Omega's test);
  * loss entry = |(P_hat_a[u] - P_a[u]) - (P_hat_b[u_b] - P_b[u_b])| per coordinate, i.e. the
    difference of the two prediction errors, averaged over all visible entries x 3 coordinates.
Geometry is fp32 element-wise arithmetic (no matmul: TF32 cannot move a visibility decision); the 4x4
camera inverses are taken in fp64.  Tracks come from GT only and are rebuilt on the GPU every step.
Optional sampling reproduces the reference: per (sample, query view) keep the foreground pixels seen
by the most other views (top half when that half exceeds num_tracks) and draw num_tracks of them at
random; num_tracks=None uses every visible pair (dense).  Both paths are free of host syncs except the
inf/NaN check shared with the other losses.
"""
import torch

NEIGHBOURS = {
    "exact": [(0, 0)],                            # the containing pixel only
    "omega4": [(0, 0), (0, 1), (1, 0), (1, 1)],   # reference: floor + {0, 1} (dy, dx)
    "centre4": [(0, 0), (0, 1), (1, 0), (1, 1)],  # the 4 pixel centres nearest to (u, v): floor(u - 0.5) + {0, 1}
}


def build_tracks(pts_gt, depth, c2w, K, tau=0.01, abs_tol=0.0, neighbours="centre4", border=4, query_views="all"):
    """GT-only correspondences between every ordered pair of views.

    pts_gt [B, V, 3, H, W] world points (0 on background), depth [B, V, H, W] z-depth (0 = background),
    c2w [B, V, 4, 4], K [B, V, 4] = (fx, fy, cx, cy) with the principal point in continuous coordinates.
    Returns idx [B, Va, Vb, H, W] (flat pixel index in view b) and valid [B, Va, Vb, H, W] (bool).
    """
    offs = NEIGHBOURS[neighbours]
    B, V, _, H, W = pts_gt.shape
    dev = pts_gt.device
    P = pts_gt.float().permute(0, 1, 3, 4, 2)                     # [B, V, H, W, 3]
    fg = depth > 0
    w2c = torch.linalg.inv(c2w.double()).float()                  # [B, V, 4, 4]
    R = w2c[:, None, :, :3, :3]                                   # [B, 1, Vb, 3, 3]
    t = w2c[:, None, :, :3, 3]                                    # [B, 1, Vb, 3]
    X, Y, Z = (P[:, :, None, ..., i] for i in range(3))           # [B, Va, 1, H, W]

    def cam_row(i):
        return (X * R[..., i, 0, None, None] + Y * R[..., i, 1, None, None] + Z * R[..., i, 2, None, None]
                + t[..., i, None, None])                          # [B, Va, Vb, H, W]
    xc, yc, zc = cam_row(0), cam_row(1), cam_row(2)
    fx, fy, cx, cy = (K.float()[:, None, :, i, None, None] for i in range(4))   # [B, 1, Vb, 1, 1]
    zs = torch.where(zc.abs() < 1e-9, torch.full_like(zc, 1e-9), zc)
    u = fx * xc / zs + cx
    v = fy * yc / zs + cy
    if neighbours == "centre4":
        u, v = u - 0.5, v - 0.5
    bx, by = torch.floor(u), torch.floor(v)
    inside = (zc > 1e-6) & (bx >= border) & (bx <= W - 1 - border) & (by >= border) & (by <= H - 1 - border)
    bx = bx.long().clamp(0, W - 1)
    by = by.long().clamp(0, H - 1)
    D_flat = depth.float().reshape(B, V, H * W)
    b_ix = torch.arange(B, device=dev)[:, None, None, None]
    vb_ix = torch.arange(V, device=dev)[None, None, :, None]
    inf = torch.full_like(zc, float("inf"))
    best_gap, best_D, best_idx = inf, torch.zeros_like(zc), torch.zeros_like(bx)
    for dy, dx in offs:
        flat = (by + dy).clamp(0, H - 1) * W + (bx + dx).clamp(0, W - 1)          # [B, Va, Vb, H, W]
        D = D_flat[b_ix, vb_ix, flat.reshape(B, V, V, H * W)].reshape(B, V, V, H, W)
        ok = inside & (D > 0)
        gap = torch.where(ok, (zc - D).abs(), inf)
        take = gap < best_gap
        best_gap = torch.where(take, gap, best_gap)
        best_D = torch.where(take, D, best_D)
        best_idx = torch.where(take, flat, best_idx)
    valid = (best_D > 0) & (best_gap < tau * best_D + abs_tol) & (best_gap < tau * zc + abs_tol)
    valid = valid & fg[:, :, None]                                             # query pixel is foreground
    valid = valid & ~torch.eye(V, dtype=torch.bool, device=dev)[None, :, :, None, None]   # a != b
    if query_views == "first":
        valid[:, 1:] = False
    elif query_views != "all":
        raise ValueError(f"query_views must be 'all' or 'first', got {query_views!r}")
    return best_idx, valid


def sample_tracks(valid, fg, num_tracks, scheme="omega", generator=None):
    """Restrict valid [B, Va, Vb, H, W] to num_tracks query pixels per (sample, query view).

    scheme 'omega'  : rank foreground pixels by how many other views see them (random tie-break); when half of
                      the foreground exceeds num_tracks keep only that top half; then num_tracks at random.
    scheme 'uniform': num_tracks at random among the foreground pixels seen by at least one other view.
    No host synchronisation (two sorts per step).
    """
    if num_tracks is None:
        return valid
    B, Va, Vb, H, W = valid.shape
    HW = H * W
    dev = valid.device
    count = valid.sum(2).reshape(B, Va, HW).float()
    fgf = fg.reshape(B, Va, HW)
    r1 = torch.rand(B, Va, HW, device=dev, generator=generator)
    if scheme == "omega":
        n_fg = fgf.sum(-1, keepdim=True)                                         # [B, Va, 1]
        pool_size = torch.where(n_fg // 2 > num_tracks, n_fg // 2, n_fg)
        key = torch.where(fgf, count + 0.5 * r1, torch.full_like(count, -1.0))   # descending: count, then random
        rank = torch.empty_like(key, dtype=torch.long)
        rank.scatter_(-1, torch.argsort(key, dim=-1, descending=True), torch.arange(HW, device=dev).expand(B, Va, HW))
        pool = fgf & (rank < pool_size)
    elif scheme == "uniform":
        pool = fgf & (count >= 1)
    else:
        raise ValueError(f"unknown sampling scheme {scheme!r}")
    r2 = torch.rand(B, Va, HW, device=dev, generator=generator)
    key2 = torch.where(pool, r2, torch.full_like(r2, 2.0))
    rank2 = torch.empty_like(key2, dtype=torch.long)
    rank2.scatter_(-1, torch.argsort(key2, dim=-1), torch.arange(HW, device=dev).expand(B, Va, HW))
    sel = pool & (rank2 < num_tracks)
    return valid & sel.reshape(B, Va, 1, H, W)


def track_consistency_loss(pts_est, pts_gt, depth, c2w, K, tau=0.01, abs_tol=0.0, neighbours="centre4", border=4,
                           num_tracks=None, sampling="omega", query_views="all", min_valid=1, hard_max=100.0,
                           generator=None):
    """pts_est / pts_gt [B, V, 3, H, W] (target views), depth [B, V, H, W], c2w [B, V, 4, 4], K [B, V, 4].
    Returns loss_consistency (scalar with graph, 0 when fewer than min_valid entries) and two detached
    diagnostics: the number of valid entries and the fraction of foreground query pixels seen by >= 1 other view."""
    from model.loss import check_and_fix_inf_nan
    dev_type = pts_est.device.type
    with torch.autocast(device_type=dev_type, enabled=False):
        B, V, _, H, W = pts_est.shape
        depth = depth.float()
        idx, valid = build_tracks(pts_gt.float(), depth, c2w.float(), K.float(), tau=tau, abs_tol=abs_tol,
                                  neighbours=neighbours, border=border, query_views=query_views)
        fg = depth > 0
        seen = (valid.sum(2) >= 1) & fg
        valid = sample_tracks(valid, fg, num_tracks, sampling, generator)
        E = (pts_est.float() - pts_gt.float()).permute(0, 1, 3, 4, 2)          # [B, V, H, W, 3] error per pixel
        E_flat = E.reshape(B, V, H * W, 3)
        b_ix = torch.arange(B, device=E.device)[:, None, None, None]
        vb_ix = torch.arange(V, device=E.device)[None, None, :, None]
        Eb = E_flat[b_ix, vb_ix, idx.reshape(B, V, V, H * W)].reshape(B, V, V, H, W, 3)   # error of b at the match
        diff = (E[:, :, None] - Eb).abs()
        diff = check_and_fix_inf_nan(diff, "loss_consistency", hard_max=hard_max)
        n_valid = valid.sum()
        loss = (diff * valid[..., None].float()).sum() / (3.0 * n_valid.float()).clamp_min(1.0)
        loss = torch.where(n_valid >= min_valid, loss, loss * 0.0)
        seen_frac = seen.sum().float() / fg.sum().float().clamp_min(1.0)
    return {"loss_consistency": loss, "consistency_valid": n_valid.float().detach(),
            "consistency_seen_frac": seen_frac.detach()}
