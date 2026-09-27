# Stage-2 geometry (plan 2026-09-27, section 二 "屏蔽表"): projections, token grids, Morton order,
# packed/padded layouts and the two per-token mask tables.
#
# Conventions (same as ProcessData.compute_rays and tools_s2/s2_candidate_probe.py):
#   * pixel (y, x) has continuous coordinate (x + 0.5, y + 0.5); projections return u, v in
#     pixel-index units: u = fx * Xc / Zc + cx - 0.5;
#   * token of a pixel-index coordinate c on a g-token grid: clamp(floor((c + 0.5) * g / img), 0, g - 1);
#   * stage-1 point maps are treated as pixel-centre points (known half-pixel loader offset
#     deliberately ignored, user 09-26).
# Everything is plain element-wise torch arithmetic (no matmul / einsum), so TF32 and autocast cannot
# flip a visibility decision at the tolerance boundary; call it under torch.autocast(enabled=False).
# Nothing here reads a tensor value on the host except build_layout (one .tolist() of the per-view
# foreground-token counts, needed for sequence lengths).

from dataclasses import dataclass

import torch
import torch.nn.functional as F

N_REG = 4                     # renderer register tokens per target view
SUM_COLS = 64                 # reverse-attention key columns reserved for target block summaries
REV_FG_OFF = 128              # first reverse key column holding a foreground target token


def roundup(x, m):
    return ((int(x) + m - 1) // m) * m


def tok_index(c, g, img):
    """pixel-index coordinate(s) -> token index on a g-token grid over an img-pixel side."""
    return torch.clamp(torch.floor((c + 0.5) * (g / img)), 0, g - 1).long()


def world_to_cam(X, c2w):
    """X [..., N, 3] world points, c2w [..., 4, 4] (broadcast over the leading dims) -> (xc, yc, zc)."""
    R = c2w[..., :3, :3]
    t = c2w[..., :3, 3]
    dx = X[..., 0] - t[..., 0, None]
    dy = X[..., 1] - t[..., 1, None]
    dz = X[..., 2] - t[..., 2, None]
    # camera coordinates = R^T (X - t): component j = sum_i (X - t)_i R_ij
    xc = dx * R[..., 0, 0, None] + dy * R[..., 1, 0, None] + dz * R[..., 2, 0, None]
    yc = dx * R[..., 0, 1, None] + dy * R[..., 1, 1, None] + dz * R[..., 2, 1, None]
    zc = dx * R[..., 0, 2, None] + dy * R[..., 1, 2, None] + dz * R[..., 2, 2, None]
    return xc, yc, zc


def project(X, c2w, K):
    """world points [..., N, 3] -> (u, v, z); u, v in pixel-index units. K [..., 4] = fx, fy, cx, cy."""
    xc, yc, zc = world_to_cam(X, c2w)
    zs = torch.where(zc.abs() < 1e-9, torch.full_like(zc, 1e-9), zc)
    u = K[..., 0, None] * xc / zs + K[..., 2, None] - 0.5
    v = K[..., 1, None] * yc / zs + K[..., 3, None] - 0.5
    return u, v, zc


def inside_image(u, v, z, H, W):
    return (u > -0.5) & (u < W - 0.5) & (v > -0.5) & (v < H - 0.5) & (z > 0)


def nearest_pixel(u, v, H, W):
    """nearest pixel indices (clamped) of continuous pixel-index coordinates."""
    ui = torch.round(u).long().clamp(0, W - 1)
    vi = torch.round(v).long().clamp(0, H - 1)
    return vi, ui


def gather_map(img, vi, ui):
    """img [L, H, W] per-leading-row map; vi/ui [L, N] -> values [L, N]."""
    L, H, W = img.shape
    flat = img.reshape(L, H * W)
    return torch.gather(flat, 1, vi * W + ui)


def morton_rank(g, device=None):
    """rank of each token (row-major index r * g + c) in the Morton (Z-curve) order of a g x g grid."""
    r = torch.arange(g, device=device).view(g, 1).expand(g, g).reshape(-1)
    c = torch.arange(g, device=device).view(1, g).expand(g, g).reshape(-1)
    code = torch.zeros_like(r)
    for bit in range(max(1, (g - 1).bit_length())):
        code |= ((c >> bit) & 1) << (2 * bit)
        code |= ((r >> bit) & 1) << (2 * bit + 1)
    order = torch.argsort(code, stable=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(order.numel(), device=device)
    return rank


def plucker_rays(c2w, K, H, W):
    """Plücker rays [o x d, d] for pixel centres, c2w [B, V, 4, 4], K [B, V, 4] -> [B, V, 6, H, W]."""
    dev, dt = c2w.device, c2w.dtype
    y, x = torch.meshgrid(torch.arange(H, device=dev, dtype=dt), torch.arange(W, device=dev, dtype=dt), indexing="ij")
    x = x.reshape(1, 1, -1)
    y = y.reshape(1, 1, -1)
    dxc = (x + 0.5 - K[..., 2, None]) / K[..., 0, None]
    dyc = (y + 0.5 - K[..., 3, None]) / K[..., 1, None]
    R = c2w[..., :3, :3]
    # world direction = R @ [dxc, dyc, 1]
    d0 = R[..., 0, 0, None] * dxc + R[..., 0, 1, None] * dyc + R[..., 0, 2, None]
    d1 = R[..., 1, 0, None] * dxc + R[..., 1, 1, None] * dyc + R[..., 1, 2, None]
    d2 = R[..., 2, 0, None] * dxc + R[..., 2, 1, None] * dyc + R[..., 2, 2, None]
    d = torch.stack([d0, d1, d2], dim=2)                                # [B, V, 3, HW]
    d = d / torch.linalg.vector_norm(d, dim=2, keepdim=True)
    o = c2w[..., :3, 3, None].expand_as(d)                              # [B, V, 3, HW]
    m = torch.cross(o, d, dim=2)
    B, V = c2w.shape[:2]
    return torch.cat([m, d], dim=2).reshape(B, V, 6, H, W)


def dilate_bool(m, radius):
    """Chebyshev dilation of a boolean map [..., h, w] (separable OR of shifts; exact, no float copy)."""
    if radius <= 0:
        return m
    out = m.clone()
    for s in range(1, radius + 1):                  # along w
        out[..., :, s:] |= m[..., :, :-s]
        out[..., :, :-s] |= m[..., :, s:]
    src = out.clone()
    for s in range(1, radius + 1):                  # along h
        out[..., s:, :] |= src[..., :-s, :]
        out[..., :-s, :] |= src[..., s:, :]
    return out


def scene_block_ids(scene_g, n_views, block, device=None):
    """block id of every scene token (raster order view-major) for block x block token blocks."""
    nb = (scene_g + block - 1) // block
    r = torch.arange(scene_g, device=device).view(scene_g, 1).expand(scene_g, scene_g) // block
    c = torch.arange(scene_g, device=device).view(1, scene_g).expand(scene_g, scene_g) // block
    per_view = (r * nb + c).reshape(-1)
    ids = torch.cat([per_view + k * nb * nb for k in range(n_views)])
    return ids, n_views * nb * nb


# ----------------------------------------------------------------------------------------- layout
@dataclass
class Layout:
    BV: int
    g: int                      # target token grid side
    n: list                     # foreground tokens per view (host ints)
    fg_tok: torch.Tensor        # [BV, g*g] bool
    slot2tok: torch.Tensor      # [BV, g*g] token index of slot j (valid for j < n[bv])
    tok2slot: torch.Tensor      # [BV, g*g] slot of token (valid where fg_tok)
    offsets: list               # packed start of each view (host ints)
    T: int                      # packed length = sum(N_REG + n)
    Lq: int                     # padded query length of the forward masked attention
    Kr: int                     # padded key length of the reverse masked attention
    pack_src: torch.Tensor      # [BV, Lq] packed row feeding each padded forward row (T = zero row)
    pad_dst: torch.Tensor       # [T] flat padded index (bv * Lq + row) of each packed row
    fg_packed: torch.Tensor     # [sum n] packed row of every foreground token (view-major, slot order)
    fg_bv: torch.Tensor         # [sum n] view of every foreground token
    fg_tok_flat: torch.Tensor   # [sum n] token index of every foreground token
    reg_packed: torch.Tensor    # [BV * N_REG] packed rows of the registers
    revkey_src: torch.Tensor    # [BV, Kr - REV_FG_OFF] packed row of fg key column j (T = zero row)
    tblock_id: torch.Tensor     # [BV, g*g] target 32-px block id of each token (0..63)
    tblock_cnt: torch.Tensor    # [BV, SUM_COLS] fg tokens per target block
    tblock_valid: torch.Tensor  # [BV, SUM_COLS] block has at least one fg token


def build_layout(alpha_t, patch, bucket, img, tblock_px=32):
    """alpha_t [B, Vt, 1, H, W] (GT target alpha) -> Layout (one host sync: the fg counts)."""
    B, Vt = alpha_t.shape[:2]
    BV = B * Vt
    dev = alpha_t.device
    g = img // patch
    fg_px = (alpha_t.reshape(BV, 1, img, img) > 0.5).float()
    fg_tok = (F.max_pool2d(fg_px, patch) > 0).reshape(BV, g * g)
    n_t = fg_tok.sum(-1)
    n = [int(x) for x in n_t.tolist()]                         # the single host sync
    rank = morton_rank(g, dev)
    big = g * g
    key = torch.where(fg_tok, rank.view(1, -1).expand(BV, -1), torch.full_like(fg_tok, big, dtype=torch.long))
    slot2tok = torch.argsort(key, dim=1, stable=True)
    tok2slot = torch.empty_like(slot2tok)
    tok2slot.scatter_(1, slot2tok, torch.arange(g * g, device=dev).view(1, -1).expand(BV, -1))

    seg = [N_REG + k for k in n]
    offsets = [0]
    for s in seg[:-1]:
        offsets.append(offsets[-1] + s)
    T = offsets[-1] + seg[-1]
    max_n = max(n) if n else 0
    Lq = roundup(N_REG + max_n, bucket)
    Kr = REV_FG_OFF + roundup(max(max_n, 1), bucket)

    off_t = torch.tensor(offsets, device=dev)
    n_dev = n_t
    rows = torch.arange(Lq, device=dev).view(1, Lq)
    valid = rows < (N_REG + n_dev).view(BV, 1)
    pack_src = torch.where(valid, off_t.view(BV, 1) + rows, torch.full_like(rows.expand(BV, -1), T))

    seg_t = torch.tensor(seg, device=dev)
    bv_of_row = torch.repeat_interleave(torch.arange(BV, device=dev), seg_t, output_size=T)
    row_in_seg = torch.arange(T, device=dev) - off_t[bv_of_row]
    pad_dst = bv_of_row * Lq + row_in_seg

    nfg = int(sum(n))
    fg_bv = torch.repeat_interleave(torch.arange(BV, device=dev), n_dev, output_size=nfg)
    fg_start = torch.cumsum(n_dev, 0) - n_dev
    slot = torch.arange(nfg, device=dev) - fg_start[fg_bv]
    fg_packed = off_t[fg_bv] + N_REG + slot
    fg_tok_flat = slot2tok[fg_bv, slot]
    reg_packed = (off_t.view(BV, 1) + torch.arange(N_REG, device=dev).view(1, N_REG)).reshape(-1)

    kcols = torch.arange(Kr - REV_FG_OFF, device=dev).view(1, -1)
    kvalid = kcols < n_dev.view(BV, 1)
    revkey_src = torch.where(kvalid, off_t.view(BV, 1) + N_REG + kcols, torch.full_like(kcols.expand(BV, -1), T))

    bs = tblock_px // patch
    nb = img // tblock_px
    tr = torch.arange(g, device=dev).view(g, 1).expand(g, g) // bs
    tc = torch.arange(g, device=dev).view(1, g).expand(g, g) // bs
    tblock_id = (tr * nb + tc).reshape(1, -1).expand(BV, -1)
    assert nb * nb <= SUM_COLS
    tblock_cnt = torch.zeros(BV, SUM_COLS, dtype=torch.long, device=dev)
    tblock_cnt.scatter_add_(1, tblock_id.contiguous(), fg_tok.long())
    tblock_valid = tblock_cnt > 0

    return Layout(BV=BV, g=g, n=n, fg_tok=fg_tok, slot2tok=slot2tok, tok2slot=tok2slot, offsets=offsets, T=T,
                  Lq=Lq, Kr=Kr, pack_src=pack_src, pad_dst=pad_dst, fg_packed=fg_packed, fg_bv=fg_bv,
                  fg_tok_flat=fg_tok_flat, reg_packed=reg_packed, revkey_src=revkey_src,
                  tblock_id=tblock_id, tblock_cnt=tblock_cnt, tblock_valid=tblock_valid)


# ----------------------------------------------------------------------------------------- tables
def build_forward_table(P_t, alpha_t, layout, c2w_in, K_in, P_in, alpha_in, patch, scene_g, img,
                        radius=2, tau=0.03, s_pad=None):
    """Forward (target -> scene) selected-branch table.

    P_t [B, Vt, 3, H, W] stage-1 target points; alpha_t [B, Vt, 1, H, W] GT target alpha;
    c2w_in [B, Vi, 4, 4] input cameras used for projection (GT when posed, predicted when unposed);
    K_in [B, Vi, 4] GT intrinsics; P_in [B, Vi, 3, H, W] pass-2 input-view points; alpha_in GT input alpha.
    Returns bool [BV, Lq, s_pad]: rows = [registers | fg slots | pad], cols = scene tokens (raster,
    view-major) | null column | pad.  Every row allows at least one column.
    """
    B, Vt = P_t.shape[:2]
    Vi = c2w_in.shape[1]
    H = W = img
    BV, Lq = layout.BV, layout.Lq
    S = Vi * scene_g * scene_g
    s_pad = s_pad or roundup(S + 1, 128)
    dev = P_t.device

    # pass-2 depth of each input view in its own (projection) camera
    Pin = P_in.permute(0, 1, 3, 4, 2).reshape(B, Vi, H * W, 3)
    D_in = world_to_cam(Pin, c2w_in)[2]                                      # [B, Vi, HW]
    A_in = alpha_in.reshape(B, Vi, H * W) > 0.5

    X = P_t.permute(0, 1, 3, 4, 2).reshape(B, Vt, 1, H * W, 3)
    u, v, z = project(X, c2w_in[:, None], K_in[:, None])                    # [B, Vt, Vi, HW]
    ins = inside_image(u, v, z, H, W)
    vi, ui = nearest_pixel(u, v, H, W)
    pix = vi * W + ui
    Dk = torch.gather(D_in[:, None].expand(B, Vt, Vi, H * W), 3, pix)
    Ak = torch.gather(A_in[:, None].expand(B, Vt, Vi, H * W), 3, pix)
    fg_t = (alpha_t.reshape(B, Vt, 1, H * W) > 0.5)
    vis = fg_t & ins & Ak & (Dk > 0) & ((z - Dk).abs() <= tau * Dk)

    su = tok_index(u, scene_g, img)
    sv = tok_index(v, scene_g, img)
    g = layout.g
    py = torch.arange(H, device=dev).view(H, 1).expand(H, W).reshape(-1)
    px = torch.arange(W, device=dev).view(1, W).expand(H, W).reshape(-1)
    ptok = (py // patch) * g + (px // patch)                                # [HW]
    slot = layout.tok2slot.view(B, Vt, 1, g * g)[..., ptok].expand(B, Vt, Vi, H * W)
    bv = torch.arange(BV, device=dev).view(B, Vt, 1, 1)
    k = torch.arange(Vi, device=dev).view(1, 1, Vi, 1)
    nv = scene_g * scene_g
    flat = ((bv * Lq + N_REG + slot) * Vi + k) * nv + sv * scene_g + su
    dump = BV * Lq * Vi * nv
    M0 = torch.zeros(dump + 1, dtype=torch.bool, device=dev)
    # index_fill_ takes the fill value as a kernel scalar (no host->device copy, no sync)
    M0.index_fill_(0, torch.where(vis, flat, torch.full_like(flat, dump)).reshape(-1), True)
    M0 = M0[:dump].view(BV, Lq, Vi, scene_g, scene_g)
    M0 = dilate_bool(M0, radius).view(BV, Lq, S)

    Ftab = torch.zeros(BV, Lq, s_pad, dtype=torch.bool, device=dev)
    Ftab[:, :, :S] = M0
    rows = torch.arange(Lq, device=dev).view(1, Lq)
    n_dev = layout.fg_tok.sum(-1).view(BV, 1)
    is_reg = rows < N_REG
    is_fg = (rows >= N_REG) & (rows < N_REG + n_dev)
    Ftab[:, :, :S] &= is_fg.unsqueeze(-1)                                   # pads / registers cleared
    Ftab[:, :N_REG, :S] = True                                              # registers see every token
    Ftab[:, :, S] = ~Ftab[:, :, :S].any(-1)                                 # null key for empty rows
    return Ftab


def build_reverse_table(P_in, alpha_in, layout, c2w_t, K_t, P_t, alpha_t, patch, scene_g, img,
                        radius, tau=0.03, s_pad=None):
    """Reverse (scene -> target) table.

    P_in [B, Vi, 3, H, W] pass-2 input-view points; alpha_in GT input alpha; c2w_t / K_t GT target
    cameras; P_t stage-1 target points (their depth is the visibility reference); alpha_t GT target alpha.
    Returns bool [BV, s_pad, Kr]: rows = scene tokens (raster, view-major) | pad rows,
    cols = [64 target block summaries | 4 registers | pad | fg target tokens in slot order | pad].
    """
    B, Vi = P_in.shape[:2]
    Vt = c2w_t.shape[1]
    H = W = img
    BV, Kr, g = layout.BV, layout.Kr, layout.g
    S = Vi * scene_g * scene_g
    s_pad = s_pad or roundup(S + 1, 128)
    dev = P_in.device

    Pt = P_t.permute(0, 1, 3, 4, 2).reshape(B, Vt, H * W, 3)
    D_t = world_to_cam(Pt, c2w_t)[2]                                         # [B, Vt, HW]
    A_t = alpha_t.reshape(B, Vt, H * W) > 0.5

    Y = P_in.permute(0, 1, 3, 4, 2).reshape(B, 1, Vi, H * W, 3)
    u, v, z = project(Y, c2w_t[:, :, None], K_t[:, :, None])               # [B, Vt, Vi, HW]
    ins = inside_image(u, v, z, H, W)
    vi, ui = nearest_pixel(u, v, H, W)
    pix = vi * W + ui
    Dt = torch.gather(D_t[:, :, None].expand(B, Vt, Vi, H * W), 3, pix)
    At = torch.gather(A_t[:, :, None].expand(B, Vt, Vi, H * W), 3, pix)
    fg_in = (alpha_in.reshape(B, 1, Vi, H * W) > 0.5)
    covis = fg_in & ins & At & (Dt > 0) & ((z - Dt).abs() <= tau * Dt)

    py = torch.arange(H, device=dev).view(H, 1).expand(H, W).reshape(-1)
    px = torch.arange(W, device=dev).view(1, W).expand(H, W).reshape(-1)
    nv = scene_g * scene_g
    srow = tok_index(py.to(P_in.dtype), scene_g, img) * scene_g + tok_index(px.to(P_in.dtype), scene_g, img)
    k = torch.arange(Vi, device=dev).view(1, 1, Vi, 1)
    s = k * nv + srow.view(1, 1, 1, -1)                                     # scene token of the input pixel
    tt = tok_index(v, g, img) * g + tok_index(u, g, img)
    bv = torch.arange(BV, device=dev).view(B, Vt, 1, 1)
    flat = (bv * S + s) * (g * g) + tt
    dump = BV * S * g * g
    R0 = torch.zeros(dump + 1, dtype=torch.bool, device=dev)
    R0.index_fill_(0, torch.where(covis, flat, torch.full_like(flat, dump)).reshape(-1), True)
    R0 = R0[:dump].view(BV, S, g, g)
    R0 = dilate_bool(R0, radius).view(BV, S, g * g)

    nk = Kr - REV_FG_OFF
    cols = torch.arange(nk, device=dev).view(1, nk)
    n_dev = layout.fg_tok.sum(-1).view(BV, 1)
    kvalid = cols < n_dev
    tok = torch.gather(layout.slot2tok, 1, torch.where(kvalid, cols.expand(BV, -1), torch.zeros_like(cols.expand(BV, -1))))
    fgcols = torch.gather(R0, 2, tok.view(BV, 1, nk).expand(BV, S, nk)) & kvalid.view(BV, 1, nk)

    Rtab = torch.zeros(BV, s_pad, Kr, dtype=torch.bool, device=dev)
    Rtab[:, :S, REV_FG_OFF:] = fgcols
    Rtab[:, :, :SUM_COLS] = layout.tblock_valid.view(BV, 1, SUM_COLS)
    Rtab[:, :, SUM_COLS:SUM_COLS + N_REG] = True
    return Rtab


def depth_in_camera(P, c2w):
    """point map [B, V, 3, H, W] -> z in the matching camera [B, V, H, W]."""
    B, V, _, H, W = P.shape
    return world_to_cam(P.permute(0, 1, 3, 4, 2).reshape(B, V, H * W, 3), c2w)[2].reshape(B, V, H, W)


def c2w_from_pose_enc(pose_enc, img):
    """[B, V, 9] absT_quaR_FoV (stage-1 camera head) -> c2w [B, V, 4, 4] (fp64 inverse, as the probe)."""
    from model.vggt.utils.pose_enc import pose_encoding_to_extri_intri
    ext, _ = pose_encoding_to_extri_intri(pose_enc.float(), (img, img))           # w2c [B, V, 3, 4]
    B, V = ext.shape[:2]
    E = torch.eye(4, device=ext.device, dtype=torch.float64).repeat(B, V, 1, 1)
    E[..., :3, :] = ext.double()
    return torch.inverse(E).float()
