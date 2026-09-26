# Stage-2 pre-check (2026-09-26, user-approved): how far are the cross-attention candidates that the
# frozen stage-1 model (uni3t) would produce from the ground-truth correspondences?  Read-only, 1 GPU.
#
# Forward  (target -> scene): target fg pixel p -> predicted 3D point (point head, pass 1) -> project
#          into input view k with the GT (posed) / predicted (unposed) input camera.
# Reverse  (scene -> target), two candidate generators:
#   R-depth: input fg pixel q -> predicted 3D point from rendering the input cameras as targets
#            (pass 2) -> project into the (always known) target camera.
#   R-inv  : invert the forward projections: scene token s collects the target tokens whose predicted
#            projection lands in s (no visibility test).
# Ground truth: GT depth + GT cameras with the pixel convention of ProcessData.compute_rays (pixel x
# has continuous coordinate x + 0.5, principal point cx = 128 -> 127.5 in index units; exp.1).
# Visibility = depth test with TAU = 1 % (same as tools/attn_geometry_probe.py in the softmoe repo).
# Grids (tokens per side of the 256 image): scene 37 (VGGT), 64 (patch 4), 128 (patch 2);
# target 32 (patch 8), 64, 128.  Recall(r) = share of GT correspondences whose predicted token is
# within Chebyshev distance r (tokens) of the GT token.
#
#   python tools_s2/s2_candidate_probe.py --ckpt <ckpt_70000> --out ~/tmp/s2_probe/cand_70k.json
#   python tools_s2/s2_candidate_probe.py --selftest            # CPU checks of the geometry helpers

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

TAU = 0.01
PP_OFF = 0.5
IMG = 256
SCENE_GRIDS = (37, 64, 128)
TARGET_GRIDS = (32, 64, 128)
RINV_PAIRS = ((37, 32), (64, 64), (128, 128))
RADII = (0, 1, 2, 3, 4)
EDGE_BAND = 4                 # px: fg pixels within 4 px of the silhouette
DEPTH_EDGE = 0.05             # relative depth jump to any fg 8-neighbour
BIN, NBIN = 0.25, 257         # error histogram: 0.25 px bins on [0, 64) + overflow bin
TAU_PRED = (0.01, 0.03, 0.05)
ANGLE_EDGES = (30.0, 60.0, 90.0)
CATS = ("interior", "silhouette", "depth_edge")


# ------------------------------------------------------------------------------ geometry helpers
def unproject(depth, c2w, fxfycxcy, pp_off=PP_OFF):
    """depth [H, W] (z) -> world points [H, W, 3] (float64)."""
    H, W = depth.shape
    fx, fy, cx, cy = [float(t) for t in fxfycxcy]
    y, x = torch.meshgrid(torch.arange(H, device=depth.device, dtype=torch.float64),
                          torch.arange(W, device=depth.device, dtype=torch.float64), indexing="ij")
    d = depth.double()
    pc = torch.stack([(x + pp_off - cx) / fx * d, (y + pp_off - cy) / fy * d, d], -1)
    c2w = c2w.double()
    return pc @ c2w[:3, :3].T + c2w[:3, 3]


def project(X, c2w, fxfycxcy, pp_off=PP_OFF):
    """world points [..., 3] -> (u, v, z) with u, v in pixel-index units."""
    c2w = c2w.double()
    pc = (X.double() - c2w[:3, 3]) @ c2w[:3, :3]
    z = pc[..., 2]
    fx, fy, cx, cy = [float(t) for t in fxfycxcy]
    zs = torch.where(z.abs() < 1e-9, torch.full_like(z, 1e-9), z)
    return fx * pc[..., 0] / zs + cx - pp_off, fy * pc[..., 1] / zs + cy - pp_off, z


def lookup(img, u, v):
    """nearest-pixel lookup; returns (value, inside)."""
    H, W = img.shape
    ui, vi = torch.round(u).long(), torch.round(v).long()
    inside = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    val = img[vi.clamp(0, H - 1), ui.clamp(0, W - 1)]
    return val, inside


def tok1d(c, g):
    """pixel-index coordinate -> token index along one axis for a grid of g tokens per side."""
    return torch.clamp(torch.floor((c.double() + PP_OFF) * g / IMG), 0, g - 1).long()


def cheb(u1, v1, u2, v2, g):
    return torch.maximum((tok1d(u1, g) - tok1d(u2, g)).abs(), (tok1d(v1, g) - tok1d(v2, g)).abs())


def categories(depth, fg):
    """per-pixel category index into CATS for fg pixels (silhouette band > depth edge > interior)."""
    bg = (~fg).float()[None, None]
    band = (F.max_pool2d(bg, 2 * EDGE_BAND + 1, 1, EDGE_BAND)[0, 0] > 0) & fg
    d = depth.double()
    dp = F.pad(d[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
    fp = F.pad(fg[None, None].float(), (1, 1, 1, 1))[0, 0] > 0
    H, W = d.shape
    mx = torch.zeros_like(d)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            dn = dp[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
            fn = fp[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
            rel = torch.where(fg & fn, (d - dn).abs() / d.clamp_min(1e-9), torch.zeros_like(d))
            mx = torch.maximum(mx, rel)
    dedge = fg & (mx > DEPTH_EDGE) & ~band
    cat = torch.zeros_like(d, dtype=torch.long)
    cat[band] = 1
    cat[dedge] = 2
    return cat


def hist(err):
    b = torch.clamp(torch.floor(err / BIN), 0, NBIN - 1).long()
    return torch.bincount(b, minlength=NBIN).cpu().numpy().astype(np.int64)


def angle_bin(deg):
    b = torch.zeros_like(deg, dtype=torch.long)
    for e in ANGLE_EDGES:
        b += (deg >= e).long()
    return b


# ------------------------------------------------------------------------------ accumulator
class Acc:
    """per-object records so every number can be bootstrapped over objects."""

    def __init__(self):
        self.obj = []

    def new(self, name):
        r = dict(name=name, h={}, n={}, s={})
        self.obj.append(r)
        return r


def add_h(r, key, err):
    if err.numel() == 0:
        return
    h = hist(err)
    r["h"][key] = r["h"].get(key, np.zeros(NBIN, np.int64)) + h
    r["s"][key + "#sum"] = r["s"].get(key + "#sum", 0.0) + float(err.sum())


def add_n(r, key, hits, total):
    a = r["n"].get(key, [0, 0])
    r["n"][key] = [a[0] + int(hits), a[1] + int(total)]


def add_s(r, key, val):
    r["s"][key] = r["s"].get(key, 0.0) + float(val)


def quant(h, q):
    tot = h.sum()
    if tot == 0:
        return float("nan")
    c = np.cumsum(h)
    i = int(np.searchsorted(c, q * tot))
    return float("inf") if i >= NBIN - 1 else (i + 1) * BIN


def summarize(acc, n_boot=1000, seed=0):
    objs = acc.obj
    rng = np.random.default_rng(seed)
    idx_b = [rng.integers(0, len(objs), len(objs)) for _ in range(n_boot)]
    out = dict(n_objects=len(objs), hist={}, ratio={}, sums={})
    hkeys = sorted({k for o in objs for k in o["h"]})
    for k in hkeys:
        per = np.stack([o["h"].get(k, np.zeros(NBIN, np.int64)) for o in objs])
        tot = per.sum(0)
        sums = np.array([o["s"].get(k + "#sum", 0.0) for o in objs])
        row = dict(n=int(tot.sum()), mean=float(sums.sum() / max(tot.sum(), 1)))
        for q in (0.5, 0.75, 0.9, 0.99):
            row[f"q{int(q * 100)}"] = quant(tot, q)
        for x in (1, 2, 4, 8):
            row[f"le{x}px"] = float(tot[: int(x / BIN)].sum() / max(tot.sum(), 1))
        med_b = [quant(per[ix].sum(0), 0.5) for ix in idx_b]
        row["q50_ci95"] = [float(np.nanpercentile(med_b, 2.5)), float(np.nanpercentile(med_b, 97.5))]
        row["counts"] = tot.tolist()
        out["hist"][k] = row
    nkeys = sorted({k for o in objs for k in o["n"]})
    for k in nkeys:
        hits = np.array([o["n"].get(k, [0, 0])[0] for o in objs], np.float64)
        tots = np.array([o["n"].get(k, [0, 0])[1] for o in objs], np.float64)
        val = hits.sum() / max(tots.sum(), 1)
        bs = [hits[ix].sum() / max(tots[ix].sum(), 1) for ix in idx_b]
        out["ratio"][k] = dict(value=float(val), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                               n=int(tots.sum()))
    skeys = sorted({k for o in objs for k in o["s"] if not k.endswith("#sum")})
    for k in skeys:
        out["sums"][k] = float(sum(o["s"].get(k, 0.0) for o in objs))
    return out


# ------------------------------------------------------------------------------ per-object analysis
@torch.no_grad()
def analyse(r, ret, ret2, mode, pred_c2w, pred_fxy):
    inp, tgt = ret.input, ret.target
    dev = ret.points.device
    V_in, V_t = inp.c2w.shape[1], tgt.c2w.shape[1]
    H = W = IMG

    def dm(x, v):          # [1, v, (1,) H, W] -> [v, H, W]
        return x[0].reshape(v, H, W).to(dev)

    D_in, D_t = dm(inp.depth_map, V_in).double(), dm(tgt.depth_map, V_t).double()
    A_in, A_t = dm(inp.alpha_mask, V_in) > 0.5, dm(tgt.alpha_mask, V_t) > 0.5
    fg_in, fg_t = A_in & (D_in > 0), A_t & (D_t > 0)
    c2w_in, K_in = inp.c2w[0].to(dev), inp.fxfycxcy[0].to(dev)
    c2w_t, K_t = tgt.c2w[0].to(dev), tgt.fxfycxcy[0].to(dev)
    P_t = ret.points[0].permute(0, 2, 3, 1).double()                   # [Vt, H, W, 3] predicted
    P_in = ret2.points[0].permute(0, 2, 3, 1).double()                 # [Vin, H, W, 3] predicted
    PM_t = tgt.point_map[0].permute(0, 2, 3, 1).to(dev).double()       # loader GT point maps
    PM_in = inp.point_map[0].permute(0, 2, 3, 1).to(dev).double()
    cam_in = pred_c2w if mode == "unposed" else c2w_in                  # camera used for projection
    # predicted depths (for the predicted visibility tests)
    Dh_t = torch.stack([project(P_t[t], c2w_t[t], K_t[t])[2] for t in range(V_t)])
    Dh_in = torch.stack([project(P_in[k], cam_in[k], K_in[k])[2] for k in range(V_in)])

    # depth sanity (abs_rel on GT fg pixels)
    for t in range(V_t):
        m = fg_t[t]
        add_s(r, "absrel_t#num", ((Dh_t[t][m] - D_t[t][m]).abs() / D_t[t][m]).sum())
        add_s(r, "absrel_t#den", m.sum())
    for k in range(V_in):
        m = fg_in[k]
        add_s(r, "absrel_in#num", ((Dh_in[k][m] - D_in[k][m]).abs() / D_in[k][m]).sum())
        add_s(r, "absrel_in#den", m.sum())

    cat_t = [categories(D_t[t], fg_t[t]) for t in range(V_t)]
    cat_in = [categories(D_in[k], fg_in[k]) for k in range(V_in)]
    X_gt_t = [unproject(D_t[t], c2w_t[t], K_t[t]) for t in range(V_t)]
    Y_gt_in = [unproject(D_in[k], c2w_in[k], K_in[k]) for k in range(V_in)]
    C_t = c2w_t[:, :3, 3].double()
    C_in = c2w_in[:, :3, 3].double()

    for t in range(V_t):
        pm = fg_t[t]
        py, px = torch.nonzero(pm, as_tuple=True)
        X = X_gt_t[t][pm]
        Xh = P_t[t][pm]
        Xl = PM_t[t][pm]
        ct = cat_t[t][pm]
        fwd = []
        for k in range(V_in):
            u, v, z = project(X, c2w_in[k], K_in[k])
            Dk, ins = lookup(D_in[k], u, v)
            onfg = ins & (Dk > 0)
            rel = torch.where(onfg, (z - Dk) / Dk.clamp_min(1e-9), torch.zeros_like(z))
            vis = onfg & (rel.abs() <= TAU)
            # gate-1 statistics (conventions): also with the other principal-point convention
            add_s(r, "g1_onfg", onfg.sum())
            add_s(r, "g1_front", (onfg & (rel < -TAU)).sum())
            add_h(r, "g1_absrel_x1000_vis", (rel[vis].abs() * 1000.0))
            u0, v0, z0 = project(unproject(D_t[t], c2w_t[t], K_t[t], pp_off=0.0)[pm], c2w_in[k], K_in[k], pp_off=0.0)
            Dk0, ins0 = lookup(D_in[k], u0, v0)
            onfg0 = ins0 & (Dk0 > 0)
            rel0 = torch.where(onfg0, (z0 - Dk0) / Dk0.clamp_min(1e-9), torch.zeros_like(z0))
            add_s(r, "g1pp128_onfg", onfg0.sum())
            add_s(r, "g1pp128_front", (onfg0 & (rel0 < -TAU)).sum())
            fwd.append((u, v, vis))
        seen = torch.stack([f[2] for f in fwd]).sum(0)
        add_s(r, "fwd_fg_pixels", pm.sum())
        add_s(r, "fwd_unseen_pixels", (seen == 0).sum())
        for k in range(V_in):
            u, v, vis = fwd[k]
            add_s(r, "fwd_pairs", vis.numel())
            add_s(r, "fwd_pairs_vis", vis.sum())
            uh, vh, zh = project(Xh, cam_in[k], K_in[k])
            insh = (uh > -0.5) & (uh < W - 0.5) & (vh > -0.5) & (vh < H - 0.5) & (zh > 0)
            err = torch.sqrt((uh - u) ** 2 + (vh - v) ** 2)
            ev = err[vis]
            add_h(r, "fwd_all", ev)
            for ci, cn in enumerate(CATS):
                add_h(r, f"fwd_{cn}", err[vis & (ct == ci)])
            for s in range(1, 5):
                add_h(r, f"fwd_seen{s}", err[vis & (seen == s)])
            dir_t = X - C_t[t]
            dir_k = X - C_in[k]
            cosang = (dir_t * dir_k).sum(-1) / (dir_t.norm(dim=-1) * dir_k.norm(dim=-1)).clamp_min(1e-12)
            ab = angle_bin(torch.rad2deg(torch.acos(cosang.clamp(-1, 1))))
            for b in range(len(ANGLE_EDGES) + 1):
                add_h(r, f"fwd_angle{b}", err[vis & (ab == b)])
            ul, vl, _ = project(Xl, c2w_in[k], K_in[k])                # loader point map (pp 128) floor
            add_h(r, "fwd_floor_loaderpm", torch.sqrt((ul - u) ** 2 + (vl - v) ** 2)[vis])
            if mode == "unposed":                                      # predicted intrinsics as well
                up, vp, _ = project(Xh, cam_in[k], pred_fxy[k])
                add_h(r, "fwd_all_predK", torch.sqrt((up - u) ** 2 + (vp - v) ** 2)[vis])
            for g in SCENE_GRIDS:
                d = cheb(uh, vh, u, v, g)
                for rad in RADII:
                    add_n(r, f"fwd_recall_g{g}_r{rad}", (vis & insh & (d <= rad)).sum(), vis.sum())
                    if g == 64:
                        add_n(r, f"fwd_recall_g{g}_r{rad}_silhouette", (vis & insh & (d <= rad) & (ct == 1)).sum(),
                              (vis & (ct == 1)).sum())
                        add_n(r, f"fwd_recall_g{g}_r{rad}_interior", (vis & insh & (d <= rad) & (ct == 0)).sum(),
                              (vis & (ct == 0)).sum())
            # predicted visibility (depth test against the pass-2 input depth; GT input mask)
            Dhk, insp = lookup(Dh_in[k], uh, vh)
            Ak, _ = lookup(A_in[k].double(), uh, vh)
            for tau in TAU_PRED:
                pv = insp & (Ak > 0.5) & ((zh - Dhk).abs() <= tau * Dhk.abs())
                add_s(r, f"fwdvis_tau{tau}_tp", (pv & vis).sum())
                add_s(r, f"fwdvis_tau{tau}_fp", (pv & ~vis).sum())
                add_s(r, f"fwdvis_tau{tau}_fn", (~pv & vis).sum())
                add_s(r, f"fwdvis_tau{tau}_tn", (~pv & ~vis).sum())
            # R-inv: scene token <- target tokens of predicted projections (no visibility)
            fwd[k] = (u, v, vis, uh, vh, insh)
        # ---------------------------------------------------------------- reverse, per input view
        for k in range(V_in):
            qm = fg_in[k]
            qy, qx = torch.nonzero(qm, as_tuple=True)
            Y = Y_gt_in[k][qm]
            vu, vv, vz = project(Y, c2w_t[t], K_t[t])
            Dtv, ins = lookup(D_t[t], vu, vv)
            onfg = ins & (Dtv > 0)
            covis = onfg & ((vz - Dtv).abs() <= TAU * Dtv)
            add_s(r, "rev_fg_pixels", qm.sum())
            add_s(r, "rev_covis_pixels", covis.sum())
            cq = cat_in[k][qm]
            # R-depth
            Yh = P_in[k][qm]
            hu, hv, hz = project(Yh, c2w_t[t], K_t[t])
            insh = (hu > -0.5) & (hu < W - 0.5) & (hv > -0.5) & (hv < H - 0.5) & (hz > 0)
            err = torch.sqrt((hu - vu) ** 2 + (hv - vv) ** 2)
            add_h(r, "rdep_all", err[covis])
            for ci, cn in enumerate(CATS):
                add_h(r, f"rdep_{cn}", err[covis & (cq == ci)])
            Yl = PM_in[k][qm]
            lu, lv, _ = project(Yl, c2w_t[t], K_t[t])
            add_h(r, "rdep_floor_loaderpm", torch.sqrt((lu - vu) ** 2 + (lv - vv) ** 2)[covis])
            for g in TARGET_GRIDS:
                d = cheb(hu, hv, vu, vv, g)
                for rad in RADII:
                    add_n(r, f"rdep_recall_g{g}_r{rad}", (covis & insh & (d <= rad)).sum(), covis.sum())
            Dht, inst = lookup(Dh_t[t], hu, hv)
            At, _ = lookup(A_t[t].double(), hu, hv)
            for tau in TAU_PRED:
                pv = inst & (At > 0.5) & ((hz - Dht).abs() <= tau * Dht.abs())
                add_s(r, f"revvis_tau{tau}_tp", (pv & covis).sum())
                add_s(r, f"revvis_tau{tau}_fp", (pv & ~covis).sum())
                add_s(r, f"revvis_tau{tau}_fn", (~pv & covis).sum())
                add_s(r, f"revvis_tau{tau}_tn", (~pv & ~covis).sum())
            # R-inv
            u, v, vis, uh, vh, insh_f = fwd[k]
            add_s(r, "rinv_entries", insh_f.sum())
            add_s(r, "rinv_entries_occluded", (insh_f & ~vis).sum())
            if covis.sum() == 0:
                continue
            for gs, gt in RINV_PAIRS:
                s_ent = tok1d(vh[insh_f], gs) * gs + tok1d(uh[insh_f], gs)
                t_ent = tok1d(py[insh_f].double(), gt) * gt + tok1d(px[insh_f].double(), gt)
                s_q = (tok1d(qy.double(), gs) * gs + tok1d(qx.double(), gs))[covis]
                t_q = (tok1d(vv, gt) * gt + tok1d(vu, gt))[covis]
                rows, inv = torch.unique(s_q, return_inverse=True)
                pos = torch.full((gs * gs,), -1, dtype=torch.long, device=dev)
                pos[rows] = torch.arange(rows.numel(), device=dev)
                keep = pos[s_ent] >= 0
                M = torch.zeros(rows.numel(), gt * gt, dtype=torch.float16, device=dev)
                M[pos[s_ent[keep]], t_ent[keep]] = 1.0
                M = M.view(rows.numel(), 1, gt, gt)
                for rad in RADII:
                    Mr = M if rad == 0 else F.max_pool2d(M, 2 * rad + 1, 1, rad)
                    hit = Mr.view(rows.numel(), gt * gt)[inv, t_q] > 0
                    add_n(r, f"rinv_recall_s{gs}_t{gt}_r{rad}", hit.sum(), hit.numel())


# ------------------------------------------------------------------------------ model / data
def build(config_path, ckpt, dev):
    from omegaconf import OmegaConf
    from easydict import EasyDict as edict
    import importlib

    base = OmegaConf.load(config_path)
    cfgs = {}
    for roll in (0.0, 10.0):
        ov = OmegaConf.from_dotlist([
            "training.target_has_input=false",
            "training.val_dataset_cfgs.training.target_has_input=false",
            "training.val_dataset_cfgs.training.num_views=14",
            "training.val_dataset_cfgs.training.num_input_views=4",
            "training.val_dataset_cfgs.training.num_target_views=10",
            "training.val_dataset_cfgs.root_dir=/home/z50057756/data/gso_sim2real_25v",
            "training.val_dataset_cfgs.split_file=data/gso_subset64.txt",
            f"training.roll_augment_max_deg={roll}",
            "inference.generate_website=false",
        ])
        cfgs[roll] = edict(OmegaConf.to_container(OmegaConf.merge(base, ov), resolve=True))
    cfg = cfgs[0.0]
    mod, cls = cfg.model.class_name.rsplit(".", 1)
    model = importlib.import_module(mod).__dict__[cls](cfg).to(dev)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True, mmap=True)["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not k.startswith("loss_computer.")]
    print(f"[s2] ckpt {ckpt}: missing {len(missing)} (non-loss {len(bad)}) unexpected {len(unexpected)}", flush=True)
    if bad or unexpected:
        print("[s2] missing:", bad[:10], "unexpected:", list(unexpected)[:10], flush=True)
        raise SystemExit("[s2] GATE 0 FAIL: checkpoint does not match the model")
    model.eval()
    dmod, dcls = cfg.training.val_dataset_name.rsplit(".", 1)
    DS = importlib.import_module(dmod).__dict__[dcls]
    return cfgs, model, {roll: DS(c) for roll, c in cfgs.items()}


def pass2(model, batch, pred_c2w=None):
    """render the 4 input cameras as targets (pred_c2w replaces their c2w in unposed mode)."""
    idx = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    b2 = {}
    for k, v in batch.items():
        b2[k] = v[:, idx].clone() if torch.is_tensor(v) and v.dim() >= 2 and v.shape[1] == 14 else v
    if pred_c2w is not None:
        b2["c2w"][:, 4:] = pred_c2w.to(b2["c2w"].dtype)
        b2["extrinsic"][:, 4:] = torch.inverse(pred_c2w.double()).to(b2["extrinsic"].dtype)
    ct = model.process_val_data.config.training
    old = ct.num_target_views
    ct.num_target_views = 4
    try:
        return model(b2, target_has_input=False, is_valid=True)
    finally:
        ct.num_target_views = old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--config", default="configs/RnGUP_lagernvs_uni3t_b32t6_fp32lr35_const_90k_all287k.yaml")
    ap.add_argument("--out")
    ap.add_argument("--modes", default="posed,unposed,posed_roll10")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    from model.vggt.utils.pose_enc import pose_encoding_to_extri_intri

    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    t0 = time.time()
    os.chdir(REPO)
    import torch.distributed as dist          # model/loss.py calls torch.distributed at init
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(29000 + int(os.environ.get("SLURM_JOB_ID", "0")) % 997))
        dist.init_process_group("gloo", rank=0, world_size=1)
    cfgs, model, dsets = build(args.config, args.ckpt, dev)
    amp = dict(enabled=True, device_type="cuda", dtype=torch.bfloat16)
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    print(f"[s2] built in {time.time() - t0:.0f}s; modes {args.modes}", flush=True)

    for mode in [m for m in args.modes.split(",") if m]:
        if mode in res:
            print(f"[s2] skip {mode} (done)", flush=True)
            continue
        base_mode = "unposed" if mode.startswith("unposed") else "posed"
        roll = 10.0 if mode.endswith("roll10") else 0.0
        model.val_cam_cond_zero_p = 1.0 if base_mode == "unposed" else 0.0
        random.seed(1234)
        loader = torch.utils.data.DataLoader(dsets[roll], batch_size=1, shuffle=False, num_workers=0)
        acc, tm, pose_err = Acc(), time.time(), []
        with torch.inference_mode():
            for bi, batch in enumerate(loader):
                batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
                with torch.autocast(**amp):
                    ret = model(batch, target_has_input=False, is_valid=True)
                d_in = (ret.input.image[0].float() - batch["image"][0, :4].float()).abs().max()
                d_t = (ret.target.image[0].float() - batch["image"][0, 4:].float()).abs().max()
                if d_in > 1e-6 or d_t > 1e-6:
                    raise SystemExit(f"[s2] GATE 2 FAIL: view order (input {d_in}, target {d_t})")
                pe = ret.camera[-1][0].float()                                     # [4, 9]
                ext, intri = pose_encoding_to_extri_intri(pe[None], (IMG, IMG))
                E = torch.eye(4, device=dev).repeat(4, 1, 1)
                E[:, :3, :] = ext[0]
                pred_c2w = torch.inverse(E.double())
                pred_fxy = torch.stack([intri[0, :, 0, 0], intri[0, :, 1, 1], intri[0, :, 0, 2], intri[0, :, 1, 2]], -1)
                with torch.autocast(**amp):
                    ret2 = pass2(model, batch, pred_c2w if base_mode == "unposed" else None)
                d2 = (ret2.target.image[0].float() - batch["image"][0, :4].float()).abs().max()
                if d2 > 1e-6:
                    raise SystemExit(f"[s2] GATE 3 FAIL: pass-2 targets are not the input views ({d2})")
                gt = ret.input.c2w[0].double()
                Rg, Rp = gt[:, :3, :3], pred_c2w[:, :3, :3]
                cosr = ((torch.einsum("vij,vij->v", Rg, Rp) - 1) / 2).clamp(-1, 1)
                pose_err.append(dict(rot_deg=torch.rad2deg(torch.acos(cosr)).tolist(),
                                     center=(gt[:, :3, 3] - pred_c2w[:, :3, 3]).norm(dim=-1).tolist()))
                r = acc.new(batch["scene_name"][0])
                analyse(r, ret, ret2, base_mode, pred_c2w, pred_fxy)
                if bi == 3:          # early gate 1 on the first 4 objects
                    s = summarize(acc, n_boot=10)
                    med = s["hist"]["g1_absrel_x1000_vis"]["q50"] / 1000.0
                    front = s["sums"]["g1_front"] / max(s["sums"]["g1_onfg"], 1)
                    print(f"[s2] GATE 1 ({mode}, 4 objects): median |rel depth| {med:.4f} front-violation {front:.4f}", flush=True)
                    if not (med <= 0.005 and front <= 0.02):
                        raise SystemExit("[s2] GATE 1 FAIL: GT projection conventions")
                if bi % 16 == 0:
                    print(f"[s2] {mode} object {bi} ({time.time() - tm:.0f}s)", flush=True)
        summ = summarize(acc)
        su = summ["sums"]
        summ["absrel_target"] = su["absrel_t#num"] / max(su["absrel_t#den"], 1)
        summ["absrel_input_pass2"] = su["absrel_in#num"] / max(su["absrel_in#den"], 1)
        summ["g1_front"] = su["g1_front"] / max(su["g1_onfg"], 1)
        summ["g1pp128_front"] = su["g1pp128_front"] / max(su["g1pp128_onfg"], 1)
        summ["fwd_vis_share"] = su["fwd_pairs_vis"] / max(su["fwd_pairs"], 1)
        summ["fwd_unseen_share"] = su["fwd_unseen_pixels"] / max(su["fwd_fg_pixels"], 1)
        summ["rev_covis_share"] = su["rev_covis_pixels"] / max(su["rev_fg_pixels"], 1)
        summ["rinv_occluded_entry_share"] = su["rinv_entries_occluded"] / max(su["rinv_entries"], 1)
        for d in ("fwdvis", "revvis"):
            for tau in TAU_PRED:
                tp, fp, fn, tn = (su[f"{d}_tau{tau}_{x}"] for x in ("tp", "fp", "fn", "tn"))
                summ[f"{d}_tau{tau}"] = dict(precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                                             occluded_kept=fp / max(fp + tn, 1), tp=tp, fp=fp, fn=fn, tn=tn)
        pr = np.array([p["rot_deg"] for p in pose_err])
        pc = np.array([p["center"] for p in pose_err])
        summ["pose"] = dict(rot_deg_median=float(np.median(pr)), rot_deg_p90=float(np.percentile(pr, 90)),
                            center_median=float(np.median(pc)), center_p90=float(np.percentile(pc, 90)))
        res[mode] = summ
        tmp = args.out + ".tmp"
        json.dump(res, open(tmp, "w"), indent=1)
        os.replace(tmp, args.out)
        H = summ["hist"]
        print(f"[s2] {mode} done ({time.time() - tm:.0f}s): absrel target {summ['absrel_target']:.4f} "
              f"input(pass2) {summ['absrel_input_pass2']:.4f}  fwd median {H['fwd_all']['q50']:.2f}px "
              f"p90 {H['fwd_all']['q90']:.2f}px  rdep median {H['rdep_all']['q50']:.2f}px  "
              f"pose rot med {summ['pose']['rot_deg_median']:.2f} deg", flush=True)
    print(f"[s2] DONE ({time.time() - t0:.0f}s) -> {args.out}", flush=True)


# ------------------------------------------------------------------------------ CPU self-test
def selftest():
    torch.manual_seed(0)
    K = torch.tensor([351.67, 351.67, 128.0, 128.0], dtype=torch.float64)
    # 1) unproject -> project into the same camera returns the pixel grid exactly
    c2w = torch.eye(4, dtype=torch.float64)
    ang = 0.4
    c2w[:3, :3] = torch.tensor([[math.cos(ang), 0, math.sin(ang)], [0, 1, 0], [-math.sin(ang), 0, math.cos(ang)]],
                               dtype=torch.float64)
    c2w[:3, 3] = torch.tensor([0.3, -0.2, -1.2], dtype=torch.float64)
    depth = 1.0 + torch.rand(IMG, IMG, dtype=torch.float64)
    X = unproject(depth, c2w, K)
    u, v, z = project(X, c2w, K)
    yy, xx = torch.meshgrid(torch.arange(IMG), torch.arange(IMG), indexing="ij")
    assert (u - xx).abs().max() < 1e-9 and (v - yy).abs().max() < 1e-9 and (z - depth).abs().max() < 1e-9
    # 2) the principal point in index units is 127.5: a point on the optical axis lands between pixels
    u0, v0, _ = project(c2w[:3, 3] + c2w[:3, 2] * 2.0, c2w, K)
    assert abs(float(u0) - 127.5) < 1e-9 and abs(float(v0) - 127.5) < 1e-9
    # 3) token mapping matches the exp.3 formula floor((x + 0.5) * g / 256)
    for g in (37, 64, 128, 32):
        x = torch.arange(IMG, dtype=torch.float64)
        ref = torch.tensor([min(int((i + 0.5) * g // IMG), g - 1) for i in range(IMG)])
        assert torch.equal(tok1d(x, g), ref), g
    assert int(tok1d(torch.tensor([255.0]), 37)) == 36 and int(tok1d(torch.tensor([0.0]), 37)) == 0
    # 4) categories: a square object with a depth step inside
    d = torch.full((32, 32), 0.0, dtype=torch.float64)
    d[8:24, 8:24] = 1.0
    d[8:24, 16:24] = 1.5
    fg = d > 0
    c = categories(d, fg)
    assert int(c[8, 12]) == 1 and int(c[11, 11]) == 1 and int(c[15, 15]) == 2 and int(c[15, 16]) == 2
    assert int(c[12, 12]) == 0 and int(c[14, 13]) == 0 and int(c[0, 0]) == 0
    # 5) histogram quantiles
    h = hist(torch.tensor([0.1, 0.1, 0.3, 5.0], dtype=torch.float64))
    assert quant(h, 0.5) == BIN and quant(h, 0.99) == 5.25
    print("[s2] selftest PASS")


if __name__ == "__main__":
    main()
