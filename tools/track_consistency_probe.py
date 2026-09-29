"""Ground-truth statistics for the track-consistency loss, measured on REAL training samples.

The loss (VGGT-Omega open-source style) needs, per pair of target views (a -> b), the pixels of b
that see the same 3D points as the foreground pixels of a.  Every rule of that construction
(depth tolerance, neighbour lookup, border margin, track sampling) is a knob; this probe measures
what each knob does on our rendered data so the values are set from numbers, not copied.

Independent of the loader's point_map: the 3D points are unprojected here from the GT depth with
the pixel-centre convention (x + 0.5 - cx), i.e. the state after the half-pixel fix.

Per ordered pair of target views (a, b) and every foreground pixel of a:
  * project into b -> continuous (u, v) (pixel i covers [i, i+1)) and depth z in b's camera;
  * read b's GT depth D at the containing pixel and at two 4-neighbour sets
        omega4  = {floor(u), floor(u)+1} x {floor(v), floor(v)+1}     (the reference code)
        centre4 = the 4 pixel centres nearest to (u, v)
    and keep, per set, the neighbour whose depth is closest to z;
  * signed relative difference rel = (z - D) / D  (> 0: behind b's surface = occluded);
  * colour check, independent of depth: |RGB_a(pixel) - RGB_b(bilinear at u, v)| against a
    control that samples b 8 px away in a random direction.
Aggregated (no per-pair arrays kept):
  * histogram of |rel| bands x {front, behind} with pair counts and mean colour difference;
  * pass rates for several tolerances, with / without the neighbour sets, and Omega's 5 % + 0.01 form;
  * pairs accepted only thanks to the neighbours (count + colour difference);
  * foreground pixels within 1 / 4 px of the image border;
  * per query pixel, how many of the other 5 target views see it (tau = 1 %, omega4);
  * category shares (silhouette band / depth edge / interior) of the loss entries under
    dense (all pairs), Omega sampling (top half by visible count, then 512 at random) and
    uniform 512 per query view;
  * valid pairs per sample.
    python tools/track_consistency_probe.py --config configs/<uni3t const>.yaml --n_objects 128 \
        --out ~/tmp/track_consistency_probe/probe_128.json
"""
import argparse
import importlib
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

BANDS = [0.0, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10, float("inf")]
TAUS = [0.0025, 0.005, 0.01, 0.02, 0.03, 0.05]
TAU_REF = 0.01          # tolerance used for the visible-count / sampling statistics
N_TRACKS = 512
SIL_PX = 4              # silhouette band: foreground pixels within 4 px (Chebyshev) of the alpha boundary
EDGE_REL = 0.05         # depth edge: > 5 % relative depth jump to a foreground 8-neighbour


def unproject(depth, c2w, K):
    """GT depth [H, W] -> world points of every pixel [H, W, 3], pixel-centre convention, float64."""
    H, W = depth.shape
    fx, fy, cx, cy = [float(v) for v in K]
    y, x = torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(W, dtype=torch.float64), indexing="ij")
    z = depth.double()
    pc = torch.stack([(x + 0.5 - cx) / fx * z, (y + 0.5 - cy) / fy * z, z], -1)
    c2w = c2w.double()
    return pc @ c2w[:3, :3].T + c2w[:3, 3]


def project(P, c2w, K):
    """world points [N, 3] -> continuous (u, v) in view (pixel i covers [i, i+1)) and depth z."""
    fx, fy, cx, cy = [float(v) for v in K]
    w2c = torch.linalg.inv(c2w.double())
    pc = P @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    zs = torch.where(z.abs() < 1e-12, torch.full_like(z, 1e-12), z)
    return fx * pc[:, 0] / zs + cx, fy * pc[:, 1] / zs + cy, z


def categories(depth, alpha):
    """per pixel: 0 interior, 1 silhouette band, 2 depth edge (only meaningful on foreground)."""
    fg = depth > 0
    bg = (~(alpha > 0.5)).float()[None, None]
    near_bg = F.max_pool2d(bg, 2 * SIL_PX + 1, stride=1, padding=SIL_PX)[0, 0] > 0
    sil = fg & near_bg
    d = depth.double()
    edge = torch.zeros_like(fg)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            nb = torch.roll(d, shifts=(dy, dx), dims=(0, 1))
            nbfg = torch.roll(fg, shifts=(dy, dx), dims=(0, 1))
            jump = (d - nb).abs() / d.clamp_min(1e-9)
            edge |= fg & nbfg & (jump > EDGE_REL)
    cat = torch.zeros(depth.shape, dtype=torch.long)
    cat[edge] = 2
    cat[sil] = 1           # band takes precedence, as in tools_s2/s2_regions.py
    return cat


def bilinear(img, u, v):
    """img [3, H, W]; (u, v) continuous coords -> index coords (u - 0.5, v - 0.5)."""
    _, H, W = img.shape
    x = (u - 0.5).clamp(0, W - 1 - 1e-6)
    y = (v - 0.5).clamp(0, H - 1 - 1e-6)
    x0 = x.floor().long()
    y0 = y.floor().long()
    x1 = (x0 + 1).clamp(max=W - 1)
    y1 = (y0 + 1).clamp(max=H - 1)
    wx = (x - x0.double())[None]
    wy = (y - y0.double())[None]
    im = img.double()
    return ((1 - wy) * ((1 - wx) * im[:, y0, x0] + wx * im[:, y0, x1])
            + wy * ((1 - wx) * im[:, y1, x0] + wx * im[:, y1, x1]))


class Acc:
    def __init__(self):
        self.d = {}

    def add(self, key, n, s=0.0):
        e = self.d.setdefault(key, [0, 0.0])
        e[0] += int(n)
        e[1] += float(s)

    def as_dict(self):
        return {k: {"n": v[0], "mean": (v[1] / v[0] if v[0] else None)} for k, v in self.d.items()}


def best_neighbour(Db, u, v, z, offsets, inside):
    """per pair, the neighbour (from `offsets` applied to floor(u), floor(v)) whose GT depth is closest to z;
    returns (D, rel) with D = 0 where no neighbour is foreground."""
    H, W = Db.shape
    xu = u.floor().long()
    yv = v.floor().long()
    best_d = torch.zeros_like(z)
    best_gap = torch.full_like(z, float("inf"))
    for dy, dx in offsets:
        xi = (xu + dx).clamp(0, W - 1)
        yi = (yv + dy).clamp(0, H - 1)
        D = Db[yi, xi]
        ok = inside & (D > 0)
        gap = torch.where(ok, (z - D).abs(), torch.full_like(z, float("inf")))
        take = gap < best_gap
        best_gap = torch.where(take, gap, best_gap)
        best_d = torch.where(take, D, best_d)
    rel = torch.where(best_d > 0, (z - best_d) / best_d.clamp_min(1e-12), torch.full_like(z, float("nan")))
    return best_d, rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_objects", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    # respect OMP_NUM_THREADS (login node: 2); otherwise the cores this process may use
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", max(1, len(os.sched_getaffinity(0))))))
    os.chdir(REPO)
    from omegaconf import OmegaConf
    from easydict import EasyDict as edict
    cfg = edict(OmegaConf.to_container(OmegaConf.load(a.config), resolve=True))
    m, c = cfg.training.dataset_name.rsplit(".", 1)
    ds = importlib.import_module(m).__dict__[c](cfg)
    n_in = int(cfg.training.num_input_views)
    n_tgt = int(cfg.training.num_target_views)
    rng = random.Random(a.seed)
    idxs = rng.sample(range(len(ds)), a.n_objects)
    g = torch.Generator().manual_seed(a.seed)

    band = Acc()          # key: (sign, band_i, nbset) -> count, colour-diff sum
    ctrl = Acc()          # control colour diff
    passes = Acc()        # key: (nbset, tau) -> n_pass ; ("hit",) -> n candidates with a fg depth read
    nbonly = Acc()        # pairs accepted only thanks to neighbours at TAU_REF
    front = Acc()         # front violations (rel < -tau) per tau (omega4)
    border = Acc()        # fg pixels within 1 / 4 px of the border
    vis_cnt = Acc()       # per query pixel: number of other target views that see it (omega4, TAU_REF)
    cat_share = Acc()     # key: (scheme, cat) -> loss entries
    per_sample = []       # dense valid pairs per sample
    fg_per_view = []
    t0 = time.time()
    OMEGA4 = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for oi, idx in enumerate(idxs):
        random.seed(1000 + oi)
        smp = ds[idx]
        dep = smp["depth_map"].float()
        img = smp["image"].float()
        alpha = smp["alpha_mask"][:, 0].float()
        c2w = smp["c2w"].double()
        K = smp["fxfycxcy"].double()
        V, H, W = dep.shape
        tg = list(range(n_in, n_in + n_tgt))          # disjoint mode: the last n_tgt views are the targets
        P = {t: unproject(dep[t], c2w[t], K[t]) for t in tg}
        cats = {t: categories(dep[t], alpha[t]) for t in tg}
        dense_pairs = 0
        for ta in tg:
            fg = dep[ta] > 0
            ys, xs = torch.nonzero(fg, as_tuple=True)
            n = ys.numel()
            fg_per_view.append(n)
            if n == 0:
                continue
            border.add("fg", n)
            border.add("within1", ((xs < 1) | (xs >= W - 1) | (ys < 1) | (ys >= H - 1)).sum())
            border.add("within4", ((xs < 4) | (xs >= W - 4) | (ys < 4) | (ys >= H - 4)).sum())
            Pa = P[ta][ys, xs]
            ca = cats[ta][ys, xs]
            rgb_a = img[ta][:, ys, xs].double()
            count = torch.zeros(n, dtype=torch.long)
            per_view_pass = []
            for tb in tg:
                if tb == ta:
                    continue
                u, v, z = project(Pa, c2w[tb], K[tb])
                xu, yv = u.floor(), v.floor()
                inside1 = (z > 1e-6) & (xu >= 1) & (xu <= W - 2) & (yv >= 1) & (yv <= H - 2)
                inside4 = (z > 1e-6) & (xu >= 4) & (xu <= W - 5) & (yv >= 4) & (yv <= H - 5)
                Db = dep[tb].double()
                D_exact, rel_exact = best_neighbour(Db, u, v, z, [(0, 0)], inside1)
                D_om, rel_om = best_neighbour(Db, u, v, z, OMEGA4, inside1)
                uc, vc = (u - 0.5), (v - 0.5)
                D_ce, rel_ce = best_neighbour(Db, uc, vc, z, OMEGA4, inside1)   # floor(u-0.5)+{0,1}: 4 nearest centres
                hit = inside1 & (D_om > 0)
                passes.add(("hit",), hit.sum())
                passes.add(("inside1",), inside1.sum())
                passes.add(("inside4",), inside4.sum())
                passes.add(("cand",), n)
                # colour check
                rgb_b = bilinear(img[tb], u, v)
                cdiff = (rgb_a - rgb_b).abs().mean(0)
                ang = torch.rand(n, generator=g, dtype=torch.float64) * 2 * np.pi
                rgb_c = bilinear(img[tb], u + 8 * torch.cos(ang), v + 8 * torch.sin(ang))
                cdiff_c = (rgb_a - rgb_c).abs().mean(0)
                for nbname, rel, Dn in (("exact", rel_exact, D_exact), ("omega4", rel_om, D_om), ("centre4", rel_ce, D_ce)):
                    ok = inside1 & (Dn > 0)
                    ar = rel.abs()
                    for bi in range(len(BANDS) - 1):
                        inb = ok & (ar >= BANDS[bi]) & (ar < BANDS[bi + 1])
                        for sname, smask in (("behind", inb & (rel > 0)), ("front", inb & (rel <= 0))):
                            band.add((nbname, sname, bi), smask.sum(), cdiff[smask].sum())
                            ctrl.add((nbname, sname, bi), smask.sum(), cdiff_c[smask].sum())
                    for tau in TAUS:
                        passes.add((nbname, tau), (ok & (ar < tau)).sum())
                        if nbname == "omega4":
                            front.add(tau, (ok & (rel < -tau)).sum())
                    if nbname == "omega4":
                        d = (z - Dn).abs()
                        om = ok & (d < 0.05 * z + 0.01) & (d < 0.05 * Dn + 0.01)
                        passes.add(("omega_5pct_001",), om.sum())
                        band.add(("omega_5pct_001", "all", 0), om.sum(), cdiff[om].sum())
                        ctrl.add(("omega_5pct_001", "all", 0), om.sum(), cdiff_c[om].sum())
                p_exact = inside1 & (D_exact > 0) & (rel_exact.abs() < TAU_REF)
                p_om = inside1 & (D_om > 0) & (rel_om.abs() < TAU_REF)
                p_ce = inside1 & (D_ce > 0) & (rel_ce.abs() < TAU_REF)
                only_om = p_om & ~p_exact
                only_ce = p_ce & ~p_exact
                nbonly.add("omega4_only", only_om.sum(), cdiff[only_om].sum())
                nbonly.add("centre4_only", only_ce.sum(), cdiff[only_ce].sum())
                nbonly.add("exact", p_exact.sum(), cdiff[p_exact].sum())
                nbonly.add("omega4_only_ctrl", only_om.sum(), cdiff_c[only_om].sum())
                nbonly.add("centre4_only_ctrl", only_ce.sum(), cdiff_c[only_ce].sum())
                per_view_pass.append(p_om)
                count += p_om.long()
            dense_pairs += int(count.sum())
            for k in range(n_tgt):
                vis_cnt.add(k, (count == k).sum())
            # category shares of the loss entries (each valid pair = one entry)
            for k in range(3):
                cat_share.add(("dense", k), count[ca == k].sum())
            # Omega sampling: drop pixels seen by no other view, sort by count desc, keep the top half if
            # N // 2 > 512, then 512 at random
            order = torch.argsort(count, descending=True, stable=True)
            if n // 2 > N_TRACKS:
                order = order[: n // 2]
            pick = order[torch.randperm(order.numel(), generator=g)[:N_TRACKS]]
            for k in range(3):
                cat_share.add(("omega512", k), count[pick][ca[pick] == k].sum())
            cand = torch.nonzero(count >= 1, as_tuple=True)[0]
            pick_u = cand[torch.randperm(cand.numel(), generator=g)[:N_TRACKS]]
            for k in range(3):
                cat_share.add(("uniform512", k), count[pick_u][ca[pick_u] == k].sum())
                cat_share.add(("pixels", k), (ca == k).sum())
        per_sample.append(dense_pairs)
        if oi % 16 == 0:
            print(f"[probe] object {oi}/{a.n_objects} ({time.time() - t0:.0f}s) dense pairs {dense_pairs}", flush=True)

    out = dict(config=a.config, n_objects=a.n_objects, seed=a.seed, bands=BANDS, taus=TAUS, tau_ref=TAU_REF,
               n_tracks=N_TRACKS, sil_px=SIL_PX, edge_rel=EDGE_REL,
               fg_per_view=dict(mean=float(np.mean(fg_per_view)), min=int(np.min(fg_per_view)), max=int(np.max(fg_per_view))),
               band={"|".join(map(str, k)): v for k, v in band.as_dict().items()},
               ctrl={"|".join(map(str, k)): v for k, v in ctrl.as_dict().items()},
               passes={"|".join(map(str, k)): v["n"] for k, v in passes.as_dict().items()},
               front={str(k): v["n"] for k, v in front.as_dict().items()},
               nbonly=nbonly.as_dict(), border={k: v["n"] for k, v in border.as_dict().items()},
               vis_count={str(k): v["n"] for k, v in vis_cnt.as_dict().items()},
               cat_share={"|".join(map(str, k)): v["n"] for k, v in cat_share.as_dict().items()},
               dense_pairs_per_sample=dict(min=int(np.min(per_sample)), p10=float(np.percentile(per_sample, 10)),
                                           median=float(np.median(per_sample)), max=int(np.max(per_sample))))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"[probe] DONE ({time.time() - t0:.0f}s) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
