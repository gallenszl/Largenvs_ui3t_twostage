"""Check that the loader's point_map lies on the pixel-centre rays that the model uses.

For every foreground pixel of every view of a few samples, it measures how far (in pixels)
the point_map point is from the ray that the repo's own compute_rays() builds for the same
pixel.  compute_rays() uses the pixel-centre convention (x + 0.5 - cx), and it is the code
the model actually consumes, so it is the reference here; the point-map code under test is
never called by this check except through the dataset itself.

Expected result
  before the fix : dx = -0.50 px, dy = -0.50 px (point sits on the ray through the top-left
                   corner of the pixel), |offset| = 0.71 px
  after the fix  : dx, dy, |offset| all < 1e-3 px (float rounding)

Usage (from the root of any RnG / lagernvs repo; CPU only, a few seconds):
  export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
  PY=/mnt/data-alpha-sg-01/team-camera/home/z50057756/conda/envs/rng-fa3/bin/python
  $PY tools_s2/check_pointmap_on_ray.py --config configs/<x>.yaml
  # preview the fix without editing the repo (patches the loader in memory only):
  $PY tools_s2/check_pointmap_on_ray.py --config configs/<x>.yaml --preview_fix
"""
import argparse
import importlib
import os
import random
import sys

import numpy as np
import torch


def fixed_pinhole_z_depth_to_xyz(depth, f, H=512, W=512):
    """The proposed fix: pixel (y, x) is unprojected through its centre (x + 0.5, y + 0.5)."""
    if isinstance(depth, torch.Tensor):
        depth = depth.numpy()
    if not isinstance(depth, float):
        H, W = depth.shape
        z = depth
    else:
        z = np.ones((H, W), dtype=np.float32) * depth
    y, x = np.mgrid[:H, :W]
    x = x + 0.5 - W / 2
    y = H / 2 - (y + 0.5)
    x = x / f * z
    y = - y / f * z
    return np.stack([x, y, z], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", choices=["val", "train"], default="val")
    ap.add_argument("--n_samples", type=int, default=2)
    ap.add_argument("--preview_fix", action="store_true")
    ap.add_argument("--tol_px", type=float, default=1e-3)
    a = ap.parse_args()
    torch.set_num_threads(2)
    sys.path.insert(0, os.getcwd())
    from omegaconf import OmegaConf
    from easydict import EasyDict as edict
    from utils.data_utils import ProcessData

    cfg = edict(OmegaConf.to_container(OmegaConf.load(a.config), resolve=True))
    cfg.training.roll_augment_max_deg = 0.0          # no in-plane rotation, same as the evidence scripts
    key = "val_dataset_name" if a.split == "val" else "dataset_name"
    mod, cls = cfg.training[key].rsplit(".", 1)
    Dataset = importlib.import_module(mod).__dict__[cls]
    if a.preview_fix:
        owner = next(c for c in Dataset.__mro__ if "pinhole_z_depth_to_xyz" in c.__dict__)
        owner.pinhole_z_depth_to_xyz = staticmethod(fixed_pinhole_z_depth_to_xyz)
        print(f"[check] in-memory fix applied to {owner.__module__}.{owner.__name__}")
    ds = Dataset(cfg)
    rays = ProcessData(cfg)

    dxs, dys, offs = [], [], []
    for s in range(a.n_samples):
        random.seed(1000 + s)
        smp = ds[s]
        c2w = smp["c2w"].double()                     # [V, 4, 4], OpenCV: x right, y down, z forward
        K = smp["fxfycxcy"].double()                  # [V, 4]
        P = smp["point_map"].double()                 # [V, 3, H, W], world frame
        fg = smp["depth_map"] > 0                     # [V, H, W]
        V, _, H, W = P.shape
        ro, rd = rays.compute_rays(c2w[None], K[None], H, W, device="cpu")
        ro, rd = ro[0].double(), rd[0].double()       # [V, 3, H, W], rd has unit length
        for v in range(V):
            m = fg[v]
            p, o, d = P[v][:, m], ro[v][:, m], rd[v][:, m]            # [3, N]
            R = c2w[v, :3, :3]
            pc = R.T @ (p - o)                                        # point in camera frame
            dc = R.T @ d                                              # ray direction in camera frame
            fx, fy = K[v, 0], K[v, 1]
            # same image plane z = 1 for both: the difference is the pixel offset of the point
            # from where compute_rays() says this pixel's centre ray goes
            dxs.append((fx * (pc[0] / pc[2] - dc[0] / dc[2])).numpy())
            dys.append((fy * (pc[1] / pc[2] - dc[1] / dc[2])).numpy())
            offs.append(np.hypot(dxs[-1], dys[-1]))
    dx, dy, off = (np.concatenate(t) for t in (dxs, dys, offs))
    print(f"[check] {a.config} split={a.split} samples={a.n_samples} foreground pixels={dx.size}")
    print(f"[check] offset from the pixel-centre ray, in pixels: dx median {np.median(dx):+.4f} "
          f"| dy median {np.median(dy):+.4f} | |offset| median {np.median(off):.4f} p99 {np.percentile(off, 99):.4f}")
    ok = abs(np.median(dx)) < a.tol_px and abs(np.median(dy)) < a.tol_px and np.percentile(off, 99) < a.tol_px
    print("[check] PASS: point_map is on the pixel-centre rays" if ok else
          "[check] FAIL: point_map is off the pixel-centre rays (before the fix expect dx = dy = -0.5)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
