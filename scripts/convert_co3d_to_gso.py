"""Convert CO3D webdataset (nested-tar) into GSO-format loose directories.

Per plan §8.10.8 — reads outer tar per scene, sample 25 views evenly, apply
synchronized crop to (image, fg_mask, depth, depth_valid) with SAME crop_box,
encode object-alpha-filtered depth to uint16 with per-frame min/max in
transforms.json, save c2w in NeRF/Blender convention (via diag(1,-1,-1,1)
OpenCV→NeRF flip, verified by §8.10.6 sanity check → 97.8% fg overlap).

Two key invariants (per plan):
  A) SAME crop_box for image + fg + depth + depth_valid (breaks pixel-space
     correspondence otherwise).
  B) NO depth unit conversion. CO3D depth is in COLMAP scene scale (matches
     [tx,ty,tz]); dataset_objaverse first-view normalize will rescale it to
     "first-view distance = 1 unit" semantics, same as GSO's Blender-scale
     depth. Encode raw float to uint16 + save min/max in transforms.json.
     Depth validity is CO3D depth_mask_list AND object alpha (fg > 127), so
     depth/point_map follows the same foreground convention as RGB.

Usage:
    python scripts/convert_co3d_to_gso.py \
        --input_dir /mnt/data-alpha-sg-02/team-camera/datasets/yuchen/co3d/webdataset/val \
        --output_dir /home/z50057756/data/co3d_teddybear_gsoformat \
        --split_file data/co3d_teddybear_pilot100.txt \
        --n_views 25 --n_workers 16
"""
import argparse
import io
import json
import math
import multiprocessing as mp
import os
import struct
import sys
import tarfile
import traceback
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


# --------------------------------------------------------------------------- #
# CO3D depth decoding (adapted from data/dataset_co3d.py::load_16big_png_depth)
# --------------------------------------------------------------------------- #
def load_16big_png_depth(bytes_io: io.BytesIO) -> np.ndarray:
    """Decode CO3D 16-bit PNG depth (int16 bit pattern → float16 → float32)."""
    depth_pil = Image.open(bytes_io)
    depth_arr = np.array(depth_pil, dtype=np.uint16)
    depth = np.frombuffer(depth_arr.tobytes(), dtype=np.float16).astype(np.float32)
    depth = depth.reshape((depth_pil.size[1], depth_pil.size[0]))
    return depth  # float32 (H, W), COLMAP-unit Z-depth


# --------------------------------------------------------------------------- #
# PLY parser (unused in this script but kept for potential future validation)
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Camera conversion: CO3D 11-tuple → stored transform_matrix
#
# The dataset applies ObjaverseDataset.transform_pose (RPY shuffle) to every
# transform_matrix it reads. That map is only RIGID for zero-roll TRACK_TO
# matrices (probe 2026-07-01: GSO drift 0.000°, CO3D handheld drift up to 25°
# → both models scored RA5=0.00 on the naive conversion).
#
# Probe also identified the rigid form on its valid domain:
#     transform_pose(P) = W @ P @ C
# with W = world map (x,y,z)->(x,-z,y) and C = diag(1,-1,-1) — i.e. the model
# internally uses OpenCV camera axes, and  c2w_nerf @ C == c2w_opencv.
#
# So the matrix the model must see is  M = W @ c2w_opencv,  and we store
# P = transform_pose⁻¹(M) (analytic inverse below) so the dataset's
# transform_pose(P) reproduces M exactly — rigid for ANY camera roll.
# --------------------------------------------------------------------------- #
from viser import transforms as vtf   # same lib as dataset → identical RPY conventions

_W_WORLD = np.array([
    [1, 0, 0, 0],
    [0, 0, -1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
], dtype=np.float64)   # position map (x,y,z) -> (x,-z,y)


def _wrap_pi(x: float) -> float:
    """Wrap angle to (-π, π]."""
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def transform_pose_inverse(M: np.ndarray) -> np.ndarray:
    """Exact functional inverse of ObjaverseDataset.transform_pose.

    Forward: rpy(P)=(r,p,y) → rot from_rpy(r-π/2, -y, p); pos (x,y,z)→(x,-z,y).
    The naive triple-inverse fails when the remapped pitch (-y) leaves the
    canonical range [-π/2, π/2]: re-decomposition snaps to the OTHER
    Euler-XYZ representation. Every rotation has exactly two Euler-XYZ
    representations — try both, keep the one whose forward round-trip
    reproduces M.
    """
    rpy = vtf.SO3.from_matrix(M[:3, :3]).as_rpy_radians()
    a1, b1, c1 = rpy.roll, rpy.pitch, rpy.yaw
    candidates = [
        (a1, b1, c1),
        (_wrap_pi(a1 + np.pi), _wrap_pi(np.pi - b1), _wrap_pi(c1 + np.pi)),
    ]
    pos = [M[0, 3], M[2, 3], -M[1, 3]]
    for a, b, c in candidates:
        r, p, y = _wrap_pi(a + np.pi / 2.0), c, _wrap_pi(-b)
        P = np.eye(4)
        P[:3, :3] = vtf.SO3.from_rpy_radians(r, p, y).as_matrix()
        P[:3, 3] = pos
        if np.allclose(_transform_pose_forward(P)[:3, :3], M[:3, :3], atol=1e-4):
            return P
    raise ValueError("transform_pose_inverse: no valid Euler branch (gimbal lock?)")


def _transform_pose_forward(pose: np.ndarray) -> np.ndarray:
    """Copy of ObjaverseDataset.transform_pose for round-trip verification."""
    rpy = vtf.SO3.from_matrix(pose[:3, :3]).as_rpy_radians()
    r, p, y = rpy.roll, rpy.pitch, rpy.yaw
    rot = vtf.SO3.from_rpy_radians(r - np.pi / 2.0, -y, p)
    x, y_, z = pose[:3, 3]
    out = np.eye(4)
    out[:3, :3] = rot.as_matrix()
    out[:3, 3] = [x, -z, y_]
    return out


def cam11_to_stored_pose(cam11: np.ndarray) -> Tuple[np.ndarray, float, float, float, float]:
    """Return (stored transform_matrix, fx, fy, cx, cy) from CO3D 11-tuple.

    stored = transform_pose⁻¹(W @ c2w_opencv), asserted to round-trip.
    """
    qw, qx, qy, qz, tx, ty, tz, fx, fy, cx, cy = [float(x) for x in cam11]
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()   # scipy: xyzw
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R
    w2c[:3, 3] = [tx, ty, tz]
    c2w_opencv = np.linalg.inv(w2c)

    model_visible = _W_WORLD @ c2w_opencv
    stored = transform_pose_inverse(model_visible)

    # round-trip check: dataset's transform_pose(stored) must equal model_visible
    rt = _transform_pose_forward(stored)
    if not np.allclose(rt, model_visible, atol=1e-4):
        raise ValueError(
            f"transform_pose round-trip failed (max err {np.abs(rt - model_visible).max():.2e}) "
            f"— likely gimbal lock; view should be skipped"
        )
    return stored, fx, fy, cx, cy


# --------------------------------------------------------------------------- #
# Per-view processing: synchronized crop + encoding
# --------------------------------------------------------------------------- #
def process_one_view(
    rgb: np.ndarray,             # (H, W, 3) uint8
    fg: np.ndarray,              # (H, W)    uint8 soft 0-254
    depth: np.ndarray,           # (H, W)    float32 raw COLMAP-unit Z-depth
    depth_valid: np.ndarray,     # (H, W)    bool
    cx: float,
    cy: float,
    fx: float,
    max_side: int = 0,
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """Apply synchronized center-crop to principal-point-centered layout.

    All 4 tensors use the SAME crop_box. Returns:
      rgba          — (H_new, W_new, 4) uint8, RGB + binary alpha from fg mask
      depth_u16     — (H_new, W_new)    uint16, min/max in transforms.json
      d_min, d_max  — floats, for decoding
      fov           — new camera_angle_x = 2*atan((W_new/2)/fx)
    """
    H, W = rgb.shape[:2]

    # --- (1) PAD to square, principal point at canvas center ---
    # The model's token grid assumes square inputs (448/14 = 32 → 1024
    # tokens; non-square crashed job 84594). The FIRST square strategy was a
    # center-CROP with side 2*min(cx, W-cx, cy, H-cy) — on portrait CO3D
    # frames (e.g. 819x463, object spanning ~790 px) that keeps only ~56% of
    # the height and chopped 28.9% of the fg pixels (user-caught, 2026-07-02).
    # PAD instead: side = 2*max(...), canvas contains the WHOLE image and the
    # principal point still lands exactly at the canvas center (preserving
    # the loader's fxfycxcy = [f, f, S/2, S/2] assumption).
    half = int(math.ceil(max(cx, W - cx, cy, H - cy)))
    if half < 1:
        raise ValueError(f"Degenerate pad: cx={cx} cy={cy} in {H}x{W} → half={half}")
    S = 2 * half
    # paste position of the original image inside the S x S canvas
    ox = half - int(round(cx))
    oy = half - int(round(cy))

    # --- (2) SYNCHRONIZED pad: all four tensors, SAME canvas placement ---
    # pad values: RGB=white (cosmetic only — alpha=0 makes the loader's
    # white-composite overwrite it), fg=0 (background), depth=65535 handled
    # after encoding (pad region is invalid), depth_valid=False.
    rgb_c = np.full((S, S, 3), 255, dtype=rgb.dtype)
    rgb_c[oy:oy + H, ox:ox + W, :] = rgb
    fg_c = np.zeros((S, S), dtype=fg.dtype)
    fg_c[oy:oy + H, ox:ox + W] = fg
    depth_c = np.zeros((S, S), dtype=depth.dtype)
    depth_c[oy:oy + H, ox:ox + W] = depth
    dvalid_c = np.zeros((S, S), dtype=bool)
    dvalid_c[oy:oy + H, ox:ox + W] = depth_valid

    H_new = W_new = S

    # --- (3) fov from canvas half-width + original fx (fx unchanged by pad) ---
    fov = 2.0 * math.atan((S / 2.0) / fx)

    # --- (4) image → RGBA (RGB + binary alpha from soft fg threshold) ---
    alpha_mask = fg_c > 127
    alpha = (alpha_mask.astype(np.uint8)) * 255
    rgba = np.dstack([rgb_c, alpha])

    # --- (5) depth encoding: no unit conversion, just uint16 + per-frame min/max ---
    # Invalid-pixel convention MUST match the dataloader
    # (dataset_objaverse.py:234  `mask = depth < 65534`, i.e. GSO far-plane
    # semantics: 65535 = invalid). Writing invalid as 0 makes MVS holes decode
    # to min_depth → per-view ghost "curtains" (layered shards in viser,
    # 2026-07-02). Valid range is scaled to [0, 65533] to stay clear of the
    # <65534 threshold (decode-side scale error 65533/65535 ≈ 0.003%).
    # Use object-only depth for CO3D-as-GSO. CO3D's depth_mask_list marks MVS
    # depth validity and can include floor/table/background pixels; the RGBA
    # alpha is the object foreground convention consumed by the GSO loader.
    dvalid_c = dvalid_c & alpha_mask

    valid_depth = depth_c[dvalid_c]
    if valid_depth.size == 0:
        d_min, d_max = 0.0, 1.0
    else:
        d_min = float(valid_depth.min())
        d_max = float(valid_depth.max())

    depth_u16 = np.full((H_new, W_new), 65535, dtype=np.uint16)   # invalid = 65535
    if d_max > d_min:
        depth_u16[dvalid_c] = (
            (depth_c[dvalid_c] - d_min) / (d_max - d_min) * 65533
        ).astype(np.uint16)

    # --- (6) optional uniform downscale (V10: quota, matches 512px training
    # renders). PIXELS ONLY — fov/pose/depth.min-max untouched: a uniform
    # resize scales canvas half-width and effective focal by the same factor,
    # so camera_angle_x is invariant; u16 depth VALUES are resampled NEAREST
    # (no interpolation → 65535-invalid and min/max decode stay exact), alpha
    # NEAREST keeps the binary {0,255} convention. ---
    if max_side and S > max_side:
        rgb_small = np.array(Image.fromarray(rgba[..., :3]).resize(
            (max_side, max_side), resample=Image.LANCZOS))
        a_small = np.array(Image.fromarray(rgba[..., 3]).resize(
            (max_side, max_side), resample=Image.NEAREST))
        rgba = np.dstack([rgb_small, a_small])
        depth_u16 = np.array(Image.fromarray(depth_u16).resize(
            (max_side, max_side), resample=Image.NEAREST))

    return rgba, depth_u16, d_min, d_max, fov


# --------------------------------------------------------------------------- #
# Scene processor: outer tar → GSO-format directory
# --------------------------------------------------------------------------- #
def convert_scene(outer_tar_path: Path, out_dir: Path, n_views: int = 25,
                  mask_override_dir: Optional[Path] = None,
                  max_side: int = 0) -> Optional[str]:
    """Return None on success, error message on failure."""
    try:
        with tarfile.open(outer_tar_path) as outer:
            # detect prefix from meta member (looks like <prefix>.meta.json)
            names = outer.getnames()
            meta_name = next((n for n in names if n.endswith(".meta.json")), None)
            if meta_name is None:
                return f"no meta.json in {outer_tar_path}"
            prefix = meta_name[: -len(".meta.json")]

            meta = json.loads(outer.extractfile(f"{prefix}.meta.json").read())
            view_num = meta["view_num"]
            indices = np.linspace(0, view_num - 1, n_views, dtype=int).tolist()

            # load all 5 inner tars into memory
            cams_data = outer.extractfile(f"{prefix}.cameras.tar").read()
            imgs_data = outer.extractfile(f"{prefix}.images.tar").read()
            deps_data = outer.extractfile(f"{prefix}.depths.tar").read()
            fgms_data = outer.extractfile(f"{prefix}.image_masks.tar").read()
            dvms_data = outer.extractfile(f"{prefix}.depth_mask_list.tar").read()

        # open inner tars
        cams_tar = tarfile.open(fileobj=io.BytesIO(cams_data))
        imgs_tar = tarfile.open(fileobj=io.BytesIO(imgs_data))
        deps_tar = tarfile.open(fileobj=io.BytesIO(deps_data))
        fgms_tar = tarfile.open(fileobj=io.BytesIO(fgms_data))
        dvms_tar = tarfile.open(fileobj=io.BytesIO(dvms_data))

        out_dir.mkdir(parents=True, exist_ok=True)

        frames = []
        for out_idx, i in enumerate(indices):
            # --- load 5 modalities for view i ---
            cam11 = np.load(io.BytesIO(cams_tar.extractfile(f"{prefix}.camera_{i}.npy").read()))
            rgb = np.array(Image.open(
                io.BytesIO(imgs_tar.extractfile(f"{prefix}.images_{i}.jpg").read())
            ))                                              # (H, W, 3) uint8
            if mask_override_dir is not None:
                # external per-scene mask dir keyed by inner frame index i
                fg = np.array(Image.open(
                    mask_override_dir / outer_tar_path.stem / f"{i}.png"
                ))                                          # (H, W) uint8, 0/255
                assert fg.shape == rgb.shape[:2], \
                    f"override mask {fg.shape} != rgb {rgb.shape[:2]} (view {i})"
            else:
                fg = np.array(Image.open(
                    io.BytesIO(fgms_tar.extractfile(f"{prefix}.image_masks_{i}.png").read())
                ))                                          # (H, W) uint8
            depth = load_16big_png_depth(
                io.BytesIO(deps_tar.extractfile(f"{prefix}.depths_{i}.png").read())
            )                                               # (H, W) float32
            dvalid = np.array(Image.open(
                io.BytesIO(dvms_tar.extractfile(f"{prefix}.depth_mask_list_{i}.png").read())
            )).astype(bool)                                 # (H, W) bool

            # --- (a) camera: store transform_pose⁻¹(W @ c2w_opencv) so the
            #     dataset's transform_pose recovers a rigid, roll-safe matrix ---
            stored_pose, fx, fy, cx, cy = cam11_to_stored_pose(cam11)

            # --- (b) synchronized crop of 4 tensors + fov ---
            rgba, depth_u16, d_min, d_max, fov = process_one_view(
                rgb, fg, depth, dvalid, cx, cy, fx, max_side=max_side
            )

            # --- (c) save ---
            Image.fromarray(rgba).save(out_dir / f"{out_idx:03d}.png")
            Image.fromarray(depth_u16, mode="I;16").save(out_dir / f"{out_idx:03d}_depth.png")

            frames.append({
                "file_path": f"{out_idx:03d}.png",
                "camera_angle_x": fov,
                "fov": fov,
                "transform_matrix": stored_pose.tolist(),
                "depth": {"min": d_min, "max": d_max},
            })

        # write transforms.json
        with open(out_dir / "transforms.json", "w") as f:
            json.dump({
                "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],   # dummy — inference ignores
                "scale": 1.0,
                "offset": [0.0, 0.0, 0.0],
                "mask_source": "override" if mask_override_dir is not None else "co3d",
                **({"max_side": max_side} if max_side else {}),
                "frames": frames,
            }, f)

        return None  # success
    except Exception as e:
        return f"{outer_tar_path.name}: {type(e).__name__}: {e}\n{traceback.format_exc()}"


# --------------------------------------------------------------------------- #
# multiprocessing wrapper
# --------------------------------------------------------------------------- #
def _worker(args):
    outer_tar_path, out_dir, n_views, mask_override_dir, max_side = args
    return convert_scene(outer_tar_path, out_dir, n_views, mask_override_dir, max_side)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="dir containing outer .tar files")
    ap.add_argument("--output_dir", required=True, help="target for <scene>/*.png + transforms.json")
    ap.add_argument("--split_file", required=True, help="one scene_id per line (no .tar suffix)")
    ap.add_argument("--n_views", type=int, default=25)
    ap.add_argument("--n_workers", type=int, default=16)
    ap.add_argument("--mask_override_dir", default=None,
                    help="root dir of per-scene <sid>/{i}.png binary masks (e.g. SAM2 "
                         "output); falls back to the tar's image_masks when unset")
    ap.add_argument("--max_side", type=int, default=0,
                    help="downscale square canvas to this side if larger (0=off). "
                         "Pixels only: fov/pose/depth min-max metadata untouched")
    args = ap.parse_args()

    with open(args.split_file) as f:
        scene_ids = [l.strip() for l in f if l.strip()]

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for sid in scene_ids:
        outer = input_dir / f"{sid}.tar"
        if not outer.exists():
            print(f"MISSING: {outer}", file=sys.stderr)
            continue
        # skip if already converted (transforms.json present)
        if (output_dir / sid / "transforms.json").exists():
            continue
        tasks.append((outer, output_dir / sid, args.n_views,
                      Path(args.mask_override_dir) if args.mask_override_dir else None,
                      args.max_side))

    if not tasks:
        print("nothing to do (all scenes already converted)")
        return

    print(f"processing {len(tasks)} scenes with {args.n_workers} workers ...")
    n_ok, n_fail = 0, 0
    errors = []
    with mp.Pool(args.n_workers) as pool:
        for i, err in enumerate(pool.imap_unordered(_worker, tasks), 1):
            if err is None:
                n_ok += 1
            else:
                n_fail += 1
                errors.append(err)
            if i % 20 == 0:
                print(f"  {i}/{len(tasks)}  ok={n_ok}  fail={n_fail}")

    print(f"\nDONE  ok={n_ok}  fail={n_fail}")
    for e in errors[:5]:
        print(f"  FAIL: {e[:200]}")


if __name__ == "__main__":
    main()
