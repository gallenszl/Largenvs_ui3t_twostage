import io
import random
import tarfile
import traceback
import os
import math
from einops import rearrange
import numpy as np
import PIL
import torch
from torch.utils.data import Dataset
import json
import torch.nn.functional as F
from PIL import Image
from viser import transforms as vtf
import imageio


class ObjaverseDataset(Dataset):
    def __init__(self, config, is_second=False):
        self.config = config
        if is_second:
            self.root_path = self.config.training.root_path2
            data_path = self.config.training.dataset_path2
            self.total_frames_per_obj = self.config.training.total_frames_per_obj2
            self.use_tar = self.config.training.get("use_tar2", False)
            self.tar_root_path = self.config.training.get("tar_root_path2", None)
        else:
            self.root_path = self.config.training.root_path
            data_path = self.config.training.dataset_path
            self.total_frames_per_obj = self.config.training.total_frames_per_obj
            self.use_tar = self.config.training.get("use_tar", False)
            self.tar_root_path = self.config.training.get("tar_root_path", None)
        if self.use_tar and not self.tar_root_path:
            raise ValueError("use_tar is True but tar_root_path is not set")
        try:
            with open(data_path, 'r') as f:
                self.all_object_list = f.readlines()
            self.all_object_list = [l.strip() for l in self.all_object_list]
            
        except Exception as e:
            print(f"Error reading dataset paths from '{data_path}'")
            raise e
        
        self.fov = 0.6981317007977318
        # focal_length = (h_w/2) / math.tan(fov/2)
        # P2: per-view roll augment magnitude (R1 sim2real). Default 0 = disabled.
        # Sampled per view in preprocess_frames(); applied to image+depth+alpha+c2w
        # so multi-view back-projection stays consistent.
        self.roll_augment_max_deg = self.config.training.get("roll_augment_max_deg", 0.0)

        # E-arm curriculum view sampling ("anchor_pool", default "none" = legacy).
        # Easy-to-hard: pool = the K frames angularly nearest to a random anchor
        # (anchor doubles as the first cond view / gauge reference); K ~
        # U[k_floor(progress), n_frames], k_floor ramps k_start -> n_frames
        # linearly over the first `ramp_frac` of training, then stays at
        # n_frames where the selection is distribution-identical to
        # random.sample(range(n), num_views). Angular distance = arccos of the
        # dot product of unit camera-position directions (object-centric;
        # immune to R2 radius jitter and A1 lookAt offset).
        self.curriculum = str(self.config.training.get("curriculum_view_sampling", "none"))
        self.curriculum_k_start = int(self.config.training.get("curriculum_k_floor_start", 8))
        self.curriculum_ramp_frac = float(self.config.training.get("curriculum_ramp_frac", 0.6))
        # training progress in [0,1]; refreshed by train.py at each dataloader
        # epoch reset (requires persistent_workers=False so workers re-fork).
        self.curriculum_progress = 0.0

    def __len__(self):
        return len(self.all_object_list)

    def view_selector(self):
        if self.total_frames_per_obj < self.config.training.num_views:
            raise ValueError(f"total_frames_per_obj ({self.total_frames_per_obj}) is smaller than num_views ({self.config.training.num_views})")

        sampled_frames = random.sample(range(0, self.total_frames_per_obj), self.config.training.num_views)
        return sampled_frames

    def _read_camera_dirs(self, object_name):
        """Unit camera-position directions for all frames (transforms.json only)."""
        if self.use_tar:
            with tarfile.open(os.path.join(self.tar_root_path, f"{object_name}.tar"), "r:") as th:
                member = next(m for m in th.getmembers()
                              if m.isfile() and m.name.split("/")[-1] == "transforms.json")
                frames = json.loads(th.extractfile(member).read())["frames"]
        else:
            with open(os.path.join(self.root_path, object_name, "transforms.json"), "rb") as f:
                frames = json.loads(f.read())["frames"]
        t = np.array([np.array(fr["transform_matrix"], dtype=np.float64)[:3, 3] for fr in frames])
        return t / np.linalg.norm(t, axis=1, keepdims=True)

    def _curriculum_select(self, object_name):
        """Anchor-pool curriculum: [anchor(cond0)] + 6 views from the nearest-K pool."""
        n = self.total_frames_per_obj
        num_views = self.config.training.num_views
        d = self._read_camera_dirs(object_name)[:n]
        anchor = random.randrange(n)
        theta = np.arccos(np.clip(d @ d[anchor], -1.0, 1.0))
        order = np.argsort(theta)  # order[0] == anchor
        s = min(max(self.curriculum_progress, 0.0) / self.curriculum_ramp_frac, 1.0)
        # deterministic pool size (unit test finding: a per-sample random upper
        # bound U[k,n] dilutes the easy phase to p50~37 deg; mid-size pools
        # already mix near/far targets internally, so no extra stochasticity)
        K = int(round(self.curriculum_k_start + (n - self.curriculum_k_start) * s))
        pool = [int(i) for i in order[:K] if i != anchor]
        return [anchor] + random.sample(pool, num_views - 1)

    @staticmethod
    def transform_pose(pose):
        pose = np.array(pose)
        rpy = vtf.SO3.from_matrix(pose[:3,:3]).as_rpy_radians()
        r,p,y = rpy.roll, rpy.pitch, rpy.yaw
        rpy = [r-np.pi/2, -y, p]
        
        pos = pose[:3,3]
        x,y,z = pos
        xyz = [x, -z, y]

        rot = vtf.SO3.from_rpy_radians(*rpy)
        wxyz = rot.wxyz
        position = np.array(xyz)

        pose1 = np.eye(4)
        pose1[:3,:3] = rot.as_matrix()
        pose1[:3,3] = position
        return pose1

    @staticmethod
    def pinhole_z_depth_to_xyz(depth, f, H=512, W=512):
        if isinstance(depth, torch.Tensor):
            depth = depth.numpy()
        if not isinstance(depth, float):
            H, W = depth.shape
            z = depth
        else:
            z = np.ones((H, W), dtype=np.float32) * depth
        y, x = np.mgrid[:H, :W]
        x = x - W // 2
        y = H // 2 - y
        x = x / f * z
        y = - y / f * z
        return np.stack([x, y, z], -1)

    def preprocess_frames(self, object_name, image_indices):
        resize_h = self.config.model.image_tokenizer.image_size
        patch_size = self.config.model.image_tokenizer.patch_size

        images = []
        fxfycxcys = []
        intrinsics = []
        c2ws = []
        point_maps = []
        depth_maps = []
        alpha_masks = []  # P4a: per-view foreground/background mask from PNG alpha

        # tar / loose: pick a reader that returns the raw bytes of <object_name>/<fname>
        tar_handle = None
        if self.use_tar:
            tar_handle = tarfile.open(os.path.join(self.tar_root_path, f"{object_name}.tar"), "r:")
            # P5a: auto-detect tar member prefix to support multiple layouts:
            #   FluffyElephant_tar:        "./<fname>"
            #   objaverse_renders_44798:   "<uid>/<fname>"   (first member is the
            #                              bare "<uid>" dir entry; need to peek
            #                              the first file with a "/")
            tar_prefix = ""
            for m in tar_handle.getmembers():
                if m.name.startswith("./"):
                    tar_prefix = "./"
                    break
                if "/" in m.name:
                    tar_prefix = m.name.split("/")[0] + "/"
                    break
            def read_bytes(fname):
                return tar_handle.extractfile(tar_prefix + fname).read()
        else:
            def read_bytes(fname):
                with open(os.path.join(self.root_path, object_name, fname), "rb") as f:
                    return f.read()

        # P8 (plan 8.19.2): RGB views are <NNN>.webp after in-place WebP transcode,
        # <NNN>.png in untranscoded datasets. Depth stays PNG either way.
        def read_rgb_bytes(img_idx):
            try:
                return read_bytes(f"{img_idx:03d}.webp")
            except Exception:
                return read_bytes(f"{img_idx:03d}.png")

        try:
            all_poses = json.loads(read_bytes("transforms.json"))['frames']

            target_first_view = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,-1],[0,0,0,1.]])

            for v_idx, img_idx in enumerate(image_indices):
                # P2: sample one roll angle (deg) per view; applied to image,
                # depth, alpha, c2w with the SAME angle so back-projection stays
                # consistent. 0 when self.roll_augment_max_deg == 0 (default off).
                if self.roll_augment_max_deg > 0:
                    roll_deg = random.uniform(
                        -self.roll_augment_max_deg, self.roll_augment_max_deg)
                else:
                    roll_deg = 0.0

                ### image + alpha (P4a optimized — single PNG decode shared by RGB + alpha)
                # Old code did np.array(image_rgba) for alpha THEN PIL alpha_composite +
                # convert("RGB"), which re-traversed the decoded RGBA pixel buffer twice
                # (effectively 2× the per-view PNG cost). Net: ~16s/iter wasted on 8 H200
                # at batch=8 × 7 views (diagnostic plan §8.3.4 Test 1 vs Test 2).
                image_rgba = Image.open(io.BytesIO(read_rgb_bytes(img_idx)))
                if image_rgba.mode != "RGBA":
                    # P8: WebP drops a uniform-255 alpha plane on all-opaque views and
                    # decodes as RGB; restore the explicit alpha channel
                    image_rgba = image_rgba.convert("RGBA")
                rgba_arr = np.array(image_rgba, copy=False)  # [H, W, 4] uint8 — decode ONCE
                alpha_raw = rgba_arr[..., 3]                 # view, no copy
                # White-bg composite in numpy (vectorized, faster than PIL.alpha_composite)
                alpha_f = (alpha_raw.astype(np.float32) / 255.0)[..., None]   # [H, W, 1]
                rgb_arr = (rgba_arr[..., :3].astype(np.float32) * alpha_f
                           + 255.0 * (1.0 - alpha_f)).astype(np.uint8)        # [H, W, 3]
                image = Image.fromarray(rgb_arr)             # PIL with already-decoded data

                original_image_h, original_image_w = rgba_arr.shape[:2]

                resize_w = int(resize_h / original_image_h * original_image_w)
                resize_w = int(round(resize_w / patch_size) * patch_size)
                image = image.resize((resize_w, resize_h), resample=Image.LANCZOS)

                # P2: rotate image CCW by roll_deg; fill white to match bg
                if roll_deg != 0.0:
                    image = image.rotate(
                        roll_deg, resample=Image.BILINEAR, fillcolor=(255, 255, 255))

                image = np.array(image) / 255.0
                image = torch.from_numpy(image).permute(2, 0, 1).float()

                # P4a: resize alpha NEAREST to preserve 0/1 boundary
                alpha_resized_pil = Image.fromarray(alpha_raw).resize(
                    (resize_w, resize_h), resample=Image.NEAREST)
                # P2: rotate alpha with same angle; fill 0 (bg)
                if roll_deg != 0.0:
                    alpha_resized_pil = alpha_resized_pil.rotate(
                        roll_deg, resample=Image.NEAREST, fillcolor=0)
                alpha_mask = torch.from_numpy(
                    (np.array(alpha_resized_pil) > 127).astype(np.float32)
                ).unsqueeze(0)  # [1, h, w]
                alpha_masks.append(alpha_mask)

                ### intrinsic
                # P1: per-frame fov if transforms.json has it (A2 enabled);
                #     fallback to self.fov for legacy FluffyElephant (no per-frame fov).
                view_fov = all_poses[img_idx].get('fov', self.fov)
                focal_length = (resize_w/2) / math.tan(view_fov/2)
                fxfycxcy = torch.tensor([focal_length, focal_length, resize_w/2, resize_h/2])
                intrinsic_mat = torch.tensor([[focal_length, 0, resize_w/2], [0., focal_length, resize_h/2], [0., 0., 1]])
                images.append(image)
                fxfycxcys.append(fxfycxcy)
                intrinsics.append(intrinsic_mat)


                ### extrinsic
                pose = all_poses[img_idx]['transform_matrix']
                c2w = torch.from_numpy(self.transform_pose(pose)).float()
                # P2: post-multiply Rz(roll_rad) — rotation about camera local z.
                # Done BEFORE first-view normalize so view 0's roll defines the
                # canonical frame.
                if roll_deg != 0.0:
                    roll_rad = math.radians(roll_deg)
                    c, s = math.cos(roll_rad), math.sin(roll_rad)
                    Rz_cam = torch.tensor([[c, -s, 0, 0],
                                           [s,  c, 0, 0],
                                           [0,  0, 1, 0],
                                           [0,  0, 0, 1.]])
                    c2w = c2w @ Rz_cam
                if v_idx == 0:
                    norm_first_view_t = torch.norm(c2w[:3,3])
                    c2w[:3,3] /= norm_first_view_t
                    inv_first_view = torch.inverse(c2w)
                else:
                    c2w[:3,3] /= norm_first_view_t
                c2w = target_first_view @ inv_first_view @ c2w

                c2ws.append(c2w)

                ### depth & point map
                depth = Image.open(io.BytesIO(read_bytes(f'{img_idx:03d}_depth.png')))
                depth = depth.resize((resize_w, resize_h), resample=Image.NEAREST)
                # P2: rotate depth with same roll angle as image+alpha+c2w.
                # Fill 65535 = invalid so the mask < 65534 check excludes wrapped pixels.
                if roll_deg != 0.0:
                    depth = depth.rotate(
                        roll_deg, resample=Image.NEAREST, fillcolor=65535)
                depth = np.array(depth)
                mask = depth < 65534 # valid mask, 1=fg, 0=bg

                max_depth = all_poses[img_idx]['depth']['max']
                min_depth = all_poses[img_idx]['depth']['min']
                depth_range = max_depth - min_depth

                depth = depth / 65535.0 * depth_range + min_depth
                depth = depth * mask

                # normalize by first view's translation
                depth = torch.from_numpy(depth) / norm_first_view_t

                depth_maps.append(depth.float())

                pt = self.pinhole_z_depth_to_xyz(depth, focal_length)
                pt = torch.from_numpy(pt).float()

                # transform to world coordinate
                h,w,c = pt.shape
                pt = rearrange(pt, 'h w c -> (h w) c')
                ones = torch.ones((h*w, 1), dtype=pt.dtype)
                pt = torch.cat([pt, ones], dim=-1)
                pt = c2w @ pt.T
                pt = pt[:3].T
                pt = rearrange(pt, '(h w) c -> c h w', h=h, w=w)
                pt = pt * torch.from_numpy(mask)[None,:,:].float()

                point_maps.append(pt)
        finally:
            if tar_handle is not None:
                tar_handle.close()

        images = torch.stack(images, dim=0)
        fxfycxcys = torch.stack(fxfycxcys, dim=0)
        intrinsics = torch.stack(intrinsics, dim=0)
        c2ws = torch.stack(c2ws)
        depth_maps = torch.stack(depth_maps, dim=0)
        point_maps = torch.stack(point_maps, dim=0)
        alpha_masks = torch.stack(alpha_masks, dim=0)  # P4a: [v, 1, h, w]

        return images, fxfycxcys, intrinsics, c2ws, point_maps, depth_maps, alpha_masks
        

    def __getitem__(self, idx):
        object_name = self.all_object_list[idx]
        if self.curriculum == "anchor_pool":
            image_indices = self._curriculum_select(object_name)
        else:
            image_indices = self.view_selector()
        input_images, input_fxfycxcy, input_intrinsics, input_c2ws, point_maps, depth_maps, alpha_masks = self.preprocess_frames(object_name, image_indices)
        extrinsic = torch.inverse(input_c2ws)

        image_indices = torch.tensor(image_indices).long().unsqueeze(-1)  # [v, 1]
        scene_indices = torch.full_like(image_indices, idx)  # [v, 1]
        indices = torch.cat([image_indices, scene_indices], dim=-1)  # [v, 2]

        return {
            "image": input_images,
            "c2w": input_c2ws,
            "extrinsic": extrinsic,
            "fxfycxcy": input_fxfycxcy,
            "intrinsic": input_intrinsics,
            "index": indices,
            "scene_name": object_name,
            "depth_map": depth_maps,
            "point_map": point_maps,
            "alpha_mask": alpha_masks,  # P4a: [v, 1, h, w]
        }
