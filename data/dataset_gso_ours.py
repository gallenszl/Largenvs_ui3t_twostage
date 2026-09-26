import random
import os
import numpy as np
import PIL
import torch
# from torch.utils.data import Dataset
from data.dataset_objaverse import ObjaverseDataset
import json
import torch.nn.functional as F
from PIL import Image
from viser import transforms as vtf
import math
from einops import rearrange


class GSODataset_ours(ObjaverseDataset):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.root_path = self.config.training.val_dataset_cfgs.root_dir
        # GSO validation stays on loose files regardless of training's use_tar setting.
        self.use_tar = False
        self.tar_root_path = None

        del self.all_object_list
        with open(self.config.training.val_dataset_cfgs.split_file, 'r') as f:
            self.all_object_list = f.readlines()
        self.all_object_list = [i.strip() for i in self.all_object_list]

        self.fov = 0.6981317007977318

        random.seed(0)
        self.rand_idx = [random.sample(range(0, 25), 14) for _ in range(len(self.all_object_list))]
        self.view_idx_list = None
        view_idx_file_path = self.config.inference.get("view_idx_file_path", None)
        if view_idx_file_path is not None:
            with open(view_idx_file_path, 'r') as f:
                view_idx_list = json.load(f)
            view_idx_list = {k: v for k, v in view_idx_list.items() if v is not None}
            self.all_object_list = [
                object_name for object_name in self.all_object_list
                if object_name in view_idx_list
            ]
            self.view_idx_list = view_idx_list

        # rand_idx = np.array(self.rand_idx, dtype=int)
        # np.savetxt('gso_idx.txt', rand_idx, fmt='%i', delimiter=' ')

    def view_selector(self, idx):
        if self.view_idx_list is not None:
            object_name = self.all_object_list[idx]
            current_view_idx = self.view_idx_list[object_name]
            return current_view_idx["context"] + current_view_idx["target"]
        return self.rand_idx[idx]

    def find_extra_indices(self, indices, num_extra):
        full_indices = set(list(range(25)))
        remain = list(full_indices - set(indices))
        extra = random.sample(remain, num_extra)
        return indices[:4] + extra + indices[4:]

    def __getitem__(self, idx):
        object_name = self.all_object_list[idx]
        image_indices = self.view_selector(idx)

        # image_indices = self.find_extra_indices(image_indices, 4)
        # image_indices = image_indices[:4] * 2

        # P4a: parent ObjaverseDataset.preprocess_frames also returns alpha_masks
        # (7th item). Wave0 E2: keep it so eval can compute foreground texture
        # metrics (fg_psnr / hf_ratio / fg_ssim ...) with the real mask.
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
            "alpha_mask": alpha_masks
        }
