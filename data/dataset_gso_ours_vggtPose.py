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


class GSODataset_ours_vggtPose(ObjaverseDataset):
    def __init__(self, config):
        super().__init__(config)
        self.config = config 
        self.root_path = self.config.training.val_dataset_cfgs.root_dir

        del self.all_object_list
        with open(self.config.training.val_dataset_cfgs.split_file, 'r') as f:
            self.all_object_list = f.readlines()
        self.all_object_list = [i.strip() for i in self.all_object_list]

        self.fov = 0.6981317007977318

        random.seed(0)
        self.rand_idx = [random.sample(range(0, 25), 14) for _ in range(len(self.all_object_list))]

        # rand_idx = np.array(self.rand_idx, dtype=int)
        # np.savetxt('gso_idx.txt', rand_idx, fmt='%i', delimiter=' ')

        with open('data/gso_vggt_pose.json', 'r') as f:
            self.vggt_pose = json.load(f)

    def view_selector(self, idx):
        return self.rand_idx[idx]

    def __getitem__(self, idx):
        object_name = self.all_object_list[idx]
        image_indices = self.view_selector(idx)
        # P4a: parent ObjaverseDataset.preprocess_frames now also returns alpha_masks
        # (7th item). GSO val doesn't use alpha_mask, so discard with `_`.
        input_images, input_fxfycxcy, input_intrinsics, input_c2ws, point_maps, depth_maps, _ = self.preprocess_frames(object_name, image_indices)
        
        ### override input pose
        for i in range(4):
            img_id = image_indices[i]
            try:
                pose = np.array(self.vggt_pose[object_name][f'{img_id:03d}.png'])
            except:
                print(f'object name: {object_name}, frame id: {img_id}')
                exit()
            input_c2ws[i] = torch.tensor(pose)
        
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
            "point_map": point_maps
        }
