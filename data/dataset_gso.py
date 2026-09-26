import random
import os
import numpy as np
import PIL
import torch
from torch.utils.data import Dataset
import json
import torch.nn.functional as F
from PIL import Image
from viser import transforms as vtf
import math


class GSODataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config 
        self.root_dir = self.config.root_dir
        self.suffix = self.config.suffix

        with open(self.config.split_file, 'r') as f:
            self.all_obj_list = f.readlines()
        self.all_obj_list = [i.strip() for i in self.all_obj_list]

        self.fov = 0.6981317007977318

        self.inference = self.config.get("if_inference", False)
        if self.inference:
            self.view_idx_list = {
                'context': [0,1,2,3],
                'target' : [4,5,6,7,8,9,10,11,12,13]}

    def __len__(self):
        return len(self.all_obj_list)
    
    @staticmethod
    def convert_pose(w2c):
        # blender w2c -> opencv c2w
        pose = np.eye(4)
        pose[:3] = w2c
        pose = np.linalg.inv(pose)

        rpy = vtf.SO3.from_matrix(pose[:3,:3]).as_rpy_radians()
        r,p,y = rpy.roll, rpy.pitch, rpy.yaw
        rpy = [r-np.pi/2, -y, p]
        
        pos = pose[:3,3]
        x,y,z = pos
        xyz = [x, -z, y]

        rot = vtf.SO3.from_rpy_radians(*rpy)
        position = np.array(xyz)

        c2w = np.eye(4)
        c2w[:3,:3] = rot.as_matrix()
        c2w[:3, 3] = position
        return c2w
    
    def preprocess_frames(self, pose_paths_chosen, image_paths_chosen, fxfycxcy):
        resize_h = self.config.image_size
        patch_size = self.config.patch_size
        square_crop = self.config.get("square_crop", False)

        images = []
        fxfycxcys = []
        intrinsics = []
        c2ws = []
        point_maps = []
        depth_maps = []

        target_first_view = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,-1],[0,0,0,1.]])

        for v_idx, (cur_image_path, cur_pose_path) in enumerate(zip(image_paths_chosen, pose_paths_chosen)):
            image = Image.open(cur_image_path)
            white_bg = Image.new(mode='RGBA', size=image.size, color=(255,)*4)
            image = Image.alpha_composite(white_bg, image)
            image = image.convert("RGB")

            original_image_w, original_image_h = image.size
            
            resize_w = int(resize_h / original_image_h * original_image_w)
            resize_w = int(round(resize_w / patch_size) * patch_size)
            image = image.resize((resize_w, resize_h), resample=Image.LANCZOS)

            image = np.array(image) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).float()

            focal_length = (resize_w/2) / math.tan(self.fov/2)
            fxfycxcy = torch.tensor([focal_length, focal_length, resize_w/2, resize_h/2])
            intrinsic_mat = torch.tensor([[focal_length, 0, resize_w/2], [0., focal_length, resize_h/2], [0., 0., 1]])

            images.append(image)
            fxfycxcys.append(fxfycxcy)
            intrinsics.append(intrinsic_mat)

            pose = np.load(cur_pose_path, allow_pickle=True)
            c2w = torch.from_numpy(self.convert_pose(pose)).float()

            if v_idx == 0:
                norm_first_view_t = torch.norm(c2w[:3,3])
                c2w[:3,3] /= norm_first_view_t
                inv_first_view = torch.inverse(c2w)
            else:
                c2w[:3,3] /= norm_first_view_t
            c2w = target_first_view @ inv_first_view @ c2w

            c2ws.append(c2w)

        images = torch.stack(images, dim=0)
        fxfycxcys = torch.stack(fxfycxcys, dim=0)
        intrinsics = torch.stack(intrinsics, dim=0)
        c2ws = torch.stack(c2ws)

        return images, fxfycxcys, intrinsics, c2ws

    def __getitem__(self, idx):
        obj_name = self.all_obj_list[idx]
        obj_path = os.path.join(self.root_dir, obj_name, self.suffix)

        if self.inference:
            current_view_idx = self.view_idx_list
            image_indices= current_view_idx["context"] + current_view_idx["target"]
        else:
            assert False, 'GSO dataset is only for validation/evaluation'
        image_paths_chosen = [os.path.join(obj_path, f'{ic:03d}.png') for ic in image_indices]
        pose_paths_chosen = [os.path.join(obj_path, f'{ic:03d}.npy') for ic in image_indices]
        fxfycxcy = np.array([560, 560, 256, 256])
        input_images, input_fxfycxcy, input_intrinsics, input_c2ws = self.preprocess_frames(pose_paths_chosen, image_paths_chosen, fxfycxcy)
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
            "scene_name": obj_name
        }

