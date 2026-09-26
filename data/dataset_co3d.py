import random
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
import gzip


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def load_16big_png_depth(depth_png):
    with Image.open(depth_png) as depth_pil:
        # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
        # we cast it to uint16, then reinterpret as float16, then cast to float32
        depth = (
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth


def pinhole_z_depth_to_xyz(depth, K):
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    uv1 = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3).T  # (3, N)

    Kinv = np.linalg.inv(K)
    xyz = Kinv @ uv1  # (3, N)

    z = depth.reshape(-1)
    xyz = xyz * z

    return xyz.T.reshape(H, W, 3)


camera_transform_matrix = np.eye(4)
camera_transform_matrix[0, 0] *= -1
camera_transform_matrix[1, 1] *= -1

target_first_view = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,-1],[0,0,0,1.]])

class CO3DDataset(Dataset):
    def __init__(self, config, is_second=False):
        self.config = config
        self.root_path = '/root/mochu_ws/ssd1/'
        self.total_frames_per_obj = 14

        with open('/root/mochu_ws/ssd1/teddybear/teddybear_test_splits.json', 'r') as f:
            self.test_set = json.load(f)
        with gzip.open('/root/mochu_ws/ssd1/teddybear/teddybear_test.jgz', "r") as fin:
            self.annotation = json.loads(fin.read())
        self.all_object_list = sorted(list(self.test_set.keys()))
        print(len(self.all_object_list))
    
        self.fov = 0.6981317007977318
    
    def __len__(self):
        return len(self.all_object_list)
    
    def __getitem__(self, index):
        seq_name = self.all_object_list[index]
        seq_content = [self.annotation[seq_name][i] for i in self.test_set[seq_name]]

        co3d_path = self.root_path

        images = []
        c2ws = []
        extrinsics = []
        fxfycxcys = []
        intrinsics = []
        indices = []
        depth_maps = []
        point_maps = []
        for idx, view in enumerate(seq_content):
            image = imageio.imread(os.path.join(co3d_path, view['filepath']))
            depth = load_16big_png_depth(os.path.join(co3d_path, view['filepath'].replace('images', 'depths').replace('.jpg', '.jpg.geometric.png'))) 
            mask = imageio.imread(os.path.join(co3d_path, view['filepath']).replace('images', 'depth_masks').replace('.jpg', '.png')) > 0
            saliency = imageio.imread(os.path.join(co3d_path, view['filepath']).replace('images', 'masks').replace('.jpg', '.png')) > 128

            h, w = image.shape[:2]
            focal = view['focal_length'][0]

            fov = focal2fov(focal, 2)

            ### crop to align FoV
            current_focal = focal * w / 2
            target_focal = 351.6771096901917
            target_half_size = 128

            current_half_size = current_focal / target_focal * target_half_size
            
            v_trim = int(h/2 - current_half_size)

            if v_trim < 0:
                v_pad = int(current_half_size - h/2)

                image = np.pad(image, ((v_pad,v_pad), (0,0), (0,0)), 'constant', constant_values=0)
                mask = np.pad(mask, ((v_pad,v_pad), (0,0)), 'constant', constant_values=0)
                depth = np.pad(depth, ((v_pad,v_pad), (0,0)), 'constant', constant_values=0)
                saliency = np.pad(saliency, ((v_pad,v_pad), (0,0)), 'constant', constant_values=0)

            else:
                image = image[v_trim:-v_trim]
                mask = mask[v_trim:-v_trim]
                depth = depth[v_trim:-v_trim]
                saliency = saliency[v_trim:-v_trim]

            h_pad = int(current_half_size - w/2)
            if h_pad < 0:
                h_trim = -h_pad
                image = image[:, h_trim:-h_trim]
                mask = mask[:, h_trim:-h_trim]
                depth = depth[:, h_trim:-h_trim]
                saliency = saliency[:, h_trim:-h_trim]
            else:
                image = np.pad(image, ((0,0), (h_pad,h_pad), (0,0)), 'constant', constant_values=0)
                mask = np.pad(mask, ((0,0), (h_pad,h_pad)), 'constant', constant_values=0)
                depth = np.pad(depth, ((0,0), (h_pad,h_pad)), 'constant', constant_values=0)
                saliency = np.pad(saliency, ((0,0), (h_pad,h_pad)), 'constant', constant_values=0)

            h, w, c = image.shape

            image[~saliency] = np.array([255]*3)
            depth[~saliency] = 0

            R = np.array(view['R'])
            t = np.array(view['T'])
            w2c_template = np.eye(4)
            w2c_template[:3, :3] = R
            w2c_template[3:, :3] = t
            w2c = np.transpose(np.matmul(w2c_template, camera_transform_matrix))

            mat = np.linalg.inv(w2c) # mat is c2w
            if idx == 0:
                norm_first_view_t = np.linalg.norm(mat[:3,3])
                mat[:3,3] /= norm_first_view_t
                inv_first_view = np.linalg.inv(mat)
            else:
                mat[:3,3] /= norm_first_view_t
            mat = target_first_view @ inv_first_view @ mat

            rot_mat = mat[:3,:3]
            wxyz = vtf.SO3.from_matrix(rot_mat).wxyz
            position = mat[:3,3]
            
            intrinsic = np.array([
                [current_focal, 0, w/2],
                [0, current_focal, h/2],
                [0, 0, 1]
            ])
            depth = depth / norm_first_view_t
            xyz = pinhole_z_depth_to_xyz(depth, intrinsic)

            h,w,c = xyz.shape
            xyz = rearrange(xyz, 'h w c -> (h w) c')
            ones = np.ones((h*w, 1), dtype=xyz.dtype)
            xyz = np.concatenate([xyz, ones], -1)
            xyz = (mat @ xyz.T)[:3].T
            xyz = rearrange(xyz, '(h w) c -> h w c', h=h, w=w)
            xyz = xyz[..., :3]

            image = torch.from_numpy(image).float() / 255.
            image = rearrange(image, 'h w c -> 1 c h w')
            image = F.interpolate(image, (256, 256), mode='bilinear')[0]

            depth = torch.from_numpy(depth).float()
            depth = rearrange(depth, 'h w -> 1 1 h w')
            depth = F.interpolate(depth, (256, 256), mode='nearest')[0,0]

            xyz = torch.from_numpy(xyz).float()
            xyz = rearrange(xyz, 'h w c -> 1 c h w')
            xyz = F.interpolate(xyz, (256, 256), mode='nearest')[0]

            images.append(image)
            point_maps.append(xyz)
            depth_maps.append(depth)
            indices.append([idx])
            extrinsics.append(torch.from_numpy(w2c).float())
            c2ws.append(torch.from_numpy(mat).float())
            intrinsics.append(torch.from_numpy(intrinsic))
            fxfycxcys.append([351.6771096901917, 351.6771096901917, 128, 128])
        
        images = torch.stack(images)
        c2ws = torch.stack(c2ws)
        extrinsics = torch.stack(extrinsics)
        fxfycxcys = torch.tensor(fxfycxcys)
        intrinsics = torch.stack(intrinsics)
        image_indices = torch.tensor(indices)
        scene_indices = torch.full_like(image_indices, index)
        indices = torch.cat([image_indices, scene_indices], dim=-1)

        depth_maps = torch.stack(depth_maps)
        point_maps = torch.stack(point_maps)

        return {
            "image": images,
            "c2w": c2ws,
            "extrinsic": extrinsics,
            "fxfycxcy": fxfycxcys,
            "intrinsic": intrinsics,
            "index": indices,
            "scene_name": seq_name,
            "depth_map": depth_maps,
            "point_map": point_maps
        }