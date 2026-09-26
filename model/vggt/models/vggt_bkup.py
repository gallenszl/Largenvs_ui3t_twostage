# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from IPython import embed
from einops import rearrange, repeat
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from model.vggt.models.aggregator import Aggregator
from model.vggt.heads.camera_head import CameraHead
from model.vggt.heads.dpt_head import DPTHead
from einops.layers.torch import Rearrange
from utils import camera_utils, data_utils 
from model.transformer import init_weights
from model.loss import LossComputer


class VGGT4LVSM(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config, img_size=518, patch_size=14, embed_dim=1024):
        super().__init__()
        self.config = config
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")

        self.process_data = data_utils.ProcessData(config)
        self.make_modifications()
        self.loss_computer = LossComputer(config)

    def make_modifications(self):
        if hasattr(self.config.model, 'pretrained_path'):
            print("Loading pretrained weights from ", self.config.model.pretrained_path)
            checkpoint = torch.load(self.config.model.pretrained_path, map_location="cpu")
            checkpoint = {k:v for k,v in checkpoint.items() if not k.startswith('track_head')}
            self.load_state_dict(checkpoint, strict=True)
            print("Loaded pretrained weights successfully.")
        self.modify_heads()
        self.add_pose_tokenizer()

    def modify_heads(self):
        del self.camera_head
        del self.depth_head
        self.rgb_head = deepcopy(self.point_head)
        self.rgb_head.activation = "linear"
        del self.point_head

    def add_pose_tokenizer(self):
        self.pose_tokenizer = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
                ph=self.patch_size,
                pw=self.patch_size),
            nn.Linear(
                6 * (self.patch_size**2),
                self.embed_dim,
                bias=False))
        self.pose_tokenizer.apply(init_weights)

    def get_posed_input(self, images=None, ray_o=None, ray_d=None, method="default_plucker"):
        if method == "custom_plucker":
            o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
            nearest_pts = ray_o + o_dot_d * ray_d
            pose_cond = torch.cat([ray_d, nearest_pts], dim=2)
            
        elif method == "aug_plucker":
            o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
            nearest_pts = ray_o + o_dot_d * ray_d
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            pose_cond = torch.cat([o_cross_d, ray_d, nearest_pts], dim=2)
            
        else:  # default_plucker
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            pose_cond = torch.cat([o_cross_d, ray_d], dim=2)

        if images is None:
            return pose_cond
        else:
            return images, pose_cond

    def _forward(self, images: torch.Tensor):
        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}
        with torch.cuda.amp.autocast(enabled=False):
            rgb, conf = self.rgb_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx)
            predictions["rgb"] = rgb
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions

    def forward(self, data_batch, has_target_image=True):
        input, target = self.process_data(
            data_batch, 
            has_target_image=has_target_image, 
            target_has_input=self.config.training.target_has_input, 
            compute_rays=True)

        # Process input images
        input_images, input_pose_cond = self.get_posed_input(
            images=input.image, ray_o=input.ray_o, ray_d=input.ray_d)
        b, v_input, c, h, w = input_images.size()

        # Process target pose
        target_pose_cond = self.get_posed_input(ray_o=target.ray_o, ray_d=target.ray_d)
        b, v_target, c, h, w = target_pose_cond.size()

        # b, (v_input + v_target), c, h, w
        pose_cond = torch.concat([input_pose_cond, target_pose_cond], dim=1)
        # (b v) n c
        pose_tokens = self.pose_tokenizer(pose_cond)
        
        aggregated_tokens_list, patch_start_idx = self.aggregator(input_images, pose_tokens)
        # discard input views and camera/register tokens
        aggregated_tokens_list = [feat[:,-v_target:] for feat in aggregated_tokens_list]

        with torch.cuda.amp.autocast(enabled=False):
            rendered_images, conf = self.rgb_head(
                aggregated_tokens_list, images=target_pose_cond, patch_start_idx=patch_start_idx)
            rendered_images = rearrange(rendered_images, 'b v h w c -> b v c h w')
        
        if has_target_image:
            loss_metrics = self.loss_computer(
                rendered_images,
                target.image,
            )
        else:
            loss_metrics = None

        result = edict(
            input=input,
            target=target,
            loss_metrics=loss_metrics,
            render=rendered_images        
            )
        
        return result
