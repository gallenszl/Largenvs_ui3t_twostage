# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from tabnanny import check
from IPython import embed
from einops import rearrange, repeat
import torch
import traceback
import os
import math
import torch.nn as nn
from easydict import EasyDict as edict
from huggingface_hub import PyTorchModelHubMixin
from model.vggt.layers import patch_embed  # used for model hub

from model.vggt.models.aggregator import Aggregator
from model.vggt.heads.camera_head import CameraHead
from model.vggt.heads.dpt_head import DPTHead
from einops.layers.torch import Rearrange
from utils import camera_utils, data_utils 
from model.transformer import init_weights
from model.loss import LossComputer


class VGGT4LVSM(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config, embed_dim=1024, **kwargs):
        super().__init__()
        self.config = config
        img_size = self.config.model.image_tokenizer.image_size
        patch_size = self.config.model.image_tokenizer.patch_size
        self.img_size = img_size
        self.patch_size = patch_size

        self.embed_dim = embed_dim
        self.kwargs = kwargs

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, patch_size=patch_size, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, patch_size=patch_size, output_dim=2, activation="exp", conf_activation="expp1")

        self.process_data = data_utils.ProcessData(config)
        self.make_modifications()
        self.loss_computer = LossComputer(config)

    def make_modifications(self):
        if self.kwargs.get('is_debugging', False) == False:
            if hasattr(self.config.model, 'pretrained_path'):
                print("Loading pretrained weights from ", self.config.model.pretrained_path)
                checkpoint = torch.load(self.config.model.pretrained_path, map_location="cpu")

                patch_embed_weight = checkpoint['aggregator.patch_embed.patch_embed.proj.weight']
                if patch_embed_weight.shape[-1] != self.patch_size:
                    # resize patch embedding weight
                    new_patch_embed_weight = torch.nn.functional.interpolate(
                        patch_embed_weight, size=(self.patch_size, self.patch_size), mode='bilinear')
                    checkpoint['aggregator.patch_embed.patch_embed.proj.weight'] = new_patch_embed_weight
                    print(f'Resizing patch embedding weight from {patch_embed_weight.shape} to {new_patch_embed_weight.shape}.')

                    # resize pose embedding weight
                    pose_embed = checkpoint['aggregator.patch_embed.pos_embed']
                    cls_embed = pose_embed[:, :1, :]
                    img_embed = pose_embed[:, 1:, :]
                    hw = img_embed.shape[1]
                    h = int(math.sqrt(hw))
                    img_embed = rearrange(img_embed, 'b (h w) c -> b c h w', h=h)
                    new_h = self.img_size // self.patch_size
                    new_img_embed = torch.nn.functional.interpolate(
                        img_embed, size=(new_h, new_h), mode='bilinear')
                    new_img_embed = rearrange(new_img_embed, 'b c h w -> b (h w) c')
                    new_pose_embed = torch.cat([cls_embed, new_img_embed], dim=1)
                    checkpoint['aggregator.patch_embed.pos_embed'] = new_pose_embed
                    print(f'Resizing pose embedding weight from {pose_embed.shape} to {new_pose_embed.shape}.')

                checkpoint = {k:v for k,v in checkpoint.items() if not k.startswith('track_head')}
                self.load_state_dict(checkpoint, strict=True)
                print("Loaded pretrained weights successfully.")
        else:
            print("Debug mode enabled. Not loading any pretrained weights.")
        self.modify_heads()
        self.add_pose_tokenizer()
        self.freeze_dino()

    def modify_heads(self):
        del self.camera_head
        del self.depth_head
        self.rgb_head = deepcopy(self.point_head)
        self.rgb_head.activation = "sigmoid"
        self.rgb_head.scratch.output_conv1.apply(init_weights)
        self.rgb_head.scratch.output_conv2.apply(init_weights)
        del self.point_head

    def add_pose_tokenizer(self):
        self.pose_tokenizer = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> b v (hh ww) (ph pw c)",
                ph=self.patch_size,
                pw=self.patch_size),
            nn.Linear(
                6 * (self.patch_size**2),
                self.embed_dim,
                bias=False))
        self.pose_tokenizer.apply(init_weights)

    def freeze_dino(self):
        print("Freezing DINO weights...")
        for param in self.aggregator.patch_embed.parameters():
            param.requires_grad = False

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

    def forward(self, data_batch, has_target_image=True, target_has_input=None, exclude_bg=False):
        if target_has_input is None:
            target_has_input = self.config.training.target_has_input

        input, target = self.process_data(
            data_batch, 
            has_target_image=has_target_image, 
            target_has_input=target_has_input, 
            compute_rays=True)

        # Process input images
        input_pose_cond = self.get_posed_input(ray_o=input.ray_o, ray_d=input.ray_d)
        b, v_input, c, h, w = input_pose_cond.size()

        # Process target pose
        target_pose_cond = self.get_posed_input(ray_o=target.ray_o, ray_d=target.ray_d)
        b, v_target, c, h, w = target_pose_cond.size()

        # b, (v_input + v_target), c, h, w
        pose_cond = torch.concat([input_pose_cond, target_pose_cond], dim=1)
        # (b v) n c
        pose_tokens = self.pose_tokenizer(pose_cond)
        
        aggregated_tokens_list, patch_start_idx = self.aggregator(input.image, pose_tokens)
        # discard input view tokens and camera/register tokens.
        # because each feat has dimension [b, v, ...],
        # the v dimension is [i, ..., o]: only the last one is the output
        aggregated_tokens_list = [feat[:,-1:] for feat in aggregated_tokens_list]

        with torch.cuda.amp.autocast(enabled=False):
            rendered_images, conf = self.rgb_head(
                aggregated_tokens_list, images=target_pose_cond, patch_start_idx=patch_start_idx)
            rendered_images = rearrange(rendered_images, 'b v h w c -> b v c h w')
        
        if has_target_image:
            loss_metrics = self.loss_computer(
                rendered_images,
                target.image,
                exclude_bg=exclude_bg
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

    @torch.no_grad()
    def load_ckpt(self, load_path):
        if os.path.isdir(load_path):
            ckpt_names = [file_name for file_name in os.listdir(load_path) if file_name.endswith(".pt")]
            ckpt_names = sorted(ckpt_names, key=lambda x: x)
            ckpt_paths = [os.path.join(load_path, ckpt_name) for ckpt_name in ckpt_names]
        else:
            ckpt_paths = [load_path]
        try:
            checkpoint = torch.load(ckpt_paths[-1], map_location="cpu", weights_only=True)
        except:
            traceback.print_exc()
            print(f"Failed to load {ckpt_paths[-1]}")
            return None
        
        self.load_state_dict(checkpoint["model"], strict=False)
        return 0

    @torch.no_grad()
    def render_video(self, data_batch, traj_type="interpolate", num_frames=60, loop_video=False, order_poses=False):
        """
        Render a video from the model.
        
        Args:
            result: Edict from forward pass or just data
            traj_type: Type of trajectory
            num_frames: Number of frames to render
            loop_video: Whether to loop the video
            order_poses: Whether to order poses
            
        Returns:
            result: Updated with video rendering
        """
    
        if data_batch.input is None:
            input, target = self.process_data(data_batch, has_target_image=False, target_has_input=self.config.training.target_has_input, compute_rays=True)
            data_batch = edict(input=input, target=target)
        else:
            input, target = data_batch.input, data_batch.target
        
        # Prepare input tokens; [b, v, 6, h, w]
        input_pose_cond = self.get_posed_input(
            images=input.image, ray_o=input.ray_o, ray_d=input.ray_d
        )
        bs, v_input, c, h, w = input_pose_cond.size()

        # input_img_tokens = self.image_tokenizer(input.image)  # [b*v_input, n_patches, d]

        # _, n_patches, d = input_img_tokens.size()  # [b*v_input, n_patches, d]
        # input_img_tokens = input_img_tokens.reshape(bs, v_input * n_patches, d)  # [b, v_input*n_patches, d]

        # target_pose_cond_list = []
        if traj_type == "interpolate":
            c2ws = input.c2w # [b, v, 4, 4]
            fxfycxcy = input.fxfycxcy #  [b, v, 4]
            device = input.c2w.device

            # Create intrinsics from fxfycxcy
            intrinsics = torch.zeros((c2ws.shape[0], c2ws.shape[1], 3, 3), device=device) # [b, v, 3, 3]
            intrinsics[:, :,  0, 0] = fxfycxcy[:, :, 0]
            intrinsics[:, :,  1, 1] = fxfycxcy[:, :, 1]
            intrinsics[:, :,  0, 2] = fxfycxcy[:, :, 2]
            intrinsics[:, :,  1, 2] = fxfycxcy[:, :, 3]

            # Loop video if requested
            if loop_video:
                c2ws = torch.cat([c2ws, c2ws[:, [0], :]], dim=1)
                intrinsics = torch.cat([intrinsics, intrinsics[:, [0], :]], dim=1)

            # Interpolate camera poses
            all_c2ws, all_intrinsics = [], []
            for b in range(input.image.size(0)):
                cur_c2ws, cur_intrinsics = camera_utils.get_interpolated_poses_many(
                    c2ws[b, :, :3, :4], intrinsics[b], num_frames, order_poses=order_poses
                )
                all_c2ws.append(cur_c2ws.to(device))
                all_intrinsics.append(cur_intrinsics.to(device))

            all_c2ws = torch.stack(all_c2ws, dim=0) # [b, num_frames, 3, 4]
            all_intrinsics = torch.stack(all_intrinsics, dim=0) # [b, num_frames, 3, 3]

            # Add homogeneous row to c2ws
            homogeneous_row = torch.tensor([[[0, 0, 0, 1]]], device=device).expand(all_c2ws.shape[0], all_c2ws.shape[1], -1, -1)
            all_c2ws = torch.cat([all_c2ws, homogeneous_row], dim=2)

            # Convert intrinsics to fxfycxcy format
            all_fxfycxcy = torch.zeros((all_intrinsics.shape[0], all_intrinsics.shape[1], 4), device=device)
            all_fxfycxcy[:, :, 0] = all_intrinsics[:, :, 0, 0]  # fx
            all_fxfycxcy[:, :, 1] = all_intrinsics[:, :, 1, 1]  # fy
            all_fxfycxcy[:, :, 2] = all_intrinsics[:, :, 0, 2]  # cx
            all_fxfycxcy[:, :, 3] = all_intrinsics[:, :, 1, 2]  # cy

        # Compute rays for rendering
        rendering_ray_o, rendering_ray_d = self.process_data.compute_rays(
            fxfycxcy=all_fxfycxcy, c2w=all_c2ws, h=h, w=w, device=device
        )

        # Get pose conditioning for target views
        target_pose_cond = self.get_posed_input(
            ray_o=rendering_ray_o.to(input.image.device), 
            ray_d=rendering_ray_d.to(input.image.device)
        )
                
        _, num_views, c, h, w = target_pose_cond.size()
    
        target_pose_tokens = self.target_pose_tokenizer(target_pose_cond) # [bs*v_target, n_patches, d]
        _, n_patches, d = target_pose_tokens.size()  # [b*v_target, n_patches, d]
        target_pose_tokens = target_pose_tokens.reshape(bs, num_views * n_patches, d)  # [b, v_target*n_patches, d]

        view_chunk_size = 4

        video_rendering_list = []
        for cur_chunk in range(0, num_views, view_chunk_size):
            cur_view_chunk_size = min(view_chunk_size, num_views - cur_chunk)

            # [b, (v_input*n_patches), d] -> [(b * cur_v_target), (v_input*n_patches), d]
            repeated_input_img_tokens = repeat(input_img_tokens.detach(), 'b np d -> (b chunk) np d', chunk=cur_view_chunk_size, np=n_patches* v_input)

            start_idx, end_idx = cur_chunk * n_patches, (cur_chunk + cur_view_chunk_size) * n_patches            
            # [b, v_target * n_patches, d] -> [b, cur_v_target*n_patches, d] -> [b*cur_v_target, n_patches, d]
            cur_target_pose_tokens = rearrange(target_pose_tokens[:, start_idx:end_idx,: ], 
                                               "b (v_chunk p) d -> (b v_chunk) p d", 
                                               v_chunk=cur_view_chunk_size, p=n_patches)

            cur_concat_input_tokens = torch.cat((repeated_input_img_tokens, cur_target_pose_tokens,), dim=1) # [b*cur_v_target, v_input*n_patches+n_patches, d]
            cur_concat_input_tokens = self.transformer_input_layernorm(
                cur_concat_input_tokens
            )

            transformer_output_tokens = self.pass_layers(cur_concat_input_tokens, gradient_checkpoint=False)

            _, pred_target_image_tokens = transformer_output_tokens.split(
                [v_input * n_patches, n_patches], dim=1
            ) # [b * v_target, v*n_patches, d], [b * v_target, n_patches, d]

            height, width = target.image_h_w

            patch_size = self.config.model.target_pose_tokenizer.patch_size

            # [b, v_target*n_patches, p*p*3]
            video_rendering = self.image_token_decoder(pred_target_image_tokens)
            
            video_rendering = rearrange(
                video_rendering, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
                v=cur_view_chunk_size,
                h=height // patch_size, 
                w=width // patch_size, 
                p1=patch_size, 
                p2=patch_size, 
                c=3
            ).cpu()

            video_rendering_list.append(video_rendering)
        video_rendering = torch.cat(video_rendering_list, dim=1)
        data_batch.video_rendering = video_rendering


        return data_batch