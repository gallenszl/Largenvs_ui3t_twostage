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

from model.vggt.models.aggregator import Aggregator, Aggregator_with_kv_cache
from model.vggt.heads.camera_head import CameraHead
from model.vggt.heads.dpt_head import DPTHead, activate_head
from model.vggt.stage1.decoders import GeneralDecoder
from model.vggt.stage1.encoders.dinov3 import DINOv3TargetEncoder
from transformers import AutoConfig
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from utils import camera_utils, data_utils
from utils.training_utils import print_rank0
from model.transformer import init_weights
# from model.loss import LossComputer
from model.loss import MultiTaskLossComputer


class RAE_Head(nn.Module):
    def __init__(self,
        dim_in: int,
        patch_size: int,
        output_dim: int,
        activation: str,
        conf_activation: str,
        rae_latent_dim: int = 768,
        rae_decoder_config_path: str = None,
        rae_decoder_patch_size: int = 16,
        rae_num_patches: int = 1024,
        rae_normalization_stat_path: str = None,
        eps: float = 1e-5):
        super().__init__()

        self.patch_size = patch_size
        self.rae_latent_dim = rae_latent_dim
        self.rae_num_patches = rae_num_patches
        self.eps = eps

        # Trainable projection: aggregator token dim -> RAE latent dim
        self.linear = nn.Linear(dim_in, rae_latent_dim)

        # Build GeneralDecoder from config
        decoder_config = AutoConfig.from_pretrained(rae_decoder_config_path)
        decoder_config.hidden_size = rae_latent_dim  # 768
        decoder_config.patch_size = rae_decoder_patch_size  # 16
        decoder_config.image_size = int(rae_decoder_patch_size * math.sqrt(rae_num_patches))
        self.rae_decoder = GeneralDecoder(decoder_config, num_patches=rae_num_patches)

        # Freeze decoder and enable gradient checkpointing
        for param in self.rae_decoder.parameters():
            param.requires_grad = False
        self.rae_decoder.gradient_checkpointing = True

        # ImageNet mean/std for denormalization (used by RAE's DINOv2 encoder)
        self.register_buffer('encoder_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('encoder_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Latent normalization stats (z-score)
        if rae_normalization_stat_path is not None:
            stats = torch.load(rae_normalization_stat_path, map_location='cpu')
            latent_mean = stats.get('mean', None)
            latent_var = stats.get('var', None)
            if latent_mean is not None:
                # Stats shape is [C, H, W] (e.g. [768, 32, 32])
                # Reshape to [1, N, C] for broadcasting with [BV, N, C] tokens
                # where N = H*W, C = latent_dim
                if latent_mean.dim() == 3:  # [C, H, W] -> [1, H*W, C]
                    C, H_lat, W_lat = latent_mean.shape
                    latent_mean = latent_mean.reshape(C, H_lat * W_lat).permute(1, 0).unsqueeze(0)
                elif latent_mean.dim() == 4:  # [1, C, H, W] -> [1, H*W, C]
                    latent_mean = latent_mean.squeeze(0)
                    C, H_lat, W_lat = latent_mean.shape
                    latent_mean = latent_mean.reshape(C, H_lat * W_lat).permute(1, 0).unsqueeze(0)
                elif latent_mean.dim() == 1:  # [C] -> [1, 1, C]
                    latent_mean = latent_mean.unsqueeze(0).unsqueeze(0)
                self.register_buffer('latent_mean', latent_mean)
            else:
                self.register_buffer('latent_mean', torch.zeros(1))
            if latent_var is not None:
                if latent_var.dim() == 3:  # [C, H, W] -> [1, H*W, C]
                    C, H_lat, W_lat = latent_var.shape
                    latent_var = latent_var.reshape(C, H_lat * W_lat).permute(1, 0).unsqueeze(0)
                elif latent_var.dim() == 4:  # [1, C, H, W] -> [1, H*W, C]
                    latent_var = latent_var.squeeze(0)
                    C, H_lat, W_lat = latent_var.shape
                    latent_var = latent_var.reshape(C, H_lat * W_lat).permute(1, 0).unsqueeze(0)
                elif latent_var.dim() == 1:  # [C] -> [1, 1, C]
                    latent_var = latent_var.unsqueeze(0).unsqueeze(0)
                self.register_buffer('latent_var', latent_var)
            else:
                self.register_buffer('latent_var', torch.ones(1))
            self.do_normalization = True
            print(f"RAE_Head: loaded normalization stats from {rae_normalization_stat_path}")
        else:
            self.do_normalization = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.rae_decoder.eval()  # always keep decoder in eval mode
        return self

    def unfreeze_decoder(self):
        """解冻 RAE decoder 参数"""
        print("Unfreezing RAE decoder parameters...")
        for param in self.rae_decoder.parameters():
            param.requires_grad = True
        print(f"  Unfroze {sum(p.numel() for p in self.rae_decoder.parameters())} parameters")

    def forward(self,
        aggregated_tokens_list,
        images: torch.Tensor,
        patch_start_idx: int,
        frames_chunk_size: int = 10):

        B, S, _, H, W = images.shape
        # aggregated_tokens_list contains target view tokens only (last column, V=1)
        # x = aggregated_tokens_list[-1]  # [B, 1, N, 2*embed_dim]
        x = aggregated_tokens_list[-1][:,:,patch_start_idx:]
        _, V, N, C = x.shape
        x = rearrange(x, 'b v n c -> (b v) n c')  # [B, N, 2048]

        # Project to RAE latent space
        z = self.linear(x)  # [B, 1024, 768]

        # Inverse z-score normalization
        if self.do_normalization:
            z = z * torch.sqrt(self.latent_var + self.eps) + self.latent_mean

        # Decode through frozen GeneralDecoder (grads flow to self.linear via computation graph)
        output = self.rae_decoder(z, drop_cls_token=False).logits  # [B, 1024, 768]
        x_rec = self.rae_decoder.unpatchify(output)  # [B, 3, 512, 512]

        # Denormalize with ImageNet mean/std
        x_rec = x_rec * self.encoder_std + self.encoder_mean
        x_rec = x_rec.clamp(0, 1)
        
        # Reshape to caller expected format: [B, V, H, W, C]
        preds = rearrange(x_rec, '(b v) c h w -> b v h w c', b=B, v=S)

        # Confidence placeholder (not used in rgb loss computation)
        conf = torch.ones_like(preds)

        return preds, conf



class LinearHead(nn.Module):
    def __init__(self, 
        dim_in, 
        patch_size, 
        output_dim, 
        activation, 
        conf_activation):
        super().__init__()

        self.patch_size = patch_size
        self.linear = nn.Linear(dim_in, patch_size**2*output_dim)
        self.activation = activation
        self.conf_activation = conf_activation

    def forward(self,
        aggregated_tokens_list,
        images: torch.Tensor,
        patch_start_idx: int,
        frames_chunk_size: int = 10):

        B, S, _, H, W = images.shape
        x = aggregated_tokens_list[-1][:,:,patch_start_idx:]
        _,V,N,C = x.shape
        NH = int(math.sqrt(N))
        x = rearrange(x, 'b v n c -> (b v) n c')
        
        x = self.linear(x)
        x = rearrange(x, 'bv (nh nw) (ph pw c) -> bv c (nh ph) (nw pw)', 
            ph=self.patch_size, pw=self.patch_size, nh=NH, nw=NH)

        preds, conf = activate_head(x, activation=self.activation, conf_activation=self.conf_activation)        

        preds = preds.view(B, S, *preds.shape[1:])
        conf = conf.view(B, S, *conf.shape[1:])
        return preds, conf

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

        if kwargs.get('use_kv_cache', False):
            self.aggregator = Aggregator_with_kv_cache(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        else:
            self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, patch_size=patch_size, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, patch_size=patch_size, output_dim=2, activation="exp", conf_activation="expp1")

        self.process_data = data_utils.ProcessData(config)
        self.process_val_data = data_utils.ProcessData(config.training.val_dataset_cfgs)
        self.make_modifications()
        # self.loss_computer = LossComputer(config)
        self.loss_computer = MultiTaskLossComputer(config)

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
        self.add_feature_alignment()

    def add_feature_alignment(self):
        """REPA-style feature alignment: align aggregator intermediate features
        with frozen DINOv3-ViT-L patch tokens of the target image.

        Phase 1 design (single-layer REPA, flips all 4 broken knobs from the prior
        failed `rae_latent` experiment):
          - hook at aggregator layer 7 (8th of 24, ~33% depth, REPA sweet spot)
          - target encoder = DINOv3-ViT-L (different family than DINOv2 backbone)
          - projector = Conv2d 3x3 (iREPA-style, preserves spatial structure)
          - loss = negative cosine similarity (computed in MultiTaskLossComputer)
        """
        align_cfg = self.config.training.get("feature_alignment", None)
        self.feature_alignment_enabled = bool(align_cfg and align_cfg.get("enabled", False))
        if not self.feature_alignment_enabled:
            return

        self.align_layer_idx = int(align_cfg.get("layer_idx", 7))
        self.align_weight = float(align_cfg.get("weight", 0.5))
        self.align_only_target = bool(align_cfg.get("only_target_view", True))

        self.align_target_encoder = DINOv3TargetEncoder(
            weights_path=align_cfg.target_encoder_weights,
            hub_dir=align_cfg.target_encoder_hub_dir,
            # 512 -> DINOv3 grid 512/16=32 exactly matches aggregator 448/14=32 grid
            resolution=int(align_cfg.get("target_resolution", 512)),
        )
        # Aggregator concat is [frame | global] each of dim embed_dim, so the
        # global-attention output occupies the last embed_dim channels.
        self.align_projector = nn.Conv2d(
            self.embed_dim, self.embed_dim, kernel_size=3, padding=1
        )
        nn.init.kaiming_normal_(self.align_projector.weight)
        nn.init.zeros_(self.align_projector.bias)

        print_rank0(
            f"[feature_alignment] enabled: layer={self.align_layer_idx}, "
            f"weight={self.align_weight}, encoder=DINOv3-ViT-L, "
            f"projector=Conv2d({self.embed_dim},{self.embed_dim},k=3)"
        )

    def modify_heads(self):
        del self.depth_head
        # del self.point_head

        embed_dim = self.embed_dim
        patch_size = self.patch_size
        rae_cfg = self.config.model.rae_decoder

        # self.point_head = LinearHead(dim_in=2 * embed_dim, patch_size=patch_size, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.rgb_head = RAE_Head(
            dim_in=2 * embed_dim,
            patch_size=patch_size,
            output_dim=4,
            activation="sigmoid",
            conf_activation="expp1",
            rae_latent_dim=rae_cfg.latent_dim,
            rae_decoder_config_path=rae_cfg.config_path,
            rae_decoder_patch_size=rae_cfg.decoder_patch_size,
            rae_num_patches=(self.img_size // self.patch_size) ** 2,
            rae_normalization_stat_path=getattr(rae_cfg, 'normalization_stat_path', None),
        )

        # Load RAE decoder pretrained weights
        if hasattr(rae_cfg, 'pretrained_decoder_path') and rae_cfg.pretrained_decoder_path:
            print(f"Loading RAE decoder weights from {rae_cfg.pretrained_decoder_path}")
            decoder_sd = torch.load(rae_cfg.pretrained_decoder_path, map_location="cpu")
            keys = self.rgb_head.rae_decoder.load_state_dict(decoder_sd, strict=False)
            if keys.missing_keys:
                print(f"RAE decoder missing keys: {keys.missing_keys}")
            if keys.unexpected_keys:
                print(f"RAE decoder unexpected keys: {keys.unexpected_keys}")
            print("RAE decoder weights loaded successfully.")

        # 如果 unfreeze_rae_decoder_at = 0，从一开始就不冻结
        unfreeze_at = self.config.training.get("unfreeze_rae_decoder_at", 0.2)
        if unfreeze_at == 0:
            print_rank0("unfreeze_rae_decoder_at = 0, RAE decoder will be trainable from the beginning")
            for param in self.rgb_head.rae_decoder.parameters():
                param.requires_grad = True


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

    def forward(self, data_batch, has_target_image=True, target_has_input=None, 
                exclude_bg=False, is_valid=False):
        if target_has_input is None:
            target_has_input = self.config.training.target_has_input

        if is_valid:
            process_data = self.process_val_data
        else:
            process_data = self.process_data

        input, target = process_data(
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
        
        aggregated_tokens_list, patch_start_idx = self.aggregator(input.image, pose_tokens, posed_input=not self.config.unposed)

        aggregated_tokens_list = [i.float() for i in aggregated_tokens_list]
        target_pose_cond = target_pose_cond.float()

        # discard input view tokens and camera/register tokens.
        # because each feat has dimension [b, v, ...],
        # the v dimension is [i, ..., o]: only the last one is the output
        target_views_tokens_list = [feat[:,-1:] for feat in aggregated_tokens_list]

        input_views_tokens_list = [feat[:, :-1] for feat in aggregated_tokens_list]

        # --- REPA-style feature alignment (target view only) ---
        # Hook: aggregator[layer_idx], last (target) view, patch tokens only,
        #       last embed_dim channels = post-global-attention features.
        # Aggregator flattens B into B*V_out at L215 (see aggregator.py); the
        # `feat[:,-1:]` slice therefore yields per-(b, v_out) target view tokens.
        align_pkg = None
        if self.feature_alignment_enabled and self.training and has_target_image:
            with torch.cuda.amp.autocast(enabled=False):
                feat = aggregated_tokens_list[self.align_layer_idx][
                    :, -1:, patch_start_idx:, self.embed_dim:
                ]  # [B*V_out, 1, N_agg, C=embed_dim]
                BV, _, N_agg, C_ = feat.shape
                H_agg = W_agg = int(math.sqrt(N_agg))
                # Reshape to BCHW and force fp32 (outer autocast is bf16 by default).
                feat = (
                    feat.reshape(BV, H_agg, W_agg, C_)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                    .float()
                )
                feat = self.align_projector(feat)  # [B*V_out, C, H_agg, W_agg]

                # With target_resolution=512, DINOv3 grid is 512/16=32 which exactly
                # matches aggregator's 448/14=32 grid -> no spatial interpolation.
                # Kept as a guard so ablations on target_resolution still work.
                H_t = W_t = self.align_target_encoder.grid
                if (H_agg, W_agg) != (H_t, W_t):
                    feat = F.interpolate(
                        feat, size=(H_t, W_t), mode="bilinear", align_corners=False
                    )
                z_pred = feat.flatten(2).transpose(1, 2)  # [B*V_out, H_t*W_t, C]

                # target.image: [B, V_out, 3, H, W] -> flatten to [B*V_out, 3, H, W]
                target_img = rearrange(target.image, "b v c h w -> (b v) c h w")
                z_tgt = self.align_target_encoder(target_img)  # [B*V_out, H_t*W_t, C]
                assert z_pred.shape == z_tgt.shape, (
                    f"z_pred {z_pred.shape} vs z_tgt {z_tgt.shape}"
                )
                align_pkg = (z_pred, z_tgt)
        # --- end feature alignment hook ---

        with torch.cuda.amp.autocast(enabled=False):
            rendered_images, conf = self.rgb_head(
                target_views_tokens_list, images=target_pose_cond, patch_start_idx=patch_start_idx)
            rendered_images = rearrange(rendered_images, 'b v h w c -> b v c h w')

            # Resize RAE output (512) to the working image size. (h, w) come from
            # the pose-cond tensors above and equal the config image_size, so at
            # the production 448 this is bit-identical to the old hard-coded
            # (448, 448); it also unblocks test-time resolution probes (e.g. 518).
            if rendered_images.shape[-2:] != (h, w):
                B, V = rendered_images.shape[:2]
                rendered_images = rendered_images.view(B * V, *rendered_images.shape[2:])
                rendered_images = torch.nn.functional.interpolate(
                    rendered_images, size=(h, w), mode='bilinear', align_corners=True
                )
                rendered_images = rendered_images.view(B, V, *rendered_images.shape[1:])

            target_points, pts_conf = self.point_head(
                target_views_tokens_list, images=target_pose_cond, patch_start_idx=patch_start_idx)
            target_points = rearrange(target_points, 'b v h w c -> b v c h w', c=3)

            pose_enc_list = self.camera_head(input_views_tokens_list)

        pts_conf = rearrange(pts_conf, 'b v h w -> b v 1 h w')

        # print(input.extrinsic[:,0]) # must be [I_3x3, [0,0,1]^T], or [I_3x3, [0,0,-1]^T]^{-1}

        input_extrinsic = repeat(input.extrinsic, 'b v_in i j -> (b v_out) v_in i j', v_out=v_target)
        input_intrinsic = repeat(input.intrinsic, 'b v_in i j -> (b v_out) v_in i j', v_out=v_target)
        if has_target_image:
            loss_metrics = self.loss_computer(
                rendering=rendered_images,
                target=target.image,
                exclude_bg=exclude_bg,
                pts_est=target_points,
                pts_conf=pts_conf,
                # pts_gt=target.point_map,
                pts_gt = getattr(target, 'point_map', None),
                pose_enc_list=pose_enc_list,
                extrinsics=input_extrinsic,
                intrinsics=input_intrinsic,
                image_hw=(h, w),
                align_pkg=align_pkg,
                align_weight=self.align_weight if self.feature_alignment_enabled else 0.0,
                # P4c: pass real alpha mask from dataset (auto-split by ProcessData.fetch_views).
                # None when dataset doesn't emit alpha_mask -> loss falls back to white-bg.
                target_alpha_mask=getattr(target, 'alpha_mask', None),
            )
        else:
            loss_metrics = None

        result = edict(
            input=input,
            target=target,
            loss_metrics=loss_metrics,
            render=rendered_images,
            points=target_points,
            camera=pose_enc_list
            )
        
        return result

    def forward_single_target_view_unposed(self, input_batch, target_c2w):
        input, target = self.process_data.forward_single_target_view(
            input_batch, target_c2w)

        target_pose_cond = self.get_posed_input(ray_o=target.ray_o, ray_d=target.ray_d)
        b, v_target, c, h, w = target_pose_cond.size()

        b, v_input, *_ = input.image.shape
        input_pose_cond = torch.zeros(b, v_input, c, h, w, device=target_pose_cond.device, dtype=target_pose_cond.dtype)

        pose_cond = torch.concat([input_pose_cond, target_pose_cond], dim=1)
        pose_cond = self.pose_tokenizer(pose_cond)

        aggregated_tokens_list, patch_start_idx = self.aggregator(input.image, pose_cond, posed_input=False)

        aggregated_tokens_list = [i.float() for i in aggregated_tokens_list]
        target_pose_cond = target_pose_cond.float()

        # discard input view tokens and camera/register tokens.
        # because each feat has dimension [b, v, ...],
        # the v dimension is [i, ..., o]: only the last one is the output
        target_views_tokens_list = [feat[:,-1:] for feat in aggregated_tokens_list]

        input_views_tokens_list = [feat[:, :-1] for feat in aggregated_tokens_list]

        with torch.cuda.amp.autocast(enabled=False):
            rendered_images, conf = self.rgb_head(
                target_views_tokens_list, images=target_pose_cond, patch_start_idx=patch_start_idx)
            rendered_images = rearrange(rendered_images, 'b v h w c -> b v c h w')

            target_points, pts_conf = self.point_head(
                target_views_tokens_list, images=target_pose_cond, patch_start_idx=patch_start_idx)
            target_points = rearrange(target_points, 'b v h w c -> b v c h w', c=3)

            pose_enc_list = self.camera_head(input_views_tokens_list)

        pts_conf = rearrange(pts_conf, 'b v h w -> b v 1 h w')

        result = edict(
            input=input,
            target=target,
            render=rendered_images,
            points=target_points,
            points_conf=pts_conf,
            camera=pose_enc_list)

        return result


    def forward_pose_only(self, input_images, return_all=False):
        ### disable all attention mask
        for block in self.aggregator.global_blocks:
            block.attn.mask_attention = None
            block.attn.mode = 'save_cache'

        aggregated_tokens_list, patch_start_idx = self.aggregator.forward_pose_only(input_images)
        aggregated_tokens_list = [i.float() for i in aggregated_tokens_list]
        with torch.cuda.amp.autocast(enabled=False):
            pose_enc_list = self.camera_head(aggregated_tokens_list)

        ### recover all attention mask
        _, _, _, h, w = input_images.shape
        num_patch_tokens = (h // self.patch_size) * (w // self.patch_size)
        target_token_count = self.aggregator.patch_start_idx + num_patch_tokens
        for block in self.aggregator.global_blocks:
            block.attn.mask_attention = target_token_count

        if return_all:
            return pose_enc_list
        return pose_enc_list[-1]
        
    def forward_rendering_using_kv_cache(self, target_c2w=None, target=None, target_pose_cond=None):
        for block in self.aggregator.global_blocks:
            block.attn.mode = 'read_cache'

        if target_pose_cond is None:
            if target is None:
                if target_c2w is None:
                    raise ValueError("Provide target_pose_cond, target, or target_c2w for kv-cache rendering.")
                target = self.process_data.forward_target_view_without_input(target_c2w)
            target_pose_cond = self.get_posed_input(ray_o=target.ray_o, ray_d=target.ray_d)
        b, v_target, c, h, w = target_pose_cond.size()

        pose_cond = target_pose_cond
        pose_cond = self.pose_tokenizer(pose_cond)
        grid_hw = (h // self.patch_size, w // self.patch_size)

        aggregated_tokens_list, patch_start_idx = self.aggregator.forward_target_view_reading_kv_cache(
            pose_cond, grid_hw=grid_hw)

        aggregated_tokens_list = [i.float() for i in aggregated_tokens_list]
        target_pose_cond = target_pose_cond.float()

        target_views_tokens_list = aggregated_tokens_list

        with torch.cuda.amp.autocast(enabled=False):
            rendered_images, conf = self.rgb_head(
                target_views_tokens_list, images=target_pose_cond, patch_start_idx=patch_start_idx)
            rendered_images = rearrange(rendered_images, 'b v h w c -> b v c h w')
            if rendered_images.shape[-2:] != target_pose_cond.shape[-2:]:
                B, V = rendered_images.shape[:2]
                rendered_images = rendered_images.view(B * V, *rendered_images.shape[2:])
                rendered_images = torch.nn.functional.interpolate(
                    rendered_images, size=target_pose_cond.shape[-2:], mode='bilinear', align_corners=True
                )
                rendered_images = rendered_images.view(B, V, *rendered_images.shape[1:])

            target_points, pts_conf = self.point_head(
                target_views_tokens_list, images=target_pose_cond, patch_start_idx=patch_start_idx)
            target_points = rearrange(target_points, 'b v h w c -> b v c h w', c=3)

        pts_conf = rearrange(pts_conf, 'b v h w -> b v 1 h w')

        result = edict(
            target=target,
            render=rendered_images,
            points=target_points,
            points_conf=pts_conf)
        
        # for block in self.aggregator.global_blocks:
        #     block.attn.mode = 'save_cache'

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
