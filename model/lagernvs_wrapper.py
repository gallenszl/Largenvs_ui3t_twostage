# P-LN2: the vendored LagerNVS EncDec_VitB8 dropped into the RnG training
# framework. Dataset / train loop / image loss stay RnG; the model runs through
# its own wrapped forward at the dataset resolution (256), exactly like
# lagernvs's native 256 models (the encoder upsamples its own copy of the cond
# views to 518 internally).
#
# Two modes:
#   model.unified_heads.enabled: false (default) — image loss only (base
#     LossComputer); points/camera in the returned edict are placeholders for the
#     eval plumbing. Bit-identical to the pre-UNI3T behaviour.
#   model.unified_heads.enabled: true — three tasks at once: source-view pose
#     (CameraHead on the VGGT aggregator, VGGT-1B pretrained weights), target-view
#     world point map (DensePointHead on the renderer target stream, trained from
#     scratch) and NVS (the renderer's own MLP head). REPA stays off either way.

import os
import traceback

import einops
import torch
import torch.nn as nn
from easydict import EasyDict as edict

from model.loss import LossComputer, MultiTaskLossComputer
from models.encoder_decoder import EncDec_VitB8  # vendored lagernvs
from models.layers.dense_point_head import DensePointHead  # UNI3T: vendored VGGT-Omega head
from utils import data_utils
from vggt.heads.camera_head import CameraHead  # vendored lagernvs (top-level vggt/)
from vggt.utils.pose_enc import extri_intri_to_pose_encoding  # vendored lagernvs (top-level vggt/)

# Same URL the vendored Reconstructor uses; torch.hub caches it under TORCH_HOME so
# asking for it a second time is a local file read, not a download.
VGGT_PRETRAINED_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"


class LagerNVSInRnG(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        # UNI3T: three-task mode = source-view pose + target-view world point map + NVS.
        # Off by default, in which case every line below behaves exactly as before.
        _uh = config.model.get("unified_heads", {}) or {}
        self.unified_heads = bool(_uh.get("enabled", False))
        self.renderer_dpt_layers = list(_uh.get("renderer_dpt_layers", [2, 5, 8, 11]))

        # Config guards run before the 1.1B model is built, so a misconfiguration costs
        # a second instead of a full construction.
        #
        # _to_native_gauge rewrites the cameras but deliberately leaves depth_map /
        # point_map in RnG units (see its docstring), so a point loss under the native
        # gauge would silently regress against mismatched ground truth.
        self.weight_point = float(config.training.get("weight_point", 0.0))
        if str(config.training.get("camera_gauge", "rng")) == "native" and self.weight_point > 0.0:
            raise ValueError(
                "camera_gauge='native' with weight_point>0: point_map is still in the "
                "RnG gauge, convert it in _to_native_gauge before enabling the point loss"
            )

        self.model = EncDec_VitB8(
            freeze_vggt=False,          # default True would silently freeze the encoder
            pretrained_vggt=True,       # VGGT-1B via torch.hub -> TORCH_HOME local file
            attention_to_features_type="bidirectional_cross_attention",
        )

        self.process_data = data_utils.ProcessData(config)
        self.process_val_data = data_utils.ProcessData(config.training.val_dataset_cfgs)
        self.loss_computer = (
            MultiTaskLossComputer(config) if self.unified_heads else LossComputer(config)
        )

        # P-LN2-pose: probability of zeroing the cond views' camera tokens.
        # 1.0 (default) = always unposed (P-LN2 behavior, bit-identical);
        # 0.4 = lagernvs native dual-mode training; 0.0 = always posed.
        self.cam_cond_zero_p = float(config.training.get("cam_cond_zero_p", 1.0))
        self.val_cam_cond_zero_p = float(config.training.get("val_cam_cond_zero_p", self.cam_cond_zero_p))

        # D-arm: camera gauge. "rng" (default) = historical object-centric gauge
        # (first cam at (0,0,-1), unit = first-cam-to-object radius).
        # "native" = LagerNVS gauge: first cond camera at the origin, translations
        # divided by 1.35 x max conditioning baseline (camera_scale token becomes
        # the constant 1/1.35 = 0.7407407).
        self.camera_gauge = str(config.training.get("camera_gauge", "rng"))
        assert self.camera_gauge in ("rng", "native"), self.camera_gauge

        # train.py:118-121 unconditionally casts these attributes back to fp32
        # after the whole model is cast to bf16.
        self.rgb_head = self.model.renderer.final_layer  # alias: output layer stays fp32
        if self.unified_heads:
            # Pose head: reads aggregator token slot 0 of the cond views. dim_in is the
            # aggregator's [frame | global] concat width, read off the existing adapter
            # rather than hardcoded.
            _agg_dim = self.model.reconstructor.geo_feature_connector.in_features
            self.camera_head = CameraHead(dim_in=_agg_dim)
            _sd = torch.hub.load_state_dict_from_url(
                VGGT_PRETRAINED_URL, map_location="cpu"
            )
            _pref = "camera_head."
            _cam_sd = {k[len(_pref):]: v for k, v in _sd.items() if k.startswith(_pref)}
            # strict=True: the vendored CameraHead must match VGGT-1B tensor-for-tensor.
            self.camera_head.load_state_dict(_cam_sd, strict=True)
            print(f"UNI3T: loaded {len(_cam_sd)} pretrained camera_head tensors from VGGT-1B")

            # Point head: reads the renderer target stream, upstream weight init.
            # (A zero-init of its output conv would look tidy but leaves the head with
            # exactly zero gradient forever -- see the UNI3T NOTE in dense_point_head.py.)
            _hidden = self.model.renderer.final_layer.norm_final.normalized_shape[0]
            self.point_head = DensePointHead(
                dim_in=_hidden,
                patch_size=self.model.renderer.patch_size,
                intermediate_layer_idx=self.renderer_dpt_layers,
            )
        else:
            self.camera_head = nn.Identity()
            self.point_head = nn.Identity()

    @staticmethod
    def _plucker(ray_o, ray_d):
        return torch.cat([torch.cross(ray_o, ray_d, dim=2), ray_d], dim=2)

    def _to_native_gauge(self, input, target):
        """RnG object-centric gauge -> native LagerNVS gauge, in place.

        RnG normalization already puts the first cond camera at R=I, t=(0,0,-1)
        and that camera is also native's reference, so the reference step
        reduces to a pure translation; the earlier r0 scaling cancels inside
        the baseline normalization (audit doc §16). Rotations and intrinsics
        are untouched, so ray directions are unchanged — only ray origins (and
        hence Plucker moments) and the pose tokens move.

        depth_map/point_map intentionally stay in RnG units: they only feed the
        placeholder depth/pose metrics of this image-loss-only family. Convert
        them first if a point/depth loss is ever enabled with this gauge.
        """
        # under use_bf16 the whole batch (incl. c2w) arrives as bf16 (train.py:333);
        # do the gauge math in fp32 (torch.inverse rejects bf16) and cast back.
        orig_dtype = input.c2w.dtype
        q = input.c2w.new_tensor([0.0, 0.0, -1.0], dtype=torch.float32)
        t_in = input.c2w[..., :3, 3].float()
        # max conditioning baseline per sample (view 0 contributes 0 by construction)
        bnorm = (t_in - q).norm(dim=-1).amax(dim=1)                           # [b]
        scale = (1.35 * bnorm.clamp_min(1e-4)).view(-1, 1, 1)
        for d in (input, target):
            c2w = d.c2w.float().clone()
            c2w[..., :3, 3] = (c2w[..., :3, 3] - q) / scale
            d.c2w = c2w.to(orig_dtype)
            d.extrinsic = torch.inverse(c2w).to(orig_dtype)  # keep the pair consistent
            h, w = d.image_h_w
            ray_o, ray_d = self.process_data.compute_rays(
                c2w, d.fxfycxcy.float(), h, w, device=c2w.device)
            d.ray_o, d.ray_d = ray_o.to(orig_dtype), ray_d.to(orig_dtype)

    def forward(self, data_batch, has_target_image=True, target_has_input=None,
                exclude_bg=False, is_valid=False):
        if target_has_input is None:
            target_has_input = self.config.training.target_has_input

        process_data = self.process_val_data if is_valid else self.process_data
        input, target = process_data(
            data_batch,
            has_target_image=has_target_image,
            target_has_input=target_has_input,
            compute_rays=True)

        if self.camera_gauge == "native":
            self._to_native_gauge(input, target)

        # [b, v, 3, 256, 256]; the model only reads the first v_input views,
        # target slots just ride along for the output cat.
        images = torch.cat([input.image, target.image], dim=1)
        rays = torch.cat([
            self._plucker(input.ray_o, input.ray_d),
            self._plucker(target.ray_o, target.ray_d),
        ], dim=1).to(images.dtype)
        b, v = images.shape[:2]
        v_input = input.image.shape[1]
        v_target = v - v_input

        # Camera conditioning: 11-dim lagernvs tokens (9-dim absT_quaR_FoV +
        # [camera_scale, world_points_scale=0]), or zeros (unposed mode).
        zero_p = self.val_cam_cond_zero_p if is_valid else self.cam_cond_zero_p
        if zero_p >= 1.0:
            cam_token = torch.zeros(b, v, 11, device=images.device, dtype=images.dtype)
        else:
            cam_token = self._build_cam_tokens(input, target).to(images.dtype)
            if self.training and zero_p > 0.0:
                # lagernvs semantics: per-sample bernoulli, cond views only
                drop = torch.rand(b, device=images.device) < zero_p
                cam_token[drop, :v_input] = 0.0

        if self.unified_heads:
            out, aux = self.model(
                images, rays, cam_token, num_cond_views=v_input, return_aux=True
            )
        else:
            out = self.model(images, rays, cam_token, num_cond_views=v_input)
        rendering = out[:, v_input:].float().clamp(0.0, 1.0)

        if self.unified_heads:
            # Both heads run in fp32 outside autocast, same as the reference RnG.
            target_rays6 = rays[:, v_input:].float()   # only its shape is read
            with torch.autocast("cuda", enabled=False):
                # [b, v_input, 9] per refinement iteration (4 of them).
                # CameraHead only reads aggregated_tokens_list[-1], so that is all the
                # reconstructor hands back; the fp32 cast is needed because the head's
                # params are fp32 and autocast is off here.
                # UNI3T: CameraHead reads only slot 0 (vggt/heads/camera_head.py:97),
                # so slice first and cast 0.25 MiB instead of the whole
                # [b, v_in, 1374, 2048] tensor (343.5 MiB of fp32 that token_norm
                # would then keep alive until backward finishes).
                _cam_tok = aux["agg_last_tokens"][:, :, :1].float()
                pose_enc_list = self.camera_head([_cam_tok])
                # No cast here on purpose: DensePointHead slices the patch tokens of the
                # 4 layers it actually uses and casts those to fp32 itself, so casting
                # all 12 up front would allocate ~1.8 GB of fp32 copies for nothing.
                # The rearrange is a pure reshape, not a copy.
                rtok = [
                    einops.rearrange(t, "(b v) n c -> b v n c", b=b)
                    for t in aux["renderer_tokens"]
                ]
                pts, pts_conf = self.point_head(
                    rtok,
                    target_rays6,
                    patch_token_start=aux["renderer_patch_start_idx"],
                )
            points = einops.rearrange(pts, "b v h w c -> b v c h w", c=3)
            points_conf = einops.rearrange(pts_conf, "b v h w -> b v 1 h w")

            # Hard failure rather than the reference repo's silent getattr(...): with
            # DDP find_unused_parameters=False, skipping the point loss leaves the whole
            # point head without gradients for that step and the run dies at the
            # reducer with a much less obvious error.
            assert hasattr(target, "point_map"), (
                "unified_heads needs target.point_map; this dataset does not provide it"
            )
            loss_metrics = self.loss_computer(
                rendering,
                target.image.float(),
                exclude_bg,
                pts_est=points,
                pts_conf=points_conf,
                pts_gt=target.point_map.float(),
                pose_enc_list=pose_enc_list,
                # our VGGT runs once per batch, so pose_enc_list is [b, v_input, 9];
                # the reference repo repeats the GT over v_target because its aggregator
                # folds v_target into the batch dim. Here no repeat is needed.
                extrinsics=input.extrinsic.float(),
                intrinsics=input.intrinsic.float(),
                image_hw=input.image_h_w,
                align_pkg=None,
                align_weight=0.0,
                target_alpha_mask=getattr(target, "alpha_mask", None),
                # read only by the track-consistency loss (training.weight_consistency > 0)
                target_depth=(target.depth_map.float() if hasattr(target, "depth_map") else None),
                target_c2w=target.c2w.float(),
                target_K=target.fxfycxcy.float(),
            )
            # export_results slices camera[-1][batch_idx * v_target], so lay the copies
            # out as (b v_target) to match.
            camera_out = [
                einops.repeat(p, "b v i -> (b vt) v i", vt=v_target).float()
                for p in pose_enc_list
            ]
        else:
            loss_metrics = self.loss_computer(
                rendering,
                target.image.float(),
                exclude_bg=exclude_bg,
                target_alpha_mask=getattr(target, "alpha_mask", None),
            )
            points = torch.zeros_like(rendering)
            # points/camera are placeholders so export_results' depth/pose branches
            # run without crashing; their metrics are meaningless by design.
            # real RnG camera[-1] slices as [v_input, 9] per (batch*target) instance
            camera_stub = torch.zeros(b * v_target, v_input, 9, device=images.device, dtype=torch.float32)
            camera_stub[:, :, 3] = 1.0   # valid quaternion (w=1)
            camera_stub[:, :, 7:] = 0.7  # sane fov
            camera_out = [camera_stub]

        with torch.no_grad():
            # collapse canary (P-LN lesson: the white-out death shows here fast)
            loss_metrics["render_std"] = rendering.std()

        return edict(
            input=input,
            target=target,
            loss_metrics=loss_metrics,
            render=rendering,
            points=points,
            camera=camera_out,
        )

    @staticmethod
    def _build_cam_tokens(input, target):
        """lagernvs build_cam_cond semantics on RnG-canonicalized cameras."""
        c2w = torch.cat([input.c2w, target.c2w], dim=1).float()            # [b, v, 4, 4]
        fxy = torch.cat([input.fxfycxcy, target.fxfycxcy], dim=1).float()  # [b, v, 4]
        h, w = input.image.shape[-2:]
        enc9 = extri_intri_to_pose_encoding(c2w[..., :3, :], fxy, image_size_hw=(h, w))
        cam_scale = input.c2w[..., :3, 3].norm(dim=-1).amax(dim=1)         # [b] max over cond views
        scale_tok = torch.stack([cam_scale, torch.zeros_like(cam_scale)], dim=-1)
        tok = torch.cat([enc9, scale_tok[:, None, :].expand(-1, c2w.shape[1], -1)], dim=-1)
        return tok

    def load_ckpt(self, load_path):
        """Same semantics as VGGT4LVSM.load_ckpt (inference.py entry point)."""
        if os.path.isdir(load_path):
            names = sorted(f for f in os.listdir(load_path) if f.endswith(".pt"))
            paths = [os.path.join(load_path, n) for n in names]
        else:
            paths = [load_path]
        try:
            checkpoint = torch.load(paths[-1], map_location="cpu", weights_only=True)
        except Exception:
            traceback.print_exc()
            print(f"Failed to load {paths[-1]}")
            return None
        missing, unexpected = self.load_state_dict(checkpoint["model"], strict=False)
        print(f"load_ckpt: {paths[-1]} | missing {len(missing)} unexpected {len(unexpected)}")
        return 0
