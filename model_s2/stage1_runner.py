# Frozen stage 1 for the stage-2 wrapper (plan 2026-09-27, section 二 "每步数据流" 1-3).
#
# Holds the stage-1 LagerNVSInRnG (uni3t ckpt_70000) OUTSIDE the stage-2 module tree: the wrapper stores
# this runner with object.__setattr__, so it is invisible to named_parameters / state_dict / the
# optimizer / DDP.  Everything runs through the stock stage-1 submodules; no stage-1 source is modified.
#   prepare : identical to LagerNVSInRnG.forward lines 155-189 (same process_data call, same
#             per-sample Bernoulli draw for the unposed mode, driven by the *stage-2* training flag)
#   pass1   : reconstructor + renderer (+ a temporary hook on renderer block 10 for the scene tokens
#             after their last update) + camera head + point head, exactly as LagerNVSInRnG.forward
#   pass2   : the 4 input cameras rendered as targets (VGGT tokens reused), point head only used

import einops
import torch
import torch.nn as nn

from model.lagernvs_wrapper import LagerNVSInRnG
from model_s2.geometry import plucker_rays

S1_LAYERS = (2, 5, 8, 11)


class Stage1Runner:
    def __init__(self, config, ckpt_path, expect_step, device):
        model = LagerNVSInRnG(config)
        if not model.unified_heads:
            raise ValueError("stage 1 must be the uni3t three-task model (model.unified_heads.enabled)")
        if model.camera_gauge != "rng":
            raise ValueError("stage 2 assumes the RnG camera gauge")
        model.loss_computer = nn.Identity()                      # never called; drops the VGG copy
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
        step = ck.get("fwdbwd_pass_step")
        if expect_step is not None and int(step) != int(expect_step):
            raise RuntimeError(f"[stage1] {ckpt_path} is step {step}, expected {expect_step}")
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        # the stage-1 checkpoint also carries its frozen perceptual-loss VGG; the loss module is dropped here
        bad = [k for k in missing if not k.startswith("loss_computer.")]
        bad_u = [k for k in unexpected if not k.startswith("loss_computer.")]
        if bad or bad_u:
            raise RuntimeError(f"[stage1] checkpoint does not match the model: missing {bad[:8]} "
                               f"unexpected {bad_u[:8]}")
        del ck
        model.requires_grad_(False)
        model.eval()
        self.model = model.to(device)
        self.device = device
        self.ckpt_path = ckpt_path
        self.step = int(step) if step is not None else None

    # ------------------------------------------------------------------------------------------
    def check(self):
        assert not self.model.training, "stage 1 must stay in eval mode"

    def prepare(self, data_batch, has_target_image, target_has_input, is_valid, training, zero_p):
        m = self.model
        if target_has_input is None:
            target_has_input = m.config.training.target_has_input
        pdata = m.process_val_data if is_valid else m.process_data
        input, target = pdata(data_batch, has_target_image=has_target_image,
                              target_has_input=target_has_input, compute_rays=True)
        images = torch.cat([input.image, target.image], dim=1)
        rays = torch.cat([m._plucker(input.ray_o, input.ray_d), m._plucker(target.ray_o, target.ray_d)],
                         dim=1).to(images.dtype)
        b, v = images.shape[:2]
        v_input = input.image.shape[1]
        if zero_p >= 1.0:
            cam_token = torch.zeros(b, v, 11, device=images.device, dtype=images.dtype)
            posed = torch.zeros(b, dtype=torch.bool, device=images.device)
        else:
            cam_token = m._build_cam_tokens(input, target).to(images.dtype)
            posed = torch.ones(b, dtype=torch.bool, device=images.device)
            if training and zero_p > 0.0:
                drop = torch.rand(b, device=images.device) < zero_p     # same draw as LagerNVSInRnG
                cam_token[drop, :v_input] = 0.0
                posed = ~drop
        return input, target, images, rays, cam_token, posed, v_input

    @torch.no_grad()
    def pass1(self, images, rays, cam_token, v_input):
        m = self.model
        ed = m.model
        input_images = images[:, :v_input]
        rec_tok, agg_last, _ = ed.reconstructor(input_images, cam_token[:, :v_input], return_tokens=True)
        rec0 = einops.rearrange(rec_tok, "b v_input p c -> b (v_input p) c")
        target_rays = rays[:, v_input:]
        vt = target_rays.shape[1]
        rec_rep = einops.repeat(rec0, "b np d -> (b v_target) np d", v_target=vt)
        store = {}
        hook = ed.renderer.renderer_core.renderer_blocks[10].register_forward_hook(
            lambda mod, inp, out: store.__setitem__("rec10", out[1]))
        try:
            rendered, inter, psi = ed.renderer(rec_rep, target_rays, return_intermediates=True)
        finally:
            hook.remove()
        render = torch.cat([input_images, rendered], dim=1)[:, v_input:].float().clamp(0.0, 1.0)
        b = images.shape[0]
        with torch.autocast("cuda", enabled=False):
            pose_enc_list = m.camera_head([agg_last[:, :, :1].float()])
            rtok = [einops.rearrange(t, "(b v) n c -> b v n c", b=b) for t in inter]
            pts, conf = m.point_head(rtok, target_rays.float(), patch_token_start=psi)
        return dict(
            render=render,
            points=einops.rearrange(pts, "b v h w c -> b v c h w", c=3),
            conf=einops.rearrange(conf, "b v h w -> b v 1 h w"),
            pose_enc_list=pose_enc_list,
            T1={k: inter[k] for k in S1_LAYERS},
            rec0=rec0,
            rec_rep=rec_rep,
            rec10=store["rec10"],
        )

    @torch.no_grad()
    def pass2(self, rec0, c2w_in, K_in, H, W):
        m = self.model
        ed = m.model
        rays_in = plucker_rays(c2w_in.float(), K_in.float(), H, W)
        vi = rays_in.shape[1]
        rec_rep = einops.repeat(rec0, "b np d -> (b v) np d", v=vi)
        _, inter, psi = ed.renderer(rec_rep, rays_in, return_intermediates=True)
        b = rec0.shape[0]
        with torch.autocast("cuda", enabled=False):
            rtok = [einops.rearrange(t, "(b v) n c -> b v n c", b=b) for t in inter]
            pts, _ = m.point_head(rtok, rays_in, patch_token_start=psi)
        return einops.rearrange(pts, "b v h w c -> b v c h w", c=3)
