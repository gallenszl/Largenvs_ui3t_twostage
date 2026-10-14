# Stage-2 sparse refinement model (plan 2026-09-27, final plan in the plan file, section 一 / 二).
# config.model.class_name = model_s2.stage2_wrapper.Stage2LagerNVS
#
# One forward: frozen stage 1 (pass 1 [+ pass 2]) -> layout [+ per-token mask tables] (geometry) -> trainable
# stage-2 renderer on the foreground target tokens -> T~ = T1 + Lin(t2) -> colour / point heads -> loss.
# The frozen stage 1 is kept out of the module tree (object.__setattr__), so state_dict / the optimizer /
# DDP only ever see the trainable stage-2 parameters (+ the frozen VGG of the perceptual loss).

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
import einops

from model.loss import LossComputer, check_and_fix_inf_nan, compute_point_loss
from model_s2 import geometry as geo
from model_s2.blocks import S2Ctx
from model_s2.fg_scene import build_fg_scene, build_fg_windows
from model_s2.heads_s2 import (assemble_tokens, build_x_prior, make_color_head, make_point_head, render_color,
                               render_points)
from model_s2.masked_attention import MaskTable
from model_s2.renderer_s2 import Stage2Renderer
from model_s2.stage1_runner import Stage1Runner

SCENE_G = 37


class S2LossComputer(LossComputer):
    """stage 1's late-regime loss without the camera term: L2 (full image) x l2_w + perceptual x p_w +
    point x weight_point with the 'exclude_bg off' split 0.5 * (conf + reg + grad) + 0.5 * L1."""

    def forward(self, rendering, target, pts_est, pts_conf, pts_gt, target_alpha_mask=None):
        lm = super().forward(rendering, target, exclude_bg=False, target_alpha_mask=target_alpha_mask)
        pl = compute_point_loss(pts_est, pts_conf, pts_gt, weight=1.0, gradient_loss_fn="normal", valid_range=0.98)
        point_loss = pl["loss_conf_point"] + pl["loss_reg_point"] + pl["loss_grad_point"]
        point_l1 = check_and_fix_inf_nan(F.l1_loss(pts_est, pts_gt), "loss_l1_point")
        point_loss = point_loss * 0.5 + point_l1 * 0.5
        lm.update(pl)
        lm["loss_point_total"] = point_loss
        lm["loss"] = lm["loss"] + point_loss * self.config.training.weight_point
        return lm


class Stage2LagerNVS(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        s2 = config.model.stage2
        tr = config.training
        # guards against silently inheriting stage-1 training switches (plan section 五)
        if float(tr.get("exclude_bg_frac", 0.25)) != 0.0:
            raise ValueError("stage 2 has no foreground-only phase: set training.exclude_bg_frac 0.0")
        if float(tr.get("unfreeze_rae_decoder_at", 0.2)) < 1.0:
            raise ValueError("set training.unfreeze_rae_decoder_at >= 1.0 (the default 0.2 crashes)")
        if tr.get("use_bf16", False):
            raise ValueError("stage 2 trains fp32 weights under bf16 autocast (use_bf16 false)")
        if float(tr.get("weight_camera", 1.0)) != 0.0:
            raise ValueError("stage 2 has no camera loss: set training.weight_camera 0.0")
        if str(tr.get("resume_ckpt", "") or "") != "" and "PLN2uni3t" in str(tr.get("resume_ckpt")):
            raise ValueError("resume_ckpt points at a stage-1 run; stage 2 loads stage 1 via model.stage2.stage1_ckpt")

        self.enabled = bool(s2.get("enabled", True))
        self.patch = int(s2.target_patch)
        self.scene_radius = int(s2.scene_radius)
        self.target_radius = int(s2.target_radius)
        self.tau = float(s2.vis_rel_tau)
        self.scene_block = int(s2.scene_block)
        self.tblock_px = int(s2.target_block_px)
        self.pad_bucket = int(s2.pad_bucket)
        self.structure = str(s2.get("structure", "current"))
        if self.structure not in ("current", "fg_dense", "fg_dual"):
            raise ValueError(f"model.stage2.structure must be current / fg_dense / fg_dual, got {self.structure!r}")
        # which target tiles stage 2 re-renders and what its mask input channel carries (user 09-28).  "gt": the GT
        # target mask (the original design; it does not exist at test time).  target_select "stage1": the tiles
        # holding a pixel the stage-1 render painted (a colour channel > target_select_tau below white), grown by
        # target_select_dilate tiles.  mask_input "stage1": that painted-pixel mask; "zero": an all-zero channel,
        # whose weights then never get a gradient -- the same as no mask channel, with the parameters, the init and
        # the RNG stream identical to the other arms.
        self.target_select = str(s2.get("target_select", "gt"))
        self.select_dilate = int(s2.get("target_select_dilate", 1))
        self.select_tau = float(s2.get("target_select_tau", 0.1))
        self.mask_input = str(s2.get("mask_input", "gt"))
        if self.target_select not in ("gt", "stage1"):
            raise ValueError(f"model.stage2.target_select must be gt / stage1, got {self.target_select!r}")
        if self.mask_input not in ("gt", "stage1", "zero"):
            raise ValueError(f"model.stage2.mask_input must be gt / stage1 / zero, got {self.mask_input!r}")
        if (self.target_select, self.mask_input) != ("gt", "gt") and self.structure != "fg_dense":
            raise ValueError("target_select / mask_input other than gt need structure fg_dense (the window and "
                             "visibility tables of current / fg_dual read the GT target mask)")
        self.masked_backend = str(s2.get("attn_backend", "sdpa"))
        cuda = torch.cuda.is_available()
        self.dense_backend = str(s2.get("dense_backend", "fa3" if cuda else "ref"))
        if not cuda and self.masked_backend != "ref":
            self.masked_backend = "ref"
        self.img = int(config.model.target_pose_tokenizer.image_size)

        device = torch.device("cuda", torch.cuda.current_device()) if cuda else torch.device("cpu")
        runner = Stage1Runner(config, s2.stage1_ckpt, s2.get("stage1_step", None), device)
        object.__setattr__(self, "_s1", runner)
        r1 = runner.model.model.renderer
        self.s1_patch = int(r1.patch_size)
        if self.patch not in (self.s1_patch, self.s1_patch // 2):
            raise ValueError(f"target_patch must be {self.s1_patch} or {self.s1_patch // 2}")

        self.renderer = Stage2Renderer(r1, self.patch, num_heads=12, structure=self.structure)
        self.color_head = make_color_head(r1.final_layer, self.patch, self.s1_patch)
        self.point_head = make_point_head(runner.model.point_head, self.patch, self.s1_patch)
        self.loss_computer = S2LossComputer(config)
        self.cam_cond_zero_p = float(tr.get("cam_cond_zero_p", 1.0))
        self.val_cam_cond_zero_p = float(tr.get("val_cam_cond_zero_p", self.cam_cond_zero_p))

    # -------------------------------------------------------------------------------------------
    @property
    def stage1(self):
        return self._s1

    def forward(self, data_batch, has_target_image=True, target_has_input=None, exclude_bg=False, is_valid=False):
        if exclude_bg:
            raise ValueError("stage 2 must never run the foreground-only loss phase")
        s1 = self._s1
        s1.check()
        zero_p = self.val_cam_cond_zero_p if is_valid else self.cam_cond_zero_p
        input, target, images, rays, cam_token, posed, v_input = s1.prepare(
            data_batch, has_target_image, target_has_input, is_valid, self.training, zero_p)
        B, Vt = target.image.shape[:2]
        H = W = self.img
        p1 = s1.pass1(images, rays, cam_token, v_input)
        camera_out = [einops.repeat(p, "b v i -> (b vt) v i", vt=Vt).float() for p in p1["pose_enc_list"]]

        if not self.enabled:
            zero = torch.zeros((), device=images.device)
            return edict(input=input, target=target, loss_metrics=edict(loss=zero), render=p1["render"],
                         points=p1["points"], camera=camera_out, render_s1=p1["render"], points_s1=p1["points"])

        ctx, mask_in, fgs, sel_t = self.make_ctx(p1, input, target, posed, v_input)
        lay = ctx.layout

        target_rays = rays[:, v_input:]
        mask1 = mask_in.to(target_rays.dtype)
        res = self.renderer(target_rays, mask1, p1["rec_rep"], ctx)
        Ts = {m: assemble_tokens(p1["T1"][m], r, lay, self.patch, self.s1_patch) for m, r in res.items()}
        has_regs = self.patch == self.s1_patch
        render = render_color(self.color_head, Ts[11], self.patch, B, Vt, H, W, has_regs)
        points, conf = render_points(self.point_head, Ts, target_rays, B, geo.N_REG if has_regs else 0)

        loss_metrics = self.loss_computer(render, target.image.float(), points, conf, target.point_map.float(),
                                          target_alpha_mask=getattr(target, "alpha_mask", None))
        with torch.no_grad():
            loss_metrics["render_std"] = render.std()
            loss_metrics["s2_fg_tokens"] = float(sum(lay.n)) / max(1, lay.BV)
            if fgs is not None:
                loss_metrics["s2_scene_fg_share"] = fgs.share
            if self.target_select != "gt":
                # monitoring only (the model never sees it): share of GT foreground pixels outside the refined tiles
                gt = target.alpha_mask.float() > 0.5
                loss_metrics["s2_sel_gt_miss"] = (gt & (sel_t < 0.5)).sum() / gt.sum().clamp_min(1)
        return edict(input=input, target=target, loss_metrics=loss_metrics, render=render, points=points,
                     camera=camera_out, render_s1=p1["render"], points_s1=p1["points"])

    def target_selection(self, render_s1, alpha_t):
        """-> (sel [B, Vt, 1, H, W]: the pixel mask whose tiles stage 2 re-renders, mask_in [B, Vt, 1, H, W] {0, 1}:
        the renderer's mask input channel), per model.stage2.target_select / mask_input (see __init__)."""
        painted = None
        if "stage1" in (self.target_select, self.mask_input):
            painted = geo.stage1_painted(render_s1, self.select_tau)
        if self.target_select == "gt":
            sel = alpha_t
        else:
            sel = geo.tiles_from_painted(painted, self.patch, self.select_dilate)
        if self.mask_input == "gt":
            mask_in = (alpha_t > 0.5).float()
        elif self.mask_input == "stage1":
            mask_in = painted
        else:
            mask_in = torch.zeros_like(alpha_t)
        return sel, mask_in

    def make_ctx(self, p1, input, target, posed, v_input):
        """frozen stage-1 outputs -> (S2Ctx, mask input channel, kept scene tokens or None, refined-tile pixel mask):
        pass 2 and the window / visibility tables (current / fg_dual only), the layout of the refined target tiles,
        the scene-token subset (fg structures).  forward and the timing tools both call this, so a timing measures
        exactly the training / inference code."""
        s1 = self._s1
        B, Vt = target.image.shape[:2]
        H = W = self.img
        # window / visibility tables need the input cameras re-rendered (pass 2); fg_dense needs neither
        need_tables = self.structure in ("current", "fg_dual")
        if need_tables:
            with torch.no_grad(), torch.autocast("cuda", enabled=False):
                c2w_pred = geo.c2w_from_pose_enc(p1["pose_enc_list"][-1], H)
                c2w_in = torch.where(posed.view(B, 1, 1, 1), input.c2w.float(), c2w_pred)
                K_in = input.fxfycxcy.float()
            P_in2 = s1.pass2(p1["rec0"], c2w_in, K_in, H, W)

        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            alpha_t = target.alpha_mask.float()
            alpha_in = input.alpha_mask.float()
            sel_t, mask_in = self.target_selection(p1["render"], alpha_t)
            lay = geo.build_layout(sel_t, self.patch, self.pad_bucket, self.img, self.tblock_px)
            S = v_input * SCENE_G * SCENE_G
            s_pad = geo.roundup(S + 1, 128)
            P_t = p1["points"].float()
            fwd_mask = rev_mask = fgs = win = None
            if need_tables:
                Ft = geo.build_forward_table(P_t, alpha_t, lay, c2w_in, K_in, P_in2.float(), alpha_in, self.patch,
                                             SCENE_G, self.img, radius=self.scene_radius, tau=self.tau, s_pad=s_pad)
                Rt = geo.build_reverse_table(P_in2.float(), alpha_in, lay, target.c2w.float(),
                                             target.fxfycxcy.float(), P_t, alpha_t, self.patch, SCENE_G, self.img,
                                             radius=self.target_radius, tau=self.tau, s_pad=s_pad)
            if self.structure == "current":
                fwd_mask = MaskTable(Ft, self.masked_backend)
                rev_mask = MaskTable(Rt, self.masked_backend)
                rec_prior = p1["rec10"]
            else:
                fgs = build_fg_scene(alpha_in, Vt, SCENE_G)
                if self.structure == "fg_dual":
                    win = build_fg_windows(Ft, Rt, lay, fgs, self.masked_backend)
                rec_prior = p1["rec10"].reshape(-1, p1["rec10"].shape[-1])[fgs.idx]
            if need_tables:
                del Ft, Rt
            x_prior = build_x_prior(p1["T1"][11], lay, self.patch, self.s1_patch)
        ctx = S2Ctx(layout=lay, q_seqlens=[geo.N_REG + n for n in lay.n], fwd_mask=fwd_mask, rev_mask=rev_mask,
                    x_prior=x_prior, rec_prior=rec_prior, scene_g=SCENE_G, n_views=v_input,
                    scene_block=self.scene_block, s_pad=s_pad, dense_backend=self.dense_backend,
                    masked_backend=self.masked_backend, structure=self.structure,
                    s_seqlens=fgs.seq if fgs is not None else None,
                    scene_idx=fgs.idx if fgs is not None else None, win=win)
        return ctx, mask_in, fgs, sel_t

    # -------------------------------------------------------------------------------------------
    def trainable_state_keys(self):
        from utils.training_utils import trainable_state_keys
        return trainable_state_keys(self)

    def load_ckpt(self, load_path):
        """inference entry point: stage-2 weights only; stage 1 comes from model.stage2.stage1_ckpt."""
        import os
        if os.path.isdir(load_path):
            names = sorted(f for f in os.listdir(load_path) if f.endswith(".pt"))
            if not names:
                raise FileNotFoundError(f"no checkpoint in {load_path}")
            load_path = os.path.join(load_path, names[-1])
        ck = torch.load(load_path, map_location="cpu", weights_only=True)
        missing, unexpected = self.load_state_dict(ck["model"], strict=False)
        bad = [k for k in missing if not k.startswith("loss_computer.")]
        if bad or unexpected:
            raise RuntimeError(f"[stage2] {load_path} does not match the model: missing {bad[:8]} "
                               f"unexpected {list(unexpected)[:8]}")
        print(f"[stage2] loaded {load_path} (step {ck.get('fwdbwd_pass_step')})")
        return 0
