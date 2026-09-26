"""UNI3T GPU self-check: build the real three-task model and run one synthetic step.

Everything here is synthetic, so it needs no dataset and no checkpoint. It verifies
what the CPU unit tests cannot: that the whole forward wires up, that the pretrained
camera head landed, that every returned tensor has the shape export_results expects,
and that all three loss channels produce finite numbers.

    srun -p gpu --qos=lowest --gres=gpu:h200:1 ... python tools/uni3t_selfcheck.py <config>
"""

import sys
from pathlib import Path

import torch
from easydict import EasyDict as edict
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.lagernvs_wrapper import LagerNVSInRnG  # noqa: E402


def load_config(path):
    cfg = OmegaConf.load(path)
    return edict(OmegaConf.to_container(cfg, resolve=True))


def synthetic_batch(cfg, device, batch=1):
    n_views = cfg.training.num_views
    hw = cfg.model.image_tokenizer.image_size
    g = torch.Generator().manual_seed(0)

    # Cameras on a ring around the origin, first one at the RnG canonical pose.
    c2w = torch.eye(4).view(1, 1, 4, 4).repeat(batch, n_views, 1, 1)
    ang = torch.linspace(0, 1.2, n_views)
    c2w[:, :, 0, 3] = torch.sin(ang) * 0.3
    c2w[:, :, 1, 3] = torch.cos(ang) * 0.1
    c2w[:, :, 2, 3] = -1.0
    f = 0.9 * hw
    fxfycxcy = torch.tensor([f, f, hw / 2, hw / 2]).view(1, 1, 4).repeat(batch, n_views, 1)
    intrinsic = torch.eye(3).view(1, 1, 3, 3).repeat(batch, n_views, 1, 1)
    intrinsic[:, :, 0, 0] = f
    intrinsic[:, :, 1, 1] = f
    intrinsic[:, :, 0, 2] = hw / 2
    intrinsic[:, :, 1, 2] = hw / 2

    point_map = torch.randn(batch, n_views, 3, hw, hw, generator=g) * 0.2
    depth_map = point_map[:, :, 2].abs() + 0.5

    batch_dict = {
        "image": torch.rand(batch, n_views, 3, hw, hw, generator=g),
        "c2w": c2w,
        "extrinsic": torch.inverse(c2w),
        "fxfycxcy": fxfycxcy,
        "intrinsic": intrinsic,
        "index": torch.arange(n_views).view(1, n_views, 1).repeat(batch, 1, 2),
        "scene_name": ["synthetic"] * batch,
        "depth_map": depth_map,
        "point_map": point_map,
        "alpha_mask": (torch.rand(batch, n_views, 1, hw, hw, generator=g) > 0.3).float(),
    }
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch_dict.items()}


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else (
        "configs/RnGUP_lagernvs_uni3t_b32t6_fp32lr35_const_90k_all287k.yaml"
    )
    cfg = load_config(cfg_path)
    device = "cuda"
    print(f"[uni3t] config: {cfg_path}")
    print(f"[uni3t] unified_heads: {cfg.model.get('unified_heads')}")
    print(f"[uni3t] weight_camera={cfg.training.weight_camera} "
          f"weight_point={cfg.training.weight_point} "
          f"cam_cond_zero_p={cfg.training.cam_cond_zero_p}")

    model = LagerNVSInRnG(cfg).to(device)
    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cam = sum(p.numel() for p in model.camera_head.parameters())
    pt = sum(p.numel() for p in model.point_head.parameters())
    print(f"[uni3t] trainable params: {trainable:,}")
    print(f"[uni3t]   camera_head: {cam:,}   point_head: {pt:,}   new total: {cam + pt:,}")
    assert cam == 216_174_610, cam
    assert pt == 28_719_632, pt

    # The pretrained camera head must not be a fresh random init.
    w = model.camera_head.trunk[0].attn.qkv.weight
    print(f"[uni3t] camera_head.trunk[0].attn.qkv.weight  std={w.std():.6f} "
          f"absmax={w.abs().max():.6f}  (random init would be ~0.02 std)")

    batch = synthetic_batch(cfg, device)
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(batch, exclude_bg=True)

    print(f"[uni3t] render : {tuple(out.render.shape)}  dtype={out.render.dtype}")
    print(f"[uni3t] points : {tuple(out.points.shape)}  dtype={out.points.dtype}")
    print(f"[uni3t] camera : list of {len(out.camera)}, last {tuple(out.camera[-1].shape)}")
    b = batch["image"].shape[0]
    v_t = cfg.training.num_target_views
    v_i = cfg.training.num_input_views
    hw = cfg.model.image_tokenizer.image_size
    assert tuple(out.points.shape) == (b, v_t, 3, hw, hw), out.points.shape
    assert tuple(out.camera[-1].shape) == (b * v_t, v_i, 9), out.camera[-1].shape

    print("[uni3t] loss channels:")
    for k, val in out.loss_metrics.items():
        fv = float(val)
        flag = "" if fv == fv and abs(fv) != float("inf") else "   <-- NOT FINITE"
        print(f"    {k:22s} {fv:.6f}{flag}")
        assert fv == fv and abs(fv) != float("inf"), k
    for need in ("loss_camera", "loss_T", "loss_R", "loss_FL",
                 "loss_conf_point", "loss_reg_point", "loss_grad_point"):
        assert need in out.loss_metrics, f"missing loss key {need}"

    out.loss_metrics.loss.backward()
    no_grad = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    print(f"[uni3t] params with requires_grad but no grad after backward: {len(no_grad)}")
    if no_grad:
        for n in no_grad[:20]:
            print(f"    {n}")
    # DDP runs with find_unused_parameters=False, so this must be empty.
    assert not no_grad, "these would make DDP's reducer abort"

    # A zeros .grad tensor is not the same as having a gradient. The first version of
    # this script only checked for None and happily passed a point head whose output
    # conv was zero-initialised and therefore permanently frozen.
    print("[uni3t] gradient norms of the new heads (all must be > 0):")
    for name, mod in (("camera_head", model.camera_head), ("point_head", model.point_head)):
        tot = sum(float(p.grad.norm()) ** 2 for p in mod.parameters() if p.grad is not None) ** 0.5
        dead = [n for n, p in mod.named_parameters()
                if p.requires_grad and (p.grad is None or float(p.grad.abs().max()) == 0.0)]
        print(f"    {name:12s} ||grad||={tot:.4e}   all-zero-grad tensors: {len(dead)}")
        if dead:
            for n in dead[:10]:
                print(f"        {n}")
        assert tot > 0.0, f"{name} received no gradient at all"
    # The output conv of the point head is the one the zero-init bug killed.
    g = model.point_head.proj.weight.grad
    print(f"[uni3t] point_head.proj.weight grad: |max|={float(g.abs().max()):.4e} "
          f"||.||={float(g.norm()):.4e}")
    assert float(g.abs().max()) > 0.0, "point_head.proj got an all-zero gradient"

    # ------------------------------------------------------------------ dtype
    # The gradient checks above cannot see a parameter whose *update* is thrown
    # away by its storage dtype: its gradient is non-zero and .grad is not None,
    # yet AdamW moves it by nothing. That is how the inherited bf16
    # per_view_register_tokens went unnoticed -- bf16 keeps 8 significant bits, so
    # once |p| > lr / (0.5 * 2**-7) = 256*lr the update rounds back to where it was.
    bad_dtype = [(n, p.dtype) for n, p in model.named_parameters()
                 if p.requires_grad and p.dtype is not torch.float32]
    print(f"[uni3t] trainable params not in fp32: {len(bad_dtype)}")
    for n, dt in bad_dtype[:20]:
        print(f"    {n}  {dt}")
    assert not bad_dtype, "a trainable parameter is not fp32; its AdamW updates may be rounded away"

    # -------------------------------------------------------- movement probe
    # Build the SAME optimizer training builds (so the parameter grouping is the
    # thing under test too), feed a constant same-sign gradient for 20 steps, and
    # measure how far each parameter actually moved. AdamW's per-coordinate step
    # is about lr under a coherent gradient, so
    #     ratio = mean |p_20 - p_0| / (20 * lr)
    # should land near 1.0. A parameter left out of every group, or in a group with
    # lr 0, or whose dtype eats the update, reads 0.
    from utils.training_utils import create_optimizer  # noqa: E402
    lr = float(cfg.training.lr)
    probe_opt, probe_params, _ = create_optimizer(
        model,
        cfg.training.weight_decay,
        lr,
        (cfg.training.beta1, cfg.training.beta2),
        decoder_lr=cfg.training.get("decoder_lr", None),
        decoder_weight_decay=cfg.training.get("decoder_weight_decay", None),
    )
    n_in_groups = sum(len(g["params"]) for g in probe_opt.param_groups)
    print(f"[uni3t] movement probe: {len(probe_params)} trainable tensors, "
          f"{n_in_groups} of them in an optimizer group, lr={lr:.2e}")
    assert n_in_groups == len(probe_params), "a trainable tensor is in no optimizer group"

    watch = {
        "renderer.per_view_register_tokens": model.model.renderer.per_view_register_tokens,
        "point_head.proj.weight": model.point_head.proj.weight,
        "camera_head.empty_pose_tokens": model.camera_head.empty_pose_tokens,
    }
    before = {k: v.detach().clone().float() for k, v in watch.items()}
    n_probe_steps = 20
    with torch.no_grad():
        for _ in range(n_probe_steps):
            for prm in probe_params.values():
                prm.grad = torch.full_like(prm, 1.0, dtype=prm.dtype)
            probe_opt.step()
    print(f"[uni3t] after {n_probe_steps} constant-gradient AdamW steps "
          f"(ratio = mean|dp| / ({n_probe_steps} * lr), expect ~0.5-1.0):")
    stalled = []
    for k, v in watch.items():
        dp = (v.detach().float() - before[k]).abs().mean().item()
        ratio = dp / (n_probe_steps * lr)
        print(f"    {k:42s} mean|dp|={dp:.3e}  ratio={ratio:.3f}")
        if ratio < 0.05:
            stalled.append(k)
    assert not stalled, f"these parameters barely moved under a coherent gradient: {stalled}"
    # The probe has scribbled on the weights and allocated Adam state; nothing below
    # reads the weights again, but drop the state so the reported peak stays honest.
    del probe_opt
    torch.cuda.empty_cache()

    # pts_to_depth / _save_metrics_depth path
    from utils.metric_utils import pts_to_depth
    pts_to_depth(out)
    print(f"[uni3t] derived depth: {tuple(out.depth.shape)} "
          f"(GT {tuple(out.target.depth_map.shape)})")

    print(f"[uni3t] peak GPU mem: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB "
          f"(batch={b}, this is NOT the training peak)")
    print("[uni3t] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
