# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import builtins
import math
import torch
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
import torch.distributed as dist
import os
from rich import print
import traceback
from torch.nn.parallel import DistributedDataParallel as DDP


def print_rank0(*args, **kwargs):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)


def format_number(num):
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    return str(num)

def create_optimizer(model, weight_decay, learning_rate, betas, decoder_lr=None, decoder_weight_decay=None, fused=False):
    # start with all of the candidate parameters
    all_param_dict = {name: param for name, param in model.named_parameters()}
    # filter out those that do not require grad
    optimized_param_dict = {name: param for name, param in all_param_dict.items() if param.requires_grad}

    # Separate decoder params from other params for differential learning rate
    _decoder_key = 'rgb_head.rae_decoder'
    use_decoder_lr = decoder_lr is not None
    _decoder_wd = decoder_weight_decay if decoder_weight_decay is not None else weight_decay

    decay_params, nodecay_params = [], []
    decoder_decay_params, decoder_nodecay_params = [], []

    for name, param in optimized_param_dict.items():
        is_decoder = _decoder_key in name
        is_nodecay = param.dim() == 1 or getattr(param, '_no_weight_decay', False)

        if is_decoder and use_decoder_lr:
            if is_nodecay:
                decoder_nodecay_params.append(param)
            else:
                decoder_decay_params.append(param)
        else:
            if is_nodecay:
                nodecay_params.append(param)
            else:
                decay_params.append(param)

    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay, 'is_decoder': False},
        {'params': nodecay_params, 'weight_decay': 0.0, 'is_decoder': False},
    ]
    if use_decoder_lr:
        if decoder_decay_params:
            optim_groups.append({'params': decoder_decay_params, 'weight_decay': _decoder_wd, 'lr': decoder_lr, 'is_decoder': True})
        if decoder_nodecay_params:
            optim_groups.append({'params': decoder_nodecay_params, 'weight_decay': 0.0, 'lr': decoder_lr, 'is_decoder': True})

    # fused=True matches native lagernvs (single-kernel step; mathematically identical)
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=fused)

    # Print Model Information
    if dist.get_rank() == 0:
        def get_module_name(name):
            parts = name.split('.')
            if len(parts) > 2 and parts[0] == 'module':
                return parts[1] + '.' + parts[2]
            return parts[0]  # Fallback to first part if no 'module.' prefix
        print(f'Optimizer: AdamW, learning rate: {learning_rate}, weight decay: {weight_decay}, betas: {betas}')
        if use_decoder_lr:
            decoder_param_count = sum(p.numel() for p in decoder_decay_params + decoder_nodecay_params)
            print(f'  Decoder lr: {decoder_lr}, decoder weight_decay: {_decoder_wd}, decoder params: {format_number(decoder_param_count)}')
        # Number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in optimized_param_dict.values())
        optim_module_names = sorted(set(get_module_name(name) for name in optimized_param_dict.keys()))
        frozen_module_names = sorted(set(get_module_name(name) for name in set(all_param_dict.keys()) - set(optimized_param_dict.keys())))

        print(f'Total parameters: {format_number(total_params)}, Trainable parameters: {format_number(trainable_params)}')
        print(f'Optimized parameters: {optim_module_names}')
        print(f'Frozen parameters: {frozen_module_names}')

    return optimizer, optimized_param_dict, all_param_dict


# ---------------------------------------------------------------------------
# Grouped gradient clipping. Ported verbatim from RnG_fa3_repro_bench
# (utils/training_utils.py, branch FA3_repro_final @ dc1ae2f), where it shipped as
# the M1 arm FA3_repro_l7_b32_clipgrp. Gated by `training.clip_group_prefixes`;
# with the key absent the historical single global clip runs unchanged.
# ---------------------------------------------------------------------------
def best_effort_write(what, step, fn, *args, **kwargs):
    """Run a write the run can survive without (debug images, validation exports).

    On OSError -- disk full, quota, a transient filesystem error -- print one line and
    return False instead of letting the exception kill the rank. Anything that is not
    an OSError is a real bug and still propagates. Returns True when fn completed.
    """
    try:
        fn(*args, **kwargs)
        return True
    except OSError as e:
        # builtins.print, not rich: rich parses "[...]" as markup, strips the "[io]" tag
        # and raises MarkupError on text like "[/x]" -- which would re-kill the rank
        # from inside this handler.
        builtins.print(f"[io] WARNING: {what} failed at step {step}: {e}; training continues", flush=True)
        return False


def build_clip_groups(optimized_param_dict, group_prefixes):
    """Partition the optimized params into clipping groups by name prefix.

    group_prefixes: list of str; each prefix (matched against the param name with a
    leading 'module.' stripped) forms its own group, all remaining params form the
    last group ('rest'). Returns list of (group_name, [params]). Every optimized
    param lands in exactly one group, so the global norm is recoverable as the
    root-sum-square of the group norms.
    """
    groups = [(pre, []) for pre in group_prefixes] + [("rest", [])]
    for name, p in optimized_param_dict.items():
        n = name[len("module."):] if name.startswith("module.") else name
        for gi, pre in enumerate(group_prefixes):
            if n.startswith(pre):
                groups[gi][1].append(p)
                break
        else:
            groups[-1][1].append(p)
    return [(g, ps) for g, ps in groups if len(ps) > 0]


def clip_grad_norm_grouped(clip_groups, max_norm):
    """Clip each group to max_norm independently (torch clip_grad_norm_ per group).

    Returns (global_pre_clip_norm, {group_name: pre_clip_norm}). The global norm is
    the root-sum-square of the group norms, i.e. exactly what a single global
    clip_grad_norm_ over the union would have reported.
    """
    per_group = {}
    for g, ps in clip_groups:
        per_group[g] = torch.nn.utils.clip_grad_norm_(ps, max_norm=max_norm).item()
    total = math.sqrt(sum(v * v for v in per_group.values()))
    return total, per_group


def create_lr_scheduler(optimizer, param_update_steps, warm_up_steps, scheduler_type='cosine', anneal_from_step=None):
    if scheduler_type == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, warm_up_steps, param_update_steps)
    elif scheduler_type == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(optimizer, warm_up_steps, param_update_steps)
    elif scheduler_type == 'constant':
        scheduler = get_constant_schedule_with_warmup(optimizer, warm_up_steps)
    elif scheduler_type == 'constant_anneal':
        # WSD branch-anneal: warmup -> constant -> (1-sqrt) cooldown to 0 over
        # (anneal_from_step, param_update_steps]  (Hägele et al., arXiv:2405.18392)
        if anneal_from_step is None or not (warm_up_steps <= anneal_from_step < param_update_steps):
            raise ValueError(
                f'constant_anneal requires warmup <= anneal_from_step < train_steps, '
                f'got anneal_from_step={anneal_from_step}, warmup={warm_up_steps}, total={param_update_steps}')
        def _wsd_branch_lambda(step, _w=warm_up_steps, _a=anneal_from_step, _t=param_update_steps):
            if step < _w:
                return float(step) / float(max(1, _w))
            if step < _a:
                return 1.0
            prog = min(1.0, (step - _a) / float(_t - _a))
            return max(0.0, 1.0 - math.sqrt(prog))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _wsd_branch_lambda)
    else:
        raise ValueError(f'Invalid scheduler type: {scheduler_type}')
    return scheduler



def find_checkpoints(load_path):
    if os.path.isdir(load_path):
        ckpt_names = [file_name for file_name in os.listdir(load_path) if file_name.endswith(".pt")]
        ckpt_names = sorted(ckpt_names, key=lambda x: x)
        ckpt_paths = [os.path.join(load_path, ckpt_name) for ckpt_name in ckpt_names]
    else:
        if load_path.endswith(".pt"):
            ckpt_paths = [load_path]
        else:
            ckpt_paths = []
    return ckpt_paths



def prune_checkpoints(ckpt_dir, keep_latest, keep_steps=(), protect=None):
    """Delete ckpt_<16 digits>.pt files except the newest keep_latest, the steps in keep_steps and protect.

    Only exact trainer checkpoint names are touched (no .tmp, no other files); every removal is logged.
    Returns the removed file names.
    """
    import re
    pat = re.compile(r"^ckpt_(\d{16})\.pt$")
    items = sorted((int(m.group(1)), f) for f in os.listdir(ckpt_dir) for m in [pat.match(f)] if m)
    keep = {st for st, _ in items[-int(keep_latest):]} if int(keep_latest) > 0 else set()
    keep |= {int(x) for x in (keep_steps or [])}
    removed = []
    for st, f in items:
        path = os.path.join(ckpt_dir, f)
        if st in keep or (protect is not None and os.path.abspath(path) == os.path.abspath(protect)):
            continue
        os.remove(path)
        removed.append(f)
        builtins.print(f"[ckpt] pruned {path}", flush=True)
    return removed


def trainable_state_keys(model):
    """state_dict keys a trainable-only (stage-2) checkpoint carries: every name of a trainable parameter and
    every persistent buffer that is not inside a module whose parameters are all frozen.  Names come from
    named_parameters(remove_duplicate=False): a module registered twice (the perceptual VGG is reachable as
    `vgg.features.*` and as `blocks.*`) appears under both names in state_dict, and the de-duplicated
    named_parameters() would miss the second one."""
    pnames = dict(model.named_parameters(remove_duplicate=False))
    keep = {n for n, p in pnames.items() if p.requires_grad}
    frozen_prefixes = []
    for mn, mod in model.named_modules(remove_duplicate=False):
        ps = list(mod.parameters())
        if mn and ps and not any(p.requires_grad for p in ps):
            frozen_prefixes.append(mn + ".")
    for k in model.state_dict().keys():
        if k not in pnames and not any(k.startswith(f) for f in frozen_prefixes):
            keep.add(k)
    return keep


def auto_resume_job(
    load_path,
    model,
    optimizer,
    lr_scheduler,
    reset_training_state,
    override_lr=None,
    fail_closed=False,
):
    """
    Resume training from the latest checkpoint in the specified directory.
    Returns the fwdbwd_pass_step and param_update_step.

    Args:
        load_path: If dir, load the last checkpoint in the directory.
            O.w., assume it's a ckpt and load it.
        model: model to be loaded
        optimizer: optimizer to be loaded
        lr_scheduler: lr scheduler to be loaded
        reset_training_state: whether to reset the training state

    Returns:
        optimizer, lr_scheduler, forward_pass_step, param_update_step

    fail_closed (stage 2): once a checkpoint exists, never fall back silently -- the newest checkpoint must
    load, its model keys must match (missing only frozen parameters, nothing unexpected), and the optimizer
    and lr_scheduler must restore; otherwise raise.
    """
    forward_pass_step = 0
    param_update_step = 0
    all_ckpt_paths = find_checkpoints(load_path)
    if len(all_ckpt_paths) == 0:
        print_rank0(f"No checkpoint found in {load_path}, we will start from scratch")
        return optimizer, lr_scheduler, forward_pass_step, param_update_step
    # A truncated newest ckpt (e.g. killed mid-write on disk quota) must not
    # silently discard the run: fall back to the next-older ckpt, newest first.
    checkpoint = None
    for ckpt_path in reversed(all_ckpt_paths):
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            break
        except Exception as _e:
            if fail_closed:
                raise RuntimeError(f"[resume] newest checkpoint {ckpt_path} does not load: {_e}") from _e
            traceback.print_exc()
            print_rank0(f"Failed to load {ckpt_path}, trying next-older checkpoint")
    if checkpoint is None:
        print_rank0(f"All checkpoints in {load_path} unloadable, we will start from scratch")
        return optimizer, lr_scheduler, forward_pass_step, param_update_step

    # Load model weights
    _m = model.module if isinstance(model, DDP) else model
    status = _m.load_state_dict(checkpoint['model'], strict=False)
    print_rank0(f"Loaded model from {os.path.abspath(ckpt_path)}, the status is {status}")
    if fail_closed:
        _keep = trainable_state_keys(_m)
        _bad_missing = [k for k in status.missing_keys if k in _keep]
        if _bad_missing or status.unexpected_keys:
            raise RuntimeError(f"[resume] {ckpt_path} does not match the model: missing trainable "
                               f"{_bad_missing[:8]} unexpected {list(status.unexpected_keys)[:8]}")

    # resume training state
    if not reset_training_state:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
            if fail_closed:
                # torch only checks group / parameter counts; a state from another model loads silently
                for _g in optimizer.param_groups:
                    for _p in _g["params"]:
                        for _k, _v in optimizer.state.get(_p, {}).items():
                            if torch.is_tensor(_v) and _v.dim() > 0 and _v.shape != _p.shape:
                                raise ValueError(f"optimizer state '{_k}' has shape {tuple(_v.shape)} "
                                                 f"for a parameter of shape {tuple(_p.shape)}")
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            forward_pass_step = checkpoint["fwdbwd_pass_step"]
            param_update_step = checkpoint["param_update_step"]
            print_rank0(f"Resumed optimizer and lr_scheduler from {ckpt_path}")
            # Branch-with-new-peak support: loaded state carries the trunk's lr /
            # base_lrs, which would silently override the config peak.
            if override_lr is not None:
                for g in optimizer.param_groups:
                    g["lr"] = override_lr
                    g["initial_lr"] = override_lr
                lr_scheduler.base_lrs = [override_lr] * len(lr_scheduler.base_lrs)
                print_rank0(f"Overrode resumed peak lr/base_lrs to {override_lr}")
        except Exception as _e:
            if fail_closed:
                raise RuntimeError(f"[resume] optimizer/lr_scheduler state of {ckpt_path} does not restore: {_e}") from _e
            traceback.print_exc()
            print_rank0(f"Failed to load optimizer and lr_scheduler from {ckpt_path}")
    
    return optimizer, lr_scheduler, forward_pass_step, param_update_step


