# Stage-2 dynamics probe (plan section 三 "动力学探针"; rule 13: bugs that leave the forward pass intact only show
# up in the optimizer dynamics).  One fixed real training batch, AdamW exactly as train.py builds it
# (create_optimizer: weight decay on >= 2-D parameters, the config betas, fused), gradient clip at the config value,
# constant lr = the config peak, bf16 autocast, --steps steps on the same batch with the same random draws.
# Reports, per parameter group:
#   step/lr   mean |parameter change| / lr.  Adam moves a coordinate by about lr per step while its gradient keeps
#             its sign, so ~1 = the group trains at full speed, ~0 = it gets no gradient, >> 1 = a scale bug;
#   |grad|    gradient norm of the group before clipping;
#   wd        weight decay of the group in the optimizer.
# Gates (exit 1 if any fails): all trainable parameters fp32; loss and gradients finite at every step; the
# zero-initialised tensors (Lin_m weights and biases, the P4 colour head's linear weight) are nonzero after
# step 1 and larger at the end than after step 1; the RGB L2 loss on the batch falls to <= 0.9 x its step-0 value.
#   python tools_s2/s2_dynamics_probe.py --config configs/S2P8_...yaml [--batch 4] [--steps 40]
import argparse
import importlib
import os
import random
import re
import sys
from collections import OrderedDict

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def group_of(name, p4):
    if name.startswith("renderer.out_lin"):
        return "Lin_m (zero init)"
    if ".inj_" in name:
        return "injection"
    if ".gate." in name:
        return "gate"
    if re.search(r"\.r?comp_[kv]\.", name):
        return "compression ResBlock"
    if name.startswith("renderer.tgt_proj"):
        return "target patch embed"
    if name.startswith(("renderer.registers", "renderer.tgt_norm")):
        return "registers / tgt_norm"
    if name.startswith("renderer.blocks"):
        return "blocks (copied)"
    if name.startswith("color_head"):
        return "colour head (zero init)" if p4 else "colour head (copied)"
    if name.startswith("point_head"):
        return "point head"
    return "other"


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=40)
    args = ap.parse_args()
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(29300 + int(os.environ.get("SLURM_JOB_ID", "3")) % 400))
        dist.init_process_group("gloo", rank=0, world_size=1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.chdir(REPO)
    from easydict import EasyDict as edict
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from model_s2.stage2_wrapper import Stage2LagerNVS
    from utils.training_utils import create_optimizer
    cfg = edict(OmegaConf.to_container(OmegaConf.load(args.config), resolve=True))
    cfg.training.batch_size_per_gpu = args.batch
    tr = cfg.training
    p4 = int(cfg.model.stage2.target_patch) != 8

    mod, cls = tr.dataset_name.rsplit(".", 1)
    ds = importlib.import_module(mod).__dict__[cls](cfg)
    seed_all(0)
    batch = next(iter(DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True)))
    batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}

    model = Stage2LagerNVS(cfg).cuda().train()
    opt, _, _ = create_optimizer(model, tr.weight_decay, tr.lr, (tr.beta1, tr.beta2), fused=True)
    lr, clip = float(tr.lr), float(tr.grad_clip_norm)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    wd_of = {id(p): g["weight_decay"] for g in opt.param_groups for p in g["params"]}
    groups = OrderedDict()
    for n, p in named:
        groups.setdefault(group_of(n, p4), []).append((n, p))
    fails = []
    bad_dtype = [n for n, p in named if p.dtype != torch.float32]
    print(f"[dyn] {len(named)} trainable tensors, {sum(p.numel() for _, p in named) / 1e6:.1f}M params, "
          f"non-fp32: {bad_dtype[:5]}", flush=True)
    if bad_dtype:
        fails.append("non-fp32 trainable parameters")
    for g, ps in groups.items():
        wds = sorted({wd_of[id(p)] for _, p in ps})
        print(f"[dyn] group {g:26s} tensors {len(ps):4d} params {sum(p.numel() for _, p in ps) / 1e6:8.2f}M "
              f"wd {wds}", flush=True)
    # the zero-initialised tensors themselves (a group such as the P4 colour head also holds a LayerNorm weight = 1)
    zero_t = [(n, p) for n, p in named if float(p.detach().abs().max()) == 0.0]
    print(f"[dyn] zero-initialised tensors: {[n for n, _ in zero_t]}", flush=True)
    if not zero_t:
        fails.append("no zero-initialised tensor found")
    norm_after = {}
    rows = []
    l2_0 = None
    report_at = {1, 2, 3, 5, 10, 20, args.steps}
    for step in range(1, args.steps + 1):
        before = {id(p): p.detach().clone() for _, p in named}
        seed_all(1)                                   # same posed/unposed and target-has-input draws every step
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(batch)
        loss = out.loss_metrics.loss
        l2 = float(out.loss_metrics.l2_loss)
        if l2_0 is None:
            l2_0 = l2
        if not bool(torch.isfinite(loss)):
            fails.append(f"non-finite loss at step {step}")
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = {g: float(torch.sqrt(sum((p.grad.float() ** 2).sum() for _, p in ps if p.grad is not None)))
                 for g, ps in groups.items()}
        missing_grad = [n for n, p in named if p.grad is None]
        if missing_grad:
            fails.append(f"no gradient for {missing_grad[:3]} at step {step}")
        if not all(np.isfinite(v) for v in gnorm.values()):
            fails.append(f"non-finite gradient at step {step}")
            break
        total = float(torch.nn.utils.clip_grad_norm_([p for _, p in named], clip))
        opt.step()
        if step in report_at:
            for g, ps in groups.items():
                num = sum(float((p.detach() - before[id(p)]).abs().sum()) for _, p in ps)
                cnt = sum(p.numel() for _, p in ps)
                wn = float(torch.sqrt(sum((p.detach().float() ** 2).sum() for _, p in ps)))
                rows.append((step, g, num / cnt / lr, gnorm[g], wn))
            print(f"[dyn] step {step:3d} loss {float(loss):+.5f} l2 {l2:.6f} grad-norm {total:.3f} "
                  f"(clip {clip})", flush=True)
        for n, p in zero_t:
            norm_after.setdefault(n, {})[step] = float(p.detach().float().norm())
        del before
    print(f"[dyn] {'step':>4s} {'group':26s} {'step/lr':>9s} {'|grad|':>10s} {'|theta|':>10s}", flush=True)
    for step, g, r, gn, wn in rows:
        print(f"[dyn] {step:4d} {g:26s} {r:9.3f} {gn:10.3e} {wn:10.3e}", flush=True)
    for n, _ in zero_t:
        last = max(norm_after[n])
        n1, nl = norm_after[n].get(1, 0.0), norm_after[n][last]
        print(f"[dyn] zero-init {n}: |theta| after step 1 {n1:.4e}, after step {last} {nl:.4e}", flush=True)
        if not (n1 > 0 and nl > n1):
            fails.append(f"{n} did not leave zero and grow")
    l2_last = l2
    print(f"[dyn] RGB L2 on the fixed batch: step 0 {l2_0:.6f} -> step {args.steps} {l2_last:.6f} "
          f"(ratio {l2_last / l2_0:.3f}, gate <= 0.9)", flush=True)
    if not l2_last <= 0.9 * l2_0:
        fails.append("RGB L2 did not fall to 0.9x on a fixed batch")
    print(f"[dyn] peak memory {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB", flush=True)
    print(f"[dyn] RESULT {'PASS' if not fails else 'FAIL: ' + '; '.join(fails)}", flush=True)
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
