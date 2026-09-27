# Stage-2 evaluation entry point (plan 2026-09-27, section 一.10 / G6).
#   python -m torch.distributed.run --standalone --nproc_per_node=1 inference_s2.py --config <stage-2 config> \
#       inference.s2_ckpt=<ckpt file | none> inference_out_dir=<dir> [inference.first_n=64] [overrides ...]
# One forward per object writes the stage-2 results to <dir> and the frozen stage-1 results of the same
# forward to <dir>_s1 (paired by construction); both get metrics.json / metrics_depth.json /
# metrics_pose.json + regions.json and the three summaries.  s2_ckpt=none switches stage 2 off
# (model.stage2.enabled=false): <dir> then reproduces stage 1 and must match the stored stage-1 evaluation.

import os

import torch
import torch.distributed as dist
from easydict import EasyDict as edict
from torch.utils.data import DataLoader, DistributedSampler

from setup import init_config, init_distributed
from utils.metric_utils import export_results, summarize_evaluation, summarize_evaluation_depth, summarize_evaluation_pose

config = init_config()
s2_ckpt = str(config.inference.get("s2_ckpt", "none"))
if s2_ckpt == "none":
    config.model.stage2.enabled = False
first_n = config.inference.get("first_n", None)
os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))
ddp_info = init_distributed(seed=777)
torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
torch.backends.cudnn.allow_tf32 = config.training.use_tf32

from data.dataset_gso_ours import GSODataset_ours  # noqa: E402
from model_s2.stage2_wrapper import Stage2LagerNVS  # noqa: E402
from tools_s2.s2_regions import save_region_metrics  # noqa: E402

dataset = GSODataset_ours(config)
if first_n:
    # keep the view draw of the full list (GSODataset_ours seeds its view sampling over the whole list)
    dataset.all_object_list = dataset.all_object_list[: int(first_n)]
sampler = DistributedSampler(dataset, shuffle=False)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.training.num_workers,
                    prefetch_factor=config.training.prefetch_factor, persistent_workers=False, pin_memory=False,
                    drop_last=False, sampler=sampler)

model = Stage2LagerNVS(config).to(ddp_info.device)
if s2_ckpt != "none":
    model.load_ckpt(s2_ckpt)
model.eval()
out_s2 = config.inference_out_dir
out_s1 = out_s2.rstrip("/") + "_s1"
if ddp_info.is_main_process:
    print(f"[s2-eval] s2_ckpt={s2_ckpt} objects={len(dataset)} zero_p={model.val_cam_cond_zero_p} -> {out_s2}", flush=True)
dist.barrier()

amp = dict(enabled=config.training.use_amp, device_type="cuda", dtype=torch.bfloat16)
with torch.no_grad(), torch.autocast(**amp):
    for i, batch in enumerate(loader):
        batch = {k: v.to(ddp_info.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        res = model(batch, target_has_input=False, is_valid=True)
        export_results(res, out_s2, compute_metrics=True)
        save_region_metrics(res, out_s2)
        if model.enabled:
            r1 = edict(input=res.input, target=res.target, render=res.render_s1, points=res.points_s1,
                       camera=res.camera)
            export_results(r1, out_s1, compute_metrics=True)
            save_region_metrics(r1, out_s1)
        if ddp_info.is_main_process and i % 50 == 0:
            print(f"[s2-eval] object {i}", flush=True)
    torch.cuda.empty_cache()
dist.barrier()
if ddp_info.is_main_process:
    for d in ([out_s2, out_s1] if model.enabled else [out_s2]):
        summarize_evaluation(d)
        summarize_evaluation_depth(d)
        summarize_evaluation_pose(d)
    print("[s2-eval] DONE", flush=True)
dist.barrier()
dist.destroy_process_group()
