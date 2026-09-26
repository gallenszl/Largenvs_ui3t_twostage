# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import importlib
import os

import torch
import torch.distributed as dist
from easydict import EasyDict as edict
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from setup import init_config, init_distributed
from utils.metric_utils import (
    export_results,
    summarize_evaluation,
    summarize_evaluation_depth,
    summarize_evaluation_pose,
)


config = init_config()

os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))

ddp_info = init_distributed(seed=777)
dist.barrier()

torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
torch.backends.cudnn.allow_tf32 = config.training.use_tf32
amp_dtype_mapping = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "tf32": torch.float32,
}


dataset_name = config.training.get("val_dataset_name")
module, class_name = dataset_name.rsplit(".", 1)
Dataset = importlib.import_module(module).__dict__[class_name]
dataset = Dataset(config)

datasampler = DistributedSampler(dataset)
dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=config.training.num_workers,
    prefetch_factor=config.training.prefetch_factor,
    persistent_workers=True,
    pin_memory=False,
    drop_last=True,
    sampler=datasampler,
)

dist.barrier()


module, class_name = config.model.class_name.rsplit(".", 1)
LVSM = importlib.import_module(module).__dict__[class_name]
model = LVSM(config, use_kv_cache=True).to(ddp_info.device)

model = model.to(amp_dtype_mapping["bf16"])
model.camera_head.to(torch.float32)
model.point_head.to(torch.float32)
model.rgb_head.to(torch.float32)
model.loss_computer.to(torch.float32)

model = DDP(model, device_ids=[ddp_info.local_rank])
model.module.load_ckpt(config.training.checkpoint_dir)


if ddp_info.is_main_process:
    print(f"Running KV-cache inference; save results to: {config.inference_out_dir}")
    import lpips  # noqa: F401
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)

dist.barrier()


def _to_device(batch, device):
    return {k: v.to(device) if type(v) == torch.Tensor else v for k, v in batch.items()}


def _repeat_camera_for_targets(pose_enc_list, num_target_views):
    return [pose_enc.repeat_interleave(num_target_views, dim=0) for pose_enc in pose_enc_list]


def _forward_kv_cache_batch(model_module, batch):
    input_data, target_data = model_module.process_val_data(
        batch,
        has_target_image=True,
        target_has_input=False,
        compute_rays=True,
    )

    pose_enc_list = model_module.forward_pose_only(input_data.image, return_all=True)
    num_target_views = target_data.ray_o.shape[1]

    rendered_images = []
    target_points = []
    points_conf = []
    for view_idx in range(num_target_views):
        target_pose_cond = model_module.get_posed_input(
            ray_o=target_data.ray_o[:, view_idx:view_idx + 1],
            ray_d=target_data.ray_d[:, view_idx:view_idx + 1],
        )
        render_pack = model_module.forward_rendering_using_kv_cache(
            target_pose_cond=target_pose_cond,
        )
        rendered_images.append(render_pack.render)
        target_points.append(render_pack.points)
        points_conf.append(render_pack.points_conf)

    return edict(
        input=input_data,
        target=target_data,
        render=torch.cat(rendered_images, dim=1),
        points=torch.cat(target_points, dim=1),
        points_conf=torch.cat(points_conf, dim=1),
        camera=_repeat_camera_for_targets(pose_enc_list, num_target_views),
    )


datasampler.set_epoch(0)
model.eval()

with torch.no_grad(), torch.autocast(
    enabled=config.training.use_amp,
    device_type="cuda",
    dtype=amp_dtype_mapping[config.training.amp_dtype],
):
    for batch in dataloader:
        batch = _to_device(batch, ddp_info.device)
        result = _forward_kv_cache_batch(model.module, batch)
        export_results(
            result,
            config.inference_out_dir,
            compute_metrics=config.inference.get("compute_metrics"),
        )
    torch.cuda.empty_cache()


dist.barrier()

if ddp_info.is_main_process and config.inference.get("compute_metrics", False):
    summarize_evaluation(config.inference_out_dir)
    summarize_evaluation_depth(config.inference_out_dir)
    summarize_evaluation_pose(config.inference_out_dir)
    if config.inference.get("generate_website", True):
        os.system(f"python generate_html.py {config.inference_out_dir}")
dist.barrier()
dist.destroy_process_group()
exit(0)
