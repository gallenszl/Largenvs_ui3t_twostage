# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import builtins
import contextlib
import csv
import importlib
import json
import os
import socket
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from setup import init_config, init_distributed


def _pop_arg(argv, name, default=None):
    for idx, arg in enumerate(list(argv)):
        if arg == name:
            if idx + 1 >= len(argv):
                raise ValueError(f"{name} requires a value")
            value = argv[idx + 1]
            del argv[idx:idx + 2]
            return value
        if arg.startswith(f"{name}="):
            value = arg.split("=", 1)[1]
            del argv[idx]
            return value
    return default


argv = sys.argv[:]
BENCHMARK_LABEL = _pop_arg(argv, "--benchmark-label", "kv_cache")
BENCHMARK_OUTPUT_DIR = Path(_pop_arg(argv, "--benchmark-output-dir", "./latency_benchmark"))
BENCHMARK_WARMUP = int(_pop_arg(argv, "--benchmark-warmup", "3"))
BENCHMARK_SAMPLES = int(_pop_arg(argv, "--benchmark-samples", "20"))
sys.argv = argv

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


@contextlib.contextmanager
def _suppress_model_debug_prints():
    original_print = builtins.print
    builtins.print = lambda *args, **kwargs: None
    try:
        yield
    finally:
        builtins.print = original_print


def _to_device(batch, device):
    return {k: v.to(device) if type(v) == torch.Tensor else v for k, v in batch.items()}


def _synchronize():
    torch.cuda.synchronize()


def _measure(callable_obj):
    _synchronize()
    start = time.perf_counter()
    with _suppress_model_debug_prints():
        result = callable_obj()
    _synchronize()
    return (time.perf_counter() - start) * 1000.0, result


def _percentile(values, percentile):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _stats(values):
    values = [float(value) for value in values]
    return {
        "mean_ms": sum(values) / len(values),
        "p50_ms": _percentile(values, 0.50),
        "p90_ms": _percentile(values, 0.90),
    }


def _flatten(rows, key):
    if key == "new_view_ms":
        values = []
        for row in rows:
            values.extend(row[key])
        return values
    return [row[key] for row in rows]


def _extract_scene_name(batch):
    for key in ("scene_name", "scene_names", "name"):
        if key not in batch:
            continue
        value = batch[key]
        if isinstance(value, (list, tuple)) and value:
            return str(value[0])
        return str(value)
    return ""


def _benchmark_batch(model_module, batch):
    input_data, target_data = model_module.process_val_data(
        batch,
        has_target_image=True,
        target_has_input=False,
        compute_rays=True,
    )

    cache_build_ms, _ = _measure(lambda: model_module.forward_pose_only(input_data.image, return_all=True))
    num_target_views = int(target_data.ray_o.shape[1])
    if num_target_views < 2:
        raise ValueError(f"Expected at least 2 target views, got {num_target_views}")

    def render_one(view_idx):
        target_pose_cond = model_module.get_posed_input(
            ray_o=target_data.ray_o[:, view_idx:view_idx + 1],
            ray_d=target_data.ray_d[:, view_idx:view_idx + 1],
        )
        return model_module.forward_rendering_using_kv_cache(target_pose_cond=target_pose_cond)

    first_render_ms, _ = _measure(lambda: render_one(0))
    new_view_ms = []
    for view_idx in range(1, num_target_views):
        elapsed_ms, _ = _measure(lambda idx=view_idx: render_one(idx))
        new_view_ms.append(elapsed_ms)

    return {
        "scene_name": _extract_scene_name(batch),
        "num_target_views": num_target_views,
        "cache_build_ms": cache_build_ms,
        "first_render_ms": first_render_ms,
        "first_total_ms": cache_build_ms + first_render_ms,
        "new_view_ms": new_view_ms,
        "new_view_mean_ms": sum(new_view_ms) / len(new_view_ms),
    }


def _build_summary(rows):
    summary = {}
    for key, prefix in (
        ("cache_build_ms", "cache_build"),
        ("first_render_ms", "first_render"),
        ("first_total_ms", "first_total"),
        ("new_view_ms", "new_view"),
    ):
        stats = _stats(_flatten(rows, key))
        summary[f"{prefix}_mean_ms"] = stats["mean_ms"]
        summary[f"{prefix}_p50_ms"] = stats["p50_ms"]
        summary[f"{prefix}_p90_ms"] = stats["p90_ms"]
    summary["new_view_fps"] = 1000.0 / summary["new_view_mean_ms"]
    return summary


datasampler.set_epoch(0)
model.eval()

total_needed = BENCHMARK_WARMUP + BENCHMARK_SAMPLES
timed_rows = []
processed = 0

with torch.no_grad(), torch.autocast(
    enabled=config.training.use_amp,
    device_type="cuda",
    dtype=amp_dtype_mapping[config.training.amp_dtype],
):
    for batch in dataloader:
        batch = _to_device(batch, ddp_info.device)
        row = _benchmark_batch(model.module, batch)
        if processed >= BENCHMARK_WARMUP:
            row["sample_index"] = processed - BENCHMARK_WARMUP
            timed_rows.append(row)
        processed += 1
        if processed >= total_needed:
            break
    torch.cuda.empty_cache()

if len(timed_rows) != BENCHMARK_SAMPLES:
    raise RuntimeError(
        f"Only collected {len(timed_rows)} timed samples; expected {BENCHMARK_SAMPLES}."
    )

dist.barrier()

if ddp_info.is_main_process:
    BENCHMARK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gpu_name = torch.cuda.get_device_name(ddp_info.local_rank)
    summary = _build_summary(timed_rows)
    payload = {
        "label": BENCHMARK_LABEL,
        "samples": BENCHMARK_SAMPLES,
        "warmup_samples": BENCHMARK_WARMUP,
        "node": socket.gethostname(),
        "gpu": gpu_name,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "config_path": config.get("_config_path", ""),
        "checkpoint_dir": config.training.checkpoint_dir,
        "summary": summary,
        "timings": timed_rows,
    }

    json_path = BENCHMARK_OUTPUT_DIR / f"{BENCHMARK_LABEL}_latency.json"
    csv_path = BENCHMARK_OUTPUT_DIR / f"{BENCHMARK_LABEL}_latency.csv"
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_index",
                "scene_name",
                "num_target_views",
                "cache_build_ms",
                "first_render_ms",
                "first_total_ms",
                "new_view_mean_ms",
            ],
        )
        writer.writeheader()
        for row in timed_rows:
            writer.writerow({
                "sample_index": row["sample_index"],
                "scene_name": row["scene_name"],
                "num_target_views": row["num_target_views"],
                "cache_build_ms": row["cache_build_ms"],
                "first_render_ms": row["first_render_ms"],
                "first_total_ms": row["first_total_ms"],
                "new_view_mean_ms": row["new_view_mean_ms"],
            })

    print(f"Wrote latency JSON: {json_path}")
    print(f"Wrote latency CSV: {csv_path}")
    print(json.dumps(summary, indent=2))

dist.barrier()
dist.destroy_process_group()
