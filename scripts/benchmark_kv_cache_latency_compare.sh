#!/bin/bash
#SBATCH --job-name=rng-kvcache-latency
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/zs3325/runs/%x-%j.out
#SBATCH --error=/scratch/zs3325/runs/%x-%j.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" && "${RUN_LOCAL:-0}" != "1" ]]; then
    exec sbatch "$0" "$@"
fi

module purge
module load cuda/12.8 miniconda/latest

eval "$(conda shell.bash hook)"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

export CONDA_ENVS_PATH=/scratch/zs3325/conda/envs
export CONDA_PKGS_DIRS=/scratch/zs3325/conda/pkgs
export PIP_CACHE_DIR=/scratch/zs3325/pip-cache
export TMPDIR=/scratch/zs3325/tmp
export HF_HOME=/scratch/zs3325/hf
export WANDB_DIR=/scratch/zs3325/wandb
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1

ulimit -c 0

OUTPUT_DIR="${OUTPUT_DIR:-/home/zs3325/code/RnG-fa3/experiments/evaluation/latency_benchmark}"
WARMUP_SAMPLES="${WARMUP_SAMPLES:-3}"
TIMED_SAMPLES="${TIMED_SAMPLES:-20}"
SPLIT_FILE="${SPLIT_FILE:-data/gso.txt}"

mkdir -p /scratch/zs3325/runs
mkdir -p /scratch/zs3325/tmp
mkdir -p /scratch/zs3325/wandb
mkdir -p "${OUTPUT_DIR}"

echo "Job: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Output dir: ${OUTPUT_DIR}"
echo "Warmup samples: ${WARMUP_SAMPLES}"
echo "Timed samples: ${TIMED_SAMPLES}"
nvidia-smi

run_fa3() {
    echo "=== Running FA3 kv-cache latency benchmark ==="
    conda activate /scratch/zs3325/conda/envs/rng-fa3
    export RNG_ATTENTION_BACKEND=auto
    cd /home/zs3325/code/RnG-fa3
    python - <<'PY'
import os
import torch
print("env:", os.environ.get("CONDA_DEFAULT_ENV"))
print("python:", os.sys.executable)
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("gpu_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("RNG_ATTENTION_BACKEND:", os.environ.get("RNG_ATTENTION_BACKEND"))
PY
    torchrun --standalone --nnodes=1 --nproc_per_node=1 \
        benchmark_kv_cache_latency.py \
        --benchmark-label fa3_kvcache \
        --benchmark-output-dir "${OUTPUT_DIR}" \
        --benchmark-warmup "${WARMUP_SAMPLES}" \
        --benchmark-samples "${TIMED_SAMPLES}" \
        --config configs/RnGUP_obj_448_bf16_15k.yaml \
        training.checkpoint_dir=./experiments/checkpoints/RnGUP_448_freeze_RAEHead_DPTHead_FA3 \
        training.val_dataset_cfgs.split_file="${SPLIT_FILE}" \
        training.target_has_input=false \
        training.val_dataset_cfgs.training.target_has_input=false \
        training.val_dataset_cfgs.training.num_views=14 \
        training.val_dataset_cfgs.training.num_input_views=4 \
        training.val_dataset_cfgs.training.num_target_views=10 \
        inference.compute_metrics=false \
        inference.generate_website=false
}

run_rng() {
    echo "=== Running RnG unfreeze_sft kv-cache latency benchmark ==="
    conda activate /scratch/zs3325/conda/envs/rng
    unset RNG_ATTENTION_BACKEND || true
    cd /home/zs3325/code/RnG
    python - <<'PY'
import os
import torch
print("env:", os.environ.get("CONDA_DEFAULT_ENV"))
print("python:", os.sys.executable)
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("gpu_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("RNG_ATTENTION_BACKEND:", os.environ.get("RNG_ATTENTION_BACKEND"))
PY
    torchrun --standalone --nnodes=1 --nproc_per_node=1 \
        benchmark_kv_cache_latency.py \
        --benchmark-label rng_unfreeze_sft_kvcache \
        --benchmark-output-dir "${OUTPUT_DIR}" \
        --benchmark-warmup "${WARMUP_SAMPLES}" \
        --benchmark-samples "${TIMED_SAMPLES}" \
        --config configs/RnGUP_obj_448_bf16_15k_unfreeze_sft.yaml \
        training.checkpoint_dir=./experiments/checkpoints/RnGUP_448_freeze_RAEHead_DPTHead_unfreeze_sft \
        training.val_dataset_cfgs.split_file="${SPLIT_FILE}" \
        training.target_has_input=false \
        training.val_dataset_cfgs.training.target_has_input=false \
        training.val_dataset_cfgs.training.num_views=14 \
        training.val_dataset_cfgs.training.num_input_views=4 \
        training.val_dataset_cfgs.training.num_target_views=10 \
        inference.compute_metrics=false \
        inference.generate_website=false
}

run_fa3
run_rng

export OUTPUT_DIR
python - <<'PY'
import csv
import json
import os
from pathlib import Path

output_dir = Path(os.environ["OUTPUT_DIR"])
labels = ["fa3_kvcache", "rng_unfreeze_sft_kvcache"]
rows = []
for label in labels:
    path = output_dir / f"{label}_latency.json"
    with path.open() as f:
        payload = json.load(f)
    summary = payload["summary"]
    rows.append({
        "version": payload["label"],
        "cache_build_mean_ms": summary["cache_build_mean_ms"],
        "cache_build_p50_ms": summary["cache_build_p50_ms"],
        "cache_build_p90_ms": summary["cache_build_p90_ms"],
        "first_render_mean_ms": summary["first_render_mean_ms"],
        "first_render_p50_ms": summary["first_render_p50_ms"],
        "first_render_p90_ms": summary["first_render_p90_ms"],
        "first_total_mean_ms": summary["first_total_mean_ms"],
        "first_total_p50_ms": summary["first_total_p50_ms"],
        "first_total_p90_ms": summary["first_total_p90_ms"],
        "new_view_mean_ms": summary["new_view_mean_ms"],
        "new_view_p50_ms": summary["new_view_p50_ms"],
        "new_view_p90_ms": summary["new_view_p90_ms"],
        "new_view_fps": summary["new_view_fps"],
        "samples": payload["samples"],
        "gpu/node/env": f"{payload['gpu']} / {payload['node']} / {payload.get('conda_env', '')}",
    })

csv_path = output_dir / "latency_benchmark_summary.csv"
json_path = output_dir / "latency_benchmark_summary.json"
md_path = output_dir / "latency_benchmark_summary.md"

headers = list(rows[0].keys())
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(rows)

with json_path.open("w") as f:
    json.dump(rows, f, indent=2)

def fmt(value):
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)

md_lines = [
    "# KV-Cache Latency Benchmark",
    "",
    "| " + " | ".join(headers) + " |",
    "| " + " | ".join(["---"] * len(headers)) + " |",
]
for row in rows:
    md_lines.append("| " + " | ".join(fmt(row[h]) for h in headers) + " |")
md_lines.append("")
md_lines.append("Timing excludes model/checkpoint load, dataloader I/O, result export, metrics, and HTML generation.")
md_lines.append("first_total_ms = cache_build_ms + first_render_ms; new_view_ms uses target views 1..9.")
md_path.write_text("\n".join(md_lines) + "\n")

print(md_path.read_text())
print(f"Wrote summary CSV: {csv_path}")
print(f"Wrote summary JSON: {json_path}")
print(f"Wrote summary Markdown: {md_path}")
PY
