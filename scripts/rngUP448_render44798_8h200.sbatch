#!/bin/bash
#SBATCH --job-name=rng448-render44798-8h200
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=208
#SBATCH --mem=1500G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=lrc-alpha-sg-gpu05
#SBATCH --output=/home/z50057756/code/RnG_feature_allignment/slurm_logs/%x-%j.out
#SBATCH --error=/home/z50057756/code/RnG_feature_allignment/slurm_logs/%x-%j.err

# Phase 1a: full 44,798 obj × 40 view training run with all sim2real patches enabled
# (A1/A2/A3/A4 in render side already baked into data, R1/R2/exclude_bg via P1-P5
# dataloader patches in this branch render44798-dataset-patches).
#
# Plan ref: ~/.claude/plans/trallis-sota-3d-generaton-clever-manatee.md §8.1

set -euo pipefail

source /home/z50057756/miniconda3/etc/profile.d/conda.sh
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"   # satisfy conda env's cuda-nvcc activate hook under set -u
conda activate /home/z50057756/conda/envs/rng-fa3

export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RNG_ATTENTION_BACKEND=auto
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

ulimit -c 0

mkdir -p /home/z50057756/code/RnG_feature_allignment/slurm_logs

cd /home/z50057756/code/RnG_feature_allignment

echo "Job ID: ${SLURM_JOB_ID}"
echo "Restart count: ${SLURM_RESTART_COUNT:-0}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Git branch: $(git branch --show-current)"
echo "Git HEAD: $(git rev-parse --short HEAD)"
nvidia-smi

python - <<'PY'
import importlib.util
import torch

print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("gpu_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu_name", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
for name in ("flash_attn_interface", "xformers"):
    print(f"{name}_available", importlib.util.find_spec(name) is not None)
PY

# Phase 1a training: 44,798 × 40v with P1-P5 dataloader patches (sim2real A1-A4 + R1/R2).
# All settings (use_tar / tar_root_path / roll_augment_max_deg / num_views=40 /
# total_frames=40 / feature_alignment.enabled=true / layer_idx=7) are inside the
# config yaml — no cmdline override needed.
torchrun --nproc_per_node=8 --nnodes=1 \
  --rdzv_id=${SLURM_JOB_ID} --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:29513 \
  train.py --config configs/RnGUP_obj_448_bf16_15k_render44798.yaml
