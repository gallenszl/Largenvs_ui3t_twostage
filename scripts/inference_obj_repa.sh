#!/bin/bash
#SBATCH --job-name=rng448-fa3-infer-repa
#SBATCH --partition=gpu
#SBATCH --qos=lowest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --exclude=lrc-alpha-sg-gpu05
#SBATCH --output=/home/z50057756/code/RnG_feature_allignment/slurm_logs/%x-%j.out
#SBATCH --error=/home/z50057756/code/RnG_feature_allignment/slurm_logs/%x-%j.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" && "${RUN_LOCAL:-0}" != "1" ]]; then
    exec sbatch "$0" "$@"
fi

source /home/z50057756/miniconda3/etc/profile.d/conda.sh
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"   # satisfy conda env's cuda-nvcc activate hook under set -u
conda activate /home/z50057756/conda/envs/rng-fa3

export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RNG_ATTENTION_BACKEND=auto
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1

ulimit -c 0

mkdir -p /home/z50057756/code/RnG_feature_allignment/slurm_logs

cd /home/z50057756/code/RnG_feature_allignment
mkdir -p ./experiments/evaluation

CONFIG_PATH="${CONFIG_PATH:-configs/RnGUP_obj_448_bf16_15k.yaml}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./experiments/checkpoints/FA3_repa_dinov3_l7}"
INFERENCE_OUT_DIR="${INFERENCE_OUT_DIR:-./experiments/evaluation/FA3_repa_dinov3_l7}"
SPLIT_FILE="${SPLIT_FILE:-data/gso.txt}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}"

echo "Job: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Config: ${CONFIG_PATH}"
echo "Checkpoint dir: ${CHECKPOINT_DIR}"
echo "Inference out dir: ${INFERENCE_OUT_DIR}"
echo "Split file: ${SPLIT_FILE}"
echo "NPROC_PER_NODE: ${NPROC_PER_NODE}"
nvidia-smi

python - <<'PY'
import os
import torch

print("python:", os.sys.executable)
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("gpu_count:", torch.cuda.device_count())
for idx in range(torch.cuda.device_count()):
    print(f"gpu[{idx}]:", torch.cuda.get_device_name(idx))
try:
    import flash_attn_interface  # noqa: F401
    print("flash_attn_interface: OK")
except Exception as exc:
    print("flash_attn_interface:", repr(exc))
try:
    import xformers  # noqa: F401
    print("xformers: OK")
except Exception as exc:
    print("xformers:", repr(exc))
print("RNG_ATTENTION_BACKEND:", os.environ.get("RNG_ATTENTION_BACKEND"))
PY

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    inference.py \
    --config "${CONFIG_PATH}" \
    training.checkpoint_dir="${CHECKPOINT_DIR}" \
    inference_out_dir="${INFERENCE_OUT_DIR}" \
    training.val_dataset_cfgs.split_file="${SPLIT_FILE}" \
    training.target_has_input=false \
    training.val_dataset_cfgs.training.target_has_input=false \
    training.val_dataset_cfgs.training.num_views=14 \
    training.val_dataset_cfgs.training.num_input_views=4 \
    training.val_dataset_cfgs.training.num_target_views=10
