#!/bin/bash
#SBATCH --job-name=repa-probe
#SBATCH --partition=gpu
#SBATCH --qos=lowest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --exclude=lrc-alpha-sg-gpu05
#SBATCH --output=/home/z50057756/code/RnG_feature_allignment/slurm_logs/%x-%j.out
#SBATCH --error=/home/z50057756/code/RnG_feature_allignment/slurm_logs/%x-%j.err

# Probe one baseline checkpoint vs DINOv3.
# Required env vars (set by launch_probe_parallel.sh):
#   PROBE_CKPT_PATH   — absolute path to ckpt .pt file
#   PROBE_OUT_CSV     — absolute path to output CSV
#   PROBE_LABEL       — short tag (e.g., "baseline_step2k") for CSV row
# Optional:
#   PROBE_LAYERS      — comma-separated layer indices (default 3,7,12,17,22)
#   PROBE_MAX_OBJECTS — cap val objects (default 0 = all 63)

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" && "${RUN_LOCAL:-0}" != "1" ]]; then
    exec sbatch "$0" "$@"
fi

source /home/z50057756/miniconda3/etc/profile.d/conda.sh
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
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

CONFIG_PATH="${CONFIG_PATH:-configs/RnGUP_obj_448_bf16_15k.yaml}"
SPLIT_FILE="${SPLIT_FILE:-data/gso_subset64.txt}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}"

# Required
: "${PROBE_CKPT_PATH:?PROBE_CKPT_PATH must be set}"
: "${PROBE_OUT_CSV:?PROBE_OUT_CSV must be set}"
PROBE_LABEL="${PROBE_LABEL:-unknown}"
PROBE_LAYERS="${PROBE_LAYERS:-3,7,12,17,22}"
PROBE_MAX_OBJECTS="${PROBE_MAX_OBJECTS:-0}"

export PROBE_CKPT_PATH PROBE_OUT_CSV PROBE_LABEL PROBE_LAYERS PROBE_MAX_OBJECTS

echo "============================================================"
echo "PROBE CONFIG"
echo "  CKPT_PATH : ${PROBE_CKPT_PATH}"
echo "  OUT_CSV   : ${PROBE_OUT_CSV}"
echo "  LABEL     : ${PROBE_LABEL}"
echo "  LAYERS    : ${PROBE_LAYERS}"
echo "  MAX_OBJ   : ${PROBE_MAX_OBJECTS}"
echo "  SPLIT     : ${SPLIT_FILE}"
echo "  NPROC     : ${NPROC_PER_NODE}"
echo "============================================================"
nvidia-smi

# Find an unused port for rendezvous (4 jobs run in parallel; default 29400 collides)
RDZV_PORT="${RDZV_PORT:-$((30000 + RANDOM % 5000))}"
echo "RDZV port = ${RDZV_PORT}"

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --rdzv_endpoint=localhost:${RDZV_PORT} \
    tools/probe_baseline_cossim.py \
    --config "${CONFIG_PATH}" \
    training.val_dataset_cfgs.split_file="${SPLIT_FILE}" \
    training.target_has_input=false \
    training.val_dataset_cfgs.training.target_has_input=false \
    training.val_dataset_cfgs.training.num_views=14 \
    training.val_dataset_cfgs.training.num_input_views=4 \
    training.val_dataset_cfgs.training.num_target_views=10 \
    training.feature_alignment.enabled=true \
    training.feature_alignment.layer_idx=7 \
    training.feature_alignment.weight=0.0
