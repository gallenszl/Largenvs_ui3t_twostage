#!/bin/bash
#SBATCH --job-name=rng448-valsmoke
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=208
#SBATCH --mem=1500G
#SBATCH --time=00:30:00
#SBATCH --exclude=lrc-alpha-sg-gpu05
#SBATCH --output=/home/z50057756/code/RnG_feature_allignment/slurm_logs/%x-%j.out
#SBATCH --error=/home/z50057756/code/RnG_feature_allignment/slurm_logs/%x-%j.err

# Smoke test: trigger val pass at step 2 (checkpoint_every=2) to verify P4a
# follow-up unpack fix (dataset_gso_ours.py + dataset_gso_ours_vggtPose.py).
# Uses a separate exp_name so we don't touch the real Phase 1a checkpoint dir.

set -euo pipefail

source /home/z50057756/miniconda3/etc/profile.d/conda.sh
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
conda activate /home/z50057756/conda/envs/rng-fa3

export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RNG_ATTENTION_BACKEND=auto
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

ulimit -c 0
mkdir -p /home/z50057756/code/RnG_feature_allignment/slurm_logs
cd /home/z50057756/code/RnG_feature_allignment

echo "Smoke Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
nvidia-smi -L

torchrun --nproc_per_node=8 --nnodes=1 \
  --rdzv_id=${SLURM_JOB_ID} --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:29515 \
  train.py --config configs/RnGUP_obj_448_bf16_15k_render44798.yaml \
  exp_name=RnGUP_obj_448_render44798_valsmoke \
  training.train_steps=4 \
  training.checkpoint_every=2 \
  training.vis_every=2
