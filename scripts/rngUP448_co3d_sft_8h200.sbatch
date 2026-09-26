#!/bin/bash
#SBATCH --job-name=rng448-co3d-sft-8h200
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=208
#SBATCH --mem=1500G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/z50057756/code/RnG_feature_allignment/slurm_logs/%x-%j.out
#SBATCH --error=/home/z50057756/code/RnG_feature_allignment/slurm_logs/%x-%j.err

# V11 CO3D SFT: init model weights from render44798 ckpt_15000 (fresh optimizer,
# step 0), train 4000 steps @ lr 1e-4 on 21,803 certified CO3D scenes (46 cats;
# eval blacklist + whole toybus category excluded). Val every 500 steps on the
# 5cat eval set. Plan ref: ~/.claude/plans/...co3d-to-gso-co-reactive-mochi.md §V11

set -euo pipefail

# Env lives on the OLD home (new-home miniconda3/conda were not migrated); same
# PATH-prepend approach as scripts/inference_gso_sim2real.sh.
CONDA_ENV=/mnt/data-alpha-sg-01/team-camera/home/z50057756/conda/envs/rng-fa3
export PATH="$CONDA_ENV/bin:$PATH"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export TORCH_HOME=/mnt/data-alpha-sg-01/team-camera/home/z50057756/.cache/torch

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

# First launch: load model weights only from the pretrain ckpt (fresh optimizer,
# step 0). On requeue/restart: the SFT run's own ckpts exist in CKPT_DIR, so drop
# the overrides and let auto-resume continue from the latest SFT ckpt instead of
# restarting from the base checkpoint.
CKPT_DIR=/home/z50057756/code/RnG_feature_allignment/experiments/checkpoints/RnGUP_co3d_448_sft
BASE_CKPT=/mnt/data-alpha-sg-01/team-camera/home/z50057756/code/RnG_feature_allignment/experiments/checkpoints/RnGUP_obj_448_render44798/ckpt_0000000000015000.pt
RESUME_ARGS=()
if ls "$CKPT_DIR"/ckpt_*.pt >/dev/null 2>&1; then
  echo "Resume mode: auto-resume from latest SFT ckpt in $CKPT_DIR"
else
  echo "Resume mode: fresh SFT init (weights only) from $BASE_CKPT"
  RESUME_ARGS=(training.resume_ckpt="$BASE_CKPT" training.reset_training_state=true)
fi

python -m torch.distributed.run --nproc_per_node=8 --nnodes=1 \
  --rdzv_id=${SLURM_JOB_ID} --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:29517 \
  train.py --config configs/RnGUP_co3d_448_sft_4k.yaml \
  ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
