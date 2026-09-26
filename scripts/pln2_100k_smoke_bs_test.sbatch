#!/bin/bash
#SBATCH --job-name=pln2-100k-bstest
#SBATCH --partition=gpu
#SBATCH --qos=lowest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=208
#SBATCH --mem=1500G
#SBATCH --time=02:00:00
#SBATCH --exclude=lrc-alpha-sg-gpu01
#SBATCH --output=/home/z50057756/code/RnG_lagernvs/slurm_logs/%x-%j.out
#SBATCH --error=/home/z50057756/code/RnG_lagernvs/slurm_logs/%x-%j.err

# 100k-arm batch-size fair test (user 07-18): ONE job, same node/8 GPUs,
# sequentially run global batch 128 (16/GPU) then 256 (32/GPU), 50 iters each,
# with the 4cond+6tgt recipe. Reports per-segment Iter Time, OOM, peak VRAM.
# NO formal training is submitted by this script.

set -uo pipefail

CONDA_ENV=/mnt/data-alpha-sg-01/team-camera/home/z50057756/conda/envs/rng-fa3
export PATH="$CONDA_ENV/bin:$PATH"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export TORCH_HOME=/home/z50057756/torch_home_lagernvs
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RNG_ATTENTION_BACKEND=auto
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=offline
ulimit -c 0

REPO=/home/z50057756/code/RnG_lagernvs
cd "$REPO"
CONFIG=configs/RnGUP_lagernvs_rgb256pose_v2color_100k.yaml
echo "Job: ${SLURM_JOB_ID} | Node: $(hostname) | Git: $(git rev-parse --short HEAD)"

run_round() {
  local NAME=$1 BS=$2 PORT=$3
  local LOG=slurm_logs/bstest-${SLURM_JOB_ID}-${NAME}.log
  local MEMLOG=slurm_logs/bstest-${SLURM_JOB_ID}-${NAME}.mem
  echo "==== ROUND ${NAME}: batch_size_per_gpu=${BS} (global $((BS*8))) ===="
  # VRAM sampler (5s cadence) for peak-memory readout
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1; sleep 5; done ) > "$MEMLOG" &
  local SAMPLER=$!
  python -m torch.distributed.run --nproc_per_node=8 --nnodes=1 \
    --rdzv_id=${SLURM_JOB_ID}${PORT} --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:${PORT} \
    train.py --config "$CONFIG" \
    exp_name=SMK100k_${NAME} \
    training.batch_size_per_gpu=${BS} \
    training.train_steps=50 \
    training.print_every=10 \
    training.checkpoint_every=100000 \
    training.vis_every=100000 \
    > "$LOG" 2>&1
  local RC=$?
  kill $SAMPLER 2>/dev/null
  echo "ROUND ${NAME} exit=${RC}"
  echo "-- Iter Time 序列 --"
  grep -oE "Forwad step:\s+[0-9]+.*Iter Time: [0-9.]+s" "$LOG" | tail -6
  echo "-- 峰值显存 (单卡最大, MiB) --"
  sort -n "$MEMLOG" | tail -1
  if grep -qi "out of memory" "$LOG"; then echo "ROUND ${NAME}: OOM 实锤"; grep -i -m1 -A2 "out of memory" "$LOG" | head -5; fi
  rm -rf experiments/checkpoints/SMK100k_${NAME}
  return 0
}

run_round bs128 16 29611
run_round bs256 32 29613
echo "==== 双 batch 测试完毕 ===="
