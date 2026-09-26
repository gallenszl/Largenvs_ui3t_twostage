#!/bin/bash
#SBATCH --job-name=uni3t-smoke
#SBATCH --partition=gpu
#SBATCH --qos=lowest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --no-requeue
#SBATCH --exclude=lrc-alpha-sg-gpu01
#SBATCH --output=/home/z50057756/code/RnG_lagernvs/slurm_logs/%x-%j.out
#SBATCH --error=/home/z50057756/code/RnG_lagernvs/slurm_logs/%x-%j.err

# Resources sized from the 132964 run (4m07s, MaxRSS 35 GB): a 2h/160G ask never
# backfills on the lowest queue -- Slurm was projecting a 3-day wait.
# UNI3T gate: three stages, each must pass before the next runs.
#   1. GPU unit tests that need xformers (renderer intermediates)
#   2. tools/uni3t_selfcheck.py  -- full model, synthetic batch, all three losses,
#      shape contract with export_results, and "no parameter without a gradient"
#   3. 60 training steps on the real dataset + one validation + one checkpoint
# Anything non-zero aborts the rest and the job exits non-zero.

set -uo pipefail

CONDA_ENV=/mnt/data-alpha-sg-01/team-camera/home/z50057756/conda/envs/rng-fa3
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export TORCH_HOME=/home/z50057756/torch_home_lagernvs
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RNG_ATTENTION_BACKEND=auto
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1
ulimit -c 0

REPO=/home/z50057756/code/RnG_lagernvs
cd "$REPO"
mkdir -p "$REPO/slurm_logs"

echo "### Job ${SLURM_JOB_ID} on $(hostname) | branch $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo; echo "=== STAGE 1/3: GPU unit tests ==="
python -m unittest tests.test_unified_heads -v
RC=$?
if [ $RC -ne 0 ]; then echo "STAGE 1 FAILED rc=$RC"; exit $RC; fi
echo "STAGE 1 PASSED"

echo; echo "=== STAGE 2/3: full-model self-check (synthetic batch) ==="
python tools/uni3t_selfcheck.py configs/RnGUP_lagernvs_uni3t_b32t6_fp32lr35_const_90k_all287k.yaml
RC=$?
if [ $RC -ne 0 ]; then echo "STAGE 2 FAILED rc=$RC"; exit $RC; fi
echo "STAGE 2 PASSED"

echo; echo "=== STAGE 3/3: 60-step smoke on the real dataset ==="
# train.py auto-resumes from checkpoint_dir. A checkpoint left by an earlier smoke
# makes stage 3 a no-op: it resumes at step 60, takes one step, and exits at
# train_steps -- which is exactly what happened in job 133114 and hid the fact that
# its loss_grad_point reading came from the older, buggy checkpoint. Clear the dir
# first. The case guard means this can only ever remove a path ending in
# /SMOKE_uni3t, never a real experiment directory.
SMOKE_CFG=configs/RnGUP_lagernvs_uni3t_smoke60.yaml
SMOKE_DIR=$(grep -E "^[[:space:]]*checkpoint_dir:" "$SMOKE_CFG" | head -1 | sed 's/^[^:]*:[[:space:]]*//')
SMOKE_DIR="${SMOKE_DIR//\$\{exp_name\}/SMOKE_uni3t}"
case "$SMOKE_DIR" in
  */SMOKE_uni3t)
    if [ -e "$SMOKE_DIR" ]; then
      echo "Clearing stale smoke dir $SMOKE_DIR ($(du -sh --apparent-size "$SMOKE_DIR" 2>/dev/null | cut -f1))"
      rm -rf "$SMOKE_DIR"
    else
      echo "Smoke dir $SMOKE_DIR is already clean"
    fi
    ;;
  *)
    echo "REFUSING to clear unexpected smoke dir: '$SMOKE_DIR'"; exit 1 ;;
esac
RDZV_PORT=$((29500 + SLURM_JOB_ID % 200))
python -m torch.distributed.run --nproc_per_node=1 --nnodes=1 \
  --rdzv_id=${SLURM_JOB_ID} --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:${RDZV_PORT} \
  train.py --config "$SMOKE_CFG"
RC=$?
if [ $RC -ne 0 ]; then echo "STAGE 3 FAILED rc=$RC"; exit $RC; fi
echo "STAGE 3 PASSED"
echo "### UNI3T SMOKE ALL STAGES PASSED"
