#!/bin/bash
#SBATCH --job-name=sam3_co3d_masks
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=104
#SBATCH --mem=500G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/z50057756/code/RnG_feature_allignment/slurm_logs/%x-%j.out
#SBATCH --error=/home/z50057756/code/RnG_feature_allignment/slurm_logs/%x-%j.err
# Usage: SPLIT_FILE=... EXTRA_ARGS="..." [PROCS_PER_GPU=4] sbatch scripts/sam3_text_co3d_masks.sbatch
# Preemption-safe: worker is scene-idempotent (atomic stats.json => exists==complete);
# requeue reruns only unfinished scenes. GraceTime=0 means finally-cleanup may not
# run, so leaked tmp frame dirs are swept at (re)start.
set -euo pipefail
ulimit -c 0

PY=/mnt/data-alpha-sg-01/team-camera/home/z50057756/conda/envs/sam3-py312/bin/python
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1
cd /home/z50057756/code/RnG_feature_allignment
mkdir -p slurm_logs

SPLIT_FILE=${SPLIT_FILE:-/home/z50057756/data/co3d_sam2_masks/qc/audit/fix100.txt}
EXTRA_ARGS=${EXTRA_ARGS:-}
N=${SLURM_GPUS_ON_NODE:-4}
P=${PROCS_PER_GPU:-1}
TOTAL=$((N * P))
# fair share of the 26-CPU-per-GPU budget across the P procs on each GPU
export OMP_NUM_THREADS=$(( 26 / P > 2 ? 26 / P : 2 ))

TMP_ROOT=${SLURM_TMPDIR:-/tmp}
# sweep stale frame dirs from previous preempted runs of this user (>4h old)
find "$TMP_ROOT" -maxdepth 1 -name 'sam3_co3d_*' -user "$USER" -mmin +240 \
  -exec rm -rf {} + 2>/dev/null || true

echo "[$(date)] job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} node=$(hostname) \
split=$SPLIT_FILE gpus=$N procs_per_gpu=$P omp=$OMP_NUM_THREADS extra='$EXTRA_ARGS'"
pids=()
for s in $(seq 0 $((TOTAL - 1))); do
  CUDA_VISIBLE_DEVICES=$((s / P)) $PY scripts/sam3_text_co3d_masks.py \
    --split_file "$SPLIT_FILE" --shard_idx "$s" --num_shards "$TOTAL" \
    --tmp_root "$TMP_ROOT" $EXTRA_ARGS &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[$(date)] all shards finished rc=$rc restart=${SLURM_RESTART_COUNT:-0}"
exit $rc
