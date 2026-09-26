#!/bin/bash
#SBATCH --job-name=sam2_co3d_masks
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=lrc-alpha-sg-gpu05
#SBATCH --output=/home/z50057756/code/RnG_feature_allignment/slurm_logs/%x-%j.out
#SBATCH --error=/home/z50057756/code/RnG_feature_allignment/slurm_logs/%x-%j.err
# Usage:
#   SPLIT_FILE=data/co3d_teddybear_pilot100.txt sbatch scripts/sam2_refine_co3d_masks.sbatch
#   SPLIT_FILE=data/co3d_teddybear_val_1140.txt sbatch scripts/sam2_refine_co3d_masks.sbatch
#   EXTRA_ARGS="--limit 3" for smoke; idempotent (stats.json marker), safe to resubmit.
set -euo pipefail
ulimit -c 0

PY=/home/z50057756/code/sam2/.venv/bin/python
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 TQDM_DISABLE=1
cd /home/z50057756/code/RnG_feature_allignment
mkdir -p slurm_logs

SPLIT_FILE=${SPLIT_FILE:-data/co3d_teddybear_pilot100.txt}
EXTRA_ARGS=${EXTRA_ARGS:-}
N=${SLURM_GPUS_ON_NODE:-4}

echo "[$(date)] job=$SLURM_JOB_ID split=$SPLIT_FILE gpus=$N extra='$EXTRA_ARGS'"
pids=()
for i in $(seq 0 $((N - 1))); do
  CUDA_VISIBLE_DEVICES=$i $PY scripts/sam2_refine_co3d_masks.py \
    --split_file "$SPLIT_FILE" --shard_idx "$i" --num_shards "$N" $EXTRA_ARGS &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[$(date)] all shards finished rc=$rc"
exit $rc
