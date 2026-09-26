#!/bin/bash
#SBATCH --job-name=sam3_par_sweep
#SBATCH --partition=gpu
#SBATCH --qos=lowest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=26
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --exclude=lrc-alpha-sg-gpu05
#SBATCH --output=/home/z50057756/code/RnG_feature_allignment/slurm_logs/%x-%j.out
#SBATCH --error=/home/z50057756/code/RnG_feature_allignment/slurm_logs/%x-%j.err
# Fair single-GPU throughput sweep: every config runs the SAME 12 scenes on the
# SAME GPU, back-to-back in one allocation. Headline metric = throughput derived
# from the worker's own per-scene elapsed times (model-load excluded).
set -uo pipefail
ulimit -c 0
PY=/mnt/data-alpha-sg-01/team-camera/home/z50057756/conda/envs/sam3-py312/bin/python
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1
# 26 CPU/GPU is the scheduler's hard per-GPU cap = the real production budget.
cd /home/z50057756/code/RnG_feature_allignment
SPLIT=$PWD/data/co3d_partest12.txt
IN=/mnt/data-alpha-sg-02/team-camera/datasets/yuchen/co3d/webdataset/train
ROOT=/home/z50057756/data/co3d_sam3_masks_hp/par_sweep
mkdir -p "$ROOT"

for P in 1 2 3 4 6 8; do
  OUT=$ROOT/p$P; LOG=$ROOT/log_p$P.txt
  rm -rf "$OUT"; mkdir -p "$OUT"; : > "$LOG"
  # split the 26-core budget evenly across the P processes (fair per config)
  export OMP_NUM_THREADS=$(( 26 / P > 2 ? 26 / P : 2 ))
  # mid-run GPU probe (fires once ~40% through)
  ( sleep 120; echo "PROBE P=$P $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader) load=$(cat /proc/loadavg | cut -d' ' -f1)" ) &
  probe=$!
  t0=$(date +%s)
  pids=()
  for i in $(seq 0 $((P - 1))); do
    $PY scripts/sam3_text_co3d_masks.py --split_file "$SPLIT" \
      --shard_idx "$i" --num_shards "$P" --input_dir "$IN" --output_dir "$OUT" \
      >> "$LOG" 2>&1 &
    pids+=($!)
  done
  rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  t1=$(date +%s)
  kill "$probe" 2>/dev/null || true
  echo "SWEEP P=$P wall=$((t1 - t0))s rc=$rc"
done
echo ALL_CONFIGS_DONE
