#!/bin/bash
#SBATCH --job-name=sam3_par_test
#SBATCH --partition=gpu
#SBATCH --qos=lowest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=26
#SBATCH --mem=150G
#SBATCH --time=02:00:00
#SBATCH --exclude=lrc-alpha-sg-gpu05
#SBATCH --output=/home/z50057756/code/RnG_feature_allignment/slurm_logs/%x-%j.out
#SBATCH --error=/home/z50057756/code/RnG_feature_allignment/slurm_logs/%x-%j.err
# Measure per-GPU throughput vs number of worker processes sharing one H200.
# Same 18 scenes for every config; fresh output dir each time.
set -euo pipefail
ulimit -c 0
PY=/mnt/data-alpha-sg-01/team-camera/home/z50057756/conda/envs/sam3-py312/bin/python
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 TQDM_DISABLE=1
cd /home/z50057756/code/RnG_feature_allignment
SPLIT=$PWD/data/co3d_partest18.txt
IN=/mnt/data-alpha-sg-02/team-camera/datasets/yuchen/co3d/webdataset/train

for P in 1 2 3; do
  OUT=/home/z50057756/data/co3d_sam3_masks_hp/partest/p$P
  rm -rf "$OUT"; mkdir -p "$OUT"
  ( sleep 150; echo "P=$P mid-run GPU: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)" ) &
  probe=$!
  t0=$(date +%s)
  pids=()
  for i in $(seq 0 $((P - 1))); do
    $PY scripts/sam3_text_co3d_masks.py --split_file "$SPLIT" \
      --shard_idx "$i" --num_shards "$P" \
      --input_dir "$IN" --output_dir "$OUT" &
    pids+=($!)
  done
  rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  t1=$(date +%s)
  kill "$probe" 2>/dev/null || true
  echo "PARTEST P=$P wall=$((t1 - t0))s scenes=18 rc=$rc"
done
echo ALL_CONFIGS_DONE
