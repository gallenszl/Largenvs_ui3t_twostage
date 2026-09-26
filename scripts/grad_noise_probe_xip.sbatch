#!/bin/bash
#SBATCH --job-name=gradnoise-xip
#SBATCH --partition=gpu
#SBATCH --qos=lowest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:3
#SBATCH --cpus-per-task=36
#SBATCH --mem=300G
#SBATCH --time=04:00:00
#SBATCH --exclude=lrc-alpha-sg-gpu01
#SBATCH --output=/home/z50057756/code/RnG_lagernvs/slurm_logs/%x-%j.out
#SBATCH --error=/home/z50057756/code/RnG_lagernvs/slurm_logs/%x-%j.err
# Cross-inner-product gradient noise probe (audit-recommended estimator):
# 3 points sharded to 3 GPUs on one node.
set -uo pipefail
CONDA_ENV=/mnt/data-alpha-sg-01/team-camera/home/z50057756/conda/envs/rng-fa3
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RNG_ATTENTION_BACKEND=auto
export PYTHONUNBUFFERED=1
export TORCH_HOME=/home/z50057756/torch_home_lagernvs
ulimit -c 0
cd /home/z50057756/code/RnG_lagernvs
PIN=experiments/probe_pinned
OUT=experiments/grad_noise_probe
mkdir -p "$OUT"
CCONST=configs/RnGUP_lagernvs_b32t6_fp32lr35_const_100k.yaml
CA10K=configs/RnGUP_lagernvs_b32t6_wsd60k_a10k.yaml
CFLAG=configs/RnGUP_lagernvs_b32t6_wsd90k_a13p5k.yaml
POINTS=(
  "const_90k|$CCONST|$PIN/const_90000.pt"
  "a10k_end_70k|$CA10K|$PIN/a10k_70000.pt"
  "flagship_end_103k5|$CFLAG|$PIN/flagship_103500.pt"
)
run_one() {
  local gpu=$1 idx=$2
  IFS='|' read -r NAME CFG CKPT <<< "${POINTS[$idx]}"
  if [ -s "$OUT/${NAME}_xip.json" ]; then echo "=== $NAME xip already done ==="; return; fi
  echo "=== GPU$gpu -> $NAME (xip) ==="
  CUDA_VISIBLE_DEVICES=$gpu CONFIG_PATH=$CFG CKPT=$CKPT POINT_NAME=$NAME \
    BSIZES="32,128" COUNTS="96,32" OUT_JSON=$OUT/${NAME}_xip.json SEED=0 \
    python scripts/grad_noise_probe_xip.py \
    > slurm_logs/gradnoise_xip_${NAME}.log 2>&1 \
    || echo "!!! $NAME xip FAILED"
}
run_one 0 0 & run_one 1 1 & run_one 2 2 &
wait
echo "XIP ALL DONE"; ls -la "$OUT" | grep xip
