#!/bin/bash
#SBATCH --job-name=gradnoise-probe
#SBATCH --partition=gpu
#SBATCH --qos=lowest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=/home/z50057756/code/RnG_lagernvs/slurm_logs/%x-%j.out
#SBATCH --error=/home/z50057756/code/RnG_lagernvs/slurm_logs/%x-%j.err

# Gradient noise scale probe over 9 measurement points (one node, 4 GPUs,
# points sharded across GPUs; single job => same hardware for every point).

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

C100K=configs/RnGUP_lagernvs_rgb256pose_v2color_100k_lr1e4_bs64.yaml
CBASE=configs/RnGUP_lagernvs_rgb256pose_v2color_15k.yaml
CTGT6=configs/RnGUP_lagernvs_rgb256pose_v2color_15k_tgt6.yaml
CFP32=configs/RnGUP_lagernvs_rgb256pose_v2color_15k_fp32lr5e5.yaml

# point_name|config|ckpt|ntgt_override
POINTS=(
  "budget_16k|$C100K|$PIN/budget_16000.pt|"
  "budget_40k|$C100K|$PIN/budget_40000.pt|"
  "budget_56k|$C100K|$PIN/budget_56000.pt|"
  "base_15k|$CBASE|$PIN/base_15000.pt|"
  "base_15k_ntgt6|$CBASE|$PIN/base_15000.pt|6"
  "base_4k|$CBASE|$PIN/base_4000.pt|"
  "tgt6_15k|$CTGT6|$PIN/tgt6_15000.pt|"
  "tgt6_15k_ntgt3|$CTGT6|$PIN/tgt6_15000.pt|3"
  "fp32lr5e5_15k|$CFP32|$PIN/fp32lr5e5_15000.pt|"
)

run_shard() {
  local gpu=$1; shift
  for idx in "$@"; do
    IFS='|' read -r NAME CFG CKPT NTGT <<< "${POINTS[$idx]}"
    if [ -s "$OUT/$NAME.json" ]; then
      echo "=== GPU$gpu -> $NAME already done, skipping ==="
      continue
    fi
    echo "=== GPU$gpu -> $NAME ==="
    CUDA_VISIBLE_DEVICES=$gpu CONFIG_PATH=$CFG CKPT=$CKPT POINT_NAME=$NAME \
      NTGT_OVERRIDE=$NTGT OUT_JSON=$OUT/$NAME.json SEED=0 \
      python scripts/grad_noise_probe.py \
      > slurm_logs/gradnoise_${NAME}.log 2>&1 \
      || echo "!!! $NAME FAILED (see slurm_logs/gradnoise_${NAME}.log)"
  done
}

run_shard 0 0 1 2 3 4 &
run_shard 1 5 6 7 8 &
wait
echo "ALL POINTS DONE"
ls -la "$OUT"
