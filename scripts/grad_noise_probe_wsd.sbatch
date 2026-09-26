#!/bin/bash
#SBATCH --job-name=gradnoise-wsd
#SBATCH --partition=gpu
#SBATCH --qos=lowest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
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

CCONST=configs/RnGUP_lagernvs_b32t6_fp32lr35_const_100k.yaml
CA10K=configs/RnGUP_lagernvs_b32t6_wsd60k_a10k.yaml
CFLAG=configs/RnGUP_lagernvs_b32t6_wsd90k_a13p5k.yaml
# point_name|config|ckpt|ntgt_override
POINTS=(
  "const_20k|$CCONST|$PIN/const_20000.pt|"
  "const_40k|$CCONST|$PIN/const_40000.pt|"
  "const_60k|$CCONST|$PIN/const_60000.pt|"
  "const_80k|$CCONST|$PIN/const_80000.pt|"
  "const_90k|$CCONST|$PIN/const_90000.pt|"
  "a10k_end_70k|$CA10K|$PIN/a10k_70000.pt|"
  "flagship_end_103k5|$CFLAG|$PIN/flagship_103500.pt|"
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

run_shard 0 0 4 &
run_shard 1 1 5 &
run_shard 2 2 6 &
run_shard 3 3 &
wait
echo "ALL POINTS DONE"
ls -la "$OUT"
