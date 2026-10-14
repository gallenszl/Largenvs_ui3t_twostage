#!/bin/bash
#SBATCH --job-name=s2-bench
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=03:30:00
#SBATCH --output=/home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/slurm_logs/%x-%j.out
#SBATCH --error=/home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/slurm_logs/%x-%j.err
# Stage-2 speed benchmark (plan G8): every leg in this one job on this one GPU.  $1 = output json
set -uo pipefail
source /home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/scripts_s2/s2_env.sh
OUT=${1:-/scratch/u6ga/zhelun.u6ga/S2_experiments/bench/bench.json}
mkdir -p "$(dirname "$OUT")"
python tools_s2/s2_bench.py --configs configs/S2P8_uni3t70k_b32t6_lr35_a21k26k.yaml,configs/S2P4_uni3t70k_b32t6_lr35_a21k26k.yaml \
  --backends sdpa,flex --batch 8 --warmup 100 --steps 200 --out "$OUT"
rc=$?; echo "[s2-bench] DONE rc=$rc"; exit $rc
