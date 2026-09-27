#!/bin/bash
#SBATCH --job-name=s2-bench
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=03:30:00
#SBATCH --no-requeue
#SBATCH --output=/home/z50057756/code/RnG_lagernvs_stage2/slurm_logs/%x-%j.out
#SBATCH --error=/home/z50057756/code/RnG_lagernvs_stage2/slurm_logs/%x-%j.err
# Stage-2 speed benchmark (plan G8): every leg in this one job on this one GPU.  $1 = output json
set -uo pipefail
source /home/z50057756/code/RnG_lagernvs_stage2/scripts_s2/s2_env.sh
OUT=${1:-/home/z50057756/tmp/s2_bench/bench.json}
mkdir -p "$(dirname "$OUT")"
python tools_s2/s2_bench.py --configs configs/S2P8_uni3t70k_b32t6_lr35_a21k26k.yaml,configs/S2P4_uni3t70k_b32t6_lr35_a21k26k.yaml \
  --backends sdpa,flex --batch 8 --warmup 100 --steps 200 --out "$OUT"
rc=$?; echo "[s2-bench] DONE rc=$rc"; exit $rc
