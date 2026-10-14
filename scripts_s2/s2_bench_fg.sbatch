#!/bin/bash
#SBATCH --job-name=s2-bench-fg
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:30:00
#SBATCH --output=/home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/slurm_logs/%x-%j.out
#SBATCH --error=/home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/slurm_logs/%x-%j.err
# G8-style whole-step training speed of the fg_dense / fg_dual block-8 arms (user 09-28), every leg in this one job
# on this one GPU, with the current block-8 structure (flex) as the calibration anchor against job 6906793.
# batch 8, 12 real batches, drop 100 steps, measure 200.  $1 = output json
set -uo pipefail
source /home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/scripts_s2/s2_env.sh
OUT=${1:-$S2_ROOT/bench/g8_fg_${SLURM_JOB_ID}.json}
mkdir -p "$(dirname "$OUT")"
python tools_s2/s2_bench.py \
  --configs configs/S2P8_uni3t70k_b32t6_lr35_a21k26k.yaml,configs/S2P8fgdense_uni3t70k_b32t6_lr35_a21k26k.yaml,configs/S2P8fgdual_uni3t70k_b32t6_lr35_a21k26k.yaml \
  --backends flex --batch 8 --warmup 100 --steps 200 --out "$OUT"
rc=$?; echo "[s2-bench] DONE rc=$rc"; exit $rc
