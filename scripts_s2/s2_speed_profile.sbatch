#!/bin/bash
#SBATCH --job-name=s2-speedprof
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --output=/home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/slurm_logs/%x-%j.out
#SBATCH --error=/home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/slurm_logs/%x-%j.err
# Speed profile, everything on this one GPU: attention density + per-call times (block 8, block 4), then inference
# latency (stage 1 only vs stage 1 + stage 2, both blocks, SDPA and flex).  $1 output dir.
set -uo pipefail
source /home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/scripts_s2/s2_env.sh
OUT=${1:-/scratch/u6ga/zhelun.u6ga/S2_experiments/bench}
mkdir -p "$OUT"
P8=configs/S2P8_uni3t70k_b32t6_lr35_a21k26k.yaml
P4=configs/S2P4_uni3t70k_b32t6_lr35_a21k26k.yaml
python tools_s2/s2_attn_profile.py --config $P8 --out "$OUT/attn_profile_P8.json"; echo "[speedprof] attn P8 rc=$?"
python tools_s2/s2_attn_profile.py --config $P4 --out "$OUT/attn_profile_P4.json"; echo "[speedprof] attn P4 rc=$?"
python tools_s2/s2_infer_bench.py --configs $P8,$P4 --out "$OUT/infer_bench.json"; echo "[speedprof] infer rc=$?"
echo "[speedprof] DONE"
