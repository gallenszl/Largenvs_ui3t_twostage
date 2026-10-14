#!/bin/bash
#SBATCH --job-name=s2-speed-s1sel
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:50:00
#SBATCH --output=/home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/slurm_logs/%x-%j.out
#SBATCH --error=/home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/slurm_logs/%x-%j.err
# Speed test before training the stage-1-selection arms (user 09-28, "about 15 min"), every leg on this one GPU:
#   leg 1  r0 / r1 inference time, fg_dense with the GT-mask tiles vs the stage-1-render tiles (tools_s2/s2_r1_timing.py)
#   leg 2  G8 whole-step training speed, batch 8, 12 real batches, drop 100 steps, measure 60: fg_dense (anchor:
#          1.767 s/step in job 6910164) and the stage-1-selection arm with the stage-1 mask channel.  The nomask arm
#          runs the same kernels on the same shapes (its mask channel is all zero, the conv still has 7 channels).
set -uo pipefail
source /home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/scripts_s2/s2_env.sh
cd "$REPO"
OUT=$S2_ROOT/bench/speed_s1sel_${SLURM_JOB_ID}
mkdir -p "$OUT"
A=configs/S2P8fgdense_uni3t70k_b32t6_lr35_a21k26k.yaml
B=configs/S2P8fgdense_s1sel1_s1mask_uni3t70k_b32t6_lr35_a21k26k.yaml
echo "Job $SLURM_JOB_ID | node $(hostname) | out $OUT"
rc_all=0
echo "=== [s2-speed] leg 1 r0 / r1 timing $(date +%T) ==="
PYTORCH_CUDA_ALLOC_CONF= python tools_s2/s2_r1_timing.py --configs "$A,$B" --n_scenes 4 --out "$OUT/r1_timing.json"
rc=$?; rc_all=$((rc_all | rc)); echo "[s2-speed] leg 1 rc=$rc $(date +%T)"
echo "=== [s2-speed] leg 2 G8 training step $(date +%T) ==="
python tools_s2/s2_bench.py --configs "$A,$B" --backends flex --batch 8 --warmup 100 --steps 60 --out "$OUT/g8.json"
rc=$?; rc_all=$((rc_all | rc)); echo "[s2-speed] leg 2 rc=$rc $(date +%T)"
echo "[s2-speed] DONE rc=$rc_all"
exit $rc_all
