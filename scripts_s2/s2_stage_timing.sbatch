#!/bin/bash
#SBATCH --job-name=s2-stage-timing
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --output=/home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/slurm_logs/%x-%j.out
#SBATCH --error=/home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/slurm_logs/%x-%j.err
# Inference time per stage (reconstruction / r0 / r1 / the rest), decomposed into host syncs, weight casts, launch
# overhead and GPU compute; everything on this one GPU.
#   leg 0  the H200 kernel-launch profile (tools_s2/s2_launch_profile.py, job 137900) rerun unchanged: same yardstick
#   leg 1  tools_s2/s2_stage_timing.py, block 8, smoke ckpt_90 of job 6906717
#   leg 2  tools_s2/s2_stage_timing.py, block 4, smoke ckpt_90 of job 6906719
set -uo pipefail
source /home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/scripts_s2/s2_env.sh
OUT=$S2_ROOT/bench/stage_timing_${SLURM_JOB_ID}
mkdir -p "$OUT"
rc_all=0
if [ "${LEG0:-1}" = 1 ]; then
echo "=== [s2-stage] LEG 0 launch profile (H200 tool, unchanged) $(date +%T) ==="
python tools_s2/s2_launch_profile.py --config configs/S2P8_uni3t70k_b32t6_lr35_a21k26k.yaml --out "$OUT/launch_profile_P8.json"
rc=$?; rc_all=$((rc_all | rc)); echo "[s2-stage] LEG 0 done rc=$rc $(date +%T)"
fi
for arm in P8:6906717 P4:6906719; do
  tag=${arm%%:*}; smoke=${arm##*:}
  CK=$S2_ROOT/SMOKE/S2${tag}_uni3t70k_b32t6_lr35_a21k26k_SMOKE_${smoke}/ckpt_0000000000000090.pt
  echo "=== [s2-stage] $tag stage timing $(date +%T) ==="
  PYTORCH_CUDA_ALLOC_CONF= python tools_s2/s2_stage_timing.py --config configs/S2${tag}_uni3t70k_b32t6_lr35_a21k26k.yaml \
    --s2_ckpt "$CK" --n_scenes 4 --out "$OUT/stage_timing_${tag}.json"
  rc=$?; rc_all=$((rc_all | rc)); echo "[s2-stage] $tag done rc=$rc $(date +%T)"
done
ls -la "$OUT"
echo "[s2-stage] DONE rc=$rc_all"
exit $rc_all
