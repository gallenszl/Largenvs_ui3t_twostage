#!/bin/bash
#SBATCH --job-name=s2-speed-gh200
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=05:00:00
#SBATCH --output=/home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/slurm_logs/%x-%j.out
#SBATCH --error=/home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/slurm_logs/%x-%j.err
# Re-measure the README §7.3 / §7.4 / §8.2 speed numbers on GH200, every leg in this one job on this one GPU, with the
# same tools and settings as on the H200 machine, so the ratios can be compared:
#   leg 1  renderer what-if (tools_s2/s2_fgdense_whatif.py, block 8): stage-1 renderer vs the current stage-2
#          renderer vs fg_dense / fg_static / fg_dual, inference 1 and 10 target views + training shape (jobs 137897/137899)
#   leg 2  bf16 weights + CUDA Graphs (tools_s2/s2_graph_bench.py, block 8, smoke checkpoint), one process per
#          target-view count so a failed capture in one cannot take the other down (job 137951, never ran)
#   leg 3  G8 whole-step training speed {P8, P4} x {sdpa, flex}, batch 8, drop 100 steps, measure 200 (job 137861)
#   $1 = stage-2 checkpoint for leg 2 (default: the block-8 smoke ckpt_90 of job 6906717)
set -uo pipefail
source /home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/scripts_s2/s2_env.sh
S2CK=${1:-$S2_ROOT/SMOKE/S2P8_uni3t70k_b32t6_lr35_a21k26k_SMOKE_6906717/ckpt_0000000000000090.pt}
P8=configs/S2P8_uni3t70k_b32t6_lr35_a21k26k.yaml
P4=configs/S2P4_uni3t70k_b32t6_lr35_a21k26k.yaml
OUT=$S2_ROOT/bench/gh200_${SLURM_JOB_ID}
mkdir -p "$OUT"
[ -f "$S2CK" ] || { echo "[s2-speed] FAIL: no stage-2 ckpt $S2CK"; exit 2; }
echo "[s2-speed] out $OUT  s2_ckpt $S2CK"
rc_all=0

echo "=== [s2-speed] LEG 1 what-if $(date +%T) ==="
python tools_s2/s2_fgdense_whatif.py --config $P8 --variants stage1_only,current,fg_dense,fg_static,fg_dual \
  --out "$OUT/whatif_P8.json"
rc=$?; rc_all=$((rc_all | rc)); echo "[s2-speed] LEG 1 done rc=$rc $(date +%T)"

echo "=== [s2-speed] LEG 2 graph bench $(date +%T) ==="
for vt in 1 10; do
  PYTORCH_CUDA_ALLOC_CONF= python tools_s2/s2_graph_bench.py --config $P8 --s2_ckpt "$S2CK" --vts "$vt" \
    --out "$OUT/graph_vt${vt}.json"
  rc=$?; rc_all=$((rc_all | rc)); echo "[s2-speed] LEG 2 vt=$vt done rc=$rc $(date +%T)"
done

echo "=== [s2-speed] LEG 3 G8 bench $(date +%T) ==="
python tools_s2/s2_bench.py --configs $P8,$P4 --backends sdpa,flex --batch 8 --warmup 100 --steps 200 \
  --out "$OUT/g8_bench.json"
rc=$?; rc_all=$((rc_all | rc)); echo "[s2-speed] LEG 3 done rc=$rc $(date +%T)"

ls -la "$OUT"
echo "[s2-speed] DONE rc=$rc_all"
exit $rc_all
