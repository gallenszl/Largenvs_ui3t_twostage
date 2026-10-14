#!/bin/bash
#SBATCH --job-name=s2-audit
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --output=/home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/slurm_logs/%x-%j.out
#SBATCH --error=/home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/slurm_logs/%x-%j.err
# Did the block-8 stage-2 residual learn anything? (user 09-29)  tools_s2/s2_residual_audit.py on the two
# stage-1-selection arms (ckpt 10k / 11k) and, as the positive control, fg_dense with the GT mask (ckpt 21k / 26k).
set -uo pipefail
source /home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/scripts_s2/s2_env.sh
cd "$REPO"
C=$S2_ROOT/checkpoints
OUT=$S2_ROOT/evaluation/residual_audit_${SLURM_JOB_ID}
mkdir -p "$OUT"
rc_all=0
for spec in S2P8fgdense_s1sel1_nomask:10000,11000 S2P8fgdense_s1sel1_s1mask:10000,11000 S2P8fgdense:21000,26000; do
  x=${spec%%:*}; steps=${spec##*:}; E=${x}_uni3t70k_b32t6_lr35_a21k26k
  CK=$(echo "$steps" | tr ',' '\n' | while read s; do printf "%s/%s/ckpt_%016d.pt," "$C" "$E" "$s"; done); CK=${CK%,}
  echo "=== [s2-audit] $E $(date +%T) ==="
  python tools_s2/s2_residual_audit.py --config "configs/$E.yaml" --ckpts "$CK" --n_scenes 16 --out "$OUT/$E.json"
  rc=$?; rc_all=$((rc_all | rc)); echo "[s2-audit] $E rc=$rc $(date +%T)"
done
echo "[s2-audit] DONE rc=$rc_all"
exit $rc_all
