#!/bin/bash
#SBATCH --job-name=s2-dense-equiv
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:30:00
#SBATCH --output=/home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/slurm_logs/%x-%j.out
#SBATCH --error=/home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/slurm_logs/%x-%j.err
# Sparse stage-2 attention vs full attention with -inf on the disallowed pairs (tools_s2/s2_dense_equiv.py),
# Each arm with its own smoke checkpoint (step 90) so the stage-2 residual is not zero.
#   arguments: <tag>:<smoke job id> ...  (tag = the config's prefix before _uni3t70k..., e.g. S2P8 / S2P4 /
#   S2P8fgdense / S2P8fgdual); default = the 09-27 run: S2P8:6906717 S2P4:6906719
set -uo pipefail
source /home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/scripts_s2/s2_env.sh
OUT=$S2_ROOT/dense_equiv/${SLURM_JOB_ID}
mkdir -p "$OUT"
rc_all=0
ARMS=("$@"); [ ${#ARMS[@]} -gt 0 ] || ARMS=(S2P8:6906717 S2P4:6906719)
for arm in "${ARMS[@]}"; do
  tag=${arm%%:*}; smoke=${arm##*:}
  CFG=configs/${tag}_uni3t70k_b32t6_lr35_a21k26k.yaml
  CK=$S2_ROOT/SMOKE/${tag}_uni3t70k_b32t6_lr35_a21k26k_SMOKE_${smoke}/ckpt_0000000000000090.pt
  echo "=== [s2-deq] $tag $(date +%T) ckpt $CK ==="
  python tools_s2/s2_dense_equiv.py --config "$CFG" --s2_ckpt "$CK" --n_val 2 --train_batch 1 --out "$OUT/${tag}.json"
  rc=$?; rc_all=$((rc_all | rc)); echo "[s2-deq] $tag done rc=$rc $(date +%T)"
done
echo "[s2-deq] DONE rc=$rc_all"
exit $rc_all
