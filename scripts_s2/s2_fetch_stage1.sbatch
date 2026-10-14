#!/bin/bash
#SBATCH --job-name=s2-fetch-s1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/slurm_logs/%x-%j.out
#SBATCH --error=/home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/slurm_logs/%x-%j.err
# Isambard-AI: download the frozen stage-1 checkpoint (README §3.1) from HF and verify its sha256.
# There is no CPU partition, so this takes the smallest allocation (1 GPU, unused).
set -uo pipefail
source "$HOME/rng_env.sh"            # hfdl venv, HF_HOME on scratch, xet cache on node-local disk
DEST=/projects/u6ga/zhelun/RnG_weights/stage1
REL=moe_experiments/checkpoints/PLN2uni3t_all287k_b32t6_fp32lr35_wsd60k_a10k
WANT=00cb27e01c359468037cd821cea6b596b339db87bc7eebbe5ae81437d47893f4
F=$DEST/$REL/ckpt_0000000000070000.pt
mkdir -p "$DEST"
echo "[s2-fetch] $(date +%T) node $(hostname) -> $DEST/$REL"
if [ ! -f "$F" ]; then
  hf download szlgallen/RAE_backup_checkpoints \
    "$REL/ckpt_0000000000070000.pt" "$REL/README.md" "$REL/SHA256SUMS.txt" "$REL/config.yaml" \
    --local-dir "$DEST" --max-workers 8 || { echo "[s2-fetch] FAIL download"; exit 1; }
fi
SIZE=$(stat -c %s "$F")
echo "[s2-fetch] $(date +%T) size $SIZE (want 16755969541)"
[ "$SIZE" = 16755969541 ] || { echo "[s2-fetch] FAIL size"; exit 1; }
GOT=$(sha256sum "$F" | awk '{print $1}')
echo "[s2-fetch] $(date +%T) sha256 $GOT"
[ "$GOT" = "$WANT" ] || { echo "[s2-fetch] FAIL sha256 (want $WANT)"; exit 1; }
chmod 0400 "$F"
ls -la "$DEST/$REL"
echo "[s2-fetch] PASS $F"
