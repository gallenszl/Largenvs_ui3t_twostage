#!/bin/bash
#SBATCH --job-name=uni3t-fulleval
#SBATCH --partition=gpu
#SBATCH --qos=lowest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/z50057756/code/RnG_lagernvs_stage2/slurm_logs/%x-%j.out
#SBATCH --error=/home/z50057756/code/RnG_lagernvs_stage2/slurm_logs/%x-%j.err
#
# Full GSO evaluation of ONE stage-1 (uni3t) checkpoint from THIS repo. Same protocol and overrides as
# scripts/uni3t_fulleval_gso.sbatch (EVv2: gso_sim2real_25v_v2, data/gso.txt = 1030 objects, roll 0,
# 4 cond -> 10 target), which is a snapshot of the original repo's script: that one cds into
# /home/z50057756/code/RnG_lagernvs and writes a done-marker there, so it is not used from this repo.
#   posed   : training.val_cam_cond_zero_p=0.0 ; unposed : training.val_cam_cond_zero_p=1.0
# Positional args (no --export):
#   $1 = checkpoint file  $2 = tag  $3 = modes ("posed unposed" or one of them)
#   $4 = config (model/data fields must match the checkpoint's training config)
#   $5 = output root (default: old-disk evaluation/cons)
# One mode takes ~17-25 min on one H200 (jobs 136137 / 137631).
set -euo pipefail
CKPT=${1:?ckpt file}; STEP=$(basename "$CKPT" .pt | tr -cd '0-9'); STEP=$((10#$STEP))
TAG=${2:?tag}; MODES=${3:-"posed unposed"}
CONFIG=${4:-configs/RnGUP_lagernvs_uni3t_b32t6_fp32lr35_const_90k_all287k.yaml}
OUT_ROOT=${5:-/mnt/data-alpha-sg-01/team-camera/home/z50057756/moe_experiments/evaluation/cons}
[ -f "$CKPT" ] || { echo "no such ckpt: $CKPT"; exit 2; }
source /home/z50057756/code/RnG_lagernvs_stage2/scripts_s2/s2_env.sh
mkdir -p "$OUT_ROOT"
GSO_ROOT=/home/z50057756/data/gso_sim2real_25v_v2
SPLIT=data/gso.txt
echo "=== uni3t full GSO eval (stage-2 repo) ==="
echo "ckpt   : $CKPT (step $STEP)"; echo "config : $CONFIG"; echo "gso    : $GSO_ROOT  split $SPLIT"; echo "modes  : $MODES"; echo "out    : $OUT_ROOT"

for MODE in $MODES; do
  case $MODE in posed) ZP=0.0;; unposed) ZP=1.0;; *) echo "bad mode $MODE"; exit 2;; esac
  OUT=$OUT_ROOT/${TAG}_${MODE}_novel
  echo; echo "=== [$MODE] val_cam_cond_zero_p=$ZP -> $OUT ==="; t0=$(date +%s)
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 inference.py \
    --config "$CONFIG" \
    training.checkpoint_dir="$CKPT" \
    inference_out_dir="$OUT" \
    training.val_cam_cond_zero_p=$ZP \
    training.val_dataset_cfgs.root_dir="$GSO_ROOT" \
    training.val_dataset_cfgs.split_file="$SPLIT" \
    training.target_has_input=false \
    training.val_dataset_cfgs.training.target_has_input=false \
    training.val_dataset_cfgs.training.num_views=14 \
    training.val_dataset_cfgs.training.num_input_views=4 \
    training.val_dataset_cfgs.training.num_target_views=10 \
    training.roll_augment_max_deg=0 \
    inference.generate_website=false
  echo "[fulleval] $MODE took $(( $(date +%s)-t0 ))s"
  for f in average_metrics.txt average_metrics_depth.txt average_metrics_pose.txt; do echo "--- $MODE $f ---"; cat "$OUT/$f"; done
  echo "[fulleval] DONE $MODE step=$STEP"
done
echo "[fulleval] ALL DONE step=$STEP"
