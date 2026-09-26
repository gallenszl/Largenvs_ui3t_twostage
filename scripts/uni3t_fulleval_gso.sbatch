#!/bin/bash
#SBATCH --job-name=uni3t-fulleval
#SBATCH --partition=gpu
#SBATCH --qos=lowest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:30:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/z50057756/code/RnG_lagernvs/slurm_logs/%x-%j.out
#SBATCH --error=/home/z50057756/code/RnG_lagernvs/slurm_logs/%x-%j.err
#
# Full GSO evaluation (EVv2 protocol: gso_sim2real_25v_v2, data/gso.txt = 1030 objects,
# roll 0, 4 cond -> 10 target) of ONE uni3t checkpoint, in BOTH camera modes:
#   posed   : training.val_cam_cond_zero_p=0.0  (cond views' camera tokens given)
#   unposed : training.val_cam_cond_zero_p=1.0  (all camera tokens zero; the target view's
#             token is never consumed -- models/encoder_decoder.py:65 slices it off -- so
#             this equals "cond views zeroed", i.e. the same unposed as training's 40% drop
#             and as thip01's posed_input=False)
# Positional args (no --export, see memory cluster_slurm_export_getuserenv_held):
#   $1 = checkpoint file      $2 = tag (default: derived from the step in the filename)
#   $3 = modes (default "posed unposed")
#   $4 = config (default: the uni3t config; pass the baseline config to evaluate
#        PLN2pose_all287k_b32t6_fp32lr35_const checkpoints under the same protocol)
#   $5 = output root (default: experiments/evaluation in this repo on nvme1; the 09-24
#        anneal evals pass the old disk because nvme1 has ~7 GB free)
set -euo pipefail
CKPT=${1:?ckpt file}; STEP=$(basename "$CKPT" .pt | tr -cd '0-9'); STEP=$((10#$STEP))
TAG=${2:-uni3t_step${STEP}}; MODES=${3:-"posed unposed"}
CONFIG=${4:-configs/RnGUP_lagernvs_uni3t_b32t6_fp32lr35_const_90k_all287k.yaml}
OUT_ROOT=${5:-/home/z50057756/code/RnG_lagernvs/experiments/evaluation}
[ -f "$CKPT" ] || { echo "no such ckpt: $CKPT"; exit 2; }

CONDA_ENV=/mnt/data-alpha-sg-01/team-camera/home/z50057756/conda/envs/rng-fa3
export PATH="$CONDA_ENV/bin:$PATH"; export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
export TORCH_HOME=/home/z50057756/torch_home_lagernvs
ulimit -c 0; cd /home/z50057756/code/RnG_lagernvs
GSO_ROOT=/home/z50057756/data/gso_sim2real_25v_v2
SPLIT=data/gso.txt

echo "=== uni3t full GSO eval ==="; echo "job $SLURM_JOB_ID node $(hostname) branch $(git rev-parse --abbrev-ref HEAD) commit $(git rev-parse --short HEAD)"
echo "ckpt   : $CKPT (step $STEP)"; echo "config : $CONFIG"; echo "gso    : $GSO_ROOT ($(ls $GSO_ROOT | grep -v _logs | wc -l) scenes)  split $SPLIT"; echo "modes  : $MODES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

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
case "$CKPT" in
  */PLN2uni3t_all287k_b32t6_fp32lr35_const/ckpt_*.pt)
    mkdir -p slurm_logs/uni3t_fulleval_state
    echo "$(date '+%F %T') job=$SLURM_JOB_ID modes='$MODES' tag=$TAG" > slurm_logs/uni3t_fulleval_state/fulleval_${STEP}.done
    echo "[fulleval] wrote done-marker fulleval_${STEP}.done" ;;
esac
