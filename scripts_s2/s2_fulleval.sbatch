#!/bin/bash
#SBATCH --job-name=s2-fulleval
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --output=/home/z50057756/code/RnG_lagernvs_stage2/slurm_logs/%x-%j.out
#SBATCH --error=/home/z50057756/code/RnG_lagernvs_stage2/slurm_logs/%x-%j.err
# Stage-2 evaluation (EVv2 protocol of scripts/uni3t_fulleval_gso.sbatch: gso_sim2real_25v_v2, data/gso.txt,
# roll 0, 4 cond -> 10 target), posed and unposed in parallel on two GPUs.
#   $1 stage-2 ckpt file | none (none = stage 2 off -> reproduces stage 1)   $2 tag
#   $3 modes (default "posed unposed")   $4 config   $5 output root   $6 first_n (optional)
set -uo pipefail
source /home/z50057756/code/RnG_lagernvs_stage2/scripts_s2/s2_env.sh
CKPT=${1:?ckpt|none}; TAG=${2:?tag}; MODES=${3:-"posed unposed"}; CONFIG=${4:?config}; OUT_ROOT=${5:?out root}
FIRST_N=${6:-}
GSO_ROOT=/home/z50057756/data/gso_sim2real_25v_v2
SPLIT=data/gso.txt
mkdir -p "$OUT_ROOT"
echo "[s2-eval] ckpt=$CKPT tag=$TAG modes=$MODES config=$CONFIG out=$OUT_ROOT first_n=${FIRST_N:-all}"
i=0; pids=()
for MODE in $MODES; do
  case $MODE in posed) ZP=0.0;; unposed) ZP=1.0;; *) echo "bad mode $MODE"; exit 2;; esac
  OUT=$OUT_ROOT/${TAG}_${MODE}_novel
  EXTRA=""; [ -n "$FIRST_N" ] && EXTRA="inference.first_n=$FIRST_N"
  CUDA_VISIBLE_DEVICES=$i python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 inference_s2.py \
    --config "$CONFIG" inference.s2_ckpt="$CKPT" inference_out_dir="$OUT" \
    training.val_cam_cond_zero_p=$ZP \
    training.val_dataset_cfgs.root_dir="$GSO_ROOT" training.val_dataset_cfgs.split_file="$SPLIT" \
    training.target_has_input=false training.val_dataset_cfgs.training.target_has_input=false \
    training.val_dataset_cfgs.training.num_views=14 training.val_dataset_cfgs.training.num_input_views=4 \
    training.val_dataset_cfgs.training.num_target_views=10 \
    training.roll_augment_max_deg=0 training.val_dataset_cfgs.roll_augment_max_deg=0 \
    inference.generate_website=false $EXTRA > "$OUT_ROOT/${TAG}_${MODE}.log" 2>&1 &
  pids+=($!); i=$((i+1))
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
for MODE in $MODES; do
  echo "=== $MODE ==="; tail -n 3 "$OUT_ROOT/${TAG}_${MODE}.log"
  for f in average_metrics.txt average_metrics_depth.txt average_metrics_pose.txt; do
    echo "--- $f"; cat "$OUT_ROOT/${TAG}_${MODE}_novel/$f" 2>/dev/null | head -3
  done
done
echo "[s2-eval] DONE rc=$rc"
exit $rc
