#!/bin/bash
#SBATCH --job-name=s2-s1ref
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --output=/home/z50057756/code/RnG_lagernvs_stage2/slurm_logs/%x-%j.out
#SBATCH --error=/home/z50057756/code/RnG_lagernvs_stage2/slurm_logs/%x-%j.err
# G9 reference: the stock stage-1 inference.py (LagerNVSInRnG, no stage-2 code) with ckpt_70000 on the exact set the
# stage-2 training validates on at step 0 (gso_sim2real_25v, data/gso_subset64.txt, roll 0, 4 cond -> 10 target),
# posed and unposed in parallel.  $1 output root.
set -uo pipefail
source /home/z50057756/code/RnG_lagernvs_stage2/scripts_s2/s2_env.sh
OUT_ROOT=${1:?out root}
CKPT=/mnt/data-alpha-sg-01/team-camera/home/z50057756/moe_experiments/checkpoints/PLN2uni3t_all287k_b32t6_fp32lr35_wsd60k_a10k/ckpt_0000000000070000.pt
CONFIG=configs/RnGUP_lagernvs_uni3t_b32t6_fp32lr35_wsd60k_a10k_all287k.yaml
mkdir -p "$OUT_ROOT"
i=0; pids=()
for MODE in posed unposed; do
  case $MODE in posed) ZP=0.0;; unposed) ZP=1.0;; esac
  CUDA_VISIBLE_DEVICES=$i python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 inference.py \
    --config "$CONFIG" training.checkpoint_dir="$CKPT" inference_out_dir="$OUT_ROOT/s1ref_subset64_${MODE}" \
    training.val_cam_cond_zero_p=$ZP \
    training.val_dataset_cfgs.root_dir=/home/z50057756/data/gso_sim2real_25v \
    training.val_dataset_cfgs.split_file=data/gso_subset64.txt \
    training.target_has_input=false training.val_dataset_cfgs.training.target_has_input=false \
    training.val_dataset_cfgs.training.num_views=14 training.val_dataset_cfgs.training.num_input_views=4 \
    training.val_dataset_cfgs.training.num_target_views=10 \
    training.roll_augment_max_deg=0 training.val_dataset_cfgs.roll_augment_max_deg=0 \
    inference.generate_website=false > "$OUT_ROOT/s1ref_subset64_${MODE}.log" 2>&1 &
  pids+=($!); i=$((i+1))
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
for MODE in posed unposed; do
  echo "=== $MODE"; tail -n 2 "$OUT_ROOT/s1ref_subset64_${MODE}.log"
  head -3 "$OUT_ROOT/s1ref_subset64_${MODE}/average_metrics.txt" 2>/dev/null
done
echo "[s2-s1ref] DONE rc=$rc"
exit $rc
