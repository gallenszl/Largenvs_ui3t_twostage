#!/bin/bash
#SBATCH --job-name=s2-s1ref-fp32m
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/slurm_logs/%x-%j.out
#SBATCH --error=/home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/slurm_logs/%x-%j.err
# G9 step-0 reference with the metrics computed OUTSIDE autocast (fp32), exactly as train.py's validation does:
# stage-1 ckpt_70000 through inference_s2.py with stage 2 off (bitwise equal to the stock inference.py, job 6906909),
# gso_sim2real_25v + data/gso_subset64.txt, roll 0, 4 cond -> 10 target, posed then unposed.
#   $1 output root; writes <root>/s1ref_subset64_{posed,unposed} (the names tools_s2/s2_train_check.py step0 reads)
set -uo pipefail
source /home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/scripts_s2/s2_env.sh
OUT_ROOT=${1:?out root}
mkdir -p "$OUT_ROOT"
rc_all=0
for MODE in posed unposed; do
  case $MODE in posed) ZP=0.0;; unposed) ZP=1.0;; esac
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 inference_s2.py \
    --config configs/S2P8_uni3t70k_b32t6_lr35_a21k26k.yaml inference.s2_ckpt=none inference.metrics_autocast=false \
    inference_out_dir="$OUT_ROOT/s1ref_subset64_${MODE}" training.val_cam_cond_zero_p=$ZP \
    training.val_dataset_cfgs.root_dir=/projects/u6ga/zhelun/RnG_samples_eval/gso_sim2real_25v \
    training.val_dataset_cfgs.split_file=data/gso_subset64.txt \
    training.target_has_input=false training.val_dataset_cfgs.training.target_has_input=false \
    training.val_dataset_cfgs.training.num_views=14 training.val_dataset_cfgs.training.num_input_views=4 \
    training.val_dataset_cfgs.training.num_target_views=10 \
    training.roll_augment_max_deg=0 training.val_dataset_cfgs.roll_augment_max_deg=0 \
    inference.generate_website=false > "$OUT_ROOT/s1ref_subset64_${MODE}.log" 2>&1
  rc=$?; rc_all=$((rc_all | rc)); echo "[s2-s1ref-fp32m] $MODE rc=$rc $(date +%T)"
  head -1 "$OUT_ROOT/s1ref_subset64_${MODE}/average_metrics.txt" 2>/dev/null
done
echo "[s2-s1ref-fp32m] DONE rc=$rc_all"
exit $rc_all
