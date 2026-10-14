#!/bin/bash
#SBATCH --job-name=s2-s1diag
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --output=/home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/slurm_logs/%x-%j.out
#SBATCH --error=/home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/slurm_logs/%x-%j.err
# Where do the tiny per-object differences between the stock stage-1 evaluation (inference.py, job 6906869) and the
# stage-2 entry with stage 2 off (inference_s2.py none, job 6906870) come from?  Hypothesis: cudnn.benchmark
# (setup.py) picks convolution algorithms by timing, per process, so LPIPS (VGG) and the DPT point head round
# differently in different processes.  All runs on this one GPU, v1 subset64, one process each:
#   R1  stock, posed, benchmark ON  (second run of the same path; vs 6906869)
#   R2  stock, posed + unposed, benchmark OFF
#   R3  stage-2 entry with stage 2 off, posed + unposed, benchmark OFF   -> must equal R2 exactly
#   R4  stock, posed, benchmark OFF again                                 -> must equal R2 exactly
#   $1 = output root that holds the 6906869 / 6906870 directories
set -uo pipefail
source /home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/scripts_s2/s2_env.sh
O=${1:?out root}
CKPT=/projects/u6ga/zhelun/RnG_weights/stage1/moe_experiments/checkpoints/PLN2uni3t_all287k_b32t6_fp32lr35_wsd60k_a10k/ckpt_0000000000070000.pt
GSO=/projects/u6ga/zhelun/RnG_samples_eval/gso_sim2real_25v
COMMON="training.val_dataset_cfgs.root_dir=$GSO training.val_dataset_cfgs.split_file=data/gso_subset64.txt \
 training.target_has_input=false training.val_dataset_cfgs.training.target_has_input=false \
 training.val_dataset_cfgs.training.num_views=14 training.val_dataset_cfgs.training.num_input_views=4 \
 training.val_dataset_cfgs.training.num_target_views=10 training.roll_augment_max_deg=0 \
 training.val_dataset_cfgs.roll_augment_max_deg=0 inference.generate_website=false"
zp() { [ "$1" = posed ] && echo 0.0 || echo 1.0; }
stock() {  # $1 mode  $2 out dir  $3 CUDNN_BENCHMARK
  CUDNN_BENCHMARK=$3 python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 inference.py \
    --config configs/RnGUP_lagernvs_uni3t_b32t6_fp32lr35_wsd60k_a10k_all287k.yaml training.checkpoint_dir="$CKPT" \
    inference_out_dir="$2" training.val_cam_cond_zero_p=$(zp $1) $COMMON > "$2.log" 2>&1
  echo "[s2-s1diag] stock $1 bench=$3 -> $2 rc=$? $(date +%T)"
}
s2off() {  # $1 mode  $2 out dir (the script appends nothing: stage 2 off writes <dir> only)  $3 CUDNN_BENCHMARK
  CUDNN_BENCHMARK=$3 python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 inference_s2.py \
    --config configs/S2P8_uni3t70k_b32t6_lr35_a21k26k.yaml inference.s2_ckpt=none inference_out_dir="$2" \
    training.val_cam_cond_zero_p=$(zp $1) $COMMON > "$2.log" 2>&1
  echo "[s2-s1diag] s2off $1 bench=$3 -> $2 rc=$? $(date +%T)"
}
D=$O/diag_${SLURM_JOB_ID}
mkdir -p "$D"
stock posed "$D/R1_stock_posed_bench1" 1
for m in posed unposed; do stock $m "$D/R2_stock_${m}_bench0" 0; done
for m in posed unposed; do s2off $m "$D/R3_s2off_${m}_bench0" 0; done
stock posed "$D/R4_stock_posed_bench0" 0
PD="python tools_s2/s2_pairdiff.py"
echo "=== R1 (stock, bench on, run 2) vs 6906869 (stock, bench on, run 1)"; $PD "$D/R1_stock_posed_bench1" "$O/s1ref_subset64_posed"
for m in posed unposed; do
  echo "=== R3 (s2off, bench off) vs R2 (stock, bench off), $m"; $PD "$D/R3_s2off_${m}_bench0" "$D/R2_stock_${m}_bench0"
done
echo "=== R4 vs R2 (stock, bench off, two processes), posed"; $PD "$D/R4_stock_posed_bench0" "$D/R2_stock_posed_bench0"
echo "=== R2 (stock, bench off) vs 6906869 (stock, bench on)"
for m in posed unposed; do $PD "$D/R2_stock_${m}_bench0" "$O/s1ref_subset64_${m}"; done
for m in posed unposed; do echo "--- R2 $m means"; head -1 "$D/R2_stock_${m}_bench0/average_metrics.txt"; done
echo "[s2-s1diag] DONE"
