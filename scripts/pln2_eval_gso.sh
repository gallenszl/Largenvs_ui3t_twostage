#!/bin/bash
#SBATCH --job-name=rng448-gsoSim2real
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --exclude=lrc-alpha-sg-gpu05,lrc-alpha-sg-gpu06
#SBATCH --output=/home/z50057756/code/RnG_lagernvs/slurm_logs/%x-%j.out
#SBATCH --error=/home/z50057756/code/RnG_lagernvs/slurm_logs/%x-%j.err

# Inference on sim2real-rendered GSO (plan §8.7 Step 4).
# Required env vars: ROLL_DEG (0 or 10), RUN_TAG (e.g. "noroll" or "roll10")
# Output: <eval_root>/RnGUP_render44798_gsoSim2real_<RUN_TAG>/

set -euo pipefail

CONDA_ENV=/mnt/data-alpha-sg-01/team-camera/home/z50057756/conda/envs/rng-fa3
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RNG_ATTENTION_BACKEND=auto
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1
export TORCH_HOME=/home/z50057756/torch_home_lagernvs

ulimit -c 0
mkdir -p /home/z50057756/code/RnG_lagernvs/slurm_logs
cd /home/z50057756/code/RnG_lagernvs

ROLL_DEG="${ROLL_DEG:?must set ROLL_DEG (e.g. 0 or 10)}"
RUN_TAG="${RUN_TAG:?must set RUN_TAG (e.g. noroll or roll10)}"

OLDHOME=/mnt/data-alpha-sg-01/team-camera/home/z50057756
GSO_ROOT="${GSO_ROOT:-/home/z50057756/data/gso_sim2real_25v}"

CONFIG_PATH="${CONFIG_PATH:-configs/RnGUP_lagernvs_rgb256_15k.yaml}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$OLDHOME/code/RnG_feature_allignment/experiments/checkpoints/RnGUP_obj_448_render44798}"
OUT_PREFIX="${OUT_PREFIX:-PLN2_gsoSim2real}"
# Eval outputs go to NEW home (/home/z50057756 -> /mnt/nvme1) per user 2026-06-29.
INFERENCE_OUT_DIR=/home/z50057756/code/RnG_lagernvs/experiments/evaluation/${OUT_PREFIX}_${RUN_TAG}
SPLIT_FILE="${SPLIT_FILE:-data/gso.txt}"
NPROC_PER_NODE="${SLURM_GPUS_ON_NODE:-1}"

echo "=== Sim2real GSO inference (run=$RUN_TAG, roll=${ROLL_DEG}°) ==="
echo "Job:          ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "GSO data:     $GSO_ROOT  ($(ls $GSO_ROOT 2>/dev/null | grep -v _logs | wc -l) scenes)"
echo "Ckpt:         $CHECKPOINT_DIR"
echo "Out:          $INFERENCE_OUT_DIR"
echo "Split:        $SPLIT_FILE ($(wc -l < $SPLIT_FILE) scenes)"
echo "Roll deg:     $ROLL_DEG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "===="

python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    inference.py \
    --config "${CONFIG_PATH}" \
    model.pretrained_path=$OLDHOME/model/VGGT/model.pt \
    model.rae_decoder.config_path=$OLDHOME/code/RAE/configs/decoder/ViTXL \
    model.rae_decoder.pretrained_decoder_path=$OLDHOME/model/RAE/decoders/dinov2/wReg_base/ViTXL_n08_i512/model.pt \
    model.rae_decoder.normalization_stat_path=$OLDHOME/model/RAE/stats/dinov2/wReg_base/imagenet1k_512/stat.pt \
    training.feature_alignment.target_encoder_weights=$OLDHOME/.cache/torch/hub/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
    training.feature_alignment.target_encoder_hub_dir=$OLDHOME/.cache/torch/hub/facebookresearch_dinov3_94a96ac83c2446f15f9bdcfae23cad3c6a9d4988 \
    training.checkpoint_dir="${CHECKPOINT_DIR}" \
    ${VIEW_IDX_FILE:+inference.view_idx_file_path="$VIEW_IDX_FILE"} \
    inference_out_dir="${INFERENCE_OUT_DIR}" \
    training.val_dataset_cfgs.root_dir=$GSO_ROOT \
    training.val_dataset_cfgs.split_file="${SPLIT_FILE}" \
    training.target_has_input=false \
    training.val_dataset_cfgs.training.target_has_input=false \
    training.val_dataset_cfgs.training.num_views=${NUM_VIEWS:-14} \
    training.val_dataset_cfgs.training.num_input_views=4 \
    training.val_dataset_cfgs.training.num_target_views=${NUM_TARGET_VIEWS:-10} \
    training.roll_augment_max_deg=$ROLL_DEG
