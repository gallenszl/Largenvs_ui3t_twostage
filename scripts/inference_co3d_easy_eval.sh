#!/bin/bash
#SBATCH --job-name=co3d-easy-eval
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --exclude=lrc-alpha-sg-gpu05,lrc-alpha-sg-gpu06
#SBATCH --output=/home/z50057756/code/RnG_feature_allignment/slurm_logs/%x-%j.out
#SBATCH --error=/home/z50057756/code/RnG_feature_allignment/slurm_logs/%x-%j.err

set -euo pipefail

CONDA_ENV=/mnt/data-alpha-sg-01/team-camera/home/z50057756/conda/envs/rng-fa3
export PATH="$CONDA_ENV/bin:$PATH"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RNG_ATTENTION_BACKEND=auto
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1
export TORCH_HOME=/mnt/data-alpha-sg-01/team-camera/home/z50057756/.cache/torch

ulimit -c 0
mkdir -p /home/z50057756/code/RnG_feature_allignment/slurm_logs
cd /home/z50057756/code/RnG_feature_allignment

OLDHOME=/mnt/data-alpha-sg-01/team-camera/home/z50057756

CONFIG_PATH="${CONFIG_PATH:-configs/RnGUP_obj_448_bf16_15k_render44798.yaml}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?must set CHECKPOINT_DIR}"
INFERENCE_OUT_DIR="${INFERENCE_OUT_DIR:?must set INFERENCE_OUT_DIR}"
VIEW_IDX_FILE="${VIEW_IDX_FILE:?must set VIEW_IDX_FILE}"
GSO_ROOT="${GSO_ROOT:-/home/z50057756/data/co3d_teddybear_gsoformat_objdepth}"
SPLIT_FILE="${SPLIT_FILE:-data/co3d_teddybear_pilot100.txt}"
NUM_VIEWS="${NUM_VIEWS:?must set NUM_VIEWS}"
NUM_INPUT_VIEWS="${NUM_INPUT_VIEWS:-4}"
NUM_TARGET_VIEWS="${NUM_TARGET_VIEWS:?must set NUM_TARGET_VIEWS}"
NPROC_PER_NODE="${SLURM_GPUS_ON_NODE:-1}"

echo "=== CO3D easy-eval inference ==="
echo "Job:              ${SLURM_JOB_ID:-local}"
echo "Node:             $(hostname)"
echo "Config:           ${CONFIG_PATH}"
echo "Checkpoint:       ${CHECKPOINT_DIR}"
echo "Output:           ${INFERENCE_OUT_DIR}"
echo "Data root:        ${GSO_ROOT}"
echo "Split:            ${SPLIT_FILE} ($(wc -l < "${SPLIT_FILE}") scenes before optional view filter)"
echo "View idx file:    ${VIEW_IDX_FILE}"
echo "Views/input/tgt:  ${NUM_VIEWS}/${NUM_INPUT_VIEWS}/${NUM_TARGET_VIEWS}"
echo "NPROC_PER_NODE:   ${NPROC_PER_NODE}"
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
    inference_out_dir="${INFERENCE_OUT_DIR}" \
    inference.view_idx_file_path="${VIEW_IDX_FILE}" \
    training.val_dataset_cfgs.root_dir="${GSO_ROOT}" \
    training.val_dataset_cfgs.split_file="${SPLIT_FILE}" \
    training.target_has_input=false \
    training.val_dataset_cfgs.training.target_has_input=false \
    training.val_dataset_cfgs.training.num_views="${NUM_VIEWS}" \
    training.val_dataset_cfgs.training.num_input_views="${NUM_INPUT_VIEWS}" \
    training.val_dataset_cfgs.training.num_target_views="${NUM_TARGET_VIEWS}" \
    training.roll_augment_max_deg=0
