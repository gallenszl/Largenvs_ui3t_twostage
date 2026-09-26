#!/bin/bash
#SBATCH --job-name=rng448-render44798-fullEval
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --exclude=lrc-alpha-sg-gpu05,lrc-alpha-sg-gpu06
#SBATCH --output=/home/z50057756/code/RnG_feature_allignment/slurm_logs/%x-%j.out
#SBATCH --error=/home/z50057756/code/RnG_feature_allignment/slurm_logs/%x-%j.err

# Post home-dir-migration version of scripts/inference_obj_repa.sh.
# Differences:
#   - miniconda3 + conda env now live under the pre-migration mount
#     (/mnt/data-alpha-sg-01/team-camera/home/...), not under the new
#     /home/z50057756 -> /mnt/nvme1 symlink. Use absolute paths.
#   - Checkpoint + inference output dirs are also under the pre-migration
#     mount (training was done before 2026-06-27 migration).

set -euo pipefail

# Post-migration: conda.sh has hardcoded /home/z50057756/miniconda3/bin/conda
# which now points at /mnt/nvme1 (empty). Bypass conda activate and just prepend
# the env bin to PATH directly. PyTorch/CUDA find their libs via RPATH.
CONDA_ENV=/mnt/data-alpha-sg-01/team-camera/home/z50057756/conda/envs/rng-fa3
export PATH="$CONDA_ENV/bin:$PATH"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

echo "Python: $(which python)"
python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available())"

export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RNG_ATTENTION_BACKEND=auto
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1

# Post-migration: new $HOME has empty .cache; LPIPS / DINOv3 expect cached
# weights at $HOME/.cache/torch/hub. Point TORCH_HOME at the pre-migration
# cache where vgg16-397923af.pth + dinov3_vitl16_pretrain.pth already live.
export TORCH_HOME=/mnt/data-alpha-sg-01/team-camera/home/z50057756/.cache/torch

ulimit -c 0
mkdir -p /home/z50057756/code/RnG_feature_allignment/slurm_logs
cd /home/z50057756/code/RnG_feature_allignment

CONFIG_PATH="${CONFIG_PATH:-configs/RnGUP_obj_448_bf16_15k_render44798.yaml}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/data-alpha-sg-01/team-camera/home/z50057756/code/RnG_feature_allignment/experiments/checkpoints/RnGUP_obj_448_render44798}"
# Eval outputs go to NEW home (/home/z50057756 -> /mnt/nvme1) per user 2026-06-29.
INFERENCE_OUT_DIR="${INFERENCE_OUT_DIR:-/home/z50057756/code/RnG_feature_allignment/experiments/evaluation/RnGUP_obj_448_render44798}"
SPLIT_FILE="${SPLIT_FILE:-data/gso.txt}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}"

echo "Job: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Config:            ${CONFIG_PATH}"
echo "Checkpoint dir:    ${CHECKPOINT_DIR}"
echo "Inference out dir: ${INFERENCE_OUT_DIR}"
echo "Split file:        ${SPLIT_FILE}"
echo "NPROC_PER_NODE:    ${NPROC_PER_NODE}"
nvidia-smi

# Post-migration: torchrun has shebang hardcoded to a stale python path.
# Use `python -m torch.distributed.run` directly so our PATH-resolved python wins.
#
# All model / data assets live on the pre-migration mount /mnt/data-alpha-sg-01;
# new $HOME (/mnt/nvme1) doesn't have model/, .cache/, FluffyElephant/. Override
# every hardcoded /home path in the config to the absolute pre-migration path.
OLDHOME=/mnt/data-alpha-sg-01/team-camera/home/z50057756

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
    training.val_dataset_cfgs.root_dir=$OLDHOME/FluffyElephant/gso_render_rv \
    training.val_dataset_cfgs.split_file="${SPLIT_FILE}" \
    training.target_has_input=false \
    training.val_dataset_cfgs.training.target_has_input=false \
    training.val_dataset_cfgs.training.num_views=14 \
    training.val_dataset_cfgs.training.num_input_views=4 \
    training.val_dataset_cfgs.training.num_target_views=10
