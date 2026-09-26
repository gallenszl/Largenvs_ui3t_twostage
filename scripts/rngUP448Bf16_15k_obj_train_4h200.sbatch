#!/bin/bash
#SBATCH --job-name=rng448-sft-repal7-8h200
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=208
#SBATCH --mem=1500G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=lrc-alpha-sg-gpu05
#SBATCH --output=/home/z50057756/code/RnG_feature_allignment/slurm_logs/%x-%j.out
#SBATCH --error=/home/z50057756/code/RnG_feature_allignment/slurm_logs/%x-%j.err

set -euo pipefail

source /home/z50057756/miniconda3/etc/profile.d/conda.sh
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"   # satisfy conda env's cuda-nvcc activate hook under set -u
conda activate /home/z50057756/conda/envs/rng-fa3

export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RNG_ATTENTION_BACKEND=auto
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

ulimit -c 0

mkdir -p /home/z50057756/code/RnG_feature_allignment/slurm_logs

cd /home/z50057756/code/RnG_feature_allignment

echo "Job ID: ${SLURM_JOB_ID}"
echo "Restart count: ${SLURM_RESTART_COUNT:-0}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

python - <<'PY'
import importlib.util
import torch

print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("gpu_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu_name", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
for name in ("flash_attn_interface", "xformers"):
    print(f"{name}_available", importlib.util.find_spec(name) is not None)
PY

# Training from pretrained weights, disabled:
# torchrun --nproc_per_node=8 --nnodes=1 \
#   --rdzv_id=${SLURM_JOB_ID} --rdzv_backend=c10d \
#   --rdzv_endpoint=localhost:29513 \
#   train.py --config configs/RnGUP_obj_448_bf16_15k.yaml \
#   exp_name=FA3_repa_dinov3_l17 \
#   training.use_tar=true \
#   training.tar_root_path=/home/z50057756/FluffyElephant_tar



# SFT with RAE/DPT heads unfrozen, starting from the freeze-stage checkpoint:
torchrun --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  train.py --config configs/RnGUP_obj_448_bf16_15k_unfreeze_sft.yaml \
  exp_name=FA3_repal7_rng_sft \
  training.resume_ckpt=experiments/checkpoints/FA3_repa_dinov3_l7/ckpt_0000000000015000.pt \
  training.reset_training_state=true \
  training.use_tar=true \
  training.tar_root_path=/home/z50057756/FluffyElephant_tar
