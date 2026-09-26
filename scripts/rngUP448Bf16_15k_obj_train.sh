#!/bin/bash
set -euo pipefail

module load cuda/12.8 miniconda/latest

eval "$(conda shell.bash hook)"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
conda activate /scratch/zs3325/conda/envs/rng-fa3

export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RNG_ATTENTION_BACKEND="${RNG_ATTENTION_BACKEND:-auto}"

ulimit -c 0

cd /home/zs3325/code/RnG-fa3

python - <<'PY'
import os
import importlib.util
import torch

print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("gpu_count", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is not available. Run this script inside a GPU allocation, e.g. "
        "`srun --partition=interactive --gres=gpu:nvidia_h200_nvl:1 --pty bash`."
    )
if torch.cuda.is_available():
    print("gpu_name", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
for name in ("flash_attn_interface", "xformers"):
    print(f"{name}_available", importlib.util.find_spec(name) is not None)
PY

torchrun --standalone \
  --nnodes=1 \
  --nproc_per_node=1 \
  train.py --config configs/RnGUP_obj_448_bf16_15k.yaml \
  exp_name=RnGUP_448_freeze_RAEHead_DPTHead_FA3 \
  "$@"
