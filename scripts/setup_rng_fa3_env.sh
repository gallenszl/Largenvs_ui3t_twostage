#!/bin/bash
set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/scratch/zs3325/conda/envs/rng-fa3}"
FA_REPO="${FA_REPO:-/scratch/zs3325/src/flash-attention}"

module load cuda/12.8 miniconda/latest

eval "$(conda shell.bash hook)"

mkdir -p /scratch/zs3325/conda/envs
mkdir -p /scratch/zs3325/conda/pkgs
mkdir -p /scratch/zs3325/pip-cache
mkdir -p /scratch/zs3325/src
mkdir -p /scratch/zs3325/tmp

export CONDA_ENVS_PATH=/scratch/zs3325/conda/envs
export CONDA_PKGS_DIRS=/scratch/zs3325/conda/pkgs
export PIP_CACHE_DIR=/scratch/zs3325/pip-cache
export TMPDIR=/scratch/zs3325/tmp

if [ ! -d "${ENV_PREFIX}" ]; then
  conda create -y -p "${ENV_PREFIX}" python=3.10
fi

export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
conda activate "${ENV_PREFIX}"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r /home/zs3325/code/RnG-fa3/requirements-fa3.txt
conda install -y -c nvidia cuda-nvcc=12.8

export CUDA_HOME="${CONDA_PREFIX}"
export CUDA_PATH="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"

NVIDIA_SITE="$(python - <<'PY'
from pathlib import Path
import site

paths = []
try:
    paths.extend(site.getsitepackages())
except Exception:
    pass
try:
    paths.append(site.getusersitepackages())
except Exception:
    pass

for path in paths:
    candidate = Path(path) / "nvidia"
    if candidate.exists():
        print(candidate)
        break
PY
)"

join_by_colon() {
  local IFS=:
  echo "$*"
}

cuda_include_paths=("${CUDA_HOME}/targets/x86_64-linux/include" "${CUDA_HOME}/include")
cuda_library_paths=("${CUDA_HOME}/lib" "${CUDA_HOME}/lib64" "${CUDA_HOME}/targets/x86_64-linux/lib")

if [ -n "${NVIDIA_SITE}" ]; then
  for component in cuda_runtime cuda_nvcc cublas cudnn cufft curand cusolver cusparse nccl nvtx cuda_cupti cuda_nvrtc nvjitlink cufile; do
    if [ -d "${NVIDIA_SITE}/${component}/include" ]; then
      cuda_include_paths+=("${NVIDIA_SITE}/${component}/include")
    fi
    if [ -d "${NVIDIA_SITE}/${component}/lib" ]; then
      cuda_library_paths+=("${NVIDIA_SITE}/${component}/lib")
    fi
  done
fi

CUDA_INCLUDE_PATH="$(join_by_colon "${cuda_include_paths[@]}")"
CUDA_LIBRARY_PATH="$(join_by_colon "${cuda_library_paths[@]}")"

export LD_LIBRARY_PATH="${CUDA_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${CUDA_LIBRARY_PATH}:${LIBRARY_PATH:-}"
export CPATH="${CUDA_INCLUDE_PATH}:${CPATH:-}"
export CPLUS_INCLUDE_PATH="${CUDA_INCLUDE_PATH}:${CPLUS_INCLUDE_PATH:-}"
export C_INCLUDE_PATH="${CUDA_INCLUDE_PATH}:${C_INCLUDE_PATH:-}"

if [ ! -d "${FA_REPO}" ]; then
  git clone https://github.com/Dao-AILab/flash-attention.git "${FA_REPO}"
else
  git -C "${FA_REPO}" fetch --all --tags
  git -C "${FA_REPO}" checkout main
  git -C "${FA_REPO}" pull --ff-only
fi

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0a}"
export MAX_JOBS="${MAX_JOBS:-2}"
export NVCC_THREADS="${NVCC_THREADS:-1}"

# This build is intentionally tailored to the RnG H200 BF16 training path:
# SM90 only, BF16 only, head_dim 64/128 kernels. Head dim 32 is covered by
# hdim64 in the FA3 interface.
export FLASH_ATTENTION_DISABLE_SM80="${FLASH_ATTENTION_DISABLE_SM80:-TRUE}"
export FLASH_ATTENTION_DISABLE_FP16="${FLASH_ATTENTION_DISABLE_FP16:-TRUE}"
export FLASH_ATTENTION_DISABLE_FP8="${FLASH_ATTENTION_DISABLE_FP8:-TRUE}"
export FLASH_ATTENTION_DISABLE_SPLIT="${FLASH_ATTENTION_DISABLE_SPLIT:-TRUE}"
export FLASH_ATTENTION_DISABLE_PAGEDKV="${FLASH_ATTENTION_DISABLE_PAGEDKV:-TRUE}"
export FLASH_ATTENTION_DISABLE_APPENDKV="${FLASH_ATTENTION_DISABLE_APPENDKV:-TRUE}"
export FLASH_ATTENTION_DISABLE_LOCAL="${FLASH_ATTENTION_DISABLE_LOCAL:-TRUE}"
export FLASH_ATTENTION_DISABLE_SOFTCAP="${FLASH_ATTENTION_DISABLE_SOFTCAP:-TRUE}"
export FLASH_ATTENTION_DISABLE_PACKGQA="${FLASH_ATTENTION_DISABLE_PACKGQA:-TRUE}"
export FLASH_ATTENTION_DISABLE_VARLEN="${FLASH_ATTENTION_DISABLE_VARLEN:-TRUE}"
export FLASH_ATTENTION_DISABLE_HDIM96="${FLASH_ATTENTION_DISABLE_HDIM96:-TRUE}"
export FLASH_ATTENTION_DISABLE_HDIM192="${FLASH_ATTENTION_DISABLE_HDIM192:-TRUE}"
export FLASH_ATTENTION_DISABLE_HDIM256="${FLASH_ATTENTION_DISABLE_HDIM256:-TRUE}"
export FLASH_ATTENTION_DISABLE_HDIMDIFF64="${FLASH_ATTENTION_DISABLE_HDIMDIFF64:-TRUE}"
export FLASH_ATTENTION_DISABLE_HDIMDIFF192="${FLASH_ATTENTION_DISABLE_HDIMDIFF192:-TRUE}"

cd "${FA_REPO}/hopper"
python setup.py install

python - <<'PY'
import importlib.util
import torch

print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("flash_attn_interface", importlib.util.find_spec("flash_attn_interface") is not None)
if torch.cuda.is_available():
    print("capability", torch.cuda.get_device_capability(0))
PY
