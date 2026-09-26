#!/bin/bash
set -euo pipefail

# Example installer for reproducing the RnG-fa3 environment on another server.
#
# Before running:
#   1. Clone/copy the RnG-fa3 code to PROJECT_DIR.
#   2. Make sure the server has a CUDA 12.8-capable driver/toolchain.
#   3. For FA3 acceleration, run on Hopper GPUs such as H100/H200.
#
# Example:
#   PROJECT_DIR=/home/user/code/RnG-fa3 \
#   ENV_PREFIX=/scratch/user/conda/envs/rng-fa3 \
#   WORK_ROOT=/scratch/user/rng-fa3-build \
#   bash scripts/install_rng_fa3_env_example.sh

PROJECT_DIR="${PROJECT_DIR:-${HOME}/code/RnG-fa3}"
ENV_PREFIX="${ENV_PREFIX:-${HOME}/conda/envs/rng-fa3}"
WORK_ROOT="${WORK_ROOT:-${HOME}/rng-fa3-build}"
FA_REPO="${FA_REPO:-${WORK_ROOT}/flash-attention}"
LOCK_FILE="${LOCK_FILE:-${PROJECT_DIR}/requirements-fa3-current-lock.txt}"
FALLBACK_REQUIREMENTS="${FALLBACK_REQUIREMENTS:-${PROJECT_DIR}/requirements-fa3.txt}"

# This is the flash-attention commit currently used in /home/zs3325/code/RnG-fa3.
FA_COMMIT="${FA_COMMIT:-2e53092aa70fccd3f04013a01a52dc20c619e62b}"

# On clusters with environment modules, these defaults match the current setup.
# If the new server does not use modules, set USE_MODULES=0 and ensure conda is
# already available in PATH.
USE_MODULES="${USE_MODULES:-auto}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.8}"
CONDA_MODULE="${CONDA_MODULE:-miniconda/latest}"

if [ ! -d "${PROJECT_DIR}" ]; then
  echo "PROJECT_DIR does not exist: ${PROJECT_DIR}" >&2
  exit 1
fi

if [ ! -f "${LOCK_FILE}" ] && [ ! -f "${FALLBACK_REQUIREMENTS}" ]; then
  echo "Missing both ${LOCK_FILE} and ${FALLBACK_REQUIREMENTS}" >&2
  exit 1
fi

if [ "${USE_MODULES}" != "0" ] && type module >/dev/null 2>&1; then
  module load "${CUDA_MODULE}" "${CONDA_MODULE}"
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available. Load miniconda/anaconda first, or set USE_MODULES=1 with a valid CONDA_MODULE." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

mkdir -p "$(dirname "${ENV_PREFIX}")"
mkdir -p "${WORK_ROOT}/conda-pkgs" "${WORK_ROOT}/pip-cache" "${WORK_ROOT}/src" "${WORK_ROOT}/tmp"

export CONDA_PKGS_DIRS="${WORK_ROOT}/conda-pkgs"
export PIP_CACHE_DIR="${WORK_ROOT}/pip-cache"
export TMPDIR="${WORK_ROOT}/tmp"

if [ ! -d "${ENV_PREFIX}" ]; then
  conda create -y -p "${ENV_PREFIX}" python=3.10.20
fi

export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
conda activate "${ENV_PREFIX}"

python -m pip install --upgrade pip==26.1.1 setuptools==82.0.1 wheel==0.47.0

# Prefer the current-env lock file for the closest reproduction. It pins
# transitive pip deps observed in /scratch/zs3325/conda/envs/rng-fa3.
if [ -f "${LOCK_FILE}" ]; then
  python -m pip install -r "${LOCK_FILE}"
else
  python -m pip install -r "${FALLBACK_REQUIREMENTS}"
fi

# Avoid libGL.so.1 issues on headless cluster nodes.
python -m pip uninstall -y opencv-python || true
python -m pip install --force-reinstall --no-deps opencv-python-headless==4.7.0.72

# Provide nvcc inside the conda env for compiling FA3.
conda install -y -c nvidia cuda-nvcc=12.8.93

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

if [ ! -d "${FA_REPO}/.git" ]; then
  git clone https://github.com/Dao-AILab/flash-attention.git "${FA_REPO}"
fi

git -C "${FA_REPO}" fetch --all --tags
git -C "${FA_REPO}" checkout "${FA_COMMIT}"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0a}"
export MAX_JOBS="${MAX_JOBS:-2}"
export NVCC_THREADS="${NVCC_THREADS:-1}"

# Build a compact FA3 package for the current RnG BF16 H100/H200 path.
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
print("cuda_available", torch.cuda.is_available())
print("flash_attn_interface", importlib.util.find_spec("flash_attn_interface") is not None)
print("xformers", importlib.util.find_spec("xformers") is not None)
if torch.cuda.is_available():
    print("gpu_name", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
PY

echo "Done. Activate with: conda activate ${ENV_PREFIX}"
