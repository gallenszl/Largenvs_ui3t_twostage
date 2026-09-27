# shared environment of the stage-2 job scripts (sourced)
CONDA_ENV=/mnt/data-alpha-sg-01/team-camera/home/z50057756/conda/envs/rng-fa3
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export TORCH_HOME=/home/z50057756/torch_home_lagernvs
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1
# node-local compile caches (FlexAttention / Inductor), never the shared JuiceFS
export TORCHINDUCTOR_CACHE_DIR="/tmp/inductor_${USER}_${SLURM_JOB_ID:-local}"
export TRITON_CACHE_DIR="/tmp/triton_${USER}_${SLURM_JOB_ID:-local}"
ulimit -c 0
REPO=/home/z50057756/code/RnG_lagernvs_stage2
cd "$REPO"
echo "Job ${SLURM_JOB_ID:-?} | node $(hostname) | restart ${SLURM_RESTART_COUNT:-0} | qos ${SLURM_JOB_QOS:-?} | git $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total,uuid,pci.bus_id,ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total,retired_pages.pending --format=csv,noheader 2>/dev/null | head -4
