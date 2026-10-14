# shared environment of the stage-2 job scripts (sourced) -- Isambard-AI (BriCS, GH200 aarch64)
# rng-fa3-a64 = aarch64 replica of the old cluster's rng-fa3 (torch 2.7.0+cu128 / xformers 0.0.30 / flash_attn_3 3.0.0)
source /projects/u6ga/zhelun/envs/rng-fa3-a64.env.sh      # PATH, LD_LIBRARY_PATH, TORCH_HOME, HF_HOME, CUDA_HOME
export PYTHONNOUSERSITE=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1
SCRATCH=/scratch/u6ga/zhelun.u6ga
S2_ROOT=$SCRATCH/S2_experiments                           # stage-2 ckpts, smoke runs, bench outputs (never $HOME)
export WANDB_DIR=$S2_ROOT/wandb
mkdir -p "$S2_ROOT" "$WANDB_DIR"
# node-local compile caches (FlexAttention / Inductor), never the shared Lustre; /local/user/<uid> is missing on some
# nodes (RnG_MOE site.sh, 09-22), so take the first writable candidate
for _c in /tmp "${LOCALDIR:-}" "$SCRATCH/tmp"; do
  [ -n "$_c" ] || continue
  if mkdir -p "$_c/s2tmp.${SLURM_JOB_ID:-local}" 2>/dev/null && [ -w "$_c/s2tmp.${SLURM_JOB_ID:-local}" ]; then
    _T="$_c/s2tmp.${SLURM_JOB_ID:-local}"; break
  fi
done
unset _c
export TORCHINDUCTOR_CACHE_DIR="$_T/inductor" TRITON_CACHE_DIR="$_T/triton" TMPDIR="$_T" TMP="$_T" TEMP="$_T"
ulimit -c 0
REPO=/home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage
cd "$REPO"
echo "Job ${SLURM_JOB_ID:-?} | node $(hostname) | restart ${SLURM_RESTART_COUNT:-0} | git $(git rev-parse --short HEAD) | tmp $_T"
nvidia-smi --query-gpu=name,memory.total,uuid,pci.bus_id,ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total,retired_pages.pending --format=csv,noheader 2>/dev/null | head -4
