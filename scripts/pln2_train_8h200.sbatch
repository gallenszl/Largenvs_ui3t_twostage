#!/bin/bash
#SBATCH --job-name=pln2-lagernvs-rgb256
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=208
#SBATCH --mem=1500G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/z50057756/code/RnG_lagernvs/slurm_logs/%x-%j.out
#SBATCH --error=/home/z50057756/code/RnG_lagernvs/slurm_logs/%x-%j.err

# P-LN2: lagernvs EncDec_VitB8 inside the RnG framework, image loss only.
# From-scratch model (VGGT-1B init inside the wrapper); NO resume from the
# production RnG ckpt — the architecture is different. Auto-resumes from its
# own ckpts on requeue via RnG's native mechanism.
#   CONFIG (required), NPROC (default 8), EXTRA (extra OmegaConf-style overrides)

set -euo pipefail

CONFIG="${CONFIG:?must set CONFIG (e.g. configs/RnGUP_lagernvs_rgb256_15k.yaml)}"
NPROC="${NPROC:-8}"

CONDA_ENV=/mnt/data-alpha-sg-01/team-camera/home/z50057756/conda/envs/rng-fa3
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
# vendored lagernvs Reconstructor loads VGGT-1B via torch.hub -> local file here
export TORCH_HOME=/home/z50057756/torch_home_lagernvs
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RNG_ATTENTION_BACKEND=auto
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
ulimit -c 0

REPO=/home/z50057756/code/RnG_lagernvs
mkdir -p "$REPO/slurm_logs"
cd "$REPO"

echo "Job ID: ${SLURM_JOB_ID} | Restart: ${SLURM_RESTART_COUNT:-0} | Node: $(hostname)"
echo "Config: $CONFIG | NPROC: $NPROC | QOS: ${SLURM_JOB_QOS:-?}"
echo "Git HEAD: $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

RDZV_PORT=$((29500 + SLURM_JOB_ID % 200))

# auto-resume (full state) from own latest ckpt on requeue
EXP_NAME=$(grep -E "^exp_name:" "$CONFIG" | awk '{print $2}')
for OV in ${EXTRA:-}; do case "$OV" in exp_name=*) EXP_NAME="${OV#exp_name=}";; esac; done
# UNI3T: checkpoint_dir can point off-repo (the uni3t arm keeps ckpts on the old
# disk because local home only has ~42 GB free). Read it from the config so the
# resume probe below looks in the right place; anything relative keeps the
# historical in-repo path, so existing arms behave exactly as before.
CKPT_DIR=$(grep -E "^[[:space:]]*checkpoint_dir:" "$CONFIG" | head -1 | sed 's/^[^:]*:[[:space:]]*//' | tr -d '"'"'"'"')
CKPT_DIR="${CKPT_DIR//\$\{exp_name\}/$EXP_NAME}"
case "$CKPT_DIR" in
  /*) : ;;
  *) CKPT_DIR=$REPO/experiments/checkpoints/$EXP_NAME ;;
esac
for OV in ${EXTRA:-}; do
  case "$OV" in
    training.checkpoint_dir=*)
      CKPT_DIR="${OV#training.checkpoint_dir=}"
      CKPT_DIR="${CKPT_DIR//\$\{exp_name\}/$EXP_NAME}"
      ;;
  esac
done
echo "Checkpoint dir: $CKPT_DIR"
WANDB_RUN_ID_FILE=$CKPT_DIR/.wandb_run_id
if [ -s "$WANDB_RUN_ID_FILE" ]; then
  echo "W&B run ID sidecar: $(tr -d '[:space:]' < "$WANDB_RUN_ID_FILE") ($WANDB_RUN_ID_FILE)"
elif [ -e "$WANDB_RUN_ID_FILE" ]; then
  echo "W&B run ID sidecar exists but is empty: $WANDB_RUN_ID_FILE (train.py will fail fast)"
else
  echo "W&B run ID sidecar: not present yet (rank 0 will create $WANDB_RUN_ID_FILE)"
fi
RESUME_ARGS=""
LATEST=$(ls -1 "$CKPT_DIR"/ckpt_*.pt 2>/dev/null | sort | tail -1 || true)
if [ -n "$LATEST" ]; then
  echo "Resume mode: full-state resume from $LATEST"
  RESUME_ARGS="training.resume_ckpt=$CKPT_DIR"
else
  echo "Resume mode: fresh start (no ckpt in $CKPT_DIR)"
fi

python -m torch.distributed.run --nproc_per_node="$NPROC" --nnodes=1 \
  --rdzv_id=${SLURM_JOB_ID} --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:${RDZV_PORT} \
  train.py --config "$CONFIG" ${EXTRA:-} $RESUME_ARGS
