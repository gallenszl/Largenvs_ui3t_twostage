#!/bin/bash
#SBATCH --job-name=uni3t-cons
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=104
#SBATCH --mem=750G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/z50057756/code/RnG_lagernvs_stage2/slurm_logs/%x-%j.out
#SBATCH --error=/home/z50057756/code/RnG_lagernvs_stage2/slurm_logs/%x-%j.err
# Stage-1 (uni3t) training from THIS repo (scripts/pln2_train_8h200.sbatch is the snapshot of the original
# repo's launcher and cds into /home/z50057756/code/RnG_lagernvs, i.e. code without the consistency loss).
#   sbatch scripts_s2/uni3t_train_4gpu.sbatch <config> [OmegaConf overrides ...]
# Auto-resumes (full state) from the newest ckpt_*.pt in the config's checkpoint_dir, so a requeue or a
# relay continues where it stopped; a fresh directory starts from step 0.
set -euo pipefail
CONFIG="${1:?config}"; shift
EXTRA="$*"
NPROC=4
source /home/z50057756/code/RnG_lagernvs_stage2/scripts_s2/s2_env.sh
export RNG_ATTENTION_BACKEND=auto
export OMP_NUM_THREADS=4
mkdir -p "$REPO/slurm_logs"
echo "Config: $CONFIG | NPROC: $NPROC | extra: '${EXTRA}'"
RDZV_PORT=$((29500 + SLURM_JOB_ID % 200))
EXP_NAME=$(grep -E "^exp_name:" "$CONFIG" | awk '{print $2}')
for OV in $EXTRA; do case "$OV" in exp_name=*) EXP_NAME="${OV#exp_name=}";; esac; done
CKPT_DIR=$(grep -E "^[[:space:]]*checkpoint_dir:" "$CONFIG" | head -1 | sed 's/^[^:]*:[[:space:]]*//' | tr -d '"'"'"'"')
CKPT_DIR="${CKPT_DIR//\$\{exp_name\}/$EXP_NAME}"
case "$CKPT_DIR" in /*) : ;; *) CKPT_DIR=$REPO/experiments/checkpoints/$EXP_NAME ;; esac
for OV in $EXTRA; do case "$OV" in training.checkpoint_dir=*) CKPT_DIR="${OV#training.checkpoint_dir=}"; CKPT_DIR="${CKPT_DIR//\$\{exp_name\}/$EXP_NAME}";; esac; done
echo "Checkpoint dir: $CKPT_DIR"
mkdir -p "$CKPT_DIR"
WANDB_RUN_ID_FILE=$CKPT_DIR/.wandb_run_id
if [ -s "$WANDB_RUN_ID_FILE" ]; then echo "W&B run ID sidecar: $(tr -d '[:space:]' < "$WANDB_RUN_ID_FILE")"
elif [ -e "$WANDB_RUN_ID_FILE" ]; then echo "W&B run ID sidecar exists but is empty (train.py will fail fast)"
else echo "W&B run ID sidecar: not present yet (rank 0 creates it)"; fi
# disk pre-flight on the old-disk subtree: one full checkpoint is ~16.8 GB, demand 40 GB free
FREE=$(df -BG --output=avail /mnt/data-alpha-sg-01/team-camera/home/z50057756 | tail -1 | tr -dc 0-9)
[ "${FREE:-0}" -lt 40 ] && { echo "only ${FREE} GB free on the old disk, refusing to start"; exit 3; }
RESUME_ARGS=""
LATEST=$(ls -1 "$CKPT_DIR"/ckpt_*.pt 2>/dev/null | sort | tail -1 || true)
if [ -n "$LATEST" ]; then echo "Resume mode: full-state resume from $LATEST"; RESUME_ARGS="training.resume_ckpt=$CKPT_DIR"
else echo "Resume mode: fresh start (no ckpt in $CKPT_DIR)"; fi
python -m torch.distributed.run --nproc_per_node="$NPROC" --nnodes=1 \
  --rdzv_id=${SLURM_JOB_ID} --rdzv_backend=c10d --rdzv_endpoint=localhost:${RDZV_PORT} \
  train.py --config "$CONFIG" $EXTRA $RESUME_ARGS
rc=$?; echo "[uni3t-train] DONE rc=$rc"; exit $rc
