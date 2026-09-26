#!/bin/bash
#SBATCH --job-name=uni3t-ioguard
#SBATCH --partition=gpu
#SBATCH --qos=lowest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:35:00
#SBATCH --no-requeue
#SBATCH --exclude=lrc-alpha-sg-gpu01
#SBATCH --output=/home/z50057756/code/RnG_lagernvs/slurm_logs/%x-%j.out
#SBATCH --error=/home/z50057756/code/RnG_lagernvs/slurm_logs/%x-%j.err
#
# End-to-end check of the 09-23 write guards (best_effort_write) inside the REAL training
# loop, 60 steps on 1 GPU (vis write at step 1, validation + checkpoint at step 60):
#   leg A  normal path  -- validation must still summarize and print, zero [io] warnings
#   leg B  real OSErrors injected via config/filesystem, no mocking:
#          * training.validation_out_dir=/dev/full/x  -> every export raises ENOTDIR
#          * a FILE named iter_00000001 pre-created in checkpoint_dir -> the step-1
#            visualization makedirs raises FileExistsError
#          must: warn, skip the validation summary, still save the step-60 checkpoint,
#          and exit 0.
set -uo pipefail
CONDA_ENV=/mnt/data-alpha-sg-01/team-camera/home/z50057756/conda/envs/rng-fa3
export PATH="$CONDA_ENV/bin:$PATH"; export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export WANDB_MODE=offline
cd /home/z50057756/code/RnG_lagernvs
echo "job $SLURM_JOB_ID node $(hostname) branch $(git rev-parse --abbrev-ref HEAD) commit $(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l)"
CFG=configs/RnGUP_lagernvs_uni3t_smoke60.yaml
ROOT=/home/z50057756/tmp/uni3t_smoke
run_leg () {  # $1 = leg tag, rest = extra overrides
  local tag=$1; shift
  local dir=$ROOT/SMOKE_uni3t_io$tag
  case "$dir" in /home/z50057756/tmp/uni3t_smoke/SMOKE_uni3t_io[AB]) rm -rf "$dir";; *) echo "REFUSING rm of $dir"; exit 9;; esac
  mkdir -p "$dir"
  [ "$tag" = B ] && : > "$dir/iter_00000001" && echo "[leg B] pre-created FILE $dir/iter_00000001"
  local port=$((29500 + (SLURM_JOB_ID + ${#tag}) % 200))
  echo; echo "===== LEG $tag ====="
  python -m torch.distributed.run --nproc_per_node=1 --nnodes=1 \
    --rdzv_id=${SLURM_JOB_ID}$tag --rdzv_backend=c10d --rdzv_endpoint=localhost:$port \
    train.py --config "$CFG" exp_name=SMOKE_uni3t_io$tag "$@"
  local rc=$?
  echo "[leg $tag] rc=$rc"
  return $rc
}
run_leg A;                                                  RCA=$?
run_leg B training.validation_out_dir=/dev/full/x;          RCB=$?
echo; echo "[ioguard] RC A=$RCA B=$RCB"
[ $RCA -eq 0 ] && [ $RCB -eq 0 ] && echo "### IOGUARD SMOKE BOTH LEGS EXITED 0" || { echo "### IOGUARD SMOKE FAILED"; exit 1; }
