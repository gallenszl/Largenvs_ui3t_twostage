#!/bin/bash
#SBATCH --job-name=anneal60k-smoke
#SBATCH --partition=gpu
#SBATCH --qos=lowest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --exclude=lrc-alpha-sg-gpu01
#SBATCH --output=/home/z50057756/code/RnG_lagernvs/slurm_logs/%x-%j.out
#SBATCH --error=/home/z50057756/code/RnG_lagernvs/slurm_logs/%x-%j.err
#
# 1-GPU smoke of a 60k -> 70k anneal branch, with the schedule compressed to 20 steps
# (training.train_steps=60020, so lr(s) = 3.5e-5 * (1 - sqrt((s - 60000) / 20))):
#   leg A  real resume from the seed: must print <All keys matched>, "Resumed optimizer and
#          lr_scheduler", first step 60001, the compressed LR schedule, save ckpt 60020,
#          print the validation summary, exit 0.
#   leg B  real OSErrors, no mocking: validation_out_dir=/dev/full/x (every export raises)
#          + a FILE named iter_00060001 in the ckpt dir (the step-60001 visualization
#          makedirs raises) -> [io] warnings, "[val] WARNING: N batch export(s) failed",
#          no summary, still saves ckpt 60020, exit 0.
#   leg C  resume-guard negative control: empty resume dir + require_resume_step=60000
#          -> must raise "[resume] require_resume_step" before any training step, exit != 0.
# The job exits 0 only if every leg's exit code is as expected AND the log checklist passes.
# Positional args (no --export lists on this Slurm): TREE BRANCH CONFIG SEED TAG [LEGS]
set -uo pipefail
TREE=$1; BRANCH=$2; CFG=$3; SEED=$4; TAG=$5; LEGS=${6:-"A C"}
CONDA_ENV=/mnt/data-alpha-sg-01/team-camera/home/z50057756/conda/envs/rng-fa3
export PATH="$CONDA_ENV/bin:$PATH"; export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export TORCH_HOME=/home/z50057756/torch_home_lagernvs
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RNG_ATTENTION_BACKEND=auto
cd "$TREE" || { echo "ABORT: no tree $TREE"; exit 3; }
BR=$(git rev-parse --abbrev-ref HEAD)
echo "job $SLURM_JOB_ID node $(hostname) tree $TREE branch $BR commit $(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l)"
[ "$BR" = "$BRANCH" ] || { echo "ABORT: branch $BR != $BRANCH"; exit 4; }
[ -f "$SEED" ] || { echo "ABORT: seed $SEED missing"; exit 5; }
ROOT=/mnt/data-alpha-sg-01/team-camera/home/z50057756/moe_experiments/checkpoints

run_leg () {  # $1 = leg (A|B|C)
  local leg=$1 idx
  case "$leg" in A) idx=0;; B) idx=1;; C) idx=2;; *) echo "bad leg $leg"; return 2;; esac
  local exp=SMOKE_anneal60k_${TAG}_$leg
  local dir=$ROOT/$exp
  case "$dir" in "$ROOT"/SMOKE_anneal60k_*) rm -rf "$dir";; *) echo "REFUSING rm of $dir"; exit 9;; esac
  mkdir -p "$dir"
  local val=$dir extra=""
  [ "$leg" != C ] && ln "$SEED" "$dir/ckpt_0000000000060000.pt"
  if [ "$leg" = B ]; then
    : > "$dir/iter_00060001"; echo "[leg B] pre-created FILE $dir/iter_00060001"
    val=/dev/full/x
  fi
  local port=$((29500 + (SLURM_JOB_ID * 3 + idx) % 200))
  echo; echo "===== LEG $leg ($exp) ====="; echo "===== LEG $leg ($exp) =====" >&2
  python -m torch.distributed.run --nproc_per_node=1 --nnodes=1 \
    --rdzv_id=${SLURM_JOB_ID}$leg --rdzv_backend=c10d --rdzv_endpoint=localhost:$port \
    train.py --config "$CFG" exp_name=$exp training.checkpoint_dir=$dir \
    training.validation_out_dir=$val training.resume_ckpt=$dir \
    training.train_steps=60020 training.print_every=1
  local rc=$?
  echo "[leg $leg] rc=$rc"; echo "[leg $leg] rc=$rc" >&2
  return $rc
}

declare -A RC
for L in $LEGS; do run_leg "$L"; RC[$L]=$?; done
echo; echo -n "[smoke] RC"; for L in $LEGS; do echo -n " $L=${RC[$L]}"; done; echo
ok=1
for L in $LEGS; do
  if [ "$L" = C ]; then [ "${RC[$L]}" -ne 0 ] || ok=0; else [ "${RC[$L]}" -eq 0 ] || ok=0; fi
done
[ $ok -eq 1 ] && echo "### ANNEAL SMOKE EXIT CODES AS EXPECTED" || { echo "### ANNEAL SMOKE EXIT CODES WRONG"; exit 1; }
# Full checklist (resume lines, first step, LR schedule, save, summary / guarded warnings,
# guard message). Its exit status is this job's, so an afterok-dependent main run only
# starts when every check passed.
CHECKER=/home/z50057756/tmp/uni3t_verify/check_anneal_smoke.py
python "$CHECKER" "$SLURM_JOB_ID" "$LEGS" || { echo "### ANNEAL SMOKE CHECKLIST FAILED"; exit 1; }
echo "### ANNEAL SMOKE CHECKLIST PASSED"
