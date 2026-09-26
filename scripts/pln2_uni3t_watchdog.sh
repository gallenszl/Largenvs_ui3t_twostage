#!/bin/bash
# crontab watchdog for the three-task arm (PLN2uni3t_all287k_b32t6_fp32lr35_const).
# Runs every 30 min: (1) prune checkpoints — keep the TWO newest + milestones
# {20k,40k,60k,80k,90k}; (2) resubmit if the job vanished and training is not done.
# Install:  */30 * * * * /home/z50057756/code/RnG_lagernvs/scripts/pln2_uni3t_watchdog.sh
# Log: slurm_logs/pln2_uni3t_watchdog.log
#
# NOTE: this arm keeps its checkpoints on the old disk (local home has ~42 GB free and
# one ckpt is ~16.8 GB), so CKPT_DIR is NOT under $REPO/experiments.

REPO=/home/z50057756/code/RnG_lagernvs
EXP=PLN2uni3t_all287k_b32t6_fp32lr35_const
JOBNAME=pln2-uni3t
CONFIG=configs/RnGUP_lagernvs_uni3t_b32t6_fp32lr35_const_90k_all287k.yaml
LOG=$REPO/slurm_logs/pln2_uni3t_watchdog.log
CKPT_DIR=/mnt/data-alpha-sg-01/team-camera/home/z50057756/moe_experiments/checkpoints/$EXP
exec >> "$LOG" 2>&1

# ---- (1) prune checkpoints: keep the TWO newest + milestones -----------------
# Pruning runs FIRST, before the branch assertion in section (2). That assertion
# exits non-zero, and with the old ordering it also skipped pruning: a job running
# against a work tree someone had checked back to master would keep writing 16.8 GB
# files with nobody removing them (~5-7/day would fill the quota in about a day).
#
# Two newest, not one: utils/training_utils.py:171 (auto_resume_job) falls back to
# the next-older checkpoint when the newest one will not load. Keeping a single file
# disables that path, and between 4k and 20k there is no milestone to fall back to
# either -- an unreadable newest checkpoint would make find_checkpoints return empty
# and train.py would silently restart from step 0 while this script kept it alive.
KEEP_MILESTONES="10000 16000 26000 36000 46000 56000 60000 66000 72000 76000 86000 90000"   # user 09-21: full GSO eval every 10k from 16k (+ terminal 90k)
# Anchors kept only until their full GSO eval has finished (marker written by
# scripts/uni3t_fulleval_gso.sbatch on ALL DONE), then pruned like any other file.
# user 09-23: 60000 = same-step pair with PLN2pose_all287k_b32t6_fp32lr35_const EVv2 60k;
# 72000 = same-step pair with that arm's last surviving checkpoint.
KEEP_UNTIL_EVALED=""   # 60000 (09-24, restored from trash by user order) and 72000 moved to KEEP_MILESTONES 09-24 (user): anneal fork point paired with the dense arm's only surviving ckpt_72000
EVAL_STATE=$REPO/slurm_logs/uni3t_fulleval_state
ALL=$(ls -1 "$CKPT_DIR"/ckpt_*.pt 2>/dev/null | sort)
N=$(echo "$ALL" | grep -c . || true)
if [ "$N" -gt 2 ]; then
  KEEP_NEWEST=$(echo "$ALL" | tail -2)
  for F in $ALL; do
    STEP=$(echo "$F" | sed 's/.*ckpt_0*\([0-9]*\)\.pt/\1/')
    echo "$KEEP_NEWEST" | grep -Fqx "$F" && continue
    echo " $KEEP_MILESTONES " | grep -q " $STEP " && continue
    if echo " $KEEP_UNTIL_EVALED " | grep -q " $STEP " && [ ! -e "$EVAL_STATE/fulleval_${STEP}.done" ]; then continue; fi
    rm -f "$F"
    echo "$(date '+%F %T') PRUNE: removed ckpt_$STEP"
  done
fi

# ---- (2) resubmit if gone and not done ---------------------------------------
LATEST=$(ls -1 "$CKPT_DIR"/ckpt_*.pt 2>/dev/null | sed 's/.*ckpt_0*\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
LATEST=${LATEST:-0}

if [ "$LATEST" -lt 90000 ] && [ -z "$(squeue -h -n $JOBNAME -u z50057756 2>/dev/null)" ]; then
  cd "$REPO"
  # the branch must be uni3t; resubmitting from master would silently launch the
  # NVS-only code against a config whose keys it does not understand
  BR=$(git rev-parse --abbrev-ref HEAD)
  if [ "$BR" != "uni3t" ]; then
    echo "$(date '+%F %T') ABORT: repo is on branch '$BR', expected 'uni3t'; not resubmitting"
    exit 1
  fi
  NEW=$(CONFIG=$CONFIG NPROC=4 sbatch --parsable --qos=normal --time=2-00:00:00 --exclude=lrc-alpha-sg-gpu01 --gres=gpu:h200:4 --cpus-per-task=104 --mem=750G --job-name=$JOBNAME scripts/pln2_train_8h200.sbatch)
  echo "$(date '+%F %T') RESUBMIT: latest ckpt $LATEST < 90000, no job in queue -> new job $NEW"
fi
