#!/bin/bash
# crontab watchdog for the uni3t anneal branch (PLN2uni3t_all287k_b32t6_fp32lr35_wsd60k_a10k:
# 60k -> 70k, 1-sqrt, seeded from PLN2uni3t_all287k_b32t6_fp32lr35_const ckpt_60000).
# Runs every 30 min: (1) prune -- keep the seed 60000, the newest ckpt and 70000;
# (2) resubmit if the job vanished and training is not done, unless the last job died on
# the require_resume_step guard (resubmitting would fail the same way every 30 min).
# Install:  */30 * * * * /home/z50057756/code/RnG_lagernvs/scripts/pln2_uni3t_anneal_watchdog.sh
# Log: slurm_logs/pln2_uni3t_anneal_watchdog.log

REPO=/home/z50057756/code/RnG_lagernvs
BRANCH=uni3t
EXP=PLN2uni3t_all287k_b32t6_fp32lr35_wsd60k_a10k
JOBNAME=pln2-uni3t-a10k
CONFIG=configs/RnGUP_lagernvs_uni3t_b32t6_fp32lr35_wsd60k_a10k_all287k.yaml
END=70000
LOG=/home/z50057756/code/RnG_lagernvs/slurm_logs/pln2_uni3t_anneal_watchdog.log
CKPT_DIR=/mnt/data-alpha-sg-01/team-camera/home/z50057756/moe_experiments/checkpoints/$EXP
exec >> "$LOG" 2>&1

# ---- (1) prune: keep seed + newest + terminal -----------------------------------------
# The seed is kept for good: if the newest checkpoint will not load, auto_resume_job falls
# back to the next-older file, and the seed guarantees that is 60000, never step 0.
KEEP="60000 $END"
ALL=$(ls -1 "$CKPT_DIR"/ckpt_*.pt 2>/dev/null | sort)
NEWEST=$(echo "$ALL" | tail -1)
for F in $ALL; do
  STEP=$(echo "$F" | sed 's/.*ckpt_0*\([0-9]*\)\.pt/\1/')
  [ "$F" = "$NEWEST" ] && continue
  echo " $KEEP " | grep -q " $STEP " && continue
  rm -f "$F"
  echo "$(date '+%F %T') PRUNE: removed ckpt_$STEP"
done

# ---- (2) resubmit if gone and not done ------------------------------------------------
LATEST=$(ls -1 "$CKPT_DIR"/ckpt_*.pt 2>/dev/null | sed 's/.*ckpt_0*\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
LATEST=${LATEST:-0}
if [ "$LATEST" -lt "$END" ] && [ -z "$(squeue -h -n $JOBNAME -u z50057756 2>/dev/null)" ]; then
  LASTERR=$(ls -1t /home/z50057756/code/RnG_lagernvs/slurm_logs/$JOBNAME-*.err 2>/dev/null | head -1)
  if [ -n "$LASTERR" ] && grep -q "\[resume\] require_resume_step" "$LASTERR"; then
    echo "$(date '+%F %T') ALERT: last job ($LASTERR) refused to start: optimizer not resumed; NOT resubmitting"
    exit 1
  fi
  cd "$REPO"
  BR=$(git rev-parse --abbrev-ref HEAD)
  if [ "$BR" != "$BRANCH" ]; then
    echo "$(date '+%F %T') ABORT: $REPO is on branch '$BR', expected '$BRANCH'; not resubmitting"
    exit 1
  fi
  NEW=$(CONFIG=$CONFIG NPROC=4 sbatch --parsable --qos=normal --time=2-00:00:00 --exclude=lrc-alpha-sg-gpu01 --gres=gpu:h200:4 --cpus-per-task=104 --mem=750G --job-name=$JOBNAME scripts/pln2_train_8h200.sbatch)
  echo "$(date '+%F %T') RESUBMIT: latest ckpt $LATEST < $END, no job in queue -> new job $NEW"
fi
