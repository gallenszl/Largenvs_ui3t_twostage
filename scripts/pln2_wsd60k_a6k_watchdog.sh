#!/bin/bash
# crontab watchdog for the WSD branch-anneal arm (PLN2pose_v2color_b32t6_wsd60k_a6k).
# Runs every 30 min: (1) resubmit if the job vanished and training is not done (<70000);
# (2) prune checkpoints — keep the 2 newest + milestone {70000}.
# Install:  */30 * * * * /home/z50057756/code/RnG_lagernvs/scripts/pln2_wsd60k_a6k_watchdog.sh
# Log: slurm_logs/pln2_wsd60k_a6k_watchdog.log

REPO=/home/z50057756/code/RnG_lagernvs
EXP=PLN2pose_v2color_b32t6_wsd60k_a6k
JOBNAME=pln2-wsd60k-a6k
LOG=$REPO/slurm_logs/pln2_wsd60k_a6k_watchdog.log
CKPT_DIR=$REPO/experiments/checkpoints/$EXP
exec >> "$LOG" 2>&1

LATEST=$(ls -1 "$CKPT_DIR"/ckpt_*.pt 2>/dev/null | sed 's/.*ckpt_0*\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
LATEST=${LATEST:-0}

# (1) resubmit if gone and not done
if [ "$LATEST" -lt 66000 ] && [ -z "$(squeue -h -n $JOBNAME -u z50057756 2>/dev/null)" ]; then
  cd "$REPO"
  NEW=$(CONFIG=configs/RnGUP_lagernvs_b32t6_wsd60k_a6k.yaml NPROC=4 sbatch --parsable --qos=normal --time=1-00:00:00 --exclude=lrc-alpha-sg-gpu01 --gres=gpu:h200:4 --cpus-per-task=104 --mem=750G --job-name=$JOBNAME scripts/pln2_train_8h200.sbatch)
  echo "$(date '+%F %T') RESUBMIT: latest ckpt $LATEST < 70000, no job in queue -> new job $NEW"
fi

# (2) prune checkpoints: keep 2 newest + milestone (never touches the trunk's own dir)
KEEP_MILESTONES="66000"
ALL=$(ls -1 "$CKPT_DIR"/ckpt_*.pt 2>/dev/null | sort)
N=$(echo "$ALL" | grep -c . || true)
[ "$N" -le 2 ] && exit 0
NEWEST2=$(echo "$ALL" | tail -2)
for F in $ALL; do
  STEP=$(echo "$F" | sed 's/.*ckpt_0*\([0-9]*\)\.pt/\1/')
  echo "$NEWEST2" | grep -q "$F" && continue
  echo " $KEEP_MILESTONES " | grep -q " $STEP " && continue
  rm -f "$F"
  echo "$(date '+%F %T') PRUNE: removed ckpt_$STEP"
done
