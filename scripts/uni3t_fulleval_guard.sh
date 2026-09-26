#!/bin/bash
# cron every 10 min: for every uni3t milestone checkpoint on disk that has no marker yet,
# submit scripts/uni3t_fulleval_gso.sbatch (posed + unposed, 1 GPU lowest). Modelled on
# RnG_lagernvs_softmoe/scripts_softmoe/armC_milestone_battery.sh. Markers prevent resubmission.
# Install: */10 * * * * /home/z50057756/code/RnG_lagernvs/scripts/uni3t_fulleval_guard.sh >> /home/z50057756/code/RnG_lagernvs/slurm_logs/uni3t_fulleval_guard.log 2>&1
export PATH=/usr/local/bin:/usr/bin:/bin:$PATH
REPO=/home/z50057756/code/RnG_lagernvs
EXP=PLN2uni3t_all287k_b32t6_fp32lr35_const
CKD=/mnt/data-alpha-sg-01/team-camera/home/z50057756/moe_experiments/checkpoints/$EXP
# must stay in sync with KEEP_MILESTONES in scripts/pln2_uni3t_watchdog.sh (user 09-21: full eval
# every 10k from 16k, plus the recovered 10k and the terminal 90k; 72000 added 09-23 as the
# same-step anchor with the baseline ckpt_72000 -- it lives in KEEP_UNTIL_EVALED there, not
# KEEP_MILESTONES; 60000 was submitted by hand for the same reason)
MILESTONES=${MILESTONES:-"10000 16000 26000 36000 46000 56000 66000 72000 76000 86000 90000"}
STATE=${STATE_DIR:-$REPO/slurm_logs/uni3t_fulleval_state}
DRYRUN=${DRYRUN:-0}
mkdir -p "$STATE"; cd "$REPO" || exit 1
command -v sbatch >/dev/null || { echo "$(date '+%F %T') sbatch not in PATH"; exit 2; }
for f in $(ls "$CKD"/ckpt_*.pt 2>/dev/null | sort); do
  st=$(basename "$f" .pt | tr -cd '0-9'); st=$((10#$st))
  echo " $MILESTONES " | grep -q " $st " || continue          # not an eval point
  marker="$STATE/fulleval_${st}.submitted"
  [ -e "$marker" ] && continue
  age=$(( $(date +%s) - $(stat -c %Y "$f") ))
  if [ "$age" -lt 120 ]; then echo "$(date '+%F %T') wait ckpt_$st (age ${age}s)"; continue; fi
  cmd=(sbatch --parsable scripts/uni3t_fulleval_gso.sbatch "$f")
  if [ "$DRYRUN" = 1 ]; then echo "DRYRUN: ${cmd[*]}"
  else j=$("${cmd[@]}") && echo "$(date '+%F %T') SUBMIT fulleval step=$st job=$j ckpt=$(basename "$f")" | tee "$marker"; fi
done
