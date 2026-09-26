#!/bin/bash
# Detached completion-watcher: polls the train run; once all 29,638 scenes have
# stats.json, submits the frozen-rule audit (+rescue) as a cpu-partition job,
# then exits. Survives session teardown (setsid). Stop: touch the STOP file.
REPO=/home/z50057756/code/RnG_feature_allignment
OUT=/home/z50057756/data/co3d_sam3_masks/train
IN=/mnt/data-alpha-sg-02/team-camera/datasets/yuchen/co3d/webdataset/train
PY=/home/z50057756/code/sam2/.venv/bin/python
LOG=$REPO/slurm_logs/audit_watcher.log
STOP=$REPO/slurm_logs/audit_watcher.STOP
DONE_MARK=$REPO/slurm_logs/audit_submitted.marker
TARGET=29638

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >>"$LOG"; }
log "audit-watcher START (target=$TARGET)"

while true; do
  [ -f "$STOP" ] && { log "STOP seen -> exit"; exit 0; }
  [ -f "$DONE_MARK" ] && { log "already submitted -> exit"; exit 0; }
  n=$(find "$OUT" -maxdepth 2 -name stats.json 2>/dev/null | wc -l)
  log "progress $n/$TARGET"
  if [ "$n" -ge "$TARGET" ]; then
    log "COMPLETE -> submitting audit"
    jid=$(sbatch --parsable --job-name=sam3_audit_train --partition=cpu --qos=normal \
      --cpus-per-task=64 --mem=200G --time=08:00:00 \
      --output=$REPO/slurm_logs/%x-%j.out --error=$REPO/slurm_logs/%x-%j.err \
      --wrap "$PY $REPO/scripts/sam2_eval_frame_audit.py \
        --mask_dir $OUT --input_dir $IN \
        --split_file $REPO/data/co3d_train_all.txt \
        --out_dir /home/z50057756/data/co3d_sam2_masks/qc/audit_train_full \
        --rescue --workers 64 --max_suspect_render 400 --usable_sample 200")
    echo "$jid" > "$DONE_MARK"
    log "submitted audit job $jid; marker written; exit"
    exit 0
  fi
  sleep 600
done
