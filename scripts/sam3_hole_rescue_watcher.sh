#!/bin/bash
# Detached: wait for the hole-scan job to finish, then launch depth-rescue with
# threads pinned (OMP/MKL/OpenBLAS=1, cv2 pinned in-script) and a big core count.
REPO=/home/z50057756/code/RnG_feature_allignment
Q=/home/z50057756/data/co3d_sam2_masks/qc/audit_train_full
PY=/home/z50057756/code/sam2/.venv/bin/python
LOG=$REPO/slurm_logs/hole_rescue_watcher.log
HITS=$Q/hole_hits.json
SCAN_JOB=${1:-85999}

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >>"$LOG"; }
log "watcher START (scan job=$SCAN_JOB)"

# wait until the scan job leaves the queue
while squeue -h -j "$SCAN_JOB" 2>/dev/null | grep -q .; do sleep 60; done
log "scan job gone; hits.json present=$([ -f "$HITS" ] && echo yes || echo no)"
[ -f "$HITS" ] || { log "no hits.json -> abort"; exit 1; }

ncand=$($PY -c "import json;print(len(json.load(open('$HITS'))))" 2>/dev/null)
log "hole candidates=$ncand -> submitting depth-rescue"

jid=$(sbatch --parsable --job-name=sam3_hole_rescue --partition=cpu --qos=normal \
  --cpus-per-task=96 --mem=200G --time=04:00:00 \
  --output=$REPO/slurm_logs/%x-%j.out --error=$REPO/slurm_logs/%x-%j.err \
  --wrap "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; \
    $PY $REPO/scripts/sam3_hole_depth_rescue.py \
      --hits_json $HITS \
      --mask_root /home/z50057756/data/co3d_sam3_masks/train \
      --input_dir /mnt/data-alpha-sg-02/team-camera/datasets/yuchen/co3d/webdataset/train \
      --out_dir $Q --workers 90")
log "submitted depth-rescue job $jid; exit"
