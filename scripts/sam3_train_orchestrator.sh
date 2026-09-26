#!/bin/bash
# V8.1 orchestrator: watch the 4 quadrant jobs every ~10 min until all 29,638
# scenes have stats.json. Respawns a quadrant job if it died with work left
# (MaxBatchRequeue=5 backstop); once both normal quadrants finish, migrates
# still-running lowest quadrants to normal qos (cancel + resubmit, idempotent).
REPO=/home/z50057756/code/RnG_feature_allignment
OUT=/home/z50057756/data/co3d_sam3_masks/train
IN=/mnt/data-alpha-sg-02/team-camera/datasets/yuchen/co3d/webdataset/train
declare -A QOS=([q0]=normal [q1]=normal [q2]=lowest [q3]=lowest)
declare -A TOT
for q in q0 q1 q2 q3; do TOT[$q]=$(wc -l < "$REPO/data/co3d_train_$q.txt"); done

count_done() {  # completed scenes of a quadrant
  local n=0
  while IFS= read -r s; do [ -f "$OUT/$s/stats.json" ] && n=$((n + 1)); done \
    < "$REPO/data/co3d_train_$1.txt"
  echo "$n"
}

submit() {  # $1=quadrant $2=qos
  SPLIT_FILE=$REPO/data/co3d_train_$1.txt \
  EXTRA_ARGS="--input_dir $IN --output_dir $OUT" \
  PROCS_PER_GPU=4 \
    sbatch --job-name=sam3_train_$1 --qos="$2" \
    "$REPO/scripts/sam3_text_co3d_masks.sbatch" 2>&1 | tail -1
}

migrated=0
while true; do
  all_done=1
  normal_done=1
  for q in q0 q1; do
    [ "$(count_done $q)" -lt "${TOT[$q]}" ] && normal_done=0
  done
  line="[$(date +%H:%M)]"
  for q in q0 q1 q2 q3; do
    d=$(count_done $q)
    alive=$(squeue -h -u "$USER" --name="sam3_train_$q" -o "%T %q" 2>/dev/null | head -1)
    line+=" $q=$d/${TOT[$q]}(${alive:-DEAD})"
    if [ "$d" -lt "${TOT[$q]}" ]; then
      all_done=0
      want_qos=${QOS[$q]}
      if [ "$normal_done" = 1 ]; then want_qos=normal; fi
      if [ -z "$alive" ]; then
        echo "RESPAWN $q qos=$want_qos: $(submit "$q" "$want_qos")"
        QOS[$q]=$want_qos
      elif [ "$normal_done" = 1 ] && [ "${QOS[$q]}" = lowest ] && [ "$migrated" -lt 2 ]; then
        jid=$(squeue -h -u "$USER" --name="sam3_train_$q" -o "%i" | head -1)
        echo "MIGRATE $q lowest->normal (cancel $jid)"
        scancel "$jid"
        sleep 5
        echo "MIGRATE submit: $(submit "$q" normal)"
        QOS[$q]=normal
        migrated=$((migrated + 1))
      fi
    fi
  done
  echo "$line"
  errs=$(cat "$OUT"/errors_shard*.log 2>/dev/null | grep -c ERROR || true)
  [ "${errs:-0}" -gt 0 ] && echo "WARN: $errs error lines in $OUT/errors_shard*.log"
  if [ "$all_done" = 1 ]; then
    echo "ALL_QUADRANTS_COMPLETE total=$(ls "$OUT" | wc -l)"
    break
  fi
  sleep 600
done
