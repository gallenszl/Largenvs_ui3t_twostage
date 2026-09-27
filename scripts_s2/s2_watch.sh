#!/bin/bash
# 5-minute watchdog for stage-2 test / smoke / training jobs (rule 21).
#   s2_watch.sh <jobid> <out.log> <err.log> [interval_s=300] [stall_polls=4]
# exit 0 = COMPLETED; 1 = other terminal state; 2 = failure signature in the logs; 3 = stalled (no log growth)
JOB=$1; OUT=$2; ERR=$3; IV=${4:-300}; STALL=${5:-4}
SIG='Traceback|^FAIL:|^ERROR:|RuntimeError|out of memory|OutOfMemory|illegal memory access|CUDA error|NaN or Inf loss|GATE.*FAIL|\[resume\]|Killed|NCCL error|EDQUOT|No space left|ImportError|ModuleNotFoundError|API key'
seen=0; last=-1; flat=0
while true; do
  st=$(sacct -j "$JOB" -X -n -o State 2>/dev/null | head -1 | awk '{print $1}')
  sz=$(( $(stat -c %s "$OUT" 2>/dev/null || echo 0) + $(stat -c %s "$ERR" 2>/dev/null || echo 0) ))
  n=$(cat "$OUT" "$ERR" 2>/dev/null | grep -cE "$SIG")
  ts=$(date +%H:%M:%S)
  case "$st" in
    COMPLETED) echo "[$ts] job $JOB COMPLETED"; tail -n 5 "$OUT"; exit 0;;
    FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE)
      echo "[$ts] job $JOB terminal state $st"; tail -n 20 "$ERR"; tail -n 5 "$OUT"; exit 1;;
  esac
  if [ "$n" -gt "$seen" ]; then
    echo "[$ts] job $JOB ($st): new failure signature(s):"; cat "$OUT" "$ERR" 2>/dev/null | grep -nE "$SIG" | tail -n 8; exit 2
  fi
  if [ "$st" = "RUNNING" ]; then
    if [ "$sz" -le "$last" ]; then flat=$((flat+1)); else flat=0; fi
    if [ "$flat" -ge "$STALL" ]; then echo "[$ts] job $JOB RUNNING but logs have not grown for $flat polls"; exit 3; fi
  fi
  last=$sz
  sleep "$IV"
done
