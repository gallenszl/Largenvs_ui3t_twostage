#!/bin/bash
#SBATCH --job-name=s2-train
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=104
#SBATCH --mem=750G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/z50057756/code/RnG_lagernvs_stage2/slurm_logs/%x-%j.out
#SBATCH --error=/home/z50057756/code/RnG_lagernvs_stage2/slurm_logs/%x-%j.err
# Stage-2 training, 4 x H200, with a relay:  $1 config  $2 end step (26000)
# Stops when the latest checkpoint reaches the end step or RELAY_STOP exists in the checkpoint dir;
# otherwise queues one successor (afterany) and resumes fail-closed from the latest checkpoint.
set -uo pipefail
source /home/z50057756/code/RnG_lagernvs_stage2/scripts_s2/s2_env.sh
CONFIG=${1:?config}; END=${2:?end step}
EXP=$(grep -E "^exp_name:" "$CONFIG" | awk '{print $2}')
CKPT_DIR=/mnt/data-alpha-sg-01/team-camera/home/z50057756/moe_experiments/checkpoints/$EXP
mkdir -p "$CKPT_DIR"
latest_step() { ls "$CKPT_DIR"/ckpt_*.pt 2>/dev/null | sed -E 's/.*ckpt_([0-9]{16})\.pt/\1/' | sort | tail -1 | sed 's/^0*//'; }
[ -f "$CKPT_DIR/RELAY_STOP" ] && { echo "[s2-train] RELAY_STOP present, exiting"; exit 0; }
L=$(latest_step); L=${L:-0}
[ "$L" -ge "$END" ] && { echo "[s2-train] latest ckpt $L >= $END, done"; exit 0; }
# disk pre-flight: one checkpoint is ~3 GB; demand 40 GB free on the old disk subtree
FREE=$(df -BG --output=avail /mnt/data-alpha-sg-01/team-camera/home/z50057756 | tail -1 | tr -dc 0-9)
[ "${FREE:-0}" -lt 40 ] && { echo "[s2-train] only ${FREE} GB free on the old disk, refusing to start"; exit 3; }
MARK="$CKPT_DIR/.relay_${SLURM_JOB_ID}"
if [ ! -f "$MARK" ]; then
  NEXT=$(sbatch --parsable --dependency=afterany:$SLURM_JOB_ID --kill-on-invalid-dep=yes \
         /home/z50057756/code/RnG_lagernvs_stage2/scripts_s2/s2_train.sbatch "$CONFIG" "$END")
  echo "$NEXT" > "$MARK"; echo "[s2-train] relay successor $NEXT"
fi
RESUME=""
[ "$L" -gt 0 ] && RESUME="training.resume_ckpt=$CKPT_DIR training.require_resume_step=$L"
echo "[s2-train] exp=$EXP latest=$L end=$END resume='${RESUME}'"
RDZV_PORT=$((29500 + SLURM_JOB_ID % 200))
python -m torch.distributed.run --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:$RDZV_PORT \
  train.py --config "$CONFIG" $RESUME
rc=$?
L2=$(latest_step); L2=${L2:-0}
if [ $rc -ne 0 ] && [ "$L2" -le "$L" ]; then
  echo "[s2-train] failed without progress (rc=$rc, latest $L2): writing RELAY_STOP and cancelling the successor"
  touch "$CKPT_DIR/RELAY_STOP"; scancel "$(cat "$MARK")" 2>/dev/null
fi
echo "[s2-train] DONE rc=$rc latest=$L2"
exit $rc
