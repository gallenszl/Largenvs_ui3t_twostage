#!/bin/bash
#SBATCH --job-name=s2-smoke
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --output=/home/z50057756/code/RnG_lagernvs_stage2/slurm_logs/%x-%j.out
#SBATCH --error=/home/z50057756/code/RnG_lagernvs_stage2/slurm_logs/%x-%j.err
# Stage-2 smoke (plan G7), 1 GPU: $1 config.
#   leg A  60 steps (batch 2, warmup 10, anneal 50->60, ckpt 30/60, validation posed+unposed on 4 objects)
#   leg B  resume from 60 to 90 (require_resume_step 60; must resume, never restart)
#   leg C  fail-closed: a truncated newest checkpoint must stop the run with a [resume] error
#   leg D  dynamics: zero-init output layers must have moved; checkpoint keys = trainable keys
set -uo pipefail
source /home/z50057756/code/RnG_lagernvs_stage2/scripts_s2/s2_env.sh
export WANDB_MODE=offline
CONFIG=${1:?config}
EXP=$(grep -E "^exp_name:" "$CONFIG" | awk '{print $2}')_SMOKE_${SLURM_JOB_ID}
SD=/mnt/data-alpha-sg-01/team-camera/home/z50057756/moe_experiments/SMOKE/$EXP
mkdir -p "$SD"
COMMON="exp_name=$EXP training.checkpoint_dir=$SD training.validation_out_dir=$SD training.batch_size_per_gpu=2 \
 training.num_workers=4 training.prefetch_factor=2 training.warmup=10 training.anneal_from_step=50 \
 training.checkpoint_every=30 training.checkpoint_extra_steps=[] training.ckpt_keep_steps=[] training.val_every=30 \
 training.val_dataset_cfgs.split_file=data/gso_subset4.txt training.print_every=5 training.wandb_log_every=5 training.vis_every=1000"
run() { python -m torch.distributed.run --standalone --nproc_per_node=1 train.py --config "$CONFIG" $COMMON "$@"; }
echo "=== leg A: 60 steps ==="
run training.train_steps=60 || { echo "[s2-smoke] GATE A FAIL"; exit 1; }
ls -la "$SD" | grep ckpt_
[ -f "$SD/ckpt_0000000000000060.pt" ] || { echo "[s2-smoke] GATE A FAIL: no ckpt_60"; exit 1; }
echo "=== leg B: resume 60 -> 90 ==="
run training.train_steps=90 training.resume_ckpt=$SD training.require_resume_step=60 || { echo "[s2-smoke] GATE B FAIL"; exit 1; }
[ -f "$SD/ckpt_0000000000000090.pt" ] || { echo "[s2-smoke] GATE B FAIL: no ckpt_90"; exit 1; }
echo "=== leg C: fail-closed on a truncated newest checkpoint ==="
SDC=${SD}_C; mkdir -p "$SDC"; cp "$SD/ckpt_0000000000000090.pt" "$SDC/"
head -c 1000000 "$SD/ckpt_0000000000000090.pt" > "$SDC/ckpt_0000000000000120.pt"
if run training.train_steps=150 training.checkpoint_dir=$SDC training.validation_out_dir=$SDC training.resume_ckpt=$SDC > "$SDC/legC.log" 2>&1; then
  echo "[s2-smoke] GATE C FAIL: a corrupt newest checkpoint did not stop the run"; exit 1
fi
grep -q "\[resume\]" "$SDC/legC.log" && echo "[s2-smoke] leg C ok (fail-closed)" || { echo "[s2-smoke] GATE C FAIL: no [resume] message"; tail -20 "$SDC/legC.log"; exit 1; }
echo "=== leg D: dynamics / checkpoint contract ==="
python - "$SD" <<'PY' || { echo "[s2-smoke] GATE D FAIL"; exit 1; }
import sys, torch
sd = torch.load(f"{sys.argv[1]}/ckpt_0000000000000090.pt", map_location="cpu", weights_only=True)["model"]
lin = [k for k in sd if k.startswith("renderer.out_lin.") and k.endswith("weight")]
norms = {k: float(sd[k].norm()) for k in lin}
print("[s2-smoke] Lin_m weight norms after 90 steps:", norms)
assert all(v > 0 for v in norms.values()), "a zero-init output layer never moved"
assert not any(k.startswith("loss_computer.") for k in sd), "frozen VGG saved in the checkpoint"
print("[s2-smoke] keys", len(sd), "params %.1fM" % (sum(v.numel() for v in sd.values()) / 1e6))
PY
echo "[s2-smoke] ALL GATES PASS"
