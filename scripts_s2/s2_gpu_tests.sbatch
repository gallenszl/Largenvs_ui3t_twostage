#!/bin/bash
#SBATCH --job-name=s2-gputest
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/slurm_logs/%x-%j.out
#SBATCH --error=/home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/slurm_logs/%x-%j.err
# Stage-2 GPU tests (plan G4). $1 = unittest modules (default: both GPU test files).
set -uo pipefail
source /home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/scripts_s2/s2_env.sh
# optional extra environment for diagnosis runs, e.g. "CUDA_LAUNCH_BLOCKING=1" (positional $2, space separated)
for kv in ${2:-}; do export "$kv"; echo "[s2-gpu] env $kv"; done
MODS="${1:-tests.test_s2_attention_gpu tests.test_s2_init_equivalence_gpu}"
python -m unittest $MODS -v
rc=$?
echo "[s2-gpu] DONE rc=$rc"
exit $rc
