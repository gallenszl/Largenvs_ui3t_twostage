#!/bin/bash
#SBATCH --job-name=s2-gputest
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/home/z50057756/code/RnG_lagernvs_stage2/slurm_logs/%x-%j.out
#SBATCH --error=/home/z50057756/code/RnG_lagernvs_stage2/slurm_logs/%x-%j.err
# Stage-2 GPU tests (plan G4). $1 = unittest modules (default: both GPU test files).
set -uo pipefail
source /home/z50057756/code/RnG_lagernvs_stage2/scripts_s2/s2_env.sh
# optional extra environment for diagnosis runs, e.g. "CUDA_LAUNCH_BLOCKING=1" (positional $2, space separated)
for kv in ${2:-}; do export "$kv"; echo "[s2-gpu] env $kv"; done
MODS="${1:-tests.test_s2_attention_gpu tests.test_s2_init_equivalence_gpu}"
python -m unittest $MODS -v
rc=$?
echo "[s2-gpu] DONE rc=$rc"
exit $rc
