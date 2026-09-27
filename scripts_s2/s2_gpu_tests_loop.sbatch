#!/bin/bash
#SBATCH --job-name=s2-gputest-loop
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=00:40:00
#SBATCH --no-requeue
#SBATCH --output=/home/z50057756/code/RnG_lagernvs_stage2/slurm_logs/%x-%j.out
#SBATCH --error=/home/z50057756/code/RnG_lagernvs_stage2/slurm_logs/%x-%j.err
# Diagnosis of the intermittent "P8 init != stage 1" failure: the GPU init test in separate processes on one GPU,
# one fixed PYTHONHASHSEED per process.  $1 = space-separated hash seeds (default "0 1 2 3 4 5").
set -uo pipefail
source /home/z50057756/code/RnG_lagernvs_stage2/scripts_s2/s2_env.sh
for hs in ${1:-0 1 2 3 4 5}; do
  PYTHONHASHSEED=$hs python -m unittest tests.test_s2_init_equivalence_gpu > /tmp/s2loop_$$_$hs.log 2>&1
  rc=$?
  echo "[s2-loop] hashseed $hs rc=$rc $(grep -E '^(OK|FAILED)' /tmp/s2loop_$$_$hs.log | tail -1)"
  grep -E "^AssertionError" /tmp/s2loop_$$_$hs.log | cut -c1-1500
done
echo "[s2-loop] DONE"
