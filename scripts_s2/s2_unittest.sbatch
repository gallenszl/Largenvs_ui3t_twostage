#!/bin/bash
#SBATCH --job-name=s2-unittest
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/slurm_logs/%x-%j.out
#SBATCH --error=/home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/slurm_logs/%x-%j.err
# CPU unit tests (README §5.1). Isambard has no CPU partition, so this takes 1 GPU and hides it: the GPU suites must
# report their 13 tests as skipped (expect "Ran 40 tests ... OK (skipped=13)" since the stage-1 tile-selection tests of 09-28;
# 36 / 11 before them, 33 / 9 before fg_dense / fg_dual). Then the whole-repo discover run.
set -uo pipefail
source /home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/scripts_s2/s2_env.sh
export CUDA_VISIBLE_DEVICES=
echo "=== stage-2 suites ==="
python -m unittest tests.test_s2_geometry tests.test_s2_attention_ref tests.test_s2_modules \
  tests.test_s2_train_contract tests.test_s2_attention_gpu tests.test_s2_init_equivalence_gpu
rc1=$?
echo "[s2-unit] stage-2 suites rc=$rc1"
echo "=== whole repo (discover) ==="
python -m unittest discover -s tests
rc2=$?
echo "[s2-unit] discover rc=$rc2"
echo "[s2-unit] DONE rc=$((rc1 | rc2))"
exit $((rc1 | rc2))
