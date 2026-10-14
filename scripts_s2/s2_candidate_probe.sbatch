#!/bin/bash
#SBATCH --job-name=s2-cand
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --open-mode=append
#SBATCH --output=/home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/slurm_logs/%x-%j.out
# 09-26 stage-2 candidate-error pre-check (frozen uni3t stage 1 vs GT correspondences).
# positional: $1 ckpt  $2 out json  [$3 modes, default posed,unposed,posed_roll10]
set -u
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
source /home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/scripts_s2/s2_env.sh
echo "== $(date) job $SLURM_JOB_ID node $(hostname) restart ${SLURM_RESTART_COUNT:-0} commit $(git rev-parse --short HEAD)"
python tools_s2/s2_candidate_probe.py --ckpt "$1" --out "$2" ${3:+--modes "$3"}
RC=$?; echo "== $(date) exit $RC"; exit $RC
