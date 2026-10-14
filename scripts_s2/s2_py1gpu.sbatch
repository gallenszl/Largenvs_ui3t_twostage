#!/bin/bash
#SBATCH --job-name=s2-py
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --output=/home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/slurm_logs/%x-%j.out
#SBATCH --error=/home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/slurm_logs/%x-%j.err
# run one python command on 1 GPU:  sbatch s2_py1gpu.sbatch <script.py> [args...]
set -uo pipefail
source /home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/scripts_s2/s2_env.sh
python "$@"
rc=$?; echo "[s2-py] DONE rc=$rc"; exit $rc
