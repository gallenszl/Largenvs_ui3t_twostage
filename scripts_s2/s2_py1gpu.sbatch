#!/bin/bash
#SBATCH --job-name=s2-py
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --output=/home/z50057756/code/RnG_lagernvs_stage2/slurm_logs/%x-%j.out
#SBATCH --error=/home/z50057756/code/RnG_lagernvs_stage2/slurm_logs/%x-%j.err
# run one python command on 1 GPU:  sbatch s2_py1gpu.sbatch <script.py> [args...]
set -uo pipefail
source /home/z50057756/code/RnG_lagernvs_stage2/scripts_s2/s2_env.sh
python "$@"
rc=$?; echo "[s2-py] DONE rc=$rc"; exit $rc
