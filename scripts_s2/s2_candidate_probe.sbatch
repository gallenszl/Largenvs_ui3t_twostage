#!/bin/bash
#SBATCH --job-name=s2-cand
#SBATCH --partition=gpu
#SBATCH --qos=lowest
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/z50057756/tmp/s2_probe/s2_cand_%j.log
# 09-26 stage-2 candidate-error pre-check (frozen uni3t stage 1 vs GT correspondences).
# positional: $1 ckpt  $2 out json  [$3 modes, default posed,unposed,posed_roll10]
set -u
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
export TORCH_HOME=/home/z50057756/torch_home_lagernvs
export PATH=/mnt/data-alpha-sg-01/team-camera/home/z50057756/conda/envs/rng-fa3/bin:$PATH
cd /home/z50057756/code/RnG_lagernvs_stage2
echo "== $(date) job $SLURM_JOB_ID node $(hostname) restart ${SLURM_RESTART_COUNT:-0} commit $(git rev-parse --short HEAD)"
python tools_s2/s2_candidate_probe.py --ckpt "$1" --out "$2" ${3:+--modes "$3"}
RC=$?; echo "== $(date) exit $RC"; exit $RC
