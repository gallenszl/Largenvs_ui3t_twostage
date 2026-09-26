#!/bin/bash
#SBATCH --job-name=uni3t-taskgrad
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --output=/home/z50057756/code/RnG_lagernvs/slurm_logs/%x-%j.out
#SBATCH --error=/home/z50057756/code/RnG_lagernvs/slurm_logs/%x-%j.err
# Per-task gradient decomposition (tools/uni3t_task_grad_decomp.py) on several uni3t checkpoints.
# qos=normal by user order 09-25 for this diagnostic. Positional args: CONFIG N_BATCHES CKPT...
set -uo pipefail
cd /home/z50057756/code/RnG_lagernvs
export PATH="/mnt/data-alpha-sg-01/team-camera/home/z50057756/conda/envs/rng-fa3/bin:$PATH"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export TORCH_HOME=/home/z50057756/torch_home_lagernvs PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RNG_ATTENTION_BACKEND=auto
echo "job $SLURM_JOB_ID node $(hostname) branch $(git rev-parse --abbrev-ref HEAD) commit $(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l)"
python tools/uni3t_task_grad_decomp.py "$@"
echo "[td] rc=$?"
