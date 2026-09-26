#!/bin/bash
#SBATCH --job-name=uni3t-gradnorm
#SBATCH --partition=gpu
#SBATCH --qos=lowest
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
set -euo pipefail
cd /mnt/nvme1/home/z50057756/code/RnG_lagernvs
export PATH="/mnt/data-alpha-sg-01/team-camera/home/z50057756/conda/envs/rng-fa3/bin:$PATH"
export PYTHONNOUSERSITE=1
CFG=${1:-configs/RnGUP_lagernvs_uni3t_b32t6_fp32lr35_const_90k_all287k.yaml}
CKPT=${2:-/mnt/data-alpha-sg-01/team-camera/home/z50057756/moe_experiments/checkpoints/PLN2uni3t_all287k_b32t6_fp32lr35_const/ckpt_0000000000016000.pt}
N=${3:-24}
echo "branch=$(git rev-parse --abbrev-ref HEAD) commit=$(git rev-parse --short HEAD) node=$(hostname)"
python tools/uni3t_gradnorm_groups.py "$CFG" "$CKPT" "$N"
