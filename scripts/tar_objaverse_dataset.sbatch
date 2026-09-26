#!/bin/bash
#SBATCH --job-name=tar_objaverse
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --qos=lowest
#SBATCH --output=/home/z50057756/code/RnG/slurm_logs/%x-%j.out
#SBATCH --error=/home/z50057756/code/RnG/slurm_logs/%x-%j.err

set -eo pipefail

source /home/z50057756/miniconda3/etc/profile.d/conda.sh
conda activate /home/z50057756/conda/envs/rng-fa3

set -u

mkdir -p /home/z50057756/code/RnG/slurm_logs

cd /home/z50057756/code/RnG

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"

AUDIT_OUT=${AUDIT_OUT:-"/home/z50057756/code/RnG/slurm_logs/tar_objaverse_audit-${SLURM_JOB_ID}.json"}
LIMIT=${LIMIT:-0}

python scripts/tar_objaverse_dataset.py \
  --list-file data/objaverse_v1_in_lvis_25v.txt \
  --root-path /home/z50057756/FluffyElephant \
  --tar-root /home/z50057756/FluffyElephant_tar \
  --num-workers "${SLURM_CPUS_PER_TASK}" \
  --limit "${LIMIT}" \
  --audit-out "${AUDIT_OUT}"
