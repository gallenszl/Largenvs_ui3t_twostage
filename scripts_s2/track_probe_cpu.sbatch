#!/bin/bash
#SBATCH --job-name=trkprobe
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/home/z50057756/code/RnG_lagernvs_stage2/slurm_logs/%x-%j.out
#SBATCH --error=/home/z50057756/code/RnG_lagernvs_stage2/slurm_logs/%x-%j.err
# GT statistics for the track-consistency loss on real training samples:  $1 n_objects  $2 output json
set -uo pipefail
export PATH=/mnt/data-alpha-sg-01/team-camera/home/z50057756/conda/envs/rng-fa3/bin:$PATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
cd /home/z50057756/code/RnG_lagernvs_stage2
python tools/track_consistency_probe.py --config configs/RnGUP_lagernvs_uni3t_b32t6_fp32lr35_const_90k_all287k.yaml \
  --n_objects "${1:-128}" --out "${2:-/home/z50057756/tmp/track_consistency_probe/probe_128.json}"
rc=$?; echo "[trkprobe] DONE rc=$rc"; exit $rc
