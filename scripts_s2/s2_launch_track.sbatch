#!/bin/bash
#SBATCH --job-name=s2-launchtrk
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/home/z50057756/code/RnG_lagernvs_stage2/slurm_logs/%x-%j.out
#SBATCH --error=/home/z50057756/code/RnG_lagernvs_stage2/slurm_logs/%x-%j.err
# On one GPU: (1) kernel-launch profile of the renderers at 1 / 10 target views, (2) VGGT TrackHead cost on our
# shapes (separate process: it imports the original vggt package read-only from ~/code/vggt).
set -uo pipefail
source /home/z50057756/code/RnG_lagernvs_stage2/scripts_s2/s2_env.sh
OUT=/home/z50057756/tmp/s2_bench
python tools_s2/s2_launch_profile.py --config configs/S2P8_uni3t70k_b32t6_lr35_a21k26k.yaml --out $OUT/launch_profile_P8.json; echo "[launchtrk] launch rc=$?"
python tools_s2/s2_trackhead_cost.py --out $OUT/trackhead_cost.json; echo "[launchtrk] track rc=$?"
echo "[launchtrk] DONE"
