#!/bin/bash
#SBATCH --job-name=s2-seams
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --output=/home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/slurm_logs/%x-%j.out
#SBATCH --error=/home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/slurm_logs/%x-%j.err
# token-seam statistics, zoom panels and overlays (tools_s2/s2_edge_seams.py), then the web page assets with them
# (tools_s2/s2_web_assets.py).   sbatch s2_seams_web.sbatch <exp name> <web out dir> [select_step=14000]
set -uo pipefail
source /home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/scripts_s2/s2_env.sh
EXP=$1; WEB=$2; SEL=${3:-14000}
CKPT=$S2_ROOT/checkpoints/$EXP
AN=$S2_ROOT/evaluation/seams_$EXP
mkdir -p "$AN"
cd "$REPO"
echo "Job $SLURM_JOB_ID | node $(hostname) | $EXP -> $AN, web $WEB (select $SEL)"
python tools_s2/s2_edge_seams.py --config "configs/$EXP.yaml" --exp_dir "$CKPT" --out "$AN/seams.json" \
  --panel_dir "$AN/panels" --ovl_dir "$AN/ovl"
rc=$?; echo "[s2-seams] edge_seams rc=$rc"; [ $rc -eq 0 ] || exit $rc
python tools_s2/s2_web_assets.py --exp_dir "$CKPT" --out "$WEB" --select_step "$SEL" --seams_json "$AN/seams.json" \
  --ovl_dir "$AN/ovl"
rc=$?; echo "[s2-seams] DONE rc=$rc"; exit $rc
