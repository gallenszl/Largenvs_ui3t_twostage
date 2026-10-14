#!/bin/bash
#SBATCH --job-name=s2-graph
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:50:00
#SBATCH --output=/home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/slurm_logs/%x-%j.out
#SBATCH --error=/home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/slurm_logs/%x-%j.err
# bf16 weights + CUDA Graphs at batch 1 (tools_s2/s2_graph_bench.py): one process per target-view count (1, then 10)
# on the same node, so that a failed capture in one cannot take the other down.
#   sbatch scripts_s2/s2_graph_bench.sbatch <stage-2 ckpt> <output prefix>   -> <prefix>_vt1.json, <prefix>_vt10.json
set -uo pipefail
source /home/u6ga/zhelun.u6ga/Largenvs_ui3t_twostage/scripts_s2/s2_env.sh
export PYTORCH_CUDA_ALLOC_CONF=      # default caching allocator (no expandable segments) for graph capture
CKPT=$1
OUT=$2
for vt in 1 10; do
  python tools_s2/s2_graph_bench.py --config configs/S2P8_uni3t70k_b32t6_lr35_a21k26k.yaml --s2_ckpt "$CKPT" \
    --vts "$vt" --out "${OUT}_vt${vt}.json"
  echo "[s2-graph] vt=$vt rc=$?"
done
echo "[s2-graph] DONE"
