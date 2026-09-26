#!/bin/bash
set -euo pipefail

CONDA_ENV=/mnt/data-alpha-sg-01/team-camera/home/z50057756/conda/envs/rng-fa3
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONUNBUFFERED=1
export TORCH_HOME=/mnt/data-alpha-sg-01/team-camera/home/z50057756/.cache/torch

cd /home/z50057756/code/RnG_feature_allignment

OLDHOME=/mnt/data-alpha-sg-01/team-camera/home/z50057756
CHECKPOINT_DIR=$OLDHOME/code/RnG_feature_allignment/experiments/checkpoints/RnGUP_obj_448_render44798
GSO_ROOT=/home/z50057756/data/co3d_teddybear_gsoformat_objdepth
SPLIT_FILE=/home/z50057756/tmp/co3d_easy_eval/co3d_teddybear_traj_top30.txt
VIEW_IDX_FILE=/home/z50057756/tmp/co3d_easy_eval/view_indices_traj_top30_uniform4_all21.json

python - <<'PY'
import plotly.express  # noqa: F401
PY

echo "=== RnG CO3D traj-top30 Viser demo ==="
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Data root:  $GSO_ROOT"
echo "Split:      $SPLIT_FILE"
echo "View idx:   $VIEW_IDX_FILE"
echo "Context:    [0, 8, 16, 24]"
echo "======================================="

python \
viser_demo.py --config "configs/RnGUP_obj_448_bf16_15k_render44798.yaml" \
model.pretrained_path=$OLDHOME/model/VGGT/model.pt \
model.rae_decoder.config_path=$OLDHOME/code/RAE/configs/decoder/ViTXL \
model.rae_decoder.pretrained_decoder_path=$OLDHOME/model/RAE/decoders/dinov2/wReg_base/ViTXL_n08_i512/model.pt \
model.rae_decoder.normalization_stat_path=$OLDHOME/model/RAE/stats/dinov2/wReg_base/imagenet1k_512/stat.pt \
training.feature_alignment.target_encoder_weights=$OLDHOME/.cache/torch/hub/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
training.feature_alignment.target_encoder_hub_dir=$OLDHOME/.cache/torch/hub/facebookresearch_dinov3_94a96ac83c2446f15f9bdcfae23cad3c6a9d4988 \
training.feature_alignment.enabled=false \
training.checkpoint_dir=$CHECKPOINT_DIR \
training.val_dataset_cfgs.root_dir=$GSO_ROOT \
training.val_dataset_cfgs.split_file=$SPLIT_FILE \
training.batch_size_per_gpu=1 \
training.target_has_input=false \
training.num_input_views=4 \
inference.view_idx_file_path=$VIEW_IDX_FILE \
inference.if_inference=true \
inference.compute_metrics=true \
inference.render_video=false \
training.roll_augment_max_deg=0
