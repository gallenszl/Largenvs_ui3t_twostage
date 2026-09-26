python \
viser_render_obj.py --config "configs/RnGUP_obj_small_bf16_15k.yaml" \
training.val_dataset_cfgs.split_file = 'data/gso.txt' \
training.batch_size_per_gpu = 1 \
training.target_has_input =  false \
training.num_input_views = 4 \
inference.if_inference = true \
inference.compute_metrics = true \
inference.render_video = false \