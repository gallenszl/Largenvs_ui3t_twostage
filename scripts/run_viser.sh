python \
viser_render.py --config "configs/LVSM_scene_decoder_only.yaml" \
training.dataset_path = "/workspace/mochu_workspace/data/re10k_img/test/full_list.txt" \
training.batch_size_per_gpu = 4 \
training.target_has_input =  false \
training.num_views = 5 \
training.square_crop = true \
training.num_input_views = 2 \
training.num_target_views = 3 \
inference.if_inference = true \
inference.compute_metrics = true \
inference.render_video = true \
inference_out_dir = ./experiments/evaluation/test