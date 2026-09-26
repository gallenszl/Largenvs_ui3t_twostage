torchrun --nproc_per_node 8 --nnodes 1 \
--rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29506 \
inference.py --config "/workspace/mochu_workspace/LVSM/configs/VGGT4LVSM_obj_decoder_only.yaml" \
inference_out_dir = ./experiments/evaluation/test_vggt_obj \
training.dataset_name = 'data.dataset_gso.GSODataset' \
training.val_dataset_cfgs.split_file = 'data/gso.txt' \
training.target_has_input =  false \
training.num_views = 14 \
training.num_input_views = 4 \
training.num_target_views = 10 \
training.batch_size_per_gpu = 1 \