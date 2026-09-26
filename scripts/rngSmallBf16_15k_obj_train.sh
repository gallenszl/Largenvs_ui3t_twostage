export https_proxy='http://agent.baidu.com:8891'
export OMP_NUM_THREADS=4
# # train small scale network
# torchrun --nproc_per_node 8 --nnodes 1 \
#     --rdzv_id 18635 --rdzv_backend c10d \
#     --rdzv_endpoint localhost:29502 \
#     train.py --config configs/LVSM_scene_decoder_only.yaml \
#     model.transformer.n_layer = 12 \
#     training.batch_size_per_gpu = 16

# train standard scale network
torchrun --nproc_per_node 8 --nnodes 1 \
    --rdzv_id 18639 --rdzv_backend c10d \
    --rdzv_endpoint localhost:29506 \
    train.py --config configs/RnG_obj_small_bf16_15k.yaml