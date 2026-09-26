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
torchrun --nproc_per_node 8 --nnodes 2 --rdzv_endpoint=${POD_0_IP}:29511 \
    --node_rank=${PADDLE_TRAINER_ID} \
    --rdzv_id 18639 --rdzv_backend c10d \
    train.py --config configs/RnGUP_obj_medium_bf16_20k.yaml \
    training.train_steps = 40000 \
    exp_name = RnGUP_obj_MedP_N2b6v3a3Bf16_40K