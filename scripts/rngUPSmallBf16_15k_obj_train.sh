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
# torchrun --nproc_per_node 8 --nnodes 1 \
#     --rdzv_id 18639 --rdzv_backend c10d \
#     --rdzv_endpoint localhost:29506 \
#     train.py --config configs/RnGUP_obj_small_bf16_15k.yaml

# ablation: no camera loss
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node 8 --nnodes 1 \
#     --rdzv_id 18639 --rdzv_backend c10d \
#     --rdzv_endpoint localhost:29506 \
#     train.py --config configs/RnGUP_obj_small_bf16_15k.yaml \
#     training.weight_camera = 0.0 \
#     training.batch_size_per_gpu = 8 \
#     training.grad_accum_steps = 1 \
#     exp_name = 'RnGUP_s_abNoCam_b8v3Bf16'


torchrun --nproc_per_node 8 --nnodes 1 \
    --rdzv_id 18639 --rdzv_backend c10d \
    --rdzv_endpoint localhost:29513 \
    train.py --config configs/RnGUP_obj_small_bf16_15k.yaml \
    training.batch_size_per_gpu = 2 \
    training.grad_accum_steps = 4 \
    exp_name = 'RnGUP_s_LinHead_zhelun_reimplementation'