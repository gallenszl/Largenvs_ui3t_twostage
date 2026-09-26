# 复现 `PLN2pose_all287k_b32t6_fp32lr35_const`

本分支是 **lagernvs**(EncDec ViT-B/8,VGGT-1B 初始化)在 RnG 框架下的 **dense 基线臂**代码快照。
该臂训练数据 = Objaverse **287,562** 物体渲染池,恒定学习率 3.5e-5,batch 32(4 卡 × 8),
每步 4 个输入视图 + 6 个 target 视图,预算 90k 步。它是 Soft MoE / ProMoE 系列实验的**配对参照**。

---

## 0. 目标结果(subset64 评测,`PSNR / LPIPS / FG-PSNR`)

评测集 = 64 个 GSO 物体(`data/gso_subset64.txt`),不在训练池内;每 2000 步自动评一次。

| step | PSNR | LPIPS | FG-PSNR |
|---:|---:|---:|---:|
| 2,000 | 14.72 | 0.3569 | 12.26 |
| 8,000 | 18.73 | 0.2294 | 16.76 |
| 10,000 | 20.69 | 0.1952 | 16.26 |
| 20,000 | 22.68 | 0.1416 | 18.09 |
| 30,000 | 23.08 | 0.1209 | 19.23 |
| 40,000 | 24.01 | 0.1043 | 20.24 |
| 50,000 | 24.28 | 0.0963 | 20.60 |
| 60,000 | 24.52 | 0.0901 | 20.56 |
| 70,000 | 24.74 | 0.0859 | 20.88 |
| 72,000 | 24.67 | 0.0841 | 20.67 |

> ⚠ **这条臂没有跑满 90k**:最后一个作业于 2026-08-18 被人工取消,训练停在 **73,320 步**,
> 最后一个评测点是 72k。引用该臂做配对比较时,72k 之后没有参照值。
>
> 判读口径(项目内约定):主看 LPIPS,差 **0.003** 以上才算信号;PSNR 0.6 dB、FG-PSNR 0.5 dB 为辅。

---

## 1. 代码 / 配置 / 启动脚本

| 用途 | 路径 |
|---|---|
| 训练配置 | `configs/RnGUP_lagernvs_b32t6_fp32lr35_const_90k_all287k.yaml` |
| 训练入口 | `train.py` |
| Slurm 启动器 | `scripts/pln2_train_8h200.sbatch`(靠环境变量选配置;文件名里的 `8h200` 是历史名,本臂实跑 **4 卡**) |
| 守护脚本 | `scripts/pln2_all287k_watchdog.sh`(每 30 分钟:作业掉了就重投;按里程碑清理 ckpt) |
| 训练数据清单 | `data/objaverse_all287k.txt`(287,562 行,仓内已含) |
| 评测清单 | `data/gso_subset64.txt`(63 行,仓内已含) |

训练启动时会把**解析后的完整配置**写到 `<checkpoint_dir>/config.yaml`,并把源码快照备份到
`<checkpoint_dir>/src/`。原始运行的归档配置与仓内 yaml 已逐键核对:**0 处不一致**
(归档多出的唯一键是续训时注入的 `training.resume_ckpt`)。

---

## 2. 环境

原始运行用的是集群上的 conda 环境 `rng-fa3`(PyTorch 2.7.0+cu128 / xformers 0.0.30 / FlashAttention-3):

```bash
conda create -n rng-fa3 python=3.10 -y && conda activate rng-fa3
pip install -r requirements-fa3.txt            # 主依赖(pinned,含 cu128 index)
# 需要逐位复刻时用完整锁文件:
# pip install -r requirements-fa3-current-lock.txt
export PYTHONNOUSERSITE=1                       # 必须:否则 user-site 的 protobuf 会遮蔽环境里的版本
export TORCH_HOME=/path/to/torch_home           # vendored Reconstructor 走 torch.hub 读本地 VGGT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

硬件:4 × H200(141G),实测步时约 3.9 s/步,峰值显存约 55 G/卡,90k 步约 4 天。

---

## 3. 仓库外需要准备的资源

以下都是**绝对路径写死在配置里**的,换机器必须改 `configs/RnGUP_lagernvs_b32t6_fp32lr35_const_90k_all287k.yaml`:

| 配置键 | 原始路径 | 说明 |
|---|---|---|
| `model.pretrained_path` | `/home/z50057756/model/VGGT/model.pt` | VGGT-1B 预训练权重(编码器初始化) |
| `model.rae_decoder.config_path` | `/home/z50057756/code/RAE/configs/decoder/ViTXL` | RAE 解码器结构配置 |
| `model.rae_decoder.pretrained_decoder_path` | `.../RAE/decoders/dinov2/wReg_base/ViTXL_n08_i512/model.pt` | RAE 解码器权重(默认冻结) |
| `model.rae_decoder.normalization_stat_path` | `.../RAE/stats/dinov2/wReg_base/imagenet1k_512/stat.pt` | 归一化统计量 |
| `training.root_path` / `tar_root_path` | `/mnt/data-alpha-sg-02/team-camera/datasets/trellis_processed/all_train_renders` | 287k 物体的渲染 tar(每物体一个 tar,40 视图) |
| `training.val_dataset_cfgs.root_dir` | `/home/z50057756/data/gso_sim2real_25v` | GSO 评测渲染 |
| `training.api_key_path` | `./configs/api_keys.yaml` | **不在仓里**(被 .gitignore),见下 |

`configs/api_keys.yaml` 是 **wandb** 的 key 文件,`setup.py` 里是硬断言,不存在会直接报错。
自己复现时建两行即可,并可离线跑:

```bash
printf 'wandb: "<your-wandb-key>"\n' > configs/api_keys.yaml
export WANDB_MODE=offline      # 不想上传就设这个
```

训练数据(渲染 tar)已备份在 HuggingFace `szlgallen/RnG_training_data` 的
`objaverse_all287k_renders/`(1.309 TiB,分 719 片 tar-of-tars,下载后 `tar -xf` 还原)。

---

## 4. 起训

### 4.1 Slurm(原始方式)

```bash
cd <repo>
CONFIG=configs/RnGUP_lagernvs_b32t6_fp32lr35_const_90k_all287k.yaml NPROC=4 \
sbatch --qos=normal --time=2-00:00:00 \
       --gres=gpu:h200:4 --cpus-per-task=104 --mem=750G \
       --job-name=pln2-all287k \
       scripts/pln2_train_8h200.sbatch
```

作业最长 2 天,靠 `--requeue` 和守护脚本接力;每次重启都会从
`experiments/checkpoints/PLN2pose_all287k_b32t6_fp32lr35_const/` 下最新的 `ckpt_*.pt`
**带优化器状态**续训(启动器自动注入 `training.resume_ckpt`),wandb 也接回同一个 run
(run id 存在 ckpt 目录的 `.wandb_run_id` 里)。

守护(可选,写进 crontab):

```bash
*/30 * * * * <repo>/scripts/pln2_all287k_watchdog.sh
```

### 4.2 不用 Slurm,直接单机 4 卡

```bash
cd <repo>
export PYTHONNOUSERSITE=1 TORCH_HOME=/path/to/torch_home OMP_NUM_THREADS=4
python -m torch.distributed.run --nproc_per_node=4 --nnodes=1 \
  train.py --config configs/RnGUP_lagernvs_b32t6_fp32lr35_const_90k_all287k.yaml
# 续训:追加 training.resume_ckpt=<checkpoint_dir>
```

---

## 5. 关键超参(全部已在配置里,列出便于核对)

| 项 | 值 |
|---|---|
| 学习率 / 日程 | `3.5e-05`,`scheduler_type: constant`,`warmup: 3000` |
| batch | `batch_size_per_gpu: 8` × 4 卡 = 32,`grad_accum_steps: 1` |
| 视图 | `num_input_views: 4`,`num_target_views: 6`,`num_views: 10`,`total_frames_per_obj: 40` |
| 优化器 | AdamW fused,`beta1 0.9 / beta2 0.95`,`weight_decay 0.05`,`grad_clip_norm 1.0` |
| 精度 | `use_amp: true`,`amp_dtype: bf16`,`use_bf16: false`(主权重 fp32),`use_tf32: true` |
| 损失 | `l2_loss_weight 1.0` + `perceptual_loss_weight 0.5`;`weight_camera / weight_point = 0`(纯图像损失) |
| 课程 | `exclude_bg_frac: 0.111111`(前 10k 步排背景),`roll_augment_max_deg: 10`,`target_has_input_prob: 0.1` |
| pose 条件 | `unposed: true`,`cam_cond_zero_p: 0.4`(训练时 40% 概率抹掉相机条件),评测 `val_cam_cond_zero_p: 0.0` |
| RAE 解码器 | 全程冻结(`unfreeze_rae_decoder_at: 1.05` > 1,永不解冻) |
| 存档 / 评测 | `checkpoint_every: 2000`,评测同频,`vis_every: 200` |

---

## 6. 评测口径

训练内嵌的验证就是上面那张表:`data.dataset_gso_ours.GSODataset_ours`,
每个场景 4 输入 + 10 target,`target_has_input: false`,batch 1。
独立评测入口是 `inference.py` / `eval_448.py`(见 `scripts/pln2_eval_gso.sh`)。

---

## 7. 复现注记(已知偏差,如实记录)

1. **步数**:如上,原始运行停在 73,320 步,不是 90k。
2. **`train.py` 的一行差异**:仓内当前版本把 `init_distributed(seed=777)` 改成了
   `init_distributed(seed=int(config.training.get("seed", 777)))`。该改动的文件时间是 2026-08-19,
   **晚于**这条臂结束(2026-08-18),且配置里没有 `training.seed`,默认仍取 777,数值路径不变。
3. **`models/` 的最后一次提交**(`d9a6141`,2026-08-12 22:33)发生在这条臂第一个作业启动之后约 1 小时,
   内容与 dense 主路径无关(给 b64/b128 退火臂加 lr override)。因此「代码逐字节同一版本」这句话
   不能打包票,能保证的是:**配置逐键相同,且改动都不在这条臂的数值路径上**。
4. **路径**:配置里全是集群绝对路径,换环境必须改(见第 3 节)。
5. **wandb**:key 文件是硬断言,不提供就起不来;离线跑用 `WANDB_MODE=offline`。

---

## 8. 目录速览

```
configs/   训练与评测配置(本臂 = RnGUP_lagernvs_b32t6_fp32lr35_const_90k_all287k.yaml)
data/      物体清单与评测清单(纯文本)
model/     LagerNVSInRnG wrapper 与 RnG 组件
models/    renderer / encoder-decoder / 各类 FFN 与注意力块
vggt/      vendored VGGT-1B
utils/     训练工具(优化器、日志契约等)
scripts/   Slurm 启动器 + 各实验臂的守护脚本
tools/     离线分析(wandb run 合并等)
tests/     CPU 单测
train.py   训练入口
```
