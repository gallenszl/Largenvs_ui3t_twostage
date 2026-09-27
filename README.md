# Largenvs uni3t 两阶段稀疏细化(第二阶段)

> **一句话现状**:第一阶段固定用 uni3t 模型 `PLN2uni3t_all287k_b32t6_fp32lr35_wsd60k_a10k` ckpt_70000,冻结不训。第二阶段是只处理前景的稀疏细化器(LSRM 式)。代码已通过 G0–G8 关口,但实测**第二阶段渲染器比第一阶段渲染器还慢**(训练形状 1.8 倍),达不到"必须比第一阶段快很多"的要求。所以正式训练一次也没跑,**下一步是先选定新结构**(第 8.1 节)。

- 代码仓:本仓 = 原机器 `~/code/RnG_lagernvs_stage2` 的 `stage2` 分支,推到 GitHub 后是 `main` 分支。
- 基底:`RnG_lagernvs` 仓 uni3t 分支 0abb1d7 的快照(本仓提交 79798f8)。第一阶段的模型代码(`model/`、`models/`、`vggt/`)一行没改,冻结的第一阶段就是用它构建的。
- 第一阶段权重:`szlgallen/RAE_backup_checkpoints` 仓内 `moe_experiments/checkpoints/PLN2uni3t_all287k_b32t6_fp32lr35_wsd60k_a10k/ckpt_0000000000070000.pt`(第 3.1 节)。
- 写于 2026-09-27。原机器 GPU 排满、难以调试,之后的实验都换到新机器上做。

---

## 目录

0. 新机器上手顺序
1. 仓库结构
2. 环境
3. 权重与数据
4. 迁移时要改的路径
5. 怎么测试
6. 怎么运行
7. 已完成的内容与结果
8. 还需要继续测试的内容
9. 之前定下、后续要做的消融 / 备选 / 暂缓
10. 已知问题与注意事项
11. 用到的 checkpoints 一览

---

## 0. 新机器上手顺序

1. `git clone git@github.com:gallenszl/Largenvs_ui3t_twostage.git`
2. 准备环境(第 2 节),关键是 torch 2.7 + xformers + flash_attn_3(需要 Hopper GPU)。
3. 从 HF 下载第一阶段权重,并核对 sha256(第 3.1 节)。这是新机器上唯一缺的东西。
4. 确认其余权重和数据都在(第 3.2、3.3 节,新机器上已有)。
5. 按第 4 节改路径,另建一个放 W&B key 的 yaml。
6. 依次跑 CPU 单测、GPU 单测、冒烟(第 5 节)。
7. 先定第二阶段的新结构(第 8.1 节),再开训练。旧结构的训练脚本和配置都能直接跑,但按第 7.4 节的实测,它不满足速度要求。

---

## 1. 仓库结构

第二阶段新增或改动的文件(相对快照 79798f8,共 53 个文件):

| 路径 | 内容 |
|---|---|
| `model_s2/stage1_runner.py` | 冻结的第一阶段。载入并核验 ckpt:步数必须等于 `stage1_step`,缺失或多余的键只允许 `loss_computer.*`。`pass1` = VGGT + 第一阶段渲染器 + 三个头;`pass2` = 把 4 个输入相机当目标再渲一遍,得到输入图深度 |
| `model_s2/geometry.py` | 投影、token 布局、逐 token 的前向 / 反向屏蔽表(可见性、几何窗口)。屏蔽表就是"每个 query 允许看哪些 key"的布尔表 |
| `model_s2/masked_attention.py` | 带屏蔽表的注意力后端:`flex`(FlexAttention,两组现在都用它)、`sdpa`(PyTorch 高效内核,作参照和保底)、`ref`(CPU fp64 参照,测试用)。另有块对角稠密注意力 `fa3` |
| `model_s2/blocks.py` | 第二阶段的块:注入、压缩(块摘要 + ResBlock)、几何窗口、门控、按可见性过滤的反向注意力 |
| `model_s2/renderer_s2.py` | 第二阶段渲染器,12 块,逐块重算 |
| `model_s2/heads_s2.py` | 把残差按层加回第一阶段 token(T~ = T1 + Lin_m(t2),m = 2/5/8/11),再过颜色头和点图头。块 4 用改过网格的 DPT |
| `model_s2/stage2_wrapper.py` | `Stage2LagerNVS`:整次前向、损失、只存可训练参数 |
| `inference_s2.py` | 评测入口。一次前向同时写第二阶段结果 `<dir>` 和第一阶段结果 `<dir>_s1`,两者天然按物体配对 |
| `configs/S2P8_uni3t70k_b32t6_lr35_a21k26k.yaml` | 块 8 组:第二阶段目标 token 仍是 8×8 像素一块 |
| `configs/S2P4_uni3t70k_b32t6_lr35_a21k26k.yaml` | 块 4 组:前景目标 token 细到 4×4 像素 |
| `tools_s2/` | 工具和诊断脚本(第 6.4 节) |
| `scripts_s2/` | Slurm 作业脚本与 5 分钟守护 `s2_watch.sh` |
| `tests/test_s2_*.py` | 单测(第 5 节) |
| `train.py`、`utils/training_utils.py` | 验证时依次跑有位姿和无位姿、第 0 步先验证、只存可训练参数、额外存点、自动剪存、续训出错直接报错停下(不再静默从第 0 步重来) |
| `data/dataset_gso_ours.py` | 验证集可以单独关掉平面内旋转 |
| `data/gso_subset4.txt` | 冒烟用的 4 个物体 |
| `docs/half_pixel_pointmap_bug.md`、`tools_s2/check_pointmap_on_ray.py` | 第一阶段点图半像素 bug 的说明与验收脚本(第 10 节) |
| `docs/README_upstream_LVSM.md` | 原 README(上游 LVSM),从根目录挪到这里 |

两组配置只有 5 行不同:`target_patch` 8 / 4、`target_radius` 2 / 3、`pad_bucket` 128 / 256,以及 `exp_name`。

---

## 2. 环境

原机器上的环境名是 `rng-fa3`。训练、测试、评测都用它。

| 包 | 版本 |
|---|---|
| python | 3.10.20 |
| torch / torchvision | 2.7.0+cu128 / 0.22.0+cu128 |
| CUDA / cuDNN | 12.8 / 9.7.1 |
| xformers | 0.0.30 |
| flash_attn_3 | 3.0.0(Hopper 版,源码编译;xformers 的 flash3 算子要用它) |
| triton | 3.3.0(FlexAttention 编译用) |
| numpy / Pillow / opencv-python-headless | 1.26.4 / 10.0.1 / 4.7.0.72 |
| lpips / timm / transformers | 0.1.4 / 1.0.25 / 4.48.0 |
| omegaconf / easydict / einops / PyYAML | 2.1.1 / 1.13 / 0.8.2 / 6.0.3 |
| scipy / scikit-image / wandb | 1.15.3 / 0.23.2 / 0.18.7 |

- **硬件**:只在 H200(sm_90)上测过。训练 4 卡,测试、冒烟、测速 1 卡。FA3 只能在 Hopper 上跑。
- **备用环境** `rng-v3moe-t211-te218`(torch 2.11)没用上,不需要。
- `scripts_s2/s2_env.sh` 设置的环境变量:
  - `PYTHONNOUSERSITE=1`:原机器 `~/.local` 里的包会遮蔽环境里的包。
  - `TORCH_HOME`:VGGT-1B 和 VGG 权重的缓存目录(第 3.2 节)。
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`:形状多变时减少显存碎片。
  - `TORCHINDUCTOR_CACHE_DIR`、`TRITON_CACHE_DIR`:放在节点本地 `/tmp`,不要放共享盘。FlexAttention 的编译缓存写在这里。
  - `OMP_NUM_THREADS=4`。

---

## 3. 权重与数据

### 3.1 第一阶段权重(新机器上只缺这个)

**是什么**:uni3t 三任务模型,同时做新视角合成、输入图位姿、目标图点图。训练数据是 all287k。前 60k 步按恒定学习率 3.5e-5 训练(即 `PLN2uni3t_all287k_b32t6_fp32lr35_const` ckpt_60000),之后按 1−√ 进度退火,60k→70k 步降到 0。本文件是退火终点。

**在哪**:

- HF:<https://huggingface.co/szlgallen/RAE_backup_checkpoints/blob/main/moe_experiments/checkpoints/PLN2uni3t_all287k_b32t6_fp32lr35_wsd60k_a10k/ckpt_0000000000070000.pt>
- 同一目录还放了 `README.md`、`SHA256SUMS.txt`,以及这次训练实际用的 `config.yaml`。
- 原机器:`/mnt/data-alpha-sg-01/team-camera/home/z50057756/moe_experiments/checkpoints/PLN2uni3t_all287k_b32t6_fp32lr35_wsd60k_a10k/ckpt_0000000000070000.pt`

| 项 | 值 |
|---|---|
| 大小 | 16,755,969,541 字节(15.6 GiB) |
| sha256 | `00cb27e01c359468037cd821cea6b596b339db87bc7eebbe5ae81437d47893f4` |
| 顶层键 | `model`、`optimizer`、`lr_scheduler`、`fwdbwd_pass_step` = 70000、`param_update_step` = 70000 |
| `model` | 1734 个 fp32 张量,共 15.07 亿个元素。其中含 66 个感知损失 VGG 张量(`loss_computer.*`) |

表注:上传后已用 HF 记录的 LFS sha256 和字节数,对照本地 `sha256sum` 核验,两者一致(2026-09-27)。文件里保留了优化器状态,所以也能接着训第一阶段。

**下载与核对**:

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1     # 可选,装了 hf_transfer 时更快
hf download szlgallen/RAE_backup_checkpoints \
  moe_experiments/checkpoints/PLN2uni3t_all287k_b32t6_fp32lr35_wsd60k_a10k/ckpt_0000000000070000.pt \
  --local-dir /path/to/ckpts
sha256sum /path/to/ckpts/moe_experiments/checkpoints/PLN2uni3t_all287k_b32t6_fp32lr35_wsd60k_a10k/ckpt_0000000000070000.pt
# 应为 00cb27e01c359468037cd821cea6b596b339db87bc7eebbe5ae81437d47893f4
```

**怎么用**:

- **第二阶段**:把两组配置里的 `model.stage2.stage1_ckpt` 改成下载后的路径。`stage1_step: 70000` 不动,载入时会和 ckpt 里的 `fwdbwd_pass_step` 比对。
- **单独跑第一阶段**:配置用 `configs/RnGUP_lagernvs_uni3t_b32t6_fp32lr35_wsd60k_a10k_all287k.yaml`,命令见 `scripts_s2/s2_s1ref_subset64.sbatch`(`inference.py`,`training.checkpoint_dir=<ckpt 文件>`)。
- **在代码里载入**:

```python
ck = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
missing, unexpected = model.load_state_dict(ck["model"], strict=False)   # model = LagerNVSInRnG(uni3t 配置)
# 允许的缺失 / 多余键只有 loss_computer.*(感知损失 VGG)
```

- **已知问题**:这个模型用的点图真值偏了半个像素(第 10 节),新视角合成和深度指标不受影响。

### 3.2 其余权重(新机器已有,这里只说明用了什么)

| 权重 | 谁在用 | 放在哪 |
|---|---|---|
| VGGT-1B `model.pt`(`facebook/VGGT-1B`) | 构造模型时由 `torch.hub.load_state_dict_from_url` 载入,用来初始化编码器,以及相机头的 69 个张量。随后被 ckpt_70000 覆盖,但构造时必须能读到 | `$TORCH_HOME/hub/checkpoints/model.pt` |
| `imagenet-vgg-verydeep-19.mat`(535 MB) | 感知损失 `PerceptualLoss`,`model/loss.py:39` 按相对路径读 `./metric_checkpoint/` | 仓库根目录下的 `metric_checkpoint/`。原机器上是指向 `RnG_lagernvs/metric_checkpoint` 的符号链接,不进 git |
| `vgg16-397923af.pth`、`vgg19-dcbb9e9d.pth` | LPIPS(`lpips.LPIPS(net="vgg")`,评测用)需要 torchvision 的 VGG16 权重;两个文件在原机器的 TORCH_HOME 里都有,一并准备 | `$TORCH_HOME/hub/checkpoints/` |
| lpips 包自带的线性层权重 | 评测 LPIPS | pip 包内 |

配置里还有几个从 RnG 继承下来的旧键:`model.pretrained_path`(VGGT)、RAE 解码器的三项、dinov3 的 `target_encoder_*`。第一阶段和第二阶段的代码路径都不读它们,不用准备。

### 3.3 数据(新机器已有,这里只说明用了什么)

| 用途 | 数据 | 原机器路径 | 在哪里配置 |
|---|---|---|---|
| 训练 | Objaverse all287k 渲染:每个物体一个 tar,内含 512×512 RGBA、16 位深度 PNG、transforms.json(逐帧 fov)。清单 `data/objaverse_all287k.txt` 共 287,562 个物体 | `/mnt/data-alpha-sg-02/team-camera/datasets/trellis_processed/all_train_renders` | `training.root_path`、`training.tar_root_path`、`training.dataset_path` |
| 训练中的验证、G9 参照 | GSO `gso_sim2real_25v`(v1 渲染)。训练内用 `data/gso_subset64.txt`(64 个物体),冒烟用 `data/gso_subset4.txt` | `/home/z50057756/data/gso_sim2real_25v` | `training.val_dataset_cfgs.root_dir`、`split_file` |
| 全量评测、G6 | GSO `gso_sim2real_25v_v2`,`data/gso.txt` 共 1030 个物体;每个物体 4 张输入 → 10 张目标,旋转 0 | `/home/z50057756/data/gso_sim2real_25v_v2` | `scripts_s2/s2_fulleval.sbatch` 里的 `GSO_ROOT` |

加载器把 512 缩到 256:RGB 用 LANCZOS,深度和前景掩码用最近邻。

---

## 4. 迁移时要改的路径

一条命令列出全部写死的路径:

```bash
grep -rn "/home/z50057756\|/mnt/data-alpha" configs/S2P* scripts_s2 tools_s2 model_s2 inference_s2.py
```

| 文件 | 要改的 |
|---|---|
| `configs/S2P8_…yaml`、`configs/S2P4_…yaml` | `model.stage2.stage1_ckpt`;`training.api_key_path`;`training.root_path` 与 `tar_root_path`;`training.checkpoint_dir` 与 `validation_out_dir`(原来都是旧盘上的 `…/moe_experiments/checkpoints/${exp_name}`);`training.val_dataset_cfgs.root_dir` |
| `scripts_s2/s2_env.sh` | `CONDA_ENV`、`TORCH_HOME`、`REPO` |
| `scripts_s2/*.sbatch`(全部) | `#SBATCH --output/--error` 路径;`source …/s2_env.sh` 的绝对路径;分区 `gpu`、QOS `normal`/`lowest`、`--gres=gpu:h200:N`;`--exclude=lrc-alpha-sg-gpu12`(只对原集群有意义,删掉) |
| `scripts_s2/s2_train.sbatch` | `CKPT_DIR`;磁盘预检 `df` 的路径(要求剩余 ≥ 40 GB);接力时提交自己用的绝对路径 |
| `scripts_s2/s2_fulleval.sbatch` | `GSO_ROOT` |
| `scripts_s2/s2_smoke_1gpu.sbatch`、`tools_s2/s2_diag_seq.py` | 冒烟输出目录(原来是旧盘 `…/moe_experiments/SMOKE/`) |
| `scripts_s2/s2_s1ref_subset64.sbatch` | 第一阶段 ckpt 路径、GSO v1 路径 |
| `scripts_s2/s2_candidate_probe.sbatch`、`tools_s2/s2_candidate_probe.py` | GSO v1 路径、环境路径 |
| `scripts_s2/s2_bench.sbatch`、`s2_speed_profile.sbatch`、`s2_launch_track.sbatch` | 输出目录(原来是 `~/tmp/s2_bench`) |
| `tools_s2/s2_trackhead_cost.py` | 只读导入原版 VGGT 仓 `~/code/vggt`(只有这个估算工具用到) |

**W&B key**:`setup.py` 要求 `training.api_key_path` 指向的 yaml 必须存在,内容是一行 `wandb: <你的 key>`。放在仓库外面,不要提交。冒烟脚本设了 `WANDB_MODE=offline`,但这个文件仍然要存在。

---

## 5. 怎么测试

### 5.1 CPU 单测(几十秒,不需要 GPU)

```bash
python -m unittest tests.test_s2_geometry tests.test_s2_attention_ref tests.test_s2_modules \
  tests.test_s2_train_contract tests.test_s2_attention_gpu tests.test_s2_init_equivalence_gpu
```

期望输出:`Ran 33 tests … OK (skipped=9)`,跳过的 9 项是 GPU 测试。2026-09-27 在原机器登录节点(2 核)实测:用时 29 秒,结果如上。

| 测试文件 | 项数 | 检查什么 |
|---|---|---|
| `test_s2_geometry` | 10 | token 编号公式;投影往返与主点约定;Morton 序是一一映射;膨胀等于最大池化;射线与 `compute_rays` 一致;打包 ↔ 补齐往返恒等;前向表、反向表与逐像素暴力实现逐格相等(0 处不同);遮挡会改变表;块编号 |
| `test_s2_attention_ref` | 4 | 双向块、末块的输出与反向和 fp64 朴素循环一致;空行输出与梯度恰为 0,register 能看到全部 key;第一阶段的块权重能原样载入第二阶段的块 |
| `test_s2_modules` | 5 | 块 8 拷来的张量与第一阶段相等、新层按约定初始化;块 8 开局输出等于第一阶段(小号模型);块 4 的形状与背景 token;每个可训练参数都有梯度(DDP 要求);权重衰减分组 |
| `test_s2_train_contract` | 5 | 剪存只留最新几个和里程碑;可训练键的判定(感知 VGG 的别名键算冻结部分,buffer 也要存);续训出错必须停下(坏 ckpt、优化器不符、键不符);学习率在 1000 / 21000 / 23500 / 26000 步分别是峰值的 1 / 1 / 1−√0.5 / 0 倍;`train.py` 的接线 |
| `test_s2_attention_gpu` | 4(GPU) | 见 5.2 |
| `test_s2_init_equivalence_gpu` | 5(GPU) | 见 5.2 |

全仓测试(含第一阶段的旧测试)可以用 `python -m unittest discover -s tests` 跑。09-27 在提交 a4d57b1 时是 83 项全过,之后没有重跑全仓。

### 5.2 GPU 单测(1 卡,需要第一阶段权重)

```bash
sbatch scripts_s2/s2_gpu_tests.sbatch        # $1 可指定测试模块;$2 可加诊断环境变量,如 "CUDA_LAUNCH_BLOCKING=1"
```

- `test_s2_attention_gpu`(4 项):sdpa 高效内核、flex 各自对 fp64 参照(前向与梯度);FA3 块对角路径对分段参照;建表过程在 `set_sync_debug_mode("error")` 下没有主机同步。
- `test_s2_init_equivalence_gpu`(5 项,用真实的 ckpt_70000):包装里的第一阶段路径与 `LagerNVSInRnG.forward` 逐位相同,块 8 开局输出与第一阶段逐位相同;训练模式下的随机抽签与第一阶段一致;第二遍的深度和屏蔽表合理;一步 AdamW 只更新第二阶段,第一阶段权重逐位不变;块 4 前向输出有限。
- **已知**:"块 8 开局逐位相同"在原集群 gpu12 上出现过一次无法解释的失败(作业 137836),之后 11 次运行全过。新机器上请连跑几次。`scripts_s2/s2_gpu_tests_loop.sbatch` 可以换不同的 `PYTHONHASHSEED` 在多个进程里重复跑。

### 5.3 冒烟(1 卡,不超过 1.5 小时)

```bash
sbatch scripts_s2/s2_smoke_1gpu.sbatch configs/S2P8_uni3t70k_b32t6_lr35_a21k26k.yaml
sbatch scripts_s2/s2_smoke_1gpu.sbatch configs/S2P4_uni3t70k_b32t6_lr35_a21k26k.yaml
```

| 段 | 内容 | 过关条件 |
|---|---|---|
| leg 0 | 动力学探针:同一个 batch 训 40 步 | 零初始化的输出层(Lin_m、块 4 颜色头)第 1 步就离开 0 并继续增大;其余零初始化张量到末步非零;L2 损失降到第 0 步的 0.9 倍以下 |
| leg A | 60 步(batch 2、warmup 10、第 50 步起退火,第 30、60 步存 ckpt,在 4 个物体上做有位姿 + 无位姿验证) | 跑完、无 NaN |
| leg B | 从第 60 步续训到 90 步 | 必须真的续训,不能重新开始 |
| leg C | 故意截断最新 ckpt | 必须以 `[resume]` 报错停下 |
| leg D | 检查 ckpt | 零初始化输出层已经动了;ckpt 里的键 = 可训练参数的键 |

日志里 `[s2-smoke]` 开头的门都应是 PASS。每个 ckpt 约 3 GB。

### 5.4 评测通路(G6)

```bash
# 关掉第二阶段,必须逐物体复现第一阶段的存档评测
sbatch scripts_s2/s2_fulleval.sbatch none S2off "posed unposed" configs/S2P8_uni3t70k_b32t6_lr35_a21k26k.yaml <输出根> 64
python tools_s2/s2_compare.py <输出根>/S2off_posed_novel <第一阶段存档评测目录> --gate
```

第一阶段的存档评测目录只在原机器上(第 7.2 节有数字)。新机器上可以先用原 `inference.py` 跑一份第一阶段评测(`scripts_s2/s2_s1ref_subset64.sbatch` 的写法),再拿来对照。

### 5.5 测速与守护

- 测速:`sbatch scripts_s2/s2_bench.sbatch <输出 json>`。块 8 / 块 4 × sdpa / flex 四腿在同一个作业、同一张卡上串行跑;每腿丢前 100 步、测 200 步。
- 规矩:同一组测速必须在同一作业、同一节点内完成;测试类作业要挂 5 分钟守护 `bash scripts_s2/s2_watch.sh <jobid> <out> <err> 300`。守护在出现失败特征、作业进入终态、或日志连续 4 次不增长时退出。

---

## 6. 怎么运行

### 6.1 训练(4 × H200)

```bash
sbatch scripts_s2/s2_train.sbatch configs/S2P8_uni3t70k_b32t6_lr35_a21k26k.yaml 26000
```

- **接力**:单个作业 2 天上限。脚本启动时会先排一个后继作业(`afterany`),从最新 ckpt 续训。续训出错直接停下,不会静默从第 0 步重来。要停下整条接力,执行 `touch <ckpt 目录>/RELAY_STOP`。
- **ckpt**:只存可训练参数,每个约 3 GB。只保留最新 2 个,加 21000 和 26000 两个点。
- **验证**:每 2000 步一次,包括第 0 步。有位姿和无位姿各跑一遍,数据是 subset64、旋转 0。
- **配方**:
  - 数据 all287k,每卡 batch 8 × 4 卡 = 32。每个样本 4 张输入 + 6 张目标。
  - AdamW β (0.9, 0.95);二维以上参数 wd 0.05;梯度裁剪 1.0;bf16 autocast + fp32 主权重。
  - 学习率一组 3.5e-5:warmup 1000 步,恒定到 21000 步,再按 1−√ 进度退火,26000 步到 0。
  - 增强:±10° 平面内旋转;10% 的样本目标图就是输入图;40% 的样本不给位姿。
  - 损失:L2 × 1.0 + 感知 × 0.5 + 点图 × 0.2(置信度项与 L1 各占 0.5);不用位姿损失。
- **预计时长**(G8 实测步时):块 8 为 3.885 秒/步,26k 步约 28 小时;块 4 为 6.290 秒/步,约 45 小时。
- ⚠ **结构定下来之前不要开训**(第 8.1 节)。

### 6.2 全量评测

```bash
sbatch scripts_s2/s2_fulleval.sbatch <第二阶段 ckpt 文件 | none> <标签> "posed unposed" <配置> <输出根> [前 N 个物体]
python tools_s2/s2_compare.py <输出根>/<标签>_posed_novel <输出根>/<标签>_posed_novel_s1 --out cmp_posed.json
```

- 一次前向同时写 `<标签>_<模式>_novel`(第二阶段)和 `..._novel_s1`(同一次前向里冻结的第一阶段),每个目录里都有 metrics / 深度 / 位姿三份汇总和 `regions.json`。
- `s2_compare.py` 按物体配对,报两者之差的均值,以及按物体有放回重抽 1000 次得到的 95% 区间。
- 判读(事先定的):主看 LPIPS,差值 ≤ −0.003 且 95% 区间不含 0 才算有效。PSNR 以 0.6 dB、前景 PSNR 以 0.5 dB 为辅助信号线,两者冲突时以 LPIPS 为准。0.003 / 0.6 / 0.5 这三条线是本项目沿用的噪声线。
- `regions.json` 按区域报误差:轮廓带(真值轮廓两侧 4 像素)、所有输入图都没拍到的区域、物体内部纹理区(图像梯度最大的 20% 前景像素)、深度跳变处。

### 6.3 训练关口检查(G9)

```bash
python tools_s2/s2_train_check.py step0 <ckpt 目录> <s1ref 输出根>          # 第 0 步验证必须等于第一阶段
python tools_s2/s2_train_check.py safety <ckpt 目录> 4000 <日志…> --world 4  # 4k 步安全线
```

- step0:逐物体对照第一阶段,|ΔPSNR| ≤ 0.02 dB、|ΔLPIPS| ≤ 2e-4、|Δabs_rel| ≤ 1e-4。
- safety:NaN 跳步 < 1%;render_std 没有低于自身中位数一半的点;两种设定的验证 LPIPS 比第 0 步差不超过 0.003。

### 6.4 工具一览(`tools_s2/`)

| 工具 | 用途 |
|---|---|
| `s2_candidate_probe.py` | 用冻结第一阶段的点做几何对应,误差有多大(投影误差、窗口命中率、可见性) |
| `s2_dynamics_probe.py` | 动力学探针:各参数组每步位移 / lr、梯度范数、零初始化层是否动 |
| `s2_bench.py` | 整步训练测速(第 5.5 节) |
| `s2_attn_profile.py` | 屏蔽表密度、单次注意力调用耗时 |
| `s2_infer_bench.py`、`s2_infer_profile.py` | 推理延迟:只跑第一阶段 vs 第一 + 第二阶段 |
| `s2_renderer_debug.py` | 渲染器按部件计时,以及几个结构假设的计时 |
| `s2_fgdense_whatif.py` | 只测速度的结构原型:`--variants fg_dense,fg_static,fg_dual`(第 7.4 节) |
| `s2_launch_profile.py` | 渲 1 张 / 10 张时的 kernel 数和 GPU 实际计算时间 |
| `s2_trackhead_cost.py` | VGGT TrackHead 在我们形状上的开销估算 |
| `s2_graph_bench.py`(+ `scripts_s2/s2_graph_bench.sbatch`) | 推理加速:bf16 权重 + CUDA Graph,1 张 / 10 张各一个进程 |
| `s2_graph_blockers.py` | 找出妨碍 CUDA Graph 录制的主机同步 |
| `s2_diag_seq.py`、`s2_diag_init.py` | 起点等价失败时的诊断 |
| `s2_regions.py`、`s2_compare.py`、`s2_train_check.py` | 区域指标、配对比较、训练关口 |
| `check_pointmap_on_ray.py` | 点图半像素 bug 的验收(第 10 节) |

---

## 7. 已完成的内容与结果

### 7.1 关口状态

| 关口 | 内容 | 状态 | 依据 |
|---|---|---|---|
| G0 | 开 `stage2` 分支、修 `.gitignore` | ✅ | 提交 62d9829 |
| G1 | 几何与屏蔽表 + CPU 测试 | ✅ | 与逐像素暴力实现 0 处不同(c5902b3) |
| G2 | 注意力参照 + 块 | ✅ | 与 fp64 逐行参照差在 1e-9 以内(5ecf061) |
| G3 | 包装、第一阶段调用、渲染器、头 | ✅ | CPU 模块测试(44369a5) |
| G4 | GPU 测试 | ✅ | 作业 137850–137854 全过;137836 在 gpu12 上失败过一次,原因不明(第 5.2 节) |
| G5 | 训练与续训改动 | ✅ | 全仓 83 项 CPU 测试(a4d57b1) |
| G6a | 关掉第二阶段,`gso.txt` 前 64 个物体复现第一阶段 | ✅ | 作业 137834:逐物体 PSNR / LPIPS / SSIM / 前景 PSNR / abs_rel 最大差 0 |
| G6b | 同上,全部 1030 个物体 | ⏳ 未跑 | |
| G7 | 块 8 / 块 4 冒烟 | ✅ | 137856、137857(SDPA);137872、137873、137876(flex)。各段全过 |
| G8 | 同节点测速,选后端 | ✅ | 137861:两组都用 flex(7.3 节) |
| G9 | 开训 + 第 0 步对照 + 4k 安全线 | ⏳ | 第一阶段参照已完成(137862,7.2 节);块 8 训练 137864 在 09-27 被取消,从未开跑,原因见 7.4 |
| G10 / G11 | 块 8 / 块 4 训满 26k,全量评测 + 配对重抽 | ⏳ | |

动力学探针(G7,固定 batch 4,40 步,恒定 lr):块 8 的 L2 降到第 0 步的 0.820 倍,块 4 降到 0.441 倍(flex 版)。第 1 步渲染器侧梯度恰为 0,这是 Lin_m 零初始化的预期结果;从第 2 步起,各组每步位移为 0.25–0.65 倍 lr。

### 7.2 第一阶段参照(ckpt_70000)

| 评测 | 设定 | PSNR | LPIPS | SSIM | 前景 PSNR | abs_rel | 位姿 AUC30 |
|---|---|---|---|---|---|---|---|
| 全量 v2,1030 个物体(G6b 参照) | 有位姿 | 26.2196 | 0.09774 | 0.8969 | 21.99 | 0.01207 | 99.04% |
| 同上 | 无位姿 | 25.0544 | 0.10445 | 0.8845 | 21.07 | 0.01515 | 92.38% |
| v1 subset64,原 `inference.py`(G9 第 0 步参照,137862) | 有位姿 | 25.2119 | 0.09247 | 0.8946 | 21.29 | 0.01082 | 97.86% |
| 同上 | 无位姿 | 24.0673 | 0.10201 | 0.8816 | 20.16 | 0.01412 | 87.14% |
| v2,`gso.txt` 前 64 个物体,第二阶段关闭(G6a,137834) | 有位姿 | 25.9001 | 0.09289 | 0.8934 | 21.19 | 0.01070 | 99.43% |
| 同上 | 无位姿 | 24.9508 | 0.09805 | 0.8832 | 20.52 | 0.01322 | 94.84% |

表注:所有评测都是每个物体 4 张输入 → 10 张目标,旋转 0。abs_rel 是深度相对误差,越低越好。位姿 AUC30 是输入图相对位姿误差在 30° 内的曲线下面积,越高越好。原机器上的存档目录是 `…/moe_experiments/evaluation/uni3t_a10k_step70000_{posed,unposed}_novel/` 和 `…/moe_experiments/evaluation/s2/`。

### 7.3 训练速度(G8,作业 137861,同节点 1 卡,batch 8,12 个真实 batch,丢前 100 步测 200 步)

| 组 | SDPA 步时(中位 / p90) | flex 步时(中位 / p90) | flex 快多少 | 峰值显存(SDPA / flex) |
|---|---|---|---|---|
| 块 8 | 4.606 / 4.881 秒 | 3.885 / 4.068 秒 | 15.7% | 41.7 / 41.2 GB |
| 块 4 | 8.335 / 9.724 秒 | 6.290 / 7.122 秒 | 24.5% | 52.5 / 51.0 GB |

- 省下的时间都在第二阶段渲染器。块 8 前向 671→542 ms、反向 2750→2128 ms;块 4 前向 1222→708 ms、反向 6260→4429 ms。
- 冻结第一阶段的两遍前向(不求梯度)每步 1.12 秒,占块 8 SDPA 整步的 24%(换成 flex 后约 29%)。
- 事先定的规则是:flex 整步快 10% 以上、且通过等价测试才用。两组都满足,所以都用 flex。

### 7.4 速度诊断:为什么现在的结构不行(训练因此没开)

用户 09-27 定的判据:**第二阶段渲染器必须比第一阶段渲染器快很多,否则这个设计无效,要重想结构。**

**端到端推理**(batch 1,4 张输入 → 10 张目标,有位姿;作业 137892):

- 只跑第一阶段:176 ms/场景。
- 加上第二阶段:块 8 为 348 ms(flex;SDPA 362),块 4 为 396 ms(flex;SDPA 444)。
- 块 8 多出的 172 ms = 第二遍 35 + 建表 9 + 第二阶段渲染器 119 + 头 10。

**渲染器对比**(作业 137897;第一阶段渲染器是 uni3t 原有的稠密渲染器):

| 渲染器 | 训练形状(48 张目标图) | 推理 10 张 | 推理 1 张 |
|---|---|---|---|
| 第一阶段渲染器 | 300 ms | 66.6 ms | 33.1 ms |
| 第二阶段,现结构(压缩 + 几何窗口) | 541 ms(1.8 倍) | 120.5 ms | 76.9 ms |
| 原型 fg_dense:两边都只对前景做全量注意力 | 125.5 ms(0.42 倍) | 42.2 ms | 40.9 ms |
| 原型 fg_static:fg_dense + 场景侧不更新、各目标图共用 | 34.6 ms(0.12 倍) | 19.2 ms | 28.3 ms |

表注:原型只测速度,不训练。输入图里的前景场景 token 占 35%(评测)到 41%(训练,向外扩一圈 token)。推理 1 张时的计时受 CPU 状态影响大,只比同一状态下的读数。

**第二个原型 fg_dual**(用户 09-27 提:保留几何窗口,两路 = 前景全量 + 几何窗口稀疏,门控相加,两个方向都这样;作业 137899):

| 渲染器 | 训练形状 | 推理 10 张 | 推理 1 张 |
|---|---|---|---|
| 第一阶段渲染器 | 302 ms | 66.8 ms | 约 34 ms |
| fg_dual | 180.7 ms(0.60 倍) | 63.1 ms | 61.2 ms |
| fg_dense(同一作业复测) | 107.9 ms | 43.2 ms | 42.4 ms |

保留窗口还有额外代价:几何窗口和可见性都需要"把输入相机再渲一遍"并建屏蔽表。这两步在训练前向中约多 250 ms/步,推理约多 50 ms。fg_dense 不需要它们。

**现结构慢在哪**(作业 137896,推理,块 8):

- 渲 10 张时,第二阶段渲染器 71% 的时间花在场景侧:继承来的部分 37.5 ms,新加的压缩、块平均、拷贝、注入 50.8 ms。目标侧只有 11 ms,其中 flex 注意力 10.6 ms,和第一阶段稠密 FA3 一样快。
- 根因是结构:每张目标图都保留一份场景 token,并在每一份上逐 token 做压缩。
- 把压缩融合成一个等价算子(torch.compile,差 1 个 bf16 舍入单位)后,10 张从 120 降到 86.5 ms,仍是第一阶段的 1.3 倍。

**渲 1 张为什么也慢**(作业 137900,profiler):

- 这时耗时由 Python 发射 kernel 决定,GPU 只有 40% 左右的时间在算。
- GPU 实际计算:第一阶段渲染器 4.8 ms / fg_dense 3.1 / fg_dual 4.5 / 现结构 9.6 ms。kernel 数:667 / 787 / 1054 / 1541 个,每个约 45–90 µs。
- 80% 的 kernel 是 `aten::copy_`:推理不求梯度时,autocast 不缓存权重转换,每次 Linear 都要拷一次 bf16;RMSNorm 还有 fp32 往返。

**其他测过的**:

- 输入相机再渲一遍:训练形状是 32 张图,229 ms(每张 6.4 ms,与目标图相同);推理 4 张 34 ms,其中 GPU 实算 14.3 ms。
- 用 VGGT TrackHead 代替"再渲一遍 + 建表"(作业 137901,位置编码缓存后):推理 1 张 72 ms、10 张 116 ms;训练 48 张冻结 0.57 秒,联合训练约 2.2 秒/步。都比它要替换的部分(推理约 51 ms、训练约 0.26 秒)更贵,所以不划算。
- bf16 权重(作业 137905,渲 1 张):输出与 fp32 权重逐位相同。只第一阶段 136.9→132.0 ms,加第二阶段 250.0→240.3 ms,几乎不提速。
- CUDA Graph:VGGT 里有两处主机同步挡住录制——`vggt/models/aggregator.py` 在 CPU 上建全零张量再拷到 GPU,以及 `vggt/layers/rope.py` 读 `int(positions.max())`。`s2_graph_bench.py` 在进程内打了补丁,仓库文件没改;RoPE 补丁在 CPU 上逐位等价。1 张和 10 张的测量作业 137951 提交时还在原机器排队,结果见第 8.2 节。

### 7.5 第一阶段的点能不能当几何对应(候选预检,作业 137766)

- 数据 v1 subset64,模型 ckpt_70000。
- 第一阶段预测的点投到输入图后,与真对应点的距离:有位姿时均值 2.81 像素、中位约 1.5 像素;无位姿时均值 7.39 像素、中位约 2.5 像素。
- 加载器点图真值自带的半像素偏差,投到输入图后均值 0.53 像素(第 10 节)。
- 结果文件只在原机器 `~/tmp/s2_probe/cand_70k.json`。

---

## 8. 还需要继续测试的内容

### 8.1 先定结构(挡着训练)

| 候选 | 渲染器耗时(训练形状 / 10 张 / 1 张) | 要不要"再渲一遍 + 建表" | 说明 |
|---|---|---|---|
| 现结构(压缩 + 几何窗口) | 541 / 120.5 / 76.9 ms | 要 | 不满足速度判据 |
| fg_dense | 125.5 / 42.2 / 40.9 ms | 不要 | 最简单 |
| fg_static | 34.6 / 19.2 / 28.3 ms | 不要 | 场景侧不更新 |
| fg_dual(用户 09-27 提) | 180.7 / 63.1 / 61.2 ms | 要 | 10 张时与第一阶段(66.8)持平 |

表注:第一阶段渲染器同口径为 300–302 / 66.6–66.8 / 约 33–34 ms。fg_dual 那一行出自另一个作业(137899),同作业里 fg_dense 是 107.9 / 43.2 / 42.4 ms。

定下来之后要做:

1. 实现新结构。现在的原型只替换了注意力的计时路径,不能训练。
2. 重跑 GPU 测试(至少起点等价),以及两组冒烟。
3. 同节点重测 G8。
4. 如果还想提速单张推理:再做 bf16 权重 + CUDA Graph(8.2)。

### 8.2 bf16 + CUDA Graph(渲 1 张、10 张)

- 作业 137951 在原机器上。如果没跑完,在新机器上执行:

```bash
sbatch scripts_s2/s2_graph_bench.sbatch <任一第二阶段 ckpt> <输出前缀>
```

- 这个作业需要一个第二阶段 ckpt 让残差不为 0,冒烟产出的 ckpt 就可以。
- 它会先逐段检查主机同步:有段没过就只跑 A(fp32 原路径)和 B(bf16 权重)两段,并在 JSON 的 `sync_check` 里写出位置。
- 要报的是:与原路径的一致性(最大绝对差、PSNR / LPIPS),以及第一、第二阶段各自的提速。

### 8.3 其余没做完的

1. **G6b**:关掉第二阶段,在全部 1030 个物体上复现第一阶段。门槛是 |ΔPSNR| ≤ 0.005 dB、|ΔLPIPS| ≤ 5e-5;参照值见 7.2。
2. **G9**:开训后第 0 步对照 + 4k 步安全线(第 6.3 节)。
3. **G10 / G11**:块 8、块 4 训满 26k 步;全量评测、配对重抽、四类区域。
4. **GPU 测试在新机器上连跑几次**(第 5.2 节提到的一次不明失败)。
5. **半像素**:第一阶段的底座修好并重训之后,第二阶段仓也要改加载器,并换成新的第一阶段 ckpt。在那之前,第二阶段按 09-26 的决定,把第一阶段的点当作像素中心的点来用(`model_s2/geometry.py` 顶部注释)。
6. **真实数据**:第二阶段用了真值目标掩码(既用来挑 token,也作为输入通道),第一阶段没有用。所以结论只适用于"有目标掩码"的设定。真实数据要换成 SAM2 掩码另外测。

---

## 9. 之前定下、后续要做的消融 / 备选 / 暂缓

**原则(用户 09-26)**:主方案选最可能提升效果的做法,不为"消融干净"去选方案。新部件按正常训练的标准设置初始化;"开局输出等于第一阶段"的做法放在消融里。

**消融**

1. **开局等于第一阶段**:4×4 切块层用 8×8 切块层换算来初始化;DPT 每一路走"2×2 平均后接原路 + 零初始化的新路";注入层零初始化。
2. **块 4 场景侧也切细**:插值 VGGT 特征,再加零初始化的像素项和射线项。
3. **加场景侧稀疏自注意力**,或者照 LSRM 在末尾加 3 个单向附加块。

**备选**

- 逐像素残差头 + StereoNet 式的彩色图引导小网络。
- 第二阶段的场景 token 加上 DINOv2 / v3 特征。
- 更省的 DPT 改法:D(32 GMAC),以及 F、G(通道减半,25.5 / 10.3 GMAC)。块 4 现在的 DPT 约 93 GMAC/张,块 8 约 78。
- DPT 多输出 3 个颜色通道。

**暂缓**

- 匹配头版本(A 组):用 VGGT TrackHead 求对应。7.4 节的开销估算显示它更贵。
- 块 2。
- 提高分辨率。
- 按层注入的 ② 方案。
- "第一阶段多训"对照:从 `PLN2uni3t_all287k_b32t6_fp32lr35_const` ckpt_76000 再退火 1 万步。这个 ckpt 只在原机器上(第 11 节)。
- FlexAttention 的 FA4 后端,以及 cudnn.flex_attention。

**结构候选**:fg_dense、fg_static、fg_dual 属于 8.1 节的结构选择,不算消融。

---

## 10. 已知问题与注意事项

- **第一阶段点图偏半个像素**。加载器 `data/dataset_objaverse.py:137-138` 用 `x - W // 2`(像素左上角),模型的射线用 `x + 0.5 - cx`(像素中心)。所以点图真值在 x、y 方向各偏 0.5 像素,ckpt_70000 就是按这个偏的真值训出来的。
  - 新视角合成、深度 abs_rel、位姿都不受影响。
  - 修法与验收见 `docs/half_pixel_pointmap_bug.md`、`tools_s2/check_pointmap_on_ray.py`:两行各加 0.5,**不要把 cx 改成 127.5**。
  - 本仓的加载器按 09-26 的决定还没改。
- **深度与前景掩码的最近邻缩放比 RGB 多偏 0.25 像素**(512→256)。这是另一个独立问题,建议暂不改,见同一文档第 10 节。
- **冻结的第一阶段不在模块树里**(用 `object.__setattr__` 挂上),所以不进 state_dict、优化器和 DDP。第二阶段的 ckpt 只有可训练参数,载入时仍然需要第一阶段 ckpt。
- **第一阶段每步用 `torch.no_grad` 现算,不能用 `inference_mode`**:它的输出 token 要进可训练层。
- **FlexAttention**:重编译上限设成 64。编译缓存放节点本地盘。开头几步因为编译会慢。
- 配置里的防呆设置不要改:`unfreeze_rae_decoder_at: 1.05`(默认 0.2 会崩)、`exclude_bg_frac: 0.0`、`resume_ckpt: ''`。若误用第一阶段配置里的续训键,会把第一阶段整包当成续训点,从 70000 步起跑。
- 两个仓外的只读引用:`training.api_key_path`(第 4 节);`tools_s2/s2_trackhead_cost.py` 会导入原版 VGGT 仓。

---

## 11. 用到的 checkpoints 一览

| 用途 | 实验名 / 文件 | 位置 | 新机器上需要吗 |
|---|---|---|---|
| 冻结的第一阶段 | `PLN2uni3t_all287k_b32t6_fp32lr35_wsd60k_a10k` ckpt_70000 | HF `szlgallen/RAE_backup_checkpoints`(第 3.1 节);原机器旧盘 | **需要**,从 HF 下载 |
| VGGT-1B 预训练 | `facebook/VGGT-1B` 的 `model.pt` | `$TORCH_HOME/hub/checkpoints/` | 需要(新机器已有) |
| 感知损失、LPIPS | `imagenet-vgg-verydeep-19.mat`、`vgg16-397923af.pth`、`vgg19-dcbb9e9d.pth` | `metric_checkpoint/`、`$TORCH_HOME/hub/checkpoints/` | 需要(新机器已有) |
| 第一阶段的退火起点 | `PLN2uni3t_all287k_b32t6_fp32lr35_const` ckpt_60000 | 原机器 | 不需要 |
| "第一阶段多训"对照的起点(暂缓) | `PLN2uni3t_all287k_b32t6_fp32lr35_const` ckpt_76000 | 只在原机器 | 做这个对照时才需要 |
| 第一阶段恒定学习率主干的中间点 | 同上 ckpt 16k / 26k / 36k / 46k / 56k | HF 同仓 `moe_experiments/checkpoints/PLN2uni3t_all287k_b32t6_fp32lr35_const/` | 不需要 |
| 第二阶段冒烟 ckpt(只存可训练参数,各约 3 GB) | `S2P8_…_SMOKE_1378xx`、`S2P4_…_SMOKE_1378xx` 的 ckpt_60 / ckpt_90 | 原机器旧盘 `…/moe_experiments/SMOKE/` | 不需要。新机器冒烟会自己生成;8.2 节的测速可以直接用新冒烟的 ckpt |
| 第二阶段正式训练 | 没有 | 块 8 训练 137864 从未开跑 | — |
