# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import builtins
import importlib
import os
import time
import wandb
import torch
from rich import print
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, ConcatDataset
import torch.distributed as dist
from setup import init_config, init_distributed, init_wandb_and_backup
from utils.metric_utils import visualize_intermediate_results
from utils.training_utils import create_optimizer, create_lr_scheduler, auto_resume_job, print_rank0, build_clip_groups, clip_grad_norm_grouped, best_effort_write
from utils.metric_utils import (
    export_results,
    summarize_evaluation,
    summarize_evaluation_depth,
    summarize_evaluation_pose,
)
from tqdm import tqdm 


amp_dtype_mapping = {
    "fp16": torch.float16, 
    "bf16": torch.bfloat16, 
    "fp32": torch.float32, 
    'tf32': torch.float32
}

class Trainer:
    def __init__(self, config):
        self.config = config
        self._init_data()
        self._init_model()

    def _init_data(self):
        config = self.config

        dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
        module, class_name = dataset_name.rsplit(".", 1)
        Dataset = importlib.import_module(module).__dict__[class_name]
        dataset = Dataset(config)

        if hasattr(config.training, 'dataset_name2'):
            dataset_name2 = config.training.get("dataset_name2")
            module, class_name = dataset_name2.rsplit(".", 1)
            Dataset2 = importlib.import_module(module).__dict__[class_name]
            dataset2 = Dataset2(config, is_second=True)

            dataset = ConcatDataset([dataset, dataset2])

        batch_size_per_gpu = config.training.batch_size_per_gpu

        datasampler = DistributedSampler(dataset)
        # E-arm curriculum needs workers re-forked each epoch so they pick up
        # the refreshed dataset.curriculum_progress (see fetch_data1).
        persistent = str(config.training.get("curriculum_view_sampling", "none")) == "none"
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size_per_gpu,
            shuffle=False,
            num_workers=config.training.num_workers,
            persistent_workers=persistent,
            pin_memory=False,
            drop_last=True,
            prefetch_factor=config.training.prefetch_factor,
            sampler=datasampler,
        )
        self.datasampler = datasampler
        self.dataloader_iter = iter(dataloader)

        # Validation dataset
        val_dataset_name = config.training.get("val_dataset_name")
        module, class_name = val_dataset_name.rsplit(".", 1)
        ValDataset = importlib.import_module(module).__dict__[class_name]
        val_dataset = ValDataset(config)

        val_datasampler = DistributedSampler(val_dataset)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.training.val_dataset_cfgs.training.batch_size_per_gpu,
            shuffle=False,
            num_workers=config.training.num_workers,
            persistent_workers=True,
            pin_memory=False,
            drop_last=True,
            prefetch_factor=config.training.prefetch_factor,
            sampler=val_datasampler,
        )
        self.val_dataloader = val_dataloader

        total_train_steps = config.training.train_steps
        grad_accum_steps = config.training.grad_accum_steps
        total_param_update_steps = total_train_steps
        total_train_steps = total_train_steps * grad_accum_steps # real train steps when using gradient accumulation
        total_batch_size = batch_size_per_gpu * config.ddp_info.world_size * grad_accum_steps
        total_num_epochs = int(total_param_update_steps * total_batch_size / len(dataset))

        self.dataset = dataset
        self.dataloader = dataloader
        self.total_train_steps = total_train_steps
        self.total_param_update_steps = total_param_update_steps
        self.total_batch_size = total_batch_size

        # fg-masked L2 schedule: exclude_bg=True for the first exclude_bg_frac of
        # training (0.25 = legacy `step < total//4` behavior), full-image L2 after.
        exclude_bg_frac = config.training.get("exclude_bg_frac", 0.25)
        self.exclude_bg_until = int(total_train_steps * exclude_bg_frac)
        print_rank0(f"exclude_bg (fg-masked L2) active for steps < {self.exclude_bg_until} "
                    f"(frac={exclude_bg_frac}, total={total_train_steps})")
        self.total_num_epochs = total_num_epochs
        self.grad_accum_steps = grad_accum_steps

    def _init_model(self):
        config = self.config
        ddp_info = config.ddp_info

        module, class_name = config.model.class_name.rsplit(".", 1)
        LVSM = importlib.import_module(module).__dict__[class_name]
        model = LVSM(config).to(ddp_info.device)

        if config.training.get('use_bf16', False):
            model = model.to(amp_dtype_mapping['bf16'])
            # all DPT heads use fp32 instead, won't converge with bf16
            model.camera_head.to(torch.float32)
            model.point_head.to(torch.float32)
            model.rgb_head.to(torch.float32)
            model.loss_computer.to(torch.float32)
            # Feature alignment (optional): frozen DINOv3 teacher runs under
            # @torch.no_grad and benefits from bf16 (smaller memory). The
            # projector lives inside an `autocast(enabled=False)` block in
            # forward(), so its computation stays fp32 — keep params fp32 too.
            if hasattr(model, 'align_target_encoder'):
                model.align_target_encoder.to(torch.bfloat16)
                model.align_projector.to(torch.float32)
                assert all(
                    not p.requires_grad
                    for p in model.align_target_encoder.parameters()
                ), 'DINOv3 grads leaked after bf16 cast'
            print_rank0("Using bf16 training!")

        model = DDP(model, device_ids=[ddp_info.local_rank])

        decoder_lr = config.training.get("decoder_lr", None)
        decoder_weight_decay = config.training.get("decoder_weight_decay", None)
        optimizer, optimized_param_dict, all_param_dict = create_optimizer(
            model,
            config.training.weight_decay,
            config.training.lr,
            (config.training.beta1, config.training.beta2),
            decoder_lr=decoder_lr,
            decoder_weight_decay=decoder_weight_decay,
            fused=config.training.get("fused_adamw", False),
        )
        optim_param_list = list(optimized_param_dict.values())

        scheduler_type = config.training.get("scheduler_type", "cosine")
        lr_scheduler = create_lr_scheduler(
            optimizer,
            self.total_param_update_steps,
            config.training.warmup,
            scheduler_type=scheduler_type,
            anneal_from_step=config.training.get("anneal_from_step", None),
        )

        if config.training.get("resume_ckpt", "") != "":
            ckpt_load_path = config.training.resume_ckpt
        else:
            ckpt_load_path = config.training.checkpoint_dir
        reset_training_state = config.training.get("reset_training_state", False)
        _ovr_lr = config.training.lr if config.training.get("apply_config_lr_on_resume", False) else None
        optimizer, lr_scheduler, cur_train_step, cur_param_update_step = auto_resume_job(
            ckpt_load_path,
            model,
            optimizer,
            lr_scheduler,
            reset_training_state,
            override_lr=_ovr_lr
        )
        # auto_resume_job returns step 0 (fresh optimizer) when no ckpt loads or the
        # optimizer state fails to load; a branch that must continue a trunk opts in
        # here so that case exits instead of silently retraining from step 0.
        _req = config.training.get("require_resume_step", None)
        if _req is not None and cur_train_step < int(_req):
            raise RuntimeError(f"[resume] require_resume_step={_req} but resumed step={cur_train_step} "
                               f"from {ckpt_load_path}: optimizer/lr_scheduler not resumed, refusing to start")

        enable_grad_scaler = config.training.use_amp and config.training.amp_dtype == "fp16"
        self.scaler = torch.amp.GradScaler('cuda', enabled=enable_grad_scaler)
        print_rank0(f"Grad scaler enabled: {enable_grad_scaler}")

        # 检查并打印参数冻结状态
        # self._print_param_freeze_status(model)

        dist.barrier()

        self.model = model
        self.optimized_param_dict = optimized_param_dict
        self.optim_param_list = optim_param_list
        # Grouped gradient clipping (config `training.clip_group_prefixes`, default unset =
        # historical single global clip). Each listed prefix is clipped to grad_clip_norm
        # on its own norm; everything else forms the 'rest' group with the same max_norm.
        # Ported from RnG_fa3_repro_bench where it shipped as the M1 arm.
        # NOTE: groups are built once here. The only path that adds params later is the
        # RAE-decoder unfreeze, which is unreachable in this repo (unfreeze_rae_decoder_at
        # 1.05 -> step 94500 > train_steps 90000), so the groups cannot go stale.
        self.clip_groups = None
        clip_prefixes = config.training.get("clip_group_prefixes", None)
        if clip_prefixes:
            self.clip_groups = build_clip_groups(optimized_param_dict, list(clip_prefixes))
            print_rank0("grouped grad clipping: " + ", ".join(
                f"{g}={sum(q.numel() for q in ps)/1e6:.1f}M params" for g, ps in self.clip_groups))
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.cur_train_step = cur_train_step
        self.cur_param_update_step = cur_param_update_step

        # RAE decoder 解冻配置
        unfreeze_threshold = config.training.get("unfreeze_rae_decoder_at", 0.2)
        self.unfreeze_rae_decoder_at = unfreeze_threshold

        # 如果 unfreeze_rae_decoder_at = 0，表示已在模型初始化时解冻
        # 否则设置解冻步数
        if unfreeze_threshold == 0:
            self.rae_decoder_unfrozen = True
            self.unfreeze_step = float('inf')  # 训练循环中不会再解冻
        else:
            self.rae_decoder_unfrozen = False
            self.unfreeze_step = int(self.total_train_steps * unfreeze_threshold)

    def _print_param_freeze_status(self, model):
        """检查并打印模型参数的冻结状态"""
        ddp_info = self.config.ddp_info
        if not ddp_info.is_main_process:
            return

        print_rank0("=" * 80)
        print_rank0("模型参数冻结状态检查")
        print_rank0("=" * 80)

        frozen_params = []
        trainable_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                frozen_params.append((name, param.shape))
            else:
                trainable_params.append((name, param.shape))

        # 按模块分组统计
        from collections import defaultdict
        frozen_by_module = defaultdict(list)
        trainable_by_module = defaultdict(list)

        for name, shape in frozen_params:
            module = name.split('.')[0]
            frozen_by_module[module].append((name, shape))

        for name, shape in trainable_params:
            module = name.split('.')[0]
            trainable_by_module[module].append((name, shape))

        # 打印按模块统计
        print_rank0(f"\n📊 按模块统计")
        print_rank0("-" * 80)

        all_modules = sorted(set(frozen_by_module.keys()) | set(trainable_by_module.keys()))
        for module in all_modules:
            frozen = len(frozen_by_module[module])
            trainable = len(trainable_by_module[module])
            total = frozen + trainable
            if trainable == 0:
                status = "❄️  全部冻结"
            elif frozen == 0:
                status = "🔥  全部可训练"
            else:
                status = "🔶  部分冻结"
            print_rank0(f"  {module:30s} {status:20s} 冻结: {frozen:3d} / 可训练: {trainable:3d} / 总计: {total:3d}")

        # 统计参数数量
        frozen_count = sum(p.numel() for name, p in model.named_parameters() if not p.requires_grad)
        trainable_count = sum(p.numel() for name, p in model.named_parameters() if p.requires_grad)
        total_count = frozen_count + trainable_count

        print_rank0(f"\n📈 参数数量统计")
        print_rank0("-" * 80)
        print_rank0(f"  被冻结参数数量: {frozen_count:>15,} ({frozen_count / total_count * 100:.2f}%)")
        print_rank0(f"  可训练参数数量: {trainable_count:>15,} ({trainable_count / total_count * 100:.2f}%)")
        print_rank0(f"  总参数数量:      {total_count:>15,}")
        print_rank0("=" * 80)
    
    def fetch_data1(self, cur_epoch):
        config = self.config
        ddp_info = config.ddp_info

        try:
            data = next(self.dataloader_iter)
        except StopIteration:
            print(f"Current Rank {ddp_info.local_rank} Ran out of data. Resetting dataloader epoch to {cur_epoch}; might take a while...")
            self.datasampler.set_epoch(cur_epoch)
            # E-arm curriculum clock: refresh progress before workers re-fork
            # (persistent_workers=False on that arm). Resume-safe: cur_epoch is
            # derived from the resumed step. No-op for ConcatDataset / other arms.
            if hasattr(self.dataset, "curriculum_progress"):
                self.dataset.curriculum_progress = cur_epoch / max(1, self.total_num_epochs)
            self.dataloader_iter = iter(self.dataloader)
            data = next(self.dataloader_iter)
        return data

    def run(self):
        config = self.config
        ddp_info = config.ddp_info

        self.start_train_step = self.cur_train_step
        self.model.train()

        while self.cur_train_step <= self.total_train_steps:
            # 检查是否需要解冻 RAE decoder
            if not self.rae_decoder_unfrozen and self.cur_train_step >= self.unfreeze_step:
                print_rank0(f"=" * 80)
                print_rank0(f"Unfreezing RAE decoder at step {self.cur_train_step} ({self.unfreeze_rae_decoder_at * 100:.0f}% of training)")
                print_rank0(f"=" * 80)

                # 解冻参数
                if isinstance(self.model, DDP):
                    self.model.module.rgb_head.unfreeze_decoder()
                    new_params = [p for n, p in self.model.module.named_parameters()
                                 if n.startswith('rgb_head.rae_decoder') and p.requires_grad]
                else:
                    self.model.rgb_head.unfreeze_decoder()
                    new_params = [p for n, p in self.model.named_parameters()
                                 if n.startswith('rgb_head.rae_decoder') and p.requires_grad]

                # 使用 add_param_group 添加新参数，保留原有优化状态
                rae_lr = config.training.get("decoder_lr", config.training.lr)
                rae_wd = config.training.get("decoder_weight_decay", config.training.weight_decay)
                decay_params, nodecay_params = [], []
                for param in new_params:
                    if param.dim() == 1:
                        nodecay_params.append(param)
                    else:
                        decay_params.append(param)

                if decay_params:
                    self.optimizer.add_param_group({'params': decay_params, 'weight_decay': rae_wd, 'lr': rae_lr, 'is_decoder': True})
                if nodecay_params:
                    self.optimizer.add_param_group({'params': nodecay_params, 'weight_decay': 0.0, 'lr': rae_lr, 'is_decoder': True})

                # Update optim_param_list and optimized_param_dict for gradient clipping and NaN sanitization
                self.optim_param_list.extend(new_params)
                model_ref = self.model.module if isinstance(self.model, DDP) else self.model
                for n, p in model_ref.named_parameters():
                    if 'rgb_head.rae_decoder' in n and p.requires_grad:
                        self.optimized_param_dict['module.' + n] = p

                self.rae_decoder_unfrozen = True
                print_rank0(f"  Added {len(new_params)} new parameters to optimizer")
                print_rank0(f"  Decay params: {len(decay_params)}, No-decay params: {len(nodecay_params)}")
                print_rank0(f"=" * 80)

            tic = time.time()
            cur_epoch = int(self.cur_train_step * (self.total_batch_size / self.grad_accum_steps) // len(self.dataset) )
            
            ### get data
            data = self.fetch_data1(cur_epoch)

            batch = {k: v.to(ddp_info.device) if type(v) == torch.Tensor else v for k, v in data.items()}

            if config.training.get('use_bf16', False):
                batch = {k: v.to(amp_dtype_mapping['bf16']) if type(v) == torch.Tensor else v for k, v in batch.items()}
                # ret_dict = self.model(batch, exclude_bg=self.cur_train_step<self.total_train_steps//4)

            # else:
            with torch.autocast(
                enabled=config.training.use_amp,
                device_type="cuda",
                dtype=amp_dtype_mapping[config.training.amp_dtype],
            ):
                ret_dict = self.model(batch, exclude_bg=self.cur_train_step<self.exclude_bg_until)
            
            update_grads = (self.cur_train_step + 1) % self.grad_accum_steps == 0 or self.cur_train_step == self.total_train_steps
            if not update_grads:
                with self.model.no_sync(): # no sync grads for efficiency
                    self.scaler.scale(ret_dict.loss_metrics.loss / self.grad_accum_steps).backward()
            else:
                self.scaler.scale(ret_dict.loss_metrics.loss / self.grad_accum_steps).backward()
            self.cur_train_step += 1

            export_inter_results = ((self.cur_train_step-1) == self.start_train_step) or (self.cur_train_step % config.training.vis_every == 0)

            # UNI3T: this decision must be collective. Deciding per rank lets one rank
            # skip its update while the other three apply theirs, and the four replicas
            # are then permanently out of sync (bf16 => GradScaler is disabled, so
            # nothing re-syncs them). Newly reachable on this branch: model/loss.py's
            # F.l1_loss(pts_est, pts_gt) was the one point term with no inf/NaN guard
            # and inverse_log_transform overflows fp32 at |y| >= 88.722836.
            _bad = (torch.isnan(ret_dict.loss_metrics.loss)
                    | torch.isinf(ret_dict.loss_metrics.loss)).to(torch.float32)
            if ddp_info.world_size > 1:
                dist.all_reduce(_bad, op=dist.ReduceOp.SUM)
            skip_optimizer_step = bool(_bad.item() > 0)
            if skip_optimizer_step:
                print(f"NaN or Inf loss detected on at least one rank, skip this iteration")
                ret_dict.loss_metrics.loss.data = torch.zeros_like(ret_dict.loss_metrics.loss)

            total_grad_norm = None
            # Check gradient norm and update optimizer if everything is fine
            if update_grads and (not skip_optimizer_step):
                # Unscales the gradients
                self.scaler.unscale_(self.optimizer) 
                # For all gradients, we safely change the NaN -> 0., inf -> 1e-6, -inf -> 1e-6.
                with torch.no_grad():
                    for n, p in self.optimized_param_dict.items():
                        if p.requires_grad and (p.grad is not None):
                            p.grad.nan_to_num_(nan=0.0, posinf=1e-6, neginf=-1e-6)
            
                # visualize the grad norm of each layer of our transformer (FOR DEBUG)
                if ddp_info.is_main_process and config.training.get("log_grad_norm_details", False):
                    grad_norms = {}  # Dictionary to store norms per layer
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:  # Some parameters might not have gradients
                            grad_norms[name] = param.grad.detach().norm().item()  # Detach for safety
                    grad_norm_log = {
                        "grad_norm_details/" + layer_name: grad_norm
                        for layer_name, grad_norm in grad_norms.items()
                    }
                    grad_norm_log["forward_pass_step"] = self.cur_train_step
                    wandb.log(grad_norm_log)

                total_grad_norm = 0.0
                group_grad_norms = None
                if config.training.grad_clip_norm > 0:
                    if self.clip_groups is not None:
                        total_grad_norm, group_grad_norms = clip_grad_norm_grouped(self.clip_groups, config.training.grad_clip_norm)
                    else:
                        total_grad_norm = torch.nn.utils.clip_grad_norm_(self.optim_param_list, max_norm=config.training.grad_clip_norm).item()
                    if group_grad_norms is not None and ddp_info.is_main_process and (self.cur_train_step % config.training.wandb_log_every == 0):
                        wandb.log({**{f"grad_norm_group/{g}": v for g, v in group_grad_norms.items()},
                                   "grad_norm_group/global": total_grad_norm,
                                   "forward_pass_step": self.cur_train_step})

                    if total_grad_norm > config.training.grad_clip_norm * 2.0:
                        print(f"WARNING: step {self.cur_train_step} grad norm too large {total_grad_norm} > {config.training.grad_clip_norm * 2.0}")

                    allowed_gradnorm = config.training.grad_clip_norm * config.training.get("allowed_gradnorm_factor", 5)
                    if total_grad_norm > allowed_gradnorm:
                        skip_optimizer_step = True
                        print(f"WARNING: step {self.cur_train_step} grad norm too large {total_grad_norm} > {allowed_gradnorm}, skipping optimizer step")

                    # show grad norm in wandb if it's too large
                    display_grad_norm = total_grad_norm > config.training.grad_clip_norm * 2.0 or total_grad_norm > allowed_gradnorm
                    if display_grad_norm and ddp_info.is_main_process:
                        wandb.log({
                            "grad_norm": total_grad_norm,
                            "forward_pass_step": self.cur_train_step,
                        })

                if not skip_optimizer_step:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.cur_param_update_step += 1

                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
            elif update_grads:
                # NaN/Inf-loss skip: grads must still be cleared (they accumulate across
                # backward calls and would poison the next step) and the lr schedule advanced
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            # log and save checkpoint
            if ddp_info.is_main_process:
                self.log_and_save_ckpt(ret_dict, cur_epoch, self.cur_param_update_step, tic, export_inter_results, total_grad_norm)

            # val cadence decoupled from checkpoint cadence (100k arm: val 2k / ckpt 4k);
            # default = checkpoint_every keeps the historical coupled behavior
            val_every = config.training.get("val_every", config.training.checkpoint_every)
            if (self.cur_train_step % val_every == 0) or (self.cur_train_step == self.total_train_steps):
                self.validate()

            if export_inter_results:
                torch.cuda.empty_cache()
                dist.barrier()

    @torch.inference_mode()
    def validate(self):
        config = self.config
        ddp_info = config.ddp_info
        self.model.eval()
        out_dir = os.path.join(config.training.validation_out_dir, f"eval_iter_{self.cur_train_step:08d}")

        n_export_failed = 0
        for batch in tqdm(self.val_dataloader, disable = not ddp_info.is_main_process):
            batch = {k: v.to(ddp_info.device) if type(v) == torch.Tensor else v for k, v in batch.items()}

            with torch.autocast(enabled=config.training.use_amp, device_type="cuda",
                    dtype=amp_dtype_mapping[config.training.amp_dtype]):
                ret_dict = self.model(batch, target_has_input=False, is_valid=True)

            # UNI3T: every rank writes its shard to the checkpoint disk. A full disk used
            # to raise here, kill that rank, and leave the others blocked in the barrier
            # below. Catch it per batch (OSError only -- the forward above is not guarded).
            if not best_effort_write("validation export", self.cur_train_step,
                                     export_results, ret_dict, out_dir, compute_metrics=True):
                n_export_failed += 1

        # Agree on whether ANY rank lost part of its shard. A partial directory would
        # still summarize -- over fewer objects -- and log a misleading number.
        _failed = torch.tensor([float(n_export_failed)], device=ddp_info.device)
        dist.all_reduce(_failed, op=dist.ReduceOp.SUM)
        n_export_failed_all = int(_failed.item())

        # Wait until every rank has exported its shard before rank 0 scans the
        # shared validation directory. Without this barrier, summaries can be
        # silently incomplete when a non-main rank finishes later.
        dist.barrier()

        avg_metric_dict = None
        if ddp_info.is_main_process:
            if n_export_failed_all > 0:
                builtins.print(f"[val] WARNING: {n_export_failed_all} batch export(s) failed at step "
                      f"{self.cur_train_step}; skipping this validation's summary and wandb "
                      f"log so no partial average is recorded")
            else:
                try:
                    avg_metric_dict = summarize_evaluation(out_dir, ret_dict=True)
                except Exception as _e:
                    builtins.print(f"[val] NVS summary failed at step {self.cur_train_step}: {_e}")
        if ddp_info.is_main_process and avg_metric_dict:
            # print in console
            print(f"Validation summary at step {self.cur_train_step}: ")
            for k, v in avg_metric_dict.items():
                print(f"{k}: {v}")

            # log to wandb
            val_log_dict = {"val/" + k: float(v) for k, v in avg_metric_dict.items()}
            # UNI3T: exactly two extra curves from the depth / pose tasks. Everything
            # else those summaries produce stays in the json / csv / txt files. Both
            # helpers return strings and return None when no shard wrote metrics.
            # UNI3T: both helpers parse per-shard json and reach outside their own
            # try blocks (utils/metric_utils.py:1207 IndexError on ragged keys, :1211
            # KeyError on a missing one). Escaping here would abort validate() and end
            # the run. This branch puts three json families on the every-2000-step
            # path, so a single bad file must cost one reading, not the arm.
            try:
                _depth = summarize_evaluation_depth(out_dir, ret_dict=True)
            except Exception as _e:
                builtins.print(f"[val] depth summary failed at step {self.cur_train_step}: {_e}")
                _depth = None
            try:
                _pose = summarize_evaluation_pose(out_dir, ret_dict=True)
            except Exception as _e:
                builtins.print(f"[val] pose summary failed at step {self.cur_train_step}: {_e}")
                _pose = None
            if _depth and "abs_rel" in _depth:
                val_log_dict["val/abs_rel"] = float(_depth["abs_rel"])
            if _pose and "Auc_30" in _pose:
                val_log_dict["val/auc30"] = float(_pose["Auc_30"])
            val_log_dict["forward_pass_step"] = self.cur_train_step
            wandb.log(val_log_dict)

        # Keep all ranks out of the next DDP forward until rank 0 has finished
        # reading and summarizing the shared validation outputs.
        dist.barrier()
        self.model.train()

    def log_and_save_ckpt(self, ret_dict, cur_epoch, cur_param_update_step, tic, export_inter_results, total_grad_norm):
        config = self.config

        # loss_dict = {k: float(f"{v.item():.6f}") for k, v in ret_dict.loss_metrics.items()}
        loss_dict = {}
        for k,v in ret_dict.loss_metrics.items():
            if isinstance(v, torch.Tensor):
                loss_dict[k] = float(f"{v.item():.6f}")
            elif isinstance(v, float):
                loss_dict[k] = v
            elif isinstance(v, int):
                loss_dict[k] = float(v)
            else:
                raise ValueError(f"Unknown type of loss value {type(v)}")

        # print in console
        if (self.cur_train_step % config.training.print_every == 0) or (self.cur_train_step < 100 + self.start_train_step):
            print_str = f"[Epoch {int(cur_epoch):>3d}] | Forwad step: {int(self.cur_train_step):>6d} (Param update step: {int(cur_param_update_step):>6d})"
            print_str += f" | Iter Time: {time.time() - tic:.2f}s | LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            print_str += f" | PeakMem: {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB\n"
            # Add loss values
            for k, v in loss_dict.items():
                print_str += f"{k}: {v:.6f} | "
            print(print_str)

        # log in wandb
        if (self.cur_train_step % config.training.wandb_log_every == 0) or (
            self.cur_train_step < 200 + self.start_train_step
        ):
            log_dict = {
                "iter": self.cur_train_step, 
                "forward_pass_step": self.cur_train_step,
                "param_update_step": cur_param_update_step,
                "lr": self.optimizer.param_groups[0]["lr"],
                "iter_time": time.time() - tic,
                "grad_norm": total_grad_norm,
                "epoch": cur_epoch,
                # UNI3T: peak allocated since process start. The three-task arm was budgeted
                # at 76-86 GB and this is the only way to check that against reality.
                "gpu_mem_peak_gb": torch.cuda.max_memory_allocated() / 2**30,
            }
            log_dict.update({"train/" + k: v for k, v in loss_dict.items()})
            wandb.log(log_dict)

        # save checkpoint
        if (self.cur_train_step % config.training.checkpoint_every == 0) or (self.cur_train_step == self.total_train_steps):
            if isinstance(self.model, DDP):
                model_weights = self.model.module.state_dict()
            else:
                model_weights = self.model.state_dict()
            checkpoint = {
                "model": model_weights,
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "fwdbwd_pass_step": self.cur_train_step,
                "param_update_step": cur_param_update_step,
            }
            os.makedirs(config.training.checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(config.training.checkpoint_dir, f"ckpt_{self.cur_train_step:016}.pt")
            # atomic save: a crash mid-write (quota/preemption) must never leave a
            # truncated ckpt_*.pt for auto-resume to pick up
            tmp_path = ckpt_path + ".tmp"
            # UNI3T (09-23): the [ckpt]/[val] messages below use builtins.print on purpose --
        # rich.print strips "[tag]" prefixes (ungreppable logs) and raises MarkupError on
        # exception text containing e.g. "[/x]", which would escape these except blocks.
        # UNI3T: a bare torch.save dies on the first OSError. The realistic one here
            # is quota exhaustion -- one checkpoint is 16.8 GB and the old disk is shared
            # with other arms. rank 0 dying takes the whole job down, the watchdog
            # resubmits from the last good checkpoint, and it fails at the same step
            # again: a permanent zero-progress loop that looks healthy on wandb. Retry,
            # then skip. Also remove the partial .pt.tmp, which neither find_checkpoints
            # (it only matches *.pt) nor the watchdog prune would ever clean up.
            _saved = False
            for _attempt in range(1, 4):
                try:
                    torch.save(checkpoint, tmp_path)
                    os.replace(tmp_path, ckpt_path)
                    _saved = True
                    break
                except (OSError, RuntimeError) as _e:
                    builtins.print(f"[ckpt] save attempt {_attempt}/3 failed at step {self.cur_train_step}: {_e}")
                    try:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    except OSError:
                        pass
            if _saved:
                print(f"Saved checkpoint at step {self.cur_train_step} to {os.path.abspath(ckpt_path)}")
            else:
                builtins.print(f"[ckpt] WARNING: gave up on checkpoint at step {self.cur_train_step} "
                      f"after 3 attempts; training continues without it")
        
        # export intermediate visualization results
        if export_inter_results:
            vis_path = os.path.join(config.training.checkpoint_dir, f"iter_{self.cur_train_step:08d}")
            # UNI3T: 09-22 job 133351 died here -- PIL's Image.save raised ENOSPC on a full
            # disk, killed rank 0, and torchrun reaped the other three (they were about to
            # meet rank 0 in the barrier after this call). These are debug images; losing
            # one set must not end the run.
            def _write_vis():
                os.makedirs(vis_path, exist_ok=True)
                visualize_intermediate_results(vis_path, ret_dict)
            best_effort_write("visualization", self.cur_train_step, _write_vis)
            torch.cuda.empty_cache()
            self.model.train()

                    
if __name__ == '__main__':
    # Load config and read(override) arguments from CLI
    config = init_config()

    os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))

    # Set up DDP for training/inference and Fix random seed
    ddp_info = init_distributed(seed=int(config.training.get("seed", 777)))
    dist.barrier()

    # Set up wandb and backup source code
    if ddp_info.is_main_process:
        init_wandb_and_backup(config)
    dist.barrier()

    # Set up tf32
    torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
    torch.backends.cudnn.allow_tf32 = config.training.use_tf32

    # Start training
    config.ddp_info = ddp_info
    trainer = Trainer(config)
    trainer.run()

    dist.barrier()
    dist.destroy_process_group()
