"""Gradient noise scale probe (McCandlish et al., arXiv:1812.06162).

Measures E[||g_B||^2] at several batch sizes on a frozen checkpoint using the
arm's own training loss and data sampling, WITHOUT any optimizer step. The
linear fit S_B = |G|^2 + tr(Sigma)/B (done offline) yields the simple noise
scale B_simple = tr(Sigma)/|G|^2.

Env vars:
  CONFIG_PATH  arm yaml (defines data paths, loss, num_views, thip, use_bf16)
  CKPT         checkpoint .pt (dict with key "model")
  POINT_NAME   label written into the output json
  OUT_JSON     output path
  BSIZES       comma list, default "1,4,16"   (samples per batch)
  COUNTS       comma list, default "256,96,48" (measurements per batch size)
  NTGT_OVERRIDE  optional: override num_target_views (num_views adjusted to 4+N)
  SEED         default 0
"""
import io, os, sys, json, time, random
import importlib

# repo root on sys.path: this script lives in scripts/ but imports the repo's
# top-level packages (model/, data/, utils/) the way train.py does from the root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from omegaconf import OmegaConf
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

amp_dtype_mapping = {"fp16": torch.float16, "bf16": torch.bfloat16,
                     "fp32": torch.float32, "tf32": torch.float32}


def main():
    cfg_path = os.environ["CONFIG_PATH"]
    ckpt_path = os.environ["CKPT"]
    point = os.environ.get("POINT_NAME", "probe")
    out_json = os.environ["OUT_JSON"]
    bsizes = [int(x) for x in os.environ.get("BSIZES", "1,4,16").split(",")]
    counts = [int(x) for x in os.environ.get("COUNTS", "256,96,48").split(",")]
    seed = int(os.environ.get("SEED", "0"))
    ntgt_override = os.environ.get("NTGT_OVERRIDE", "")

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    config = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    config = edict(config)
    if ntgt_override:
        n = int(ntgt_override)
        config.training.num_target_views = n
        config.training.num_views = 4 + n
    config.ddp_info = edict(device="cuda:0", local_rank=0, rank=0, world_size=1,
                            is_main_process=True)

    # ---- model (mirrors train.py _init_model, minus DDP/optimizer) ----
    module, class_name = config.model.class_name.rsplit(".", 1)
    LVSM = importlib.import_module(module).__dict__[class_name]
    model = LVSM(config).to("cuda:0")
    if config.training.get("use_bf16", False):
        model = model.to(amp_dtype_mapping["bf16"])
        model.camera_head.to(torch.float32)
        model.point_head.to(torch.float32)
        model.rgb_head.to(torch.float32)
        model.loss_computer.to(torch.float32)
        if hasattr(model, "align_target_encoder"):
            model.align_target_encoder.to(torch.bfloat16)
            model.align_projector.to(torch.float32)
    state = torch.load(ckpt_path, map_location="cpu")
    sd = state["model"] if "model" in state else state
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[{point}] ckpt loaded: missing={len(missing)} unexpected={len(unexpected)}")
    assert len(unexpected) == 0, unexpected[:5]
    model.train()

    trainables = [p for p in model.parameters() if p.requires_grad]
    print(f"[{point}] trainable params: {sum(p.numel() for p in trainables)/1e6:.1f}M")

    # ---- dataset (the arm's own sampling: thip / roll / view counts) ----
    module, class_name = config.training.get(
        "dataset_name", "data.dataset.Dataset").rsplit(".", 1)
    Dataset = importlib.import_module(module).__dict__[class_name]
    dataset = Dataset(config)
    print(f"[{point}] dataset: {len(dataset)} objects, "
          f"num_views={config.training.num_views}, "
          f"ntgt={config.training.num_target_views}")

    use_bf16 = config.training.get("use_bf16", False)
    results = {"point": point, "ckpt": ckpt_path, "config": cfg_path,
               "ntgt": int(config.training.num_target_views), "seed": seed,
               "bsizes": {}, "skipped_nan": 0}

    micro_cap = int(os.environ.get("MICRO_BATCH", "16"))
    for B, N in zip(bsizes, counts):
        # B > micro_cap is realized via gradient accumulation: the accumulated
        # gradient of sum_m (|micro|/B) * mean-loss_m equals the mean-of-B-samples
        # gradient bit-for-bit in expectation semantics (gradients are linear,
        # no BatchNorm-style cross-sample layers, no optimizer step involved).
        micro = min(B, micro_cap)
        assert B % micro == 0, f"B={B} not divisible by micro={micro}"
        n_micro = B // micro
        g = torch.Generator(); g.manual_seed(seed + B)
        loader = DataLoader(dataset, batch_size=micro, shuffle=True, generator=g,
                            num_workers=min(8, max(2, micro)), drop_last=True,
                            pin_memory=False)
        it = iter(loader)
        sqnorms, losses = [], []
        t0 = time.time()
        while len(sqnorms) < N:
            model.zero_grad(set_to_none=True)
            loss_total, bad = 0.0, False
            for _ in range(n_micro):
                try:
                    data = next(it)
                except StopIteration:
                    it = iter(loader); data = next(it)
                batch = {k: v.to("cuda:0") if torch.is_tensor(v) else v
                         for k, v in data.items()}
                if use_bf16:
                    batch = {k: v.to(amp_dtype_mapping["bf16"]) if torch.is_tensor(v) else v
                             for k, v in batch.items()}
                with torch.autocast(enabled=config.training.use_amp, device_type="cuda",
                                    dtype=amp_dtype_mapping[config.training.amp_dtype]):
                    ret = model(batch, exclude_bg=False)
                loss = ret.loss_metrics.loss
                if torch.isnan(loss) or torch.isinf(loss):
                    bad = True
                    break
                (loss * (micro / B)).backward()
                loss_total += loss.item() * micro / B
            if bad:
                results["skipped_nan"] += 1
                continue
            sq = 0.0
            for p in trainables:
                if p.grad is not None:
                    sq += p.grad.detach().float().pow(2).sum().item()
            sqnorms.append(sq); losses.append(loss_total)
            if len(sqnorms) % 32 == 0:
                print(f"[{point}] B={B}: {len(sqnorms)}/{N} "
                      f"mean||g||^2={sum(sqnorms)/len(sqnorms):.4e} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        results["bsizes"][str(B)] = {"sqnorms": sqnorms, "losses": losses,
                                     "wall_s": time.time() - t0}
        print(f"[{point}] B={B} done: mean={np.mean(sqnorms):.4e} "
              f"median={np.median(sqnorms):.4e} n={N}", flush=True)

    model.zero_grad(set_to_none=True)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f)
    print(f"[{point}] saved {out_json}")


if __name__ == "__main__":
    main()
