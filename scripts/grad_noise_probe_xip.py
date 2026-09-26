"""Cross-inner-product gradient noise probe (the audit-recommended estimator).

Per draw at total batch B: two INDEPENDENT half-batches A/B of size B/2 each.
  E[<g_A, g_B>]      = |G|^2           (exactly unbiased -- independence)
  E[|g_A - g_B|^2]   = 4 tr(Sigma)/B
=> B_simple = tr/G2 directly, no intercept extrapolation. Kills the
"large-numbers-minus-large-numbers" root cause of the classical method's
unidentifiability at anneal-end checkpoints.

Env: CONFIG_PATH CKPT POINT_NAME OUT_JSON  BSIZES="32,128" COUNTS="96,32" SEED=0
"""
import os, sys, json, time, random, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from omegaconf import OmegaConf
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

amp_dtype_mapping = {"fp16": torch.float16, "bf16": torch.bfloat16,
                     "fp32": torch.float32, "tf32": torch.float32}

def main():
    cfg_path = os.environ["CONFIG_PATH"]; ckpt_path = os.environ["CKPT"]
    point = os.environ.get("POINT_NAME", "probe"); out_json = os.environ["OUT_JSON"]
    bsizes = [int(x) for x in os.environ.get("BSIZES", "32,128").split(",")]
    counts = [int(x) for x in os.environ.get("COUNTS", "96,32").split(",")]
    seed = int(os.environ.get("SEED", "0"))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    config = edict(OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True))
    config.ddp_info = edict(device="cuda:0", local_rank=0, rank=0, world_size=1, is_main_process=True)
    module, class_name = config.model.class_name.rsplit(".", 1)
    LVSM = importlib.import_module(module).__dict__[class_name]
    model = LVSM(config).to("cuda:0")
    if config.training.get("use_bf16", False):
        model = model.to(amp_dtype_mapping["bf16"])
        model.camera_head.to(torch.float32); model.point_head.to(torch.float32)
        model.rgb_head.to(torch.float32); model.loss_computer.to(torch.float32)
    state = torch.load(ckpt_path, map_location="cpu")
    sd = state["model"] if "model" in state else state
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[{point}] ckpt loaded: missing={len(missing)} unexpected={len(unexpected)}")
    assert len(unexpected) == 0
    model.train()
    trainables = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in trainables)
    print(f"[{point}] trainable {total/1e6:.1f}M -> snapshot buffer {total*4/2**30:.1f} GiB")
    buf = torch.empty(total, dtype=torch.float32, device="cuda:0")

    module, class_name = config.training.get("dataset_name", "data.dataset.Dataset").rsplit(".", 1)
    Dataset = importlib.import_module(module).__dict__[class_name]
    dataset = Dataset(config)
    use_bf16 = config.training.get("use_bf16", False)
    micro_cap = int(os.environ.get("MICRO_BATCH", "16"))
    results = {"point": point, "ckpt": ckpt_path, "config": cfg_path, "seed": seed,
               "estimator": "cross_inner_product_halves", "bsizes": {}, "skipped_nan": 0}

    def accum_half(it, loader, H, micro):
        # returns loss or None on NaN; leaves mean-of-H gradient in p.grad
        model.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(H // micro):
            try: data = next(it[0])
            except StopIteration:
                it[0] = iter(loader); data = next(it[0])
            batch = {k: v.to("cuda:0") if torch.is_tensor(v) else v for k, v in data.items()}
            if use_bf16:
                batch = {k: v.to(amp_dtype_mapping["bf16"]) if torch.is_tensor(v) else v for k, v in batch.items()}
            with torch.autocast(enabled=config.training.use_amp, device_type="cuda",
                                dtype=amp_dtype_mapping[config.training.amp_dtype]):
                ret = model(batch, exclude_bg=False)
            loss = ret.loss_metrics.loss
            if torch.isnan(loss) or torch.isinf(loss): return None
            (loss * (micro / H)).backward()
            tot += loss.item() * micro / H
        return tot

    for B, N in zip(bsizes, counts):
        H = B // 2; micro = min(H, micro_cap)
        assert H % micro == 0
        g = torch.Generator(); g.manual_seed(seed + 1000 + B)
        loader = DataLoader(dataset, batch_size=micro, shuffle=True, generator=g,
                            num_workers=min(8, max(2, micro)), drop_last=True, pin_memory=False)
        it = [iter(loader)]
        xips, d2s, lossesA = [], [], []
        t0 = time.time()
        while len(xips) < N:
            lA = accum_half(it, loader, H, micro)
            if lA is None: results["skipped_nan"] += 1; continue
            off = 0
            for p in trainables:
                n = p.numel()
                if p.grad is not None: buf[off:off+n].copy_(p.grad.detach().float().view(-1))
                else: buf[off:off+n].zero_()
                off += n
            lB = accum_half(it, loader, H, micro)
            if lB is None: results["skipped_nan"] += 1; continue
            xip = 0.0; d2 = 0.0; off = 0
            for p in trainables:
                n = p.numel()
                gb = p.grad.detach().float().view(-1) if p.grad is not None else torch.zeros(n, device="cuda:0")
                ga = buf[off:off+n]
                xip += torch.dot(ga, gb).item()
                d2 += (ga - gb).pow(2).sum().item()
                off += n
            xips.append(xip); d2s.append(d2); lossesA.append(lA)
            if len(xips) % 8 == 0:
                G2 = float(np.mean(xips)); tr = B/4*float(np.mean(d2s))
                print(f"[{point}] B={B}: {len(xips)}/{N} G2~{G2:.4f} tr~{tr:.3f} "
                      f"Bs~{tr/G2 if G2>0 else float('inf'):.0f} ({time.time()-t0:.0f}s)", flush=True)
        results["bsizes"][str(B)] = {"xips": xips, "d2s": d2s, "lossesA": lossesA,
                                     "wall_s": time.time() - t0}
    model.zero_grad(set_to_none=True)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    json.dump(results, open(out_json, "w"))
    print(f"[{point}] saved {out_json}")

if __name__ == "__main__":
    main()
