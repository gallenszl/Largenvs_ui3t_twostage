# Cost estimate of replacing "render the input cameras again + build the mask tables" by a tracking head: VGGT's
# TrackHead (DPT feature extractor, 128 channels at half resolution + CoTracker-style tracker: 7-level correlation
# pyramid, radius 4, 6-layer update transformer, hidden 384, 4 iterations), code imported read-only from
# ~/code/vggt, random weights (timing does not depend on them), bf16 autocast.  The two parts are timed separately
# on our shapes (518 x 518 frames, 37 x 37 tokens + 5 special tokens, 2048-d tokens):
#   features  DPT on n_frames frames (4 input views per scene + the target views, whose features in our design
#             would come from the stage-1 renderer tokens through a similar DPT)
#   tracker   one 5-frame set per target view (the target frame + the 4 input frames), N query points in the target
#             frame (P8: ~310 foreground tokens per view, P4: ~1240), 4 iterations
# forward only (inference / frozen head) and forward + backward (head trained jointly).
#   python tools_s2/s2_trackhead_cost.py --out x.json
import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/z50057756/code/vggt")
from vggt.heads.track_head import TrackHead  # noqa: E402
import vggt.heads.track_modules.base_track_predictor as _btp  # noqa: E402

# The reference tracker recomputes a constant 2-D sin-cos positional embedding on the CPU inside its iteration loop
# (and copies it to the GPU each time): ~1 s per call regardless of the number of queries.  Cache it on the GPU.
_pe_orig = _btp.get_2d_sincos_pos_embed
_pe_cache = {}


def _pe_cached(embed_dim, grid_size, return_grid=False):
    key = (embed_dim, tuple(grid_size) if isinstance(grid_size, (tuple, list)) else grid_size, return_grid)
    if key not in _pe_cache:
        out = _pe_orig(embed_dim, grid_size=grid_size, return_grid=return_grid)
        _pe_cache[key] = out.cuda() if torch.is_tensor(out) else out
    return _pe_cache[key]


if "--no_cache_pe" not in sys.argv:
    _btp.get_2d_sincos_pos_embed = _pe_cached


def timeit(fn, reps=10, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_cache_pe", action="store_true")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")
    head = TrackHead(dim_in=2048).to(dev)
    P = 5 + 37 * 37
    res = {}

    def features(B, S, train):
        tok = torch.randn(B, S, P, 2048, device=dev, requires_grad=train)
        img = torch.zeros(B, S, 3, 518, 518, device=dev)
        toks = [tok] * 24

        def f():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = head.feature_extractor(toks, img, 5)
            if train:
                out.float().mean().backward()
            return out
        with torch.set_grad_enabled(train):
            return timeit(f, reps=5, warm=2)

    def tracker(Bt, N, train):
        fm = torch.randn(Bt, 5, 128, 259, 259, device=dev, requires_grad=train)
        qp = torch.rand(Bt, N, 2, device=dev) * 517.0

        def f():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                coords, vis, conf = head.tracker(query_points=qp, fmaps=fm, iters=4)
            if train:
                (coords[-1].float().mean() + vis.float().mean()).backward()
        with torch.set_grad_enabled(train):
            return timeit(f, reps=5, warm=2)

    cases = {
        "infer_1view": dict(feat=(1, 5), trk=(1,), train=False),       # 4 input frames + 1 target frame
        "infer_10views": dict(feat=(1, 14), trk=(10,), train=False),   # 4 input + 10 target frames
        "train_1of6_trained": dict(feat=(8, 10), trk=(8,), train=True),  # tracker on 8 of the 48 target views
        "train_b8x6_frozen": dict(feat=(8, 10), trk=(48,), train=False),
    }
    for name, c in cases.items():
        try:
            tf = features(*c["feat"], c["train"])
        except Exception as e:                                                    # noqa: BLE001
            tf = f"{type(e).__name__}: {e}"
        for N in (310, 1240):
            try:
                tt = tracker(c["trk"][0], N, c["train"])
            except Exception as e:                                                # noqa: BLE001
                tt = f"{type(e).__name__}: {e}"
            res[f"{name}_N{N}"] = dict(features_ms=tf, tracker_ms=tt)
            print(f"[track] {name:18s} N={N:5d}: features {tf if isinstance(tf, str) else f'{tf:8.1f} ms'} | "
                  f"tracker {tt if isinstance(tt, str) else f'{tt:8.1f} ms'}", flush=True)
            torch.cuda.empty_cache()
    res["params_M"] = dict(features=sum(p.numel() for p in head.feature_extractor.parameters()) / 1e6,
                           tracker=sum(p.numel() for p in head.tracker.parameters()) / 1e6)
    print(f"[track] params (M): {res['params_M']}", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"[track] DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
