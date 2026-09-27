# Why is rendering one view slow?  For each renderer (stage-1 renderer call on the target views, and the stage-2
# renderer variants current / fg_dense / fg_dual), one profiled forward (torch.profiler, record_function around
# the renderer call) gives: host time to issue the call, summed GPU kernel time inside it, number of kernels, and
# the kernels grouped by the op that launched them.  1 and 10 target views, batch 1.
#   python tools_s2/s2_launch_profile.py --config configs/S2P8_...yaml --out x.json
import argparse
import json
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import tools_s2.s2_fgdense_whatif as W  # noqa: E402
from tools_s2.s2_renderer_debug import patch, unpatch  # noqa: E402


def range_stats(prof, label):
    rs = [e for e in prof.events() if e.name == label]
    out = []
    for r in rs:
        n, t, by = 0, 0.0, {}
        stack = list(r.cpu_children)
        while stack:
            e = stack.pop()
            for k in getattr(e, "kernels", []):
                n += 1
                t += k.duration
                b = by.setdefault(e.name, [0, 0.0])
                b[0] += 1
                b[1] += k.duration
            stack.extend(e.cpu_children)
        out.append(dict(host_ms=r.time_range.elapsed_us() / 1000.0, kernels=n, kernel_ms=t / 1000.0,
                        by_op={k: dict(kernels=v[0], ms=v[1] / 1000.0) for k, v in
                               sorted(by.items(), key=lambda kv: -kv[1][0])[:14]}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_scenes", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(28500 + int(os.environ.get("SLURM_JOB_ID", "3")) % 400))
        dist.init_process_group("gloo", rank=0, world_size=1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.chdir(REPO)
    from easydict import EasyDict as edict
    from torch.profiler import ProfilerActivity, profile, record_function
    import model_s2.blocks as Bk
    from model_s2.stage2_wrapper import Stage2LagerNVS
    from tools_s2.s2_bench import load_cfg
    from tools_s2.s2_infer_bench import val_scenes
    cfg = load_cfg(args.config, 1)
    model = Stage2LagerNVS(cfg).cuda().eval()
    model.val_cam_cond_zero_p = 0.0
    zero = torch.zeros((), device="cuda")
    model.loss_computer.forward = lambda *a, **k: edict(loss=zero)
    for blk in model.renderer.blocks:
        if isinstance(blk, Bk.S2BidirBlock):
            blk._rgate_tmp = torch.nn.Linear(768, 1536, bias=False).cuda()
    nin = int(cfg.training.val_dataset_cfgs.training.num_input_views)
    scenes = [dict(batch=b, alpha_in=b["alpha_mask"][:, :nin].float().view(1, nin, 1, 256, 256))
              for b in val_scenes(cfg, args.n_scenes)]
    pvd = model.stage1.model.process_val_data.config.training
    ed = model.stage1.model.model
    res = {}
    for vt in (1, 10):
        pvd.num_target_views = vt
        for var in ("current", "fg_dense", "fg_dual"):
            vps = []
            if var != "current":
                W.apply(vps, var)
            state = {"p2": False}
            ps = []
            orig_p2 = model.stage1.pass2

            def p2(*a, _o=orig_p2, **k):
                state["p2"] = True
                try:
                    return _o(*a, **k)
                finally:
                    state["p2"] = False
            patch(ps, model.stage1, "pass2", p2)
            o1 = ed.renderer.forward

            def r1(*a, _o=o1, **k):
                with record_function("S1_RENDERER_P2" if state["p2"] else "S1_RENDERER"):
                    return _o(*a, **k)
            patch(ps, ed.renderer, "forward", r1)
            o2 = model.renderer.forward

            def r2(*a, _o=o2, **k):
                with record_function("S2_RENDERER"):
                    return _o(*a, **k)
            patch(ps, model.renderer, "forward", r2)
            try:
                def one(sc):
                    W.G["alpha_in"] = sc["alpha_in"]
                    W.G["scene_fg_share"] = []
                    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        model(sc["batch"], target_has_input=False, is_valid=True)
                for _ in range(2):
                    for sc in scenes:
                        one(sc)
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    for sc in scenes:
                        one(sc)
                    torch.cuda.synchronize()
                r = {lab: range_stats(prof, lab) for lab in ("S1_RENDERER", "S1_RENDERER_P2", "S2_RENDERER")}
            finally:
                unpatch(ps)
                unpatch(vps)
            key = f"vt{vt}_{var}"
            res[key] = r
            for lab, lst in r.items():
                if not lst:
                    continue
                h = sum(x["host_ms"] for x in lst) / len(lst)
                kn = sum(x["kernels"] for x in lst) / len(lst)
                km = sum(x["kernel_ms"] for x in lst) / len(lst)
                print(f"[launch] vt={vt} {var:8s} {lab:15s} host {h:7.1f} ms | GPU kernels {km:7.1f} ms | "
                      f"{kn:6.0f} kernels | host per kernel {1000 * h / max(kn, 1):5.1f} us", flush=True)
                if lab == "S2_RENDERER" or (lab == "S1_RENDERER" and var == "current"):
                    top = lst[0]["by_op"]
                    print(f"[launch]     top ops by kernel count: " +
                          ", ".join(f"{k} {v['kernels']}" for k, v in list(top.items())[:10]), flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"[launch] DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
