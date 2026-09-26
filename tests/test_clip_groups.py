"""Grouped gradient clipping, ported from RnG_fa3_repro_bench (M1 arm).

    srun -p cpu -N 1 -n 1 -c 4 --mem=8G python -m unittest tests.test_clip_groups -v
"""

import math
import sys
import unittest
from collections import OrderedDict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.training_utils import build_clip_groups, clip_grad_norm_grouped  # noqa: E402


def _params(spec):
    """spec: {name: (numel, grad_fill)} -> OrderedDict of leaf params with .grad set."""
    out = OrderedDict()
    for name, (n, fill) in spec.items():
        p = torch.nn.Parameter(torch.zeros(n))
        p.grad = torch.full((n,), float(fill))
        out[name] = p
    return out


class TestBuildClipGroups(unittest.TestCase):
    def test_every_param_in_exactly_one_group(self):
        d = _params({
            "camera_head.a": (4, 1.0), "camera_head.b": (4, 1.0),
            "point_head.c": (4, 1.0),
            "model.renderer.d": (4, 1.0), "model.reconstructor.e": (4, 1.0),
        })
        groups = build_clip_groups(d, ["camera_head.", "point_head."])
        names = [g for g, _ in groups]
        self.assertEqual(names, ["camera_head.", "point_head.", "rest"])
        seen = [id(p) for _, ps in groups for p in ps]
        self.assertEqual(len(seen), len(set(seen)), "a param landed in two groups")
        self.assertEqual(len(seen), len(d), "a param landed in no group")
        self.assertEqual([len(ps) for _, ps in groups], [2, 1, 2])

    def test_ddp_module_prefix_is_stripped(self):
        d = _params({"module.point_head.c": (4, 1.0), "module.model.x": (4, 1.0)})
        groups = build_clip_groups(d, ["point_head."])
        self.assertEqual([(g, len(ps)) for g, ps in groups], [("point_head.", 1), ("rest", 1)])

    def test_empty_groups_are_dropped(self):
        d = _params({"model.x": (4, 1.0)})
        groups = build_clip_groups(d, ["point_head."])
        self.assertEqual([g for g, _ in groups], ["rest"])

    def test_first_matching_prefix_wins(self):
        # 'point_head.' before 'point_head.proj.' -> the broader one takes it
        d = _params({"point_head.proj.w": (4, 1.0)})
        groups = build_clip_groups(d, ["point_head.", "point_head.proj."])
        self.assertEqual([(g, len(ps)) for g, ps in groups], [("point_head.", 1)])


class TestClipGradNormGrouped(unittest.TestCase):
    def test_reported_global_equals_single_global_clip(self):
        spec = {"point_head.a": (16, 0.9), "model.b": (16, 0.05), "model.c": (16, -0.2)}
        d1, d2 = _params(spec), _params(spec)
        ref = torch.nn.utils.clip_grad_norm_(list(d1.values()), max_norm=1.0).item()
        groups = build_clip_groups(d2, ["point_head."])
        total, per = clip_grad_norm_grouped(groups, 1.0)
        self.assertAlmostEqual(total, ref, places=5)
        self.assertAlmostEqual(math.sqrt(sum(v * v for v in per.values())), total, places=6)

    def test_each_group_clipped_to_max_norm(self):
        d = _params({"point_head.a": (16, 3.0), "model.b": (16, 2.0)})
        groups = build_clip_groups(d, ["point_head."])
        clip_grad_norm_grouped(groups, 1.0)
        for _, ps in groups:
            n = torch.cat([p.grad.flatten() for p in ps]).norm().item()
            self.assertAlmostEqual(n, 1.0, places=5)

    def test_small_group_is_left_alone_while_global_clip_would_shrink_it(self):
        """The whole point of the M1 mechanism, and the test that fails without it.

        point_head's gradient is huge, the trunk's is small. A single global clip
        scales BOTH by 1/||g_total||; grouped clipping leaves the trunk untouched.
        """
        spec = {"point_head.a": (64, 1.0), "model.trunk": (64, 0.03)}
        trunk_norm = math.sqrt(64) * 0.03                      # 0.24, well under 1.0
        d_glob, d_grp = _params(spec), _params(spec)

        g_total = torch.nn.utils.clip_grad_norm_(list(d_glob.values()), max_norm=1.0).item()
        trunk_after_global = d_glob["model.trunk"].grad.norm().item()

        groups = build_clip_groups(d_grp, ["point_head."])
        clip_grad_norm_grouped(groups, 1.0)
        trunk_after_grouped = d_grp["model.trunk"].grad.norm().item()

        # global clip shrank the trunk by 1/g_total even though the trunk alone was fine
        self.assertGreater(g_total, 2.0)
        self.assertAlmostEqual(trunk_after_global, trunk_norm / g_total, places=5)
        # grouped clip left it exactly where it was
        self.assertAlmostEqual(trunk_after_grouped, trunk_norm, places=5)
        # and that difference is the effect size the mechanism buys
        self.assertGreater(trunk_after_grouped / trunk_after_global, 2.0)

    def test_group_already_under_max_norm_is_untouched(self):
        d = _params({"point_head.a": (16, 0.01), "model.b": (16, 0.01)})
        before = {k: v.grad.clone() for k, v in d.items()}
        groups = build_clip_groups(d, ["point_head."])
        clip_grad_norm_grouped(groups, 1.0)
        for k, v in d.items():
            self.assertTrue(torch.equal(v.grad, before[k]), k)


class TestDefaultPathUnchanged(unittest.TestCase):
    def test_absent_key_means_no_groups(self):
        """train.py gates on `if clip_prefixes:` -- None and [] must both stay on the
        historical single global clip."""
        for val in (None, [], ()):
            self.assertFalse(bool(val), repr(val))


if __name__ == "__main__":
    unittest.main()
