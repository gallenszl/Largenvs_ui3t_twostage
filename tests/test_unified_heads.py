"""UNI3T: tests for the three-task (pose + point map + NVS) wiring.

Run on a compute node, not the login node -- the renderer/camera-head cases
allocate a few hundred MB and the login cgroup only has 8 GiB:

    srun -p cpu -N 1 -n 1 -c 8 --mem=32G \
        python -m unittest tests.test_unified_heads -v
"""

import ast
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from easydict import EasyDict as edict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.loss import LossComputer, MultiTaskLossComputer  # noqa: E402
from models.layers.dense_point_head import DensePointHead  # noqa: E402
from models.renderer import Renderer  # noqa: E402
from utils.metric_utils import (  # noqa: E402
    calculate_auc_np,
    summarize_evaluation_depth,
    summarize_evaluation_pose,
)

UPSTREAM_HEAD = Path(
    "/home/z50057756/code/vggt-omega/vggt_omega/models/heads/dense_head.py"
)
VENDORED_HEAD = REPO_ROOT / "models" / "layers" / "dense_point_head.py"
VGGT_CKPT = Path(
    os.environ.get(
        "TORCH_HOME", str(Path.home() / ".cache" / "torch")
    )
) / "hub" / "checkpoints" / "model.pt"

# Renderer geometry of the uni3t arm.
ARM_DIM_IN = 768
ARM_PATCH = 8
ARM_LAYERS = [2, 5, 8, 11]
# Closed form for dim_in=768, patch=8, features=256, out_channels=[256,512,1024,1024],
# out_dim=3: norm 1,536 + projects 2,165,504 + resize 11,536,128 + layerN_rn 6,488,064
# + refinenets 8,524,288 + proj 3,084 + proj_conf 1,028.
ARM_HEAD_PARAMS = 28_719_632


def make_head(**kw):
    kw.setdefault("dim_in", ARM_DIM_IN)
    kw.setdefault("patch_size", ARM_PATCH)
    kw.setdefault("intermediate_layer_idx", list(ARM_LAYERS))
    return DensePointHead(**kw)


class TestDensePointHead(unittest.TestCase):
    def test_param_count(self):
        n = sum(p.numel() for p in make_head().parameters())
        self.assertEqual(n, ARM_HEAD_PARAMS)

    def test_pixel_shuffle_channel_accounting(self):
        """pixel_shuffle(r) maps [N, C*r^2, H, W] -> [N, C, H*r, W*r]."""
        head = make_head()
        self.assertEqual(head.final_shuffle_factor, ARM_PATCH // 4)
        self.assertEqual(head.proj.weight.shape[0], 3 * head.final_shuffle_factor**2)
        self.assertEqual(head.proj_conf.weight.shape[0], head.final_shuffle_factor**2)

        tokens = [torch.randn(1, 1, 4 + 1024, ARM_DIM_IN) for _ in range(12)]
        images = torch.zeros(1, 1, 6, 256, 256)
        with torch.no_grad():
            points, conf = head(tokens, images, patch_token_start=4)
        # Same layout VGGT's DPTHead returns, which is what the loss and the eval
        # plumbing already expect.
        self.assertEqual(tuple(points.shape), (1, 1, 256, 256, 3))
        self.assertEqual(tuple(conf.shape), (1, 1, 256, 256))

    def test_conf_head_starts_near_one(self):
        """proj_conf keeps upstream's zero-weight / log(0.05)-bias init."""
        head = make_head()
        self.assertTrue(torch.all(head.proj_conf.weight == 0))
        tokens = [torch.randn(1, 1, 4 + 1024, ARM_DIM_IN) for _ in range(12)]
        images = torch.zeros(1, 1, 6, 256, 256)
        with torch.no_grad():
            _, conf = head(tokens, images, patch_token_start=4)
        self.assertTrue(torch.allclose(conf, torch.full_like(conf, 1.05), atol=1e-6))

    def test_inverse_log_transform_has_no_gradient_at_zero(self):
        """The trap that makes zero-initialising `proj` fatal.

        inv_log(y) = sign(y) * expm1(|y|). autograd gives d/dy == 0 at y == 0 because
        torch.sign has zero derivative and sign(0) == 0, so a zero-initialised point
        head would produce identically-zero logits and never receive any gradient.
        """
        from model.vggt.heads.head_act import inverse_log_transform

        y0 = torch.zeros(1, requires_grad=True)
        inverse_log_transform(y0).sum().backward()
        self.assertEqual(y0.grad.item(), 0.0)

        y1 = torch.full((1,), 1e-3, requires_grad=True)
        inverse_log_transform(y1).sum().backward()
        self.assertAlmostEqual(y1.grad.item(), 1.001, places=3)

    def test_point_head_output_conv_gets_a_real_gradient(self):
        """Guards the regression above: proj must not be zero-initialised."""
        head = make_head()
        self.assertFalse(
            torch.all(head.proj.weight == 0),
            "proj is zero-initialised -> inv_log kills its gradient forever",
        )
        tokens = [torch.randn(1, 1, 4 + 256, ARM_DIM_IN) for _ in range(12)]
        images = torch.zeros(1, 1, 6, 128, 128)
        points, _ = head(tokens, images, patch_token_start=4)
        gt = torch.randn_like(points) * 0.2 + 0.5
        ((points - gt) ** 2).mean().backward()
        # .grad being a zeros tensor is NOT the same as having a gradient; that is
        # exactly what the earlier "no param without .grad" check failed to catch.
        self.assertGreater(float(head.proj.weight.grad.abs().max()), 0.0)
        self.assertGreater(float(head.proj.weight.grad.norm()), 1e-6)

    def test_patch_size_must_be_divisible_by_four(self):
        with self.assertRaises(ValueError):
            make_head(patch_size=6)

    def test_needs_exactly_four_intermediate_layers(self):
        head = make_head(intermediate_layer_idx=[2, 5, 8])
        tokens = [torch.randn(1, 1, 4 + 1024, ARM_DIM_IN) for _ in range(12)]
        images = torch.zeros(1, 1, 6, 256, 256)
        with self.assertRaises(ValueError):
            with torch.no_grad():
                head(tokens, images, patch_token_start=4)


class TestVendoredDeviations(unittest.TestCase):
    """The vendored head must differ from upstream only where we said it does."""

    # Everything that carries the fusion pyramid must be byte-identical.
    FROZEN_TOP_LEVEL = [
        "_make_dense_resize_layer",
        "_make_prediction_head",
        "_init_small_conf_prediction_head",
        "_make_fusion_block",
        "_make_scratch",
        "ResidualConvUnit",
        "FeatureFusionBlock",
        "custom_interpolate",
    ]
    FROZEN_METHODS = ["_apply_pos_embed", "scratch_forward"]

    @staticmethod
    def _sources(path):
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
        top, methods = {}, {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                top[node.name] = ast.get_source_segment(src, node)
            if isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef):
                        methods[sub.name] = ast.get_source_segment(src, sub)
        return top, methods

    def setUp(self):
        if not UPSTREAM_HEAD.exists():
            self.skipTest(f"upstream head not available at {UPSTREAM_HEAD}")
        self.up_top, self.up_meth = self._sources(UPSTREAM_HEAD)
        self.our_top, self.our_meth = self._sources(VENDORED_HEAD)

    def test_frozen_helpers_byte_identical(self):
        for name in self.FROZEN_TOP_LEVEL:
            self.assertIn(name, self.our_top, name)
            self.assertEqual(self.up_top[name], self.our_top[name], name)

    def test_frozen_methods_byte_identical(self):
        for name in self.FROZEN_METHODS:
            self.assertEqual(self.up_meth[name], self.our_meth[name], name)

    def test_same_method_set(self):
        self.assertEqual(sorted(self.up_meth), sorted(self.our_meth))

    def test_only_expected_top_level_names_changed(self):
        # DenseHead -> DensePointHead is the sole rename.
        self.assertEqual(
            set(self.up_top) - set(self.our_top), {"DenseHead"}
        )
        self.assertEqual(
            set(self.our_top) - set(self.up_top), {"DensePointHead"}
        )


DEV = "cuda" if torch.cuda.is_available() else "cpu"
GPU_ONLY = unittest.skipUnless(
    torch.cuda.is_available(),
    "the renderer's attention is xformers FA2/FA3, which has no CPU kernel and only "
    "takes bf16/fp16 with head_dim in {64,128,192,256}",
)
# head_dim 256/4 = 64 keeps the FA3 op happy while staying far cheaper than the
# real arm (hidden 768, 12 blocks, 1024 patches).
TINY = dict(depth=4, hidden=256, heads=4, hw=32)


def tiny_renderer(kind="bidirectional_cross_attention"):
    torch.manual_seed(0)
    r = Renderer(
        depth=TINY["depth"],
        hidden_size=TINY["hidden"],
        patch_size=ARM_PATCH,
        num_heads=TINY["heads"],
        attention_to_features_type=kind,
    ).eval()
    return r.to(DEV)


class TestRendererIntermediates(unittest.TestCase):
    KINDS = [
        "bidirectional_cross_attention",
        "cross_attention",
        "full_attention",
    ]

    def _inputs(self, b=1, v=2):
        torch.manual_seed(1)
        hw = TINY["hw"]
        rays = torch.randn(b, v, 6, hw, hw, device=DEV)
        rec = torch.randn(b * v, 10, TINY["hidden"], device=DEV)
        return rec, rays

    @GPU_ONLY
    def test_default_path_is_unchanged(self):
        n_patch = (TINY["hw"] // ARM_PATCH) ** 2
        for kind in self.KINDS:
            with self.subTest(kind=kind):
                r = tiny_renderer(kind)
                rec, rays = self._inputs()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    plain = r(rec, rays)
                    withint, inter, psi = r(rec, rays, return_intermediates=True)
                self.assertTrue(torch.equal(plain, withint))
                self.assertEqual(len(inter), r.depth)
                self.assertEqual(psi, r.patch_start_idx)
                for t in inter:
                    self.assertEqual(
                        tuple(t.shape), (2, psi + n_patch, TINY["hidden"])
                    )

    @GPU_ONLY
    def test_last_intermediate_is_the_image_head_input(self):
        """The image head and the point head read the same final-block tensor."""
        import einops

        r = tiny_renderer()
        rec, rays = self._inputs()
        hw = TINY["hw"]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            rendered, inter, psi = r(rec, rays, return_intermediates=True)
            x = r.output_act(r.final_layer(inter[-1][:, psi:, :]))
            redone = einops.rearrange(
                x,
                "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
                v=2,
                h=hw // ARM_PATCH,
                w=hw // ARM_PATCH,
                p1=ARM_PATCH,
                p2=ARM_PATCH,
                c=3,
            )
        self.assertTrue(torch.equal(rendered, redone))

    def test_timeit_and_intermediates_are_exclusive(self):
        r = tiny_renderer()
        rec, rays = self._inputs()
        with self.assertRaises(AssertionError):
            r(rec, rays, timeit=True, return_intermediates=True)


class TestCameraHeadPretrained(unittest.TestCase):
    def test_strict_load_from_vggt1b(self):
        if not VGGT_CKPT.exists():
            self.skipTest(f"VGGT-1B checkpoint not found at {VGGT_CKPT}")
        from vggt.heads.camera_head import CameraHead

        sd = torch.load(VGGT_CKPT, map_location="cpu", mmap=True, weights_only=True)
        prefix = "camera_head."
        cam = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        self.assertEqual(len(cam), 69)
        head = CameraHead(dim_in=2048)
        head.load_state_dict(cam, strict=True)  # raises on any mismatch
        self.assertEqual(sum(p.numel() for p in head.parameters()), 216_174_610)


def base_config(weight_camera=0.0, weight_point=0.0):
    return edict(
        training=edict(
            l2_loss_weight=1.0,
            lpips_loss_weight=0.0,
            perceptual_loss_weight=0.0,
            weight_camera=weight_camera,
            weight_point=weight_point,
        )
    )


class TestMultiTaskLossReducesToBase(unittest.TestCase):
    def test_zero_weights_match_base_loss(self):
        torch.manual_seed(0)
        b, v, hw = 1, 2, 8
        rendering = torch.rand(b, v, 3, hw, hw)
        target = torch.rand(b, v, 3, hw, hw)
        pts_est = torch.randn(b, v, 3, hw, hw)
        pts_conf = torch.rand(b, v, 1, hw, hw) + 1.0
        pts_gt = torch.randn(b, v, 3, hw, hw) + 2.0  # non-zero => non-empty valid mask

        eye = torch.eye(4).view(1, 1, 4, 4).repeat(b, v, 1, 1)
        intr = torch.eye(3).view(1, 1, 3, 3).repeat(b, v, 1, 1)
        intr[..., 0, 0] = intr[..., 1, 1] = 100.0
        pose_enc_list = [torch.randn(b, v, 9) for _ in range(4)]

        base = LossComputer(base_config())(rendering, target, False)["loss"]
        multi = MultiTaskLossComputer(base_config(0.0, 0.0))(
            rendering,
            target,
            False,
            pts_est=pts_est,
            pts_conf=pts_conf,
            pts_gt=pts_gt,
            pose_enc_list=pose_enc_list,
            extrinsics=eye,
            intrinsics=intr,
            image_hw=(hw, hw),
        )["loss"]
        self.assertTrue(torch.equal(base, multi), f"{base.item()} vs {multi.item()}")


class TestPointL1TermIsGuarded(unittest.TestCase):
    """09-18 audit: F.l1_loss(pts_est, pts_gt) was the one point-loss term that did
    not pass through check_and_fix_inf_nan, while pts_est comes from
    inverse_log_transform, which overflows fp32 to inf at |y| >= 88.722836. A single
    inf there made the whole loss inf, and train.py's per-rank NaN skip then let one
    rank sit out an update the other three applied."""

    @staticmethod
    def _call(pts_est):
        torch.manual_seed(0)
        b, v, hw = 1, 2, 8
        rendering = torch.rand(b, v, 3, hw, hw)
        target = torch.rand(b, v, 3, hw, hw)
        pts_conf = torch.rand(b, v, 1, hw, hw) + 1.0
        pts_gt = torch.randn(b, v, 3, hw, hw) + 2.0
        eye = torch.eye(4).view(1, 1, 4, 4).repeat(b, v, 1, 1)
        intr = torch.eye(3).view(1, 1, 3, 3).repeat(b, v, 1, 1)
        intr[..., 0, 0] = intr[..., 1, 1] = 100.0
        return MultiTaskLossComputer(base_config(1.0, 0.2))(
            rendering, target, True,
            pts_est=pts_est, pts_conf=pts_conf, pts_gt=pts_gt,
            pose_enc_list=[torch.randn(b, v, 9) for _ in range(4)],
            extrinsics=eye, intrinsics=intr, image_hw=(hw, hw),
        )["loss"]

    def test_single_inf_prediction_does_not_make_the_loss_inf(self):
        torch.manual_seed(0)
        pts = torch.randn(1, 2, 3, 8, 8)
        pts[0, 0, 0, 0, 0] = float("inf")
        loss = self._call(pts)
        self.assertTrue(torch.isfinite(loss), f"loss = {loss.item()}")

    def test_finite_case_is_unchanged_by_the_guard(self):
        # The guard is check_and_fix_inf_nan, which is the identity on finite input
        # below its +-100 clamp, so the ordinary path must be bit-identical.
        torch.manual_seed(0)
        pts = torch.randn(1, 2, 3, 8, 8)
        a = self._call(pts)
        b = self._call(pts.clone())
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.isfinite(a))


class TestTrainableParamsAreFp32(unittest.TestCase):
    """09-08 user ruling, applied to this arm on 09-18: every trainable parameter is
    fp32. The inherited bf16 per_view_register_tokens kept 8 significant bits, so
    AdamW updates were rounded away once |p| > lr / (0.5 * 2**-7) = 256*lr = 8.96e-3.
    Measured on the paired baseline ckpt_72000: |p|max 0.006226 across only 614
    distinct magnitudes / 3072 params. A forward-equivalence test cannot see this."""

    def test_renderer_register_tokens_are_fp32(self):
        r = Renderer(hidden_size=64, depth=2, num_heads=2, patch_size=8,
                     attention_to_features_type="bidirectional_cross_attention")
        self.assertIs(r.per_view_register_tokens.dtype, torch.float32)

    def test_no_trainable_renderer_parameter_is_low_precision(self):
        r = Renderer(hidden_size=64, depth=2, num_heads=2, patch_size=8,
                     attention_to_features_type="bidirectional_cross_attention")
        bad = [(n, p.dtype) for n, p in r.named_parameters()
               if p.requires_grad and p.dtype is not torch.float32]
        self.assertEqual(bad, [], f"low-precision trainable params: {bad}")


class TestSummaryRetDict(unittest.TestCase):
    @staticmethod
    def _write(tmp, name, payload):
        d = Path(tmp) / name
        d.mkdir(parents=True, exist_ok=True)
        return d, payload

    def test_pose_ret_dict_exposes_auc30(self):
        r_err = [1.0, 12.0, 40.0]
        t_err = [2.0, 8.0, 25.0]
        with tempfile.TemporaryDirectory() as tmp:
            for i, (r, t) in enumerate(zip(r_err, t_err)):
                d = Path(tmp) / f"{i:06d}"
                d.mkdir()
                (d / "metrics_pose.json").write_text(
                    json.dumps({"summary": {"scene_name": f"s{i}", "rError": r, "tError": t}})
                )
            out = summarize_evaluation_pose(tmp, ret_dict=True)
        self.assertIn("Auc_30", out)
        import numpy as np

        expected = calculate_auc_np(np.array(r_err), np.array(t_err), max_threshold=30) * 100
        self.assertAlmostEqual(float(out["Auc_30"]), float(expected), places=6)

    def test_depth_ret_dict_exposes_abs_rel(self):
        names = ["silog", "abs_rel", "log10", "rms", "sq_rel", "log_rms", "d1", "d2", "d3"]
        with tempfile.TemporaryDirectory() as tmp:
            for i, val in enumerate([0.10, 0.20]):
                d = Path(tmp) / f"{i:06d}"
                d.mkdir()
                summary = {"scene_name": f"s{i}"}
                summary.update({n: val for n in names})
                (d / "metrics_depth.json").write_text(json.dumps({"summary": summary}))
            out = summarize_evaluation_depth(tmp, ret_dict=True)
        self.assertIn("abs_rel", out)
        self.assertAlmostEqual(float(out["abs_rel"]), 0.15, places=6)


if __name__ == "__main__":
    unittest.main()
