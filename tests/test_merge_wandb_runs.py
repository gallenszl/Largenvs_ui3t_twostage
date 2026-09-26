import json
from pathlib import Path
import tempfile
import unittest

from tools.merge_wandb_runs import (
    LINEAGE_ARTIFACT_TYPE,
    coalesce_segment,
    get_or_create_claim,
    semantic_step,
    strip_resume_ckpt,
    validate_remote_prefix,
    validation_metrics,
)


class WandbMergeTests(unittest.TestCase):
    def test_lineage_artifact_type_is_not_wandb_reserved(self):
        self.assertNotEqual(LINEAGE_ARTIFACT_TYPE, "job")
        self.assertFalse(LINEAGE_ARTIFACT_TYPE.startswith("wandb-"))

    def test_semantic_step_precedence_and_legacy_fallback(self):
        self.assertEqual(
            semantic_step({"forward_pass_step": 17, "iter": 16, "_step": 15}), 17
        )
        self.assertEqual(semantic_step({"iter": 16, "_step": 15}), 16)
        self.assertEqual(semantic_step({"_step": 15}), 15)
        self.assertIsNone(semantic_step({"_step": 1.5}))

    def test_latest_timestamp_wins_for_replayed_step(self):
        rows = [
            {
                "forward_pass_step": 14001,
                "train/loss": 1.0,
                "_timestamp": 100.0,
                "_step": 500,
            },
            {
                "forward_pass_step": 14001,
                "train/loss": 0.5,
                "val/psnr": 20.0,
                "_timestamp": 200.0,
                "_step": 900,
            },
        ]
        selected, report = coalesce_segment(rows, 14001, 14001)
        self.assertEqual(selected[14001]["train/loss"], 0.5)
        self.assertEqual(selected[14001]["val/psnr"], 20.0)
        self.assertEqual(selected[14001]["iter"], 14001)
        self.assertEqual(report["conflicts_resolved_last_write_wins"], 1)

    def test_latest_timestamp_wins_even_if_history_rows_are_reversed(self):
        rows = [
            {"iter": 3, "train/loss": 0.25, "_timestamp": 200.0},
            {"iter": 3, "train/loss": 1.0, "_timestamp": 100.0},
        ]
        selected, _ = coalesce_segment(rows, 3, 3)
        self.assertEqual(selected[3]["train/loss"], 0.25)

    def test_remote_recovery_requires_an_exact_semantic_step_prefix(self):
        expected = {
            1: {"forward_pass_step": 1, "iter": 1, "train/loss": 1.0},
            10: {"forward_pass_step": 10, "iter": 10, "train/loss": 0.5},
        }
        rows = [
            {"forward_pass_step": 1, "iter": 1, "train/loss": 1.0},
        ]
        self.assertEqual(validate_remote_prefix(rows, expected), 1)
        with self.assertRaises(RuntimeError):
            validate_remote_prefix(
                [{"forward_pass_step": 10, "iter": 10, "train/loss": 0.5}],
                expected,
            )
        with self.assertRaises(RuntimeError):
            validate_remote_prefix(
                [{"forward_pass_step": 1, "iter": 1, "train/loss": 9.0}],
                expected,
            )

    def test_canonical_claim_is_stable_and_bound_to_payload(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)
            with mock.patch(
                "tools.merge_wandb_runs.wandb.util.generate_id",
                return_value="canonical123",
            ):
                first = get_or_create_claim(checkpoint_dir, "merged", "sha-one")
                second = get_or_create_claim(checkpoint_dir, "merged", "sha-one")
            self.assertEqual(first, ("canonical123", False))
            self.assertEqual(second, first)
            with self.assertRaises(ValueError):
                get_or_create_claim(checkpoint_dir, "merged", "sha-two")

    def test_validation_metrics_recompute_scene_average(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)
            eval_dir = checkpoint_dir / "eval_iter_00002000"
            for index, psnr in enumerate((10.0, 20.0)):
                scene_dir = eval_dir / f"{index:06d}"
                scene_dir.mkdir(parents=True)
                (scene_dir / "metrics.json").write_text(
                    json.dumps(
                        {
                            "summary": {
                                "scene_name": f"scene-{index}",
                                "psnr": psnr,
                                "ssim": 0.5 + 0.1 * index,
                            }
                        }
                    ),
                    encoding="utf-8",
                )
            result = validation_metrics(checkpoint_dir, expected_scenes=2)
            self.assertEqual(result[2000]["val/psnr"], 15.0)
            self.assertAlmostEqual(result[2000]["val/ssim"], 0.55)

    def test_resume_path_is_removed_from_uploaded_config_only(self):
        original = {"training": {"resume_ckpt": "/tmp/a.pt", "lr": 1e-4}}
        cleaned = strip_resume_ckpt(original)
        self.assertNotIn("resume_ckpt", cleaned["training"])
        self.assertEqual(original["training"]["resume_ckpt"], "/tmp/a.pt")


if __name__ == "__main__":
    unittest.main()
