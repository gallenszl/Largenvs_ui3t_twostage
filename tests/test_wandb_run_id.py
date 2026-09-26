import multiprocessing
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from easydict import EasyDict as edict

import setup


def _create_run_id_worker(checkpoint_dir, start_event, result_queue):
    """Top-level multiprocessing target so the test also works with spawn."""
    os.environ.pop("WANDB_RUN_ID", None)
    start_event.wait()
    try:
        result_queue.put((True, setup.get_or_create_wandb_run_id(checkpoint_dir)))
    except Exception as exc:  # pragma: no cover - surfaced in the parent
        result_queue.put((False, repr(exc)))


class WandbRunIdSidecarTests(unittest.TestCase):
    def test_first_creation_and_subsequent_read_return_same_id(self):
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("WANDB_RUN_ID", None)
                with mock.patch.object(
                    setup.wandb.util, "generate_id", return_value="generated123"
                ) as generate_id:
                    first = setup.get_or_create_wandb_run_id(checkpoint_dir)
                    second = setup.get_or_create_wandb_run_id(checkpoint_dir)

            self.assertEqual(first, "generated123")
            self.assertEqual(second, first)
            self.assertEqual(generate_id.call_count, 1)
            self.assertEqual(
                (Path(checkpoint_dir) / ".wandb_run_id").read_text(),
                "generated123\n",
            )

    def test_concurrent_creation_converges_on_one_complete_id(self):
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            context = multiprocessing.get_context("fork")
            start_event = context.Event()
            result_queue = context.Queue()
            processes = [
                context.Process(
                    target=_create_run_id_worker,
                    args=(checkpoint_dir, start_event, result_queue),
                )
                for _ in range(2)
            ]
            for process in processes:
                process.start()
            start_event.set()
            results = [result_queue.get(timeout=10) for _ in processes]
            for process in processes:
                process.join(timeout=10)
                self.assertEqual(process.exitcode, 0)

            self.assertTrue(all(success for success, _ in results), results)
            run_ids = {value for _, value in results}
            self.assertEqual(len(run_ids), 1)
            persisted = (Path(checkpoint_dir) / ".wandb_run_id").read_text().strip()
            self.assertEqual(run_ids, {persisted})
            self.assertTrue(persisted)

    def test_environment_id_seeds_new_sidecar(self):
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            with mock.patch.dict(os.environ, {"WANDB_RUN_ID": "fromenv123"}):
                run_id = setup.get_or_create_wandb_run_id(checkpoint_dir)

            self.assertEqual(run_id, "fromenv123")
            self.assertEqual(
                (Path(checkpoint_dir) / ".wandb_run_id").read_text(),
                "fromenv123\n",
            )

    def test_environment_id_must_match_existing_sidecar(self):
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            sidecar = Path(checkpoint_dir) / ".wandb_run_id"
            sidecar.write_text("persisted123\n")
            with mock.patch.dict(os.environ, {"WANDB_RUN_ID": "different123"}):
                with self.assertRaisesRegex(ValueError, "conflicts"):
                    setup.get_or_create_wandb_run_id(checkpoint_dir)

    def test_empty_and_forbidden_ids_fail_fast(self):
        invalid_ids = ["", "   ", "a/b", "a\\b", "a#b", "a?b", "a%b", "a:b"]
        for invalid_id in invalid_ids:
            with self.subTest(run_id=invalid_id):
                with tempfile.TemporaryDirectory() as checkpoint_dir:
                    with mock.patch.dict(
                        os.environ, {"WANDB_RUN_ID": invalid_id}, clear=False
                    ):
                        with self.assertRaises(ValueError):
                            setup.get_or_create_wandb_run_id(checkpoint_dir)

    def test_invalid_persisted_id_fails_fast(self):
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            (Path(checkpoint_dir) / ".wandb_run_id").write_text("\n")
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("WANDB_RUN_ID", None)
                with self.assertRaises(ValueError):
                    setup.get_or_create_wandb_run_id(checkpoint_dir)


class WandbInitializationTests(unittest.TestCase):
    def test_init_resumes_sidecar_run_and_removes_transient_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            api_key_path = Path(temp_dir) / "api_key.yaml"
            api_key_path.write_text("wandb: test-api-key\n")
            checkpoint_dir = Path(temp_dir) / "checkpoints"
            config = edict(
                training=edict(
                    api_key_path=str(api_key_path),
                    checkpoint_dir=str(checkpoint_dir),
                    wandb_project="project",
                    wandb_exp_name="experiment",
                    resume_ckpt="/transient/ckpt.pt",
                    stable_value=17,
                )
            )

            with mock.patch.dict(os.environ, {"WANDB_RUN_ID": "stable123"}), mock.patch.object(
                setup, "wandb"
            ) as wandb_mock, mock.patch.object(setup, "local_backup_src_code"):
                setup.init_wandb_and_backup(config)

            init_kwargs = wandb_mock.init.call_args.kwargs
            self.assertEqual(init_kwargs["id"], "stable123")
            self.assertEqual(init_kwargs["resume"], "allow")
            self.assertNotIn("resume_ckpt", init_kwargs["config"].training)
            self.assertEqual(init_kwargs["config"].training.stable_value, 17)
            self.assertEqual(config.training.resume_ckpt, "/transient/ckpt.pt")

            expected_metrics = {
                "train/*",
                "val/*",
                "grad_norm",
                "grad_norm_details/*",
                "lr",
                "iter_time",
                "epoch",
                "param_update_step",
                "iter",
            }
            calls = wandb_mock.define_metric.call_args_list
            self.assertIn(mock.call("forward_pass_step"), calls)
            for metric_name in expected_metrics:
                self.assertIn(
                    mock.call(metric_name, step_metric="forward_pass_step"), calls
                )

            self.assertEqual(
                (checkpoint_dir / ".wandb_run_id").read_text(), "stable123\n"
            )
            saved_config = (checkpoint_dir / "config.yaml").read_text()
            self.assertIn("resume_ckpt", saved_config)


if __name__ == "__main__":
    unittest.main()
