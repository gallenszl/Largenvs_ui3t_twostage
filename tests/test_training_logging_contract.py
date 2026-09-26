import ast
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _qualified_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualified_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


class TrainingLoggingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse((REPO_ROOT / "train.py").read_text(encoding="utf-8"))

    def test_wandb_log_never_uses_sdk_step_argument(self):
        calls = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call) and _qualified_name(node.func) == "wandb.log"
        ]
        self.assertTrue(calls)
        for call in calls:
            self.assertNotIn("step", {keyword.arg for keyword in call.keywords})

    def test_every_wandb_payload_contains_forward_pass_step(self):
        def dict_has_step(node):
            return isinstance(node, ast.Dict) and any(
                isinstance(key, ast.Constant) and key.value == "forward_pass_step"
                for key in node.keys
            )

        checked = 0
        for function in (
            node for node in ast.walk(self.tree) if isinstance(node, ast.FunctionDef)
        ):
            calls = [
                node
                for node in ast.walk(function)
                if isinstance(node, ast.Call)
                and _qualified_name(node.func) == "wandb.log"
            ]
            for call in calls:
                payload = call.args[0]
                if dict_has_step(payload):
                    checked += 1
                    continue

                self.assertIsInstance(payload, ast.Name)
                payload_name = payload.id
                has_step = False
                for node in ast.walk(function):
                    if not isinstance(node, ast.Assign) or node.lineno >= call.lineno:
                        continue
                    if any(
                        isinstance(target, ast.Name) and target.id == payload_name
                        for target in node.targets
                    ) and dict_has_step(node.value):
                        has_step = True
                    if any(
                        isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == payload_name
                        and isinstance(target.slice, ast.Constant)
                        and target.slice.value == "forward_pass_step"
                        for target in node.targets
                    ):
                        has_step = True
                self.assertTrue(has_step, f"missing forward_pass_step in {payload_name}")
                checked += 1

        self.assertGreater(checked, 0)

    def test_validation_orders_export_barrier_summary_barrier(self):
        trainer = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Trainer"
        )
        validate = next(
            node
            for node in trainer.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "validate"
        )
        call_nodes = sorted(
            (node for node in ast.walk(validate) if isinstance(node, ast.Call)),
            key=lambda n: n.lineno,
        )
        calls = [(node.lineno, _qualified_name(node.func)) for node in call_nodes]

        def is_export(node):
            name = _qualified_name(node.func)
            if name == "export_results":
                return True
            # 09-23: the export is guarded as
            # best_effort_write(what, step, export_results, ret_dict, out_dir, ...)
            return name == "best_effort_write" and any(
                isinstance(arg, ast.Name) and arg.id == "export_results"
                for arg in node.args
            )

        export_line = next(node.lineno for node in call_nodes if is_export(node))
        summary_line = next(
            line for line, name in calls if name == "summarize_evaluation"
        )
        barrier_lines = [line for line, name in calls if name == "dist.barrier"]

        self.assertEqual(len(barrier_lines), 2)
        self.assertLess(export_line, barrier_lines[0])
        self.assertLess(barrier_lines[0], summary_line)
        self.assertLess(summary_line, barrier_lines[1])

        # 09-23: every rank agrees on the export-failure count BEFORE rank 0 summarizes.
        # An all_reduce after the first barrier would let rank 0 start reading a
        # directory some other rank failed to fill.
        allreduce_lines = [line for line, name in calls if name == "dist.all_reduce"]
        self.assertEqual(len(allreduce_lines), 1)
        self.assertLess(export_line, allreduce_lines[0])
        self.assertLess(allreduce_lines[0], barrier_lines[0])


    def test_resume_guard_follows_auto_resume_job(self):
        # 09-24: auto_resume_job returns step 0 with a fresh optimizer when nothing
        # loads or the optimizer state fails to load. A branch that must continue a
        # trunk sets training.require_resume_step; train.py must check it right after
        # the resume and raise, never train on from step 0.
        resume_end = next(
            node.end_lineno
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and _qualified_name(node.value.func) == "auto_resume_job"
        )
        key_line = next(
            node.lineno
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and _qualified_name(node.func) == "config.training.get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "require_resume_step"
        )
        guards = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.If)
            and any(
                isinstance(sub, ast.Compare)
                and isinstance(sub.left, ast.Name)
                and sub.left.id == "cur_train_step"
                and any(isinstance(op, ast.Lt) for op in sub.ops)
                for sub in ast.walk(node.test)
            )
        ]
        self.assertEqual(len(guards), 1)
        guard = guards[0]
        raises = [
            stmt
            for stmt in guard.body
            if isinstance(stmt, ast.Raise)
            and isinstance(stmt.exc, ast.Call)
            and _qualified_name(stmt.exc.func) == "RuntimeError"
        ]
        self.assertEqual(len(raises), 1)
        self.assertLess(resume_end, key_line)
        self.assertLess(key_line, guard.lineno)
        # immediately after the resume, before anything else consumes the step
        self.assertLess(guard.lineno - resume_end, 12)


if __name__ == "__main__":
    unittest.main()
