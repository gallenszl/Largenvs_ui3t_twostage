"""CPU tests of the stage-2 training plumbing (plan G5): checkpoint pruning, fail-closed resume, the LR
schedule, and the train.py wiring of val_modes / trainable-only checkpoints.

Run:  python -m unittest tests.test_s2_train_contract tests.test_training_logging_contract -v
"""
import ast
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from utils.training_utils import auto_resume_job, create_lr_scheduler, prune_checkpoints, trainable_state_keys  # noqa: E402


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.a.register_buffer("buf", torch.zeros(3))            # persistent buffer of a trainable module
        self.frozen = nn.Linear(4, 4)
        self.frozen.register_buffer("fbuf", torch.ones(2))      # buffer of a frozen module
        self.frozen.requires_grad_(False)
        self.alias = nn.ModuleList([self.frozen])               # same frozen module under a second name,
                                                                # like the perceptual VGG (vgg.features.* / blocks.*)


# what a trainable-only checkpoint of Tiny must carry, written out by hand (not derived from the code under test)
TINY_KEEP = {"a.weight", "a.bias", "a.buf"}


def make(model):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    sch = create_lr_scheduler(opt, 100, 10, scheduler_type="constant_anneal", anneal_from_step=50)
    return opt, sch


def save(path, model, opt, sch, step, trainable_only=True):
    sd = model.state_dict()
    if trainable_only:
        sd = {k: v for k, v in sd.items() if k in TINY_KEEP}
    torch.save({"model": sd, "optimizer": opt.state_dict(), "lr_scheduler": sch.state_dict(),
                "fwdbwd_pass_step": step, "param_update_step": step}, path)


class TrainContractTests(unittest.TestCase):
    def test_prune_keeps_latest_and_milestones(self):
        with tempfile.TemporaryDirectory() as d:
            for st in (2000, 4000, 6000, 20000, 21000, 22000, 24000):
                open(os.path.join(d, f"ckpt_{st:016d}.pt"), "w").close()
            open(os.path.join(d, "ckpt_0000000000024000.pt.tmp"), "w").close()
            open(os.path.join(d, "notes.txt"), "w").close()
            removed = prune_checkpoints(d, 2, [21000, 26000])
            left = sorted(os.listdir(d))
            self.assertEqual(sorted(removed), [f"ckpt_{s:016d}.pt" for s in (2000, 4000, 6000, 20000)])
            self.assertIn("ckpt_0000000000021000.pt", left)
            self.assertIn("ckpt_0000000000024000.pt.tmp", left)          # never touches other files
            self.assertIn("notes.txt", left)
            # protect: the file just written survives even if not among the newest (clock skew etc.)
            open(os.path.join(d, f"ckpt_{1000:016d}.pt"), "w").close()
            prune_checkpoints(d, 1, [], protect=os.path.join(d, f"ckpt_{1000:016d}.pt"))
            self.assertIn(f"ckpt_{1000:016d}.pt", os.listdir(d))

    def test_trainable_state_keys_alias_and_buffers(self):
        m = Tiny()
        self.assertIn("alias.0.weight", m.state_dict())          # the alias really is in the state dict
        self.assertEqual(trainable_state_keys(m), TINY_KEEP)

    def test_fail_closed_resume(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as d:
            m = Tiny()
            opt, sch = make(m)
            # empty dir -> fresh start
            _, _, st, _ = auto_resume_job(d, m, opt, sch, False, fail_closed=True)
            self.assertEqual(st, 0)
            # a good trainable-only checkpoint resumes (frozen params may be missing)
            m.a.weight.data.fill_(0.5)
            save(os.path.join(d, f"ckpt_{30:016d}.pt"), m, opt, sch, 30)
            m2 = Tiny()
            opt2, sch2 = make(m2)
            _, _, st, _ = auto_resume_job(d, m2, opt2, sch2, False, fail_closed=True)
            self.assertEqual(st, 30)
            self.assertTrue(torch.equal(m2.a.weight, m.a.weight))
            # a corrupt newest checkpoint must raise instead of falling back to the older one
            open(os.path.join(d, f"ckpt_{40:016d}.pt"), "wb").write(b"garbage")
            with self.assertRaises(RuntimeError):
                auto_resume_job(d, Tiny(), *make(Tiny()), False, fail_closed=True)
            # the legacy path still falls back silently (unchanged behaviour)
            m3 = Tiny()
            _, _, st, _ = auto_resume_job(d, m3, *make(m3), False, fail_closed=False)
            self.assertEqual(st, 30)
            os.remove(os.path.join(d, f"ckpt_{40:016d}.pt"))
            # key mismatch (missing a trainable parameter) must raise
            sd = torch.load(os.path.join(d, f"ckpt_{30:016d}.pt"))
            del sd["model"]["a.bias"]
            torch.save(sd, os.path.join(d, f"ckpt_{50:016d}.pt"))
            with self.assertRaises(RuntimeError):
                m4 = Tiny()
                auto_resume_job(d, m4, *make(m4), False, fail_closed=True)
            os.remove(os.path.join(d, f"ckpt_{50:016d}.pt"))
            sd = torch.load(os.path.join(d, f"ckpt_{30:016d}.pt"))
            del sd["model"]["a.buf"]
            torch.save(sd, os.path.join(d, f"ckpt_{50:016d}.pt"))
            with self.assertRaises(RuntimeError):
                m4b = Tiny()
                auto_resume_job(d, m4b, *make(m4b), False, fail_closed=True)
            os.remove(os.path.join(d, f"ckpt_{50:016d}.pt"))
            # optimizer state that does not fit must raise
            big = nn.Linear(8, 8)
            opt_big = torch.optim.AdamW(big.parameters(), lr=1e-3)
            big(torch.randn(2, 8)).sum().backward()
            opt_big.step()
            sd = torch.load(os.path.join(d, f"ckpt_{30:016d}.pt"))
            sd["optimizer"] = opt_big.state_dict()
            torch.save(sd, os.path.join(d, f"ckpt_{60:016d}.pt"))
            with self.assertRaises(RuntimeError):
                m5 = Tiny()
                o5, s5 = make(m5)
                m5.a(torch.randn(2, 4)).sum().backward()
                o5.step()
                auto_resume_job(d, m5, o5, s5, False, fail_closed=True)

    def test_stage2_lr_schedule_points(self):
        p = nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([p], lr=1.0)
        sch = create_lr_scheduler(opt, 26000, 1000, scheduler_type="constant_anneal", anneal_from_step=21000)
        lam = sch.lr_lambdas[0]
        self.assertAlmostEqual(lam(0), 0.0)
        self.assertAlmostEqual(lam(500), 0.5)
        self.assertAlmostEqual(lam(1000), 1.0)
        self.assertAlmostEqual(lam(21000), 1.0)
        self.assertAlmostEqual(lam(23500), 1.0 - math.sqrt(0.5), places=9)
        self.assertAlmostEqual(lam(26000), 0.0)

    def test_train_py_wiring(self):
        src = (REPO / "train.py").read_text()
        tree = ast.parse(src)
        trainer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Trainer")
        validate = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == "validate")
        self.assertEqual([a.arg for a in validate.args.args], ["self", "mode", "zero_p"])
        self.assertIn('config.training.get("val_modes", None)', src)
        self.assertIn('config.training.get("checkpoint_extra_steps", None)', src)
        self.assertIn('config.training.get("save_trainable_only", False)', src)
        self.assertIn('fail_closed=bool(config.training.get("resume_fail_closed", False))', src)
        # every validate() call passes the mode pair
        calls = [n for n in ast.walk(trainer) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "validate"]
        self.assertTrue(calls)
        for c in calls:
            self.assertEqual(len(c.args), 2)


if __name__ == "__main__":
    unittest.main()
