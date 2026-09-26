"""best_effort_write: the guard for debug-image and validation-export writes.

09-22 job 133351 died when PIL's Image.save hit ENOSPC in the rank-0 visualization
write. These tests produce a REAL ENOSPC by writing to /dev/full (Linux returns ENOSPC
on every write to it), through the same PIL call, instead of mocking the error.

    python -m unittest tests.test_best_effort_write -v
"""

import errno
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.training_utils import best_effort_write  # noqa: E402

DEV_FULL = "/dev/full"


def _pil_save_to(path):
    from PIL import Image
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(path, format="PNG")


@unittest.skipUnless(os.path.exists(DEV_FULL), "needs Linux /dev/full")
class TestRealENOSPC(unittest.TestCase):
    def test_negative_control_unguarded_pil_save_raises_enospc(self):
        """Proves the fixture really produces the 133351 failure -- without the guard
        the exact PIL call raises OSError errno 28. If this ever stops raising, the
        test below would pass vacuously."""
        with self.assertRaises(OSError) as ctx:
            _pil_save_to(DEV_FULL)
        self.assertEqual(ctx.exception.errno, errno.ENOSPC)

    def test_guarded_pil_save_returns_false_and_does_not_raise(self):
        ok = best_effort_write("visualization", 40800, _pil_save_to, DEV_FULL)
        self.assertFalse(ok)

    def test_guarded_plain_write_returns_false(self):
        def _w():
            with open(DEV_FULL, "w") as f:
                f.write("x" * 4096)
        self.assertFalse(best_effort_write("validation export", 2000, _w))


class TestContract(unittest.TestCase):
    def test_success_runs_fn_with_args_and_returns_true(self):
        seen = {}
        def _fn(a, b, *, c):
            seen.update(a=a, b=b, c=c)
        self.assertTrue(best_effort_write("x", 1, _fn, 1, 2, c=3))
        self.assertEqual(seen, {"a": 1, "b": 2, "c": 3})

    def test_non_oserror_still_propagates(self):
        """Only filesystem errors are survivable; a real bug must still stop the run."""
        def _bug():
            raise ValueError("shape mismatch")
        with self.assertRaises(ValueError):
            best_effort_write("validation export", 2000, _bug)

    def test_other_oserrors_are_caught_too(self):
        def _eio():
            raise OSError(errno.EIO, "Input/output error")
        self.assertFalse(best_effort_write("visualization", 1, _eio))


class TestMessageIsPrintedVerbatim(unittest.TestCase):
    """09-23 smoke 136772: rich.print stripped the "[io]" prefix (logs not greppable)
    and raises MarkupError on text like "[/train]". Printed from inside the except
    block, that MarkupError would escape the guard and kill the rank anyway."""

    def test_negative_control_rich_raises_on_closing_tag_text(self):
        from rich import print as rprint
        from rich.errors import MarkupError
        with self.assertRaises(MarkupError):
            rprint("[io] WARNING: failed: /data/[/train]/x.png")

    def test_bracketed_exception_text_does_not_escape_the_guard(self):
        def _fail():
            raise OSError(errno.ENOSPC, "No space left on device", "/data/[/train]/x.png")
        self.assertFalse(best_effort_write("visualization", 1, _fail))

    def test_prefix_and_path_survive_in_stdout(self):
        import io
        from contextlib import redirect_stdout
        def _fail():
            raise OSError(errno.ENOSPC, "No space left on device", "/data/[train]/x.png")
        buf = io.StringIO()
        with redirect_stdout(buf):
            best_effort_write("validation export", 60, _fail)
        out = buf.getvalue()
        self.assertIn("[io] WARNING: validation export failed at step 60:", out)
        self.assertIn("/data/[train]/x.png", out)   # not rewritten to /data//x.png


class TestAllReduceInsideInferenceMode(unittest.TestCase):
    """validate() is decorated with @torch.inference_mode() and now calls
    dist.all_reduce on a tensor created inside it. This checks that torch allows it
    (real torch.distributed on a 1-rank gloo group -- it tests the library behaviour
    the code depends on, not a re-implementation of our logic)."""

    def test_all_reduce_on_inference_tensor(self):
        import torch.distributed as dist
        if dist.is_initialized():
            self.skipTest("a process group already exists")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29731")
        dist.init_process_group("gloo", rank=0, world_size=1)
        try:
            with torch.inference_mode():
                t = torch.tensor([3.0])
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                self.assertEqual(int(t.item()), 3)
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
