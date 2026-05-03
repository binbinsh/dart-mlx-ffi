from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

RUNTIME_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNTIME_DIR))

import run_matrix
from run_matrix import _prepare_out_root


class RunMatrixOutRootFallbackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dmf_run_matrix_out_root_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_prepare_out_root_falls_back_from_broken_symlink_parent(self) -> None:
        root = self.tmp
        broken_parent = root / "benchmark" / "out"
        broken_target = root / "missing" / "out_target"
        broken_parent.parent.mkdir(parents=True, exist_ok=True)
        broken_parent.symlink_to(broken_target)
        requested = root / "benchmark" / "out" / "runtime"

        with mock.patch.object(run_matrix, "ROOT", root):
            resolved, meta = _prepare_out_root(requested)

        self.assertEqual(meta["mode"], "fallback")
        self.assertEqual(resolved, root / "benchmark" / "out_local" / "runtime")


if __name__ == "__main__":
    unittest.main()
