from __future__ import annotations

import subprocess
import sys
import time
import unittest
from pathlib import Path

RUNTIME_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNTIME_DIR))

from flutter_smoke_common import MarkerParser, capture_runtime_smoke, terminate_process


class FlutterSmokeCommonTest(unittest.TestCase):
    def test_capture_runtime_smoke_times_out_without_output(self) -> None:
        process = subprocess.Popen(
            ["sleep", "5"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        started = time.time()
        try:
            payload, timed_out, lines = capture_runtime_smoke(
                process,
                timeout_seconds=1,
                parser=MarkerParser(),
            )
        finally:
            terminate_process(process)
        self.assertIsNone(payload)
        self.assertTrue(timed_out)
        self.assertEqual(lines, [])
        self.assertLess(time.time() - started, 3.0)


if __name__ == "__main__":
    unittest.main()
