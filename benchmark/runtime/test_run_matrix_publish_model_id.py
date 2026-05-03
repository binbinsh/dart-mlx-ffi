from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

RUNTIME_DIR = Path(__file__).resolve().parent
ROOT = RUNTIME_DIR.parents[1]
sys.path.insert(0, str(RUNTIME_DIR))

from run_matrix import RuntimeCell


class RunMatrixPublishModelIdTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dinf_run_matrix_publish_id_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_runtime_cell_passes_publish_model_id_to_mlx_runner(self) -> None:
        cell = RuntimeCell(
            Namespace(
                model_id="paddle_ocr_vl",
                platform="macos",
                engine="coreml",
                artifact="artifact",
                baseline_engine="mlx",
                baseline_artifact=None,
                baseline_report=None,
                baseline_publish_report=Path("benchmark/out/publish_report.json"),
                baseline_publish_model_id="mlx-community/PaddleOCR-VL-1.5-8bit",
                candidate_report=None,
                raw_baseline_report=None,
                config=RUNTIME_DIR / "models.yaml",
                input_json=RUNTIME_DIR / "fixtures/tiny_input.json",
                prompt=None,
                prompt_file=RUNTIME_DIR / "fixtures/text_prompt.txt",
                task="vlm",
                tools_file=None,
                tools_json=None,
                embedding_query=None,
                embedding_query_file=None,
                embedding_dim=None,
                image_file=None,
                audio_file=None,
                warmup="1",
                iters="1",
                max_tokens="1",
                num_threads=None,
                provider=None,
                delegate=None,
                coreml_mode=None,
                litert_section_index=None,
                hf_cache_root=None,
                require_provider=False,
                require_delegate=False,
                out_root=self.tmp / "out",
                allow_fail=True,
                dry_run=True,
            )
        )

        cmd = cell._baseline_command()

        self.assertIn("--publish-model-id", cmd)
        index = cmd.index("--publish-model-id")
        self.assertEqual(cmd[index + 1], "mlx-community/PaddleOCR-VL-1.5-8bit")
        self.assertIn("--task", cmd)
        self.assertEqual(cmd[cmd.index("--task") + 1], "vlm")
        self.assertIn("--max-tokens", cmd)
        self.assertEqual(cmd[cmd.index("--max-tokens") + 1], "1")

    def test_mlx_runner_uses_publish_model_id_lookup(self) -> None:
        publish = self.tmp / "publish.json"
        publish.write_text(
            json.dumps(
                [
                    {
                        "model_id": "mlx-community/PaddleOCR-VL-1.5-8bit",
                        "dart_ms": 12.34,
                        "max_abs_diff": 0.0,
                    }
                ]
            ),
            encoding="utf-8",
        )
        out = self.tmp / "out.json"
        completed = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "benchmark/runtime/runners/mlx_runner.py",
                "--model-id",
                "paddle_ocr_vl",
                "--platform",
                "macos",
                "--artifact",
                "demo",
                "--publish-report",
                str(publish),
                "--publish-model-id",
                "mlx-community/PaddleOCR-VL-1.5-8bit",
                "--task",
                "vlm",
                "--warmup",
                "2",
                "--iters",
                "3",
                "--max-tokens",
                "4",
                "--out",
                str(out),
            ],
            cwd=ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        decoded = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(decoded["model_id"], "paddle_ocr_vl")
        self.assertEqual(decoded["metrics"]["end_to_end_ms"], 12.34)
        self.assertEqual(decoded["metrics"]["iteration_count"], 3)
        self.assertEqual(decoded["metrics"]["warmup_count"], 2)
        self.assertEqual(decoded["task"], "vlm")
        self.assertEqual(
            decoded["run_config"],
            {
                "format": "dart_mlx_ffi.run_config.v1",
                "task": "vlm",
                "warmup": 2,
                "iters": 3,
                "max_tokens": 4,
                "sampling_strategy": "greedy",
            },
        )


if __name__ == "__main__":
    unittest.main()
