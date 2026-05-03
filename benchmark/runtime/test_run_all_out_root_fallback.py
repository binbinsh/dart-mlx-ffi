from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import yaml

RUNTIME_DIR = Path(__file__).resolve().parent
ROOT = RUNTIME_DIR.parents[1]
sys.path.insert(0, str(RUNTIME_DIR))

import run_matrix
from run_all import build_plan


class RunAllOutRootFallbackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dmf_run_all_out_root_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_build_plan_uses_out_local_when_out_root_parent_is_broken(self) -> None:
        root = self.tmp
        broken_parent = root / "benchmark" / "out"
        broken_target = root / "missing" / "out_target"
        broken_parent.parent.mkdir(parents=True, exist_ok=True)
        broken_parent.symlink_to(broken_target)
        artifacts_path = root / "artifacts.yaml"
        artifacts_path.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "models": {
                        "qwen2_5": {
                            "source_model": "Qwen/Qwen2.5-0.5B-Instruct",
                            "platforms": {
                                "linux": {
                                    "engine": "onnx",
                                    "artifact": "hf://onnx-community/Qwen2.5-0.5B-Instruct/onnx/model_q4f16.onnx",
                                    "baseline_engine": "onnx",
                                }
                            },
                        }
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        out_root = root / "benchmark" / "out" / "runtime"
        with mock.patch.object(run_matrix, "ROOT", root):
            plan = build_plan(
                Namespace(
                    config=ROOT / "benchmark/runtime/models.yaml",
                    artifacts=artifacts_path,
                    model_id="qwen2_5",
                    platform="linux",
                    engine="onnx",
                    out_root=out_root,
                    plan_out=None,
                    run=True,
                    prepare_inputs=False,
                    allow_fail=True,
                    dry_run=False,
                    path_check="none",
                    execution_check="none",
                    artifact_health_check="none",
                )
            )
        self.assertEqual(str(root / "benchmark" / "out_local" / "runtime"), plan["out_root"])
        command = plan["cells"][0]["command"]
        self.assertIn("--out-root", command)
        self.assertIn(str(root / "benchmark" / "out_local" / "runtime"), command)


if __name__ == "__main__":
    unittest.main()
