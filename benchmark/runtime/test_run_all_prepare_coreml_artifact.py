from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import yaml

RUNTIME_DIR = Path(__file__).resolve().parent
ROOT = RUNTIME_DIR.parents[1]
sys.path.insert(0, str(RUNTIME_DIR))

from run_all import build_plan


class RunAllPrepareCoreMlArtifactTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dinf_run_all_coreml_prepare_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_prepare_inputs_passes_coreml_artifact_for_local_paths(self) -> None:
        artifact = self.tmp / "pipeline.json"
        artifact.write_text(
            json.dumps({"format": "dart_inference.coreml_pipeline.v1", "stages": []}),
            encoding="utf-8",
        )
        artifacts_path = self.tmp / "artifacts.yaml"
        artifacts_path.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "models": {
                        "qwen3_vl": {
                            "source_model": "Qwen/Qwen2.5-VL-3B-Instruct",
                            "task": "vlm",
                            "platforms": {
                                "macos": {
                                    "engine": "coreml",
                                    "artifact": str(artifact),
                                    "baseline_engine": "coreml",
                                }
                            },
                        }
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        plan = build_plan(
            Namespace(
                config=ROOT / "benchmark/runtime/models.yaml",
                artifacts=artifacts_path,
                model_id="qwen3_vl",
                platform="macos",
                engine="coreml",
                out_root=self.tmp / "out",
                plan_out=None,
                run=True,
                prepare_inputs=True,
                allow_fail=True,
                dry_run=False,
                path_check="always",
                execution_check="none",
                artifact_health_check="none",
            )
        )

        command = plan["cells"][0]["prepare_input_command"]
        self.assertIn("--coreml-artifact", command)
        self.assertIn(str(artifact), command)

    def test_prepare_inputs_does_not_pass_coreml_artifact_for_hf_uri(self) -> None:
        artifacts_path = self.tmp / "artifacts.yaml"
        artifacts_path.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "models": {
                        "qwen3_vl": {
                            "source_model": "Qwen/Qwen2.5-VL-3B-Instruct",
                            "task": "vlm",
                            "platforms": {
                                "macos": {
                                    "engine": "coreml",
                                    "artifact": "hf://org/repo/pipeline.json",
                                    "baseline_engine": "coreml",
                                }
                            },
                        }
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        plan = build_plan(
            Namespace(
                config=ROOT / "benchmark/runtime/models.yaml",
                artifacts=artifacts_path,
                model_id="qwen3_vl",
                platform="macos",
                engine="coreml",
                out_root=self.tmp / "out",
                plan_out=None,
                run=True,
                prepare_inputs=True,
                allow_fail=True,
                dry_run=False,
                path_check="always",
                execution_check="none",
                artifact_health_check="none",
            )
        )

        command = plan["cells"][0]["prepare_input_command"]
        self.assertNotIn("--coreml-artifact", command)


if __name__ == "__main__":
    unittest.main()
