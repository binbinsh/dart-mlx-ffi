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

from run_all import _artifact_health_failure, build_plan


class RunAllAndroidDeviceSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dinf_run_all_android_smoke_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_android_device_smoke_health_command_for_hf_litert(self) -> None:
        artifacts_path = self.tmp / "artifacts.yaml"
        artifacts_path.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "models": {
                        "qwen2_5": {
                            "source_model": "Qwen/Qwen2.5-0.5B-Instruct",
                            "platforms": {
                                "android": {
                                    "engine": "litert",
                                    "artifact": "hf://litert-community/Qwen2.5-0.5B-Instruct/Qwen2.5-0.5B-Instruct_seq128_q8_ekv1280.tflite",
                                    "android_device_smoke": True,
                                    "device_id": "android-device",
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
                model_id="qwen2_5",
                platform="android",
                engine="litert",
                out_root=self.tmp / "out",
                plan_out=None,
                run=True,
                prepare_inputs=False,
                allow_fail=True,
                dry_run=False,
                path_check="none",
                execution_check="none",
                artifact_health_check="always",
            )
        )

        command = plan["cells"][0]["artifact_health_command"]
        self.assertIn("benchmark/runtime/android_flutter_smoke.py", command[1])
        self.assertIn("--model-id", command)
        self.assertIn("qwen2_5", command)
        self.assertIn("--device-id", command)
        self.assertIn("android-device", command)
        self.assertIn("--artifact", command)

    def test_android_device_smoke_skips_local_artifact_paths(self) -> None:
        local_artifact = self.tmp / "model.tflite"
        local_artifact.write_text("placeholder", encoding="utf-8")
        artifacts_path = self.tmp / "artifacts.yaml"
        artifacts_path.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "models": {
                        "qwen2_5": {
                            "source_model": "Qwen/Qwen2.5-0.5B-Instruct",
                            "platforms": {
                                "android": {
                                    "engine": "litert",
                                    "artifact": str(local_artifact),
                                    "android_device_smoke": True,
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
                model_id="qwen2_5",
                platform="android",
                engine="litert",
                out_root=self.tmp / "out",
                plan_out=None,
                run=True,
                prepare_inputs=False,
                allow_fail=True,
                dry_run=False,
                path_check="none",
                execution_check="none",
                artifact_health_check="always",
            )
        )

        command = plan["cells"][0]["artifact_health_command"]
        self.assertEqual(command, [])

    def test_artifact_health_failure_reads_failure_class_and_reason(self) -> None:
        report = self.tmp / "artifact_health.json"
        report.write_text(
            json.dumps(
                {
                    "checks": [
                        {
                            "failure_class": "interpreter_create_failed",
                            "failure_reason": "LiteRT runtime loaded model but failed to create interpreter.",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        failure = _artifact_health_failure(str(report))

        self.assertIsNotNone(failure)
        self.assertEqual(failure["failure_class"], "interpreter_create_failed")
        self.assertIn("failed to create interpreter", failure["failure_reason"])

    def test_artifact_health_failure_reads_android_runtime_smoke_failure(self) -> None:
        report = self.tmp / "android_smoke.json"
        report.write_text(
            json.dumps(
                {
                    "engine": "litert",
                    "passed": False,
                    "runtime_smoke": {
                        "passed": False,
                        "error": "Bad state: TfLiteInterpreterCreate failed for /tmp/model.tflite",
                    },
                }
            ),
            encoding="utf-8",
        )

        failure = _artifact_health_failure(str(report))

        self.assertIsNotNone(failure)
        self.assertEqual(failure["failure_class"], "interpreter_create_failed")
        self.assertIn("TfLiteInterpreterCreate failed", failure["failure_reason"])

    def test_artifact_health_failure_marks_litert_runtime_version_mismatch(self) -> None:
        report = self.tmp / "android_smoke_version.json"
        report.write_text(
            json.dumps(
                {
                    "engine": "litert",
                    "runtime_smoke": {
                        "passed": False,
                        "error": "Bad state: TfLiteInterpreterCreate failed [tflite_error: Op builtin_code out of range: 206.]",
                    },
                }
            ),
            encoding="utf-8",
        )

        failure = _artifact_health_failure(str(report))

        self.assertIsNotNone(failure)
        self.assertEqual(failure["failure_class"], "runtime_version_mismatch")


if __name__ == "__main__":
    unittest.main()
