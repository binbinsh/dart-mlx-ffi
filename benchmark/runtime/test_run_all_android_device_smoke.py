from __future__ import annotations

import json
import subprocess
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

from run_all import _artifact_health_failure, build_plan, run_plan


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
                                    "delegate": "xnnpack",
                                    "require_delegate": True,
                                    "litert_section_index": 1,
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
        self.assertIn("--delegate", command)
        self.assertIn("xnnpack", command)
        self.assertIn("--require-delegate", command)
        self.assertIn("--litert-section-index", command)
        self.assertIn("1", command)

    def test_default_matrix_uses_apple_android_production_platforms(self) -> None:
        artifacts_path = self.tmp / "artifacts.yaml"
        artifacts_path.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "models": {
                        "qwen2_5": {
                            "source_model": "Qwen/Qwen2.5-0.5B-Instruct",
                            "platforms": {
                                "ios": {
                                    "engine": "coreml",
                                    "artifact": "hf://demo/qwen/coreml/model.mlmodelc",
                                    "baseline_engine": "coreml",
                                },
                                "macos": {
                                    "engine": "coreml",
                                    "artifact": "hf://demo/qwen/coreml/model.mlmodelc",
                                    "baseline_engine": "coreml",
                                },
                                "android": {
                                    "engine": "litert",
                                    "artifact": "hf://demo/qwen/litert/model.tflite",
                                    "baseline_engine": "litert",
                                },
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
                platform=None,
                engine=None,
                out_root=self.tmp / "out",
                plan_out=None,
                run=False,
                prepare_inputs=False,
                allow_fail=True,
                dry_run=False,
                path_check="none",
                execution_check="none",
                artifact_health_check="none",
                include_abandoned_platforms=False,
            )
        )

        self.assertEqual(plan["production_platforms"], ["ios", "macos", "android"])
        self.assertEqual(plan["cell_count"], 3)
        self.assertEqual(plan["production_model_count"], 0)
        self.assertEqual(plan["model_readiness_counts"]["preferred-ready"], 1)
        self.assertEqual(plan["model_readiness"][0]["readiness_class"], "preferred-ready")
        self.assertEqual(
            {cell["platform"] for cell in plan["cells"]},
            {"ios", "macos", "android"},
        )
        self.assertNotIn("windows", {cell["platform"] for cell in plan["cells"]})

    def test_run_plan_reports_readiness_classes(self) -> None:
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
                                    "engine": "onnx",
                                    "artifact": "hf://demo/qwen/onnx/model.onnx",
                                    "baseline_engine": "onnx",
                                    "fallback_from": ["litert"],
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
                engine=None,
                out_root=self.tmp / "out",
                plan_out=None,
                run=False,
                prepare_inputs=False,
                allow_fail=True,
                dry_run=False,
                path_check="none",
                execution_check="none",
                artifact_health_check="none",
                include_abandoned_platforms=False,
            )
        )

        self.assertEqual(plan["ready_count"], 1)
        self.assertEqual(plan["artifact_ready_count"], 1)
        self.assertEqual(
            plan["readiness_counts"],
            {
                "production-ready": 0,
                "preferred-ready": 0,
                "fallback-ready": 1,
                "blocked": 0,
            },
        )
        self.assertEqual(plan["fallback_ready_count"], 1)
        self.assertEqual(plan["preferred_ready_count"], 0)
        self.assertEqual(plan["production_ready_count"], 0)
        self.assertEqual(plan["production_model_count"], 0)
        self.assertEqual(plan["model_readiness_counts"]["fallback-ready"], 1)
        self.assertEqual(plan["model_readiness"][0]["readiness_class"], "fallback-ready")
        self.assertEqual(plan["cells"][0]["readiness_state"], "fallback_ready")
        self.assertEqual(plan["cells"][0]["readiness_class"], "fallback-ready")

    def test_run_plan_reports_blocked_readiness_class(self) -> None:
        artifacts_path = self.tmp / "artifacts.yaml"
        artifacts_path.write_text(
            yaml.safe_dump({"version": 1, "models": {}}, sort_keys=False),
            encoding="utf-8",
        )

        plan = build_plan(
            Namespace(
                config=ROOT / "benchmark/runtime/models.yaml",
                artifacts=artifacts_path,
                model_id="qwen2_5",
                platform="android",
                engine=None,
                out_root=self.tmp / "out",
                plan_out=None,
                run=False,
                prepare_inputs=False,
                allow_fail=True,
                dry_run=False,
                path_check="none",
                execution_check="none",
                artifact_health_check="none",
                include_abandoned_platforms=False,
            )
        )

        self.assertEqual(plan["ready_count"], 0)
        self.assertEqual(plan["artifact_ready_count"], 0)
        self.assertEqual(plan["blocked_count"], 1)
        self.assertEqual(plan["readiness_counts"]["blocked"], 1)
        self.assertEqual(plan["model_readiness_counts"]["blocked"], 1)
        self.assertEqual(plan["model_readiness"][0]["readiness_class"], "blocked")
        self.assertEqual(plan["cells"][0]["readiness_state"], "blocked")
        self.assertEqual(plan["cells"][0]["readiness_class"], "blocked")

    def test_model_android_engine_order_can_make_onnx_preferred(self) -> None:
        config_path = self.tmp / "models.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "support_policy": {
                        "production_requires": {"platforms": ["android"]},
                        "abandoned_platforms": ["windows"],
                    },
                    "first_wave": [
                        {
                            "id": "demo_audio",
                            "family": "Demo",
                            "task": "audio",
                            "source_model": "demo/audio",
                            "source_artifacts": "huggingface",
                            "support_level": "staging",
                            "engine_order": {"android": ["onnx", "litert"]},
                        }
                    ],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        artifacts_path = self.tmp / "artifacts.yaml"
        artifacts_path.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "models": {
                        "demo_audio": {
                            "source_model": "demo/audio",
                            "platforms": {
                                "android": {
                                    "engine": "onnx",
                                    "artifact": "hf://demo/audio/model.onnx",
                                    "baseline_engine": "onnx",
                                    "fallback_from": ["litert"],
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
                config=config_path,
                artifacts=artifacts_path,
                model_id="demo_audio",
                platform="android",
                engine=None,
                out_root=self.tmp / "out",
                plan_out=None,
                run=False,
                prepare_inputs=False,
                allow_fail=True,
                dry_run=False,
                path_check="none",
                execution_check="none",
                artifact_health_check="none",
                include_abandoned_platforms=False,
            )
        )

        self.assertEqual(plan["ready_count"], 1)
        self.assertEqual(plan["artifact_ready_count"], 1)
        self.assertEqual(plan["readiness_counts"]["preferred-ready"], 1)
        self.assertEqual(plan["preferred_ready_count"], 1)
        self.assertEqual(plan["fallback_ready_count"], 0)
        self.assertEqual(plan["model_readiness_counts"]["preferred-ready"], 1)
        self.assertEqual(plan["cells"][0]["preferred_engine"], "onnx")
        self.assertEqual(plan["cells"][0]["fallback_from"], [])
        self.assertEqual(plan["cells"][0]["readiness_state"], "preferred_ready")
        self.assertEqual(plan["cells"][0]["readiness_class"], "preferred-ready")

    def test_run_plan_reports_production_ready_when_verdict_passes(self) -> None:
        artifact = self.tmp / "model.mlmodelc"
        artifact.mkdir()
        out_root = self.tmp / "out"
        verdict = out_root / "qwen2_5" / "ios" / "verdict.json"
        verdict.parent.mkdir(parents=True)
        verdict.write_text(
            json.dumps(
                {
                    "verdict": {
                        "passed": True,
                        "identity": {"passed": True},
                        "correctness": {"passed": True},
                        "speed": {"passed": True},
                        "peak_memory": {"passed": True},
                        "device_profile": {"passed": True},
                    }
                }
            ),
            encoding="utf-8",
        )
        artifacts_path = self.tmp / "artifacts.yaml"
        artifacts_path.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "models": {
                        "qwen2_5": {
                            "source_model": "Qwen/Qwen2.5-0.5B-Instruct",
                            "platforms": {
                                "ios": {
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
                model_id="qwen2_5",
                platform="ios",
                engine=None,
                out_root=out_root,
                plan_out=None,
                run=False,
                prepare_inputs=False,
                allow_fail=True,
                dry_run=False,
                path_check="none",
                execution_check="none",
                artifact_health_check="none",
                include_abandoned_platforms=False,
            )
        )

        self.assertEqual(plan["ready_count"], 1)
        self.assertEqual(plan["artifact_ready_count"], 1)
        self.assertEqual(plan["readiness_counts"]["production-ready"], 1)
        self.assertEqual(plan["production_ready_count"], 1)
        self.assertEqual(plan["production_model_count"], 1)
        self.assertEqual(plan["model_readiness"][0]["readiness_class"], "production-ready")
        self.assertEqual(plan["cells"][0]["readiness_state"], "production_ready")
        self.assertEqual(plan["cells"][0]["readiness_class"], "production-ready")

    def test_android_device_smoke_health_command_for_local_artifact(self) -> None:
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
        self.assertIn("benchmark/runtime/android_flutter_smoke.py", command[1])
        self.assertIn("--artifact", command)
        self.assertIn(str(local_artifact), command)

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

    def test_artifact_health_failure_marks_required_delegate_failure(self) -> None:
        report = self.tmp / "android_smoke_delegate.json"
        report.write_text(
            json.dumps(
                {
                    "engine": "litert",
                    "runtime_smoke": {
                        "passed": False,
                        "error": (
                            "Bad state: TfLiteInterpreterCreate failed for "
                            "/tmp/model.tflite with delegates [\"nnapi\"] "
                            "[no optional support libraries loaded]"
                        ),
                    },
                }
            ),
            encoding="utf-8",
        )

        failure = _artifact_health_failure(str(report))

        self.assertIsNotNone(failure)
        self.assertEqual(
            failure["failure_class"],
            "delegate_interpreter_create_failed",
        )

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

    def test_android_device_smoke_allows_execution_check_on_macos_host(self) -> None:
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
                execution_check="always",
                artifact_health_check="always",
            )
        )

        cell = plan["cells"][0]
        self.assertEqual(cell["state"], "ready")
        self.assertNotIn(
            "Platform android cannot be executed by local runner on host macos. Run this cell on the target host with --platform, or mark the cell executor: remote/device and provide pre-collected reports.",
            cell["reasons"],
        )

    def test_device_smoke_only_run_skips_runtime_runner(self) -> None:
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
                                    "baseline_engine": "litert",
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
                prepare_inputs=True,
                allow_fail=True,
                dry_run=False,
                path_check="none",
                execution_check="none",
                artifact_health_check="always",
            )
        )
        cell = plan["cells"][0]
        self.assertTrue(cell["device_smoke_only"])
        with mock.patch(
            "run_all._run",
            return_value=subprocess.CompletedProcess(args=["ok"], returncode=0),
        ) as mocked_run:
            result = run_plan(plan, allow_fail=True, dry_run=False)
        self.assertEqual(mocked_run.call_count, 1)
        self.assertEqual(result["failed_count"], 0)
        self.assertEqual(result["results"][0]["stage"], "device_smoke_only")


if __name__ == "__main__":
    unittest.main()
