from __future__ import annotations

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

from run_all import build_plan, run_plan


class RunAllIosDeviceSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dmf_run_all_ios_smoke_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_ios_device_smoke_health_command_for_hf_coreml(self) -> None:
        artifacts_path = self.tmp / "artifacts.yaml"
        artifacts_path.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "defaults": {
                        "audio_file": str(ROOT / "benchmark/runtime/fixtures/audio.wav")
                    },
                    "models": {
                        "silero_vad": {
                            "source_model": "snakers4/silero-vad",
                            "platforms": {
                                "ios": {
                                    "engine": "coreml",
                                    "artifact": "hf://FluidInference/silero-vad-coreml/silero-vad-unified-v6.0.0.mlmodelc",
                                    "ios_device_smoke": True,
                                    "device_id": "ios-device",
                                    "ios_smoke_build_mode": "debug",
                                    "ios_smoke_wait_for_artifact_seconds": 120,
                                    "coreml_compute_units": "cpuAndGPU",
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
                model_id="silero_vad",
                platform="ios",
                engine="coreml",
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
        self.assertIn("benchmark/runtime/ios_flutter_smoke.py", command[1])
        self.assertIn("--model-id", command)
        self.assertIn("silero_vad", command)
        self.assertIn("--device-id", command)
        self.assertIn("ios-device", command)
        self.assertIn("--artifact", command)
        self.assertIn("--wait-for-artifact-seconds", command)
        self.assertIn("120", command)
        self.assertIn("--build-mode", command)
        self.assertIn("debug", command)
        self.assertIn("--coreml-compute-units", command)
        self.assertIn("cpuAndGPU", command)

    def test_ios_device_smoke_health_command_for_local_coreml_artifact(self) -> None:
        local_artifact = self.tmp / "model.mlmodelc"
        local_artifact.write_text("placeholder", encoding="utf-8")
        artifacts_path = self.tmp / "artifacts.yaml"
        artifacts_path.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "defaults": {
                        "audio_file": str(ROOT / "benchmark/runtime/fixtures/audio.wav")
                    },
                    "models": {
                        "silero_vad": {
                            "source_model": "snakers4/silero-vad",
                            "platforms": {
                                "ios": {
                                    "engine": "coreml",
                                    "artifact": str(local_artifact),
                                    "ios_device_smoke": True,
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
                model_id="silero_vad",
                platform="ios",
                engine="coreml",
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
        self.assertIn("benchmark/runtime/ios_flutter_smoke.py", command[1])
        self.assertIn("--artifact", command)
        self.assertIn(str(local_artifact), command)

    def test_ios_device_smoke_allows_execution_check_on_macos_host(self) -> None:
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
                                    "artifact": "hf://finnvoorhees/coreml-Qwen2.5-0.5B-Instruct-4bit/Qwen2.5-0.5B-Instruct-4bit.mlmodelc",
                                    "baseline_engine": "coreml-llm",
                                    "ios_device_smoke": True,
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
                engine="coreml",
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
            "Platform ios cannot be executed by local runner on host macos. Run this cell on the target host with --platform, or mark the cell executor: remote/device and provide pre-collected reports.",
            cell["reasons"],
        )

    def test_coreml_llm_baseline_can_reuse_candidate_artifact(self) -> None:
        artifacts_path = self.tmp / "artifacts.yaml"
        artifacts_path.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "models": {
                        "qwen2_5": {
                            "source_model": "Qwen/Qwen2.5-0.5B-Instruct",
                            "platforms": {
                                "macos": {
                                    "engine": "coreml",
                                    "artifact": "hf://finnvoorhees/coreml-Qwen2.5-0.5B-Instruct-4bit/Qwen2.5-0.5B-Instruct-4bit.mlmodelc",
                                    "baseline_engine": "coreml-llm",
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
                platform="macos",
                engine="coreml",
                out_root=self.tmp / "out",
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

        cell = plan["cells"][0]
        self.assertEqual(cell["state"], "ready")
        self.assertNotIn(
            "Missing CoreML-LLM baseline artifact or raw report",
            cell["reasons"],
        )

    def test_device_smoke_only_run_skips_runtime_runner(self) -> None:
        local_artifact = self.tmp / "model.mlmodelc"
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
                                "ios": {
                                    "engine": "coreml",
                                    "artifact": str(local_artifact),
                                    "baseline_engine": "coreml-llm",
                                    "ios_device_smoke": True,
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
                engine="coreml",
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
