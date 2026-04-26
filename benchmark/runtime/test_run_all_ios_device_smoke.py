from __future__ import annotations

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


class RunAllIosDeviceSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dinf_run_all_ios_smoke_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_ios_device_smoke_health_command_for_hf_coreml(self) -> None:
        artifacts_path = self.tmp / "artifacts.yaml"
        artifacts_path.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "models": {
                        "silero_vad": {
                            "source_model": "snakers4/silero-vad",
                            "platforms": {
                                "ios": {
                                    "engine": "coreml",
                                    "artifact": "hf://FluidInference/silero-vad-coreml/silero-vad-unified-v6.0.0.mlmodelc",
                                    "ios_device_smoke": True,
                                    "device_id": "ios-device",
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

    def test_ios_device_smoke_skips_local_artifact_paths(self) -> None:
        local_artifact = self.tmp / "model.mlmodelc"
        local_artifact.write_text("placeholder", encoding="utf-8")
        artifacts_path = self.tmp / "artifacts.yaml"
        artifacts_path.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
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
        self.assertEqual(command, [])


if __name__ == "__main__":
    unittest.main()
