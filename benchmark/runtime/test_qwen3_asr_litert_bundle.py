from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


RUNTIME_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNTIME_DIR))

from artifact_health import validate_artifact
from convert_artifacts_support import _find_artifact


class Qwen3AsrLiteRtBundleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dmf_qwen3_asr_bundle_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_bundle_json_is_runtime_artifact_and_health_runs_model_probe(self) -> None:
        bundle = self._write_bundle()

        found = _find_artifact(self.tmp, ["qwen3_asr_litert_bundle.json"])
        self.assertEqual(found, bundle)

        payload = {
            "passed": True,
            "device_profile": {
                "runtime_diagnostics": {
                    "engine": "litert",
                    "model_level_runner": "Qwen3AsrNativeRunner",
                }
            },
        }
        with (
            mock.patch(
                "artifact_health.prepare_runtime_environment",
                return_value=(
                    {"TEST_LITERT_ENV": "1"},
                    {"litert_env": {"ready": True}},
                ),
            ),
            mock.patch("artifact_health.subprocess.run") as run,
        ):
            run.return_value = subprocess.CompletedProcess(
                args=["dart"],
                returncode=0,
                stdout=json.dumps(payload),
                stderr="",
            )

            result = validate_artifact(
                engine="litert",
                artifact=bundle,
                platform="android",
            )

        command = run.call_args.args[0]
        self.assertTrue(result["passed"])
        self.assertEqual(result["bundle"], "qwen3_asr")
        self.assertTrue(any("qwen3_asr_litert_bundle.json" in item for item in command))
        self.assertTrue(any(check["name"] == "runner" for check in result["checks"]))

    def test_missing_tokenizer_fails_before_probe(self) -> None:
        bundle = self._write_bundle(include_tokenizer=False)

        with mock.patch("artifact_health.subprocess.run") as run:
            result = validate_artifact(
                engine="litert",
                artifact=bundle,
                platform="android",
            )

        run.assert_not_called()
        self.assertFalse(result["passed"])
        self.assertEqual(result["checks"][0]["name"], "tokenizer")

    def _write_bundle(self, *, include_tokenizer: bool = True) -> Path:
        for name in (
            "encoder.tflite",
            "decoder_init.tflite",
            "decoder_step.tflite",
            "config.json",
            "embed_tokens.bin",
        ):
            (self.tmp / name).write_bytes(b"data")
        if include_tokenizer:
            (self.tmp / "tokenizer.json").write_bytes(b"data")
        bundle = self.tmp / "qwen3_asr_litert_bundle.json"
        bundle.write_text(
            json.dumps(
                {
                    "format": "dart_mlx_ffi.qwen3_asr_litert_bundle.v1",
                    "runner": "Qwen3AsrNativeRunner.loadLiteRtBundle",
                    "components": {
                        "encoder": "encoder.tflite",
                        "decoder_init": "decoder_init.tflite",
                        "decoder_step": "decoder_step.tflite",
                    },
                }
            ),
            encoding="utf-8",
        )
        return bundle


if __name__ == "__main__":
    unittest.main()
