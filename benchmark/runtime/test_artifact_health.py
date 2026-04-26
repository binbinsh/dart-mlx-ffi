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

from artifact_health import _classify_dart_runtime_failure, validate_artifact


class ArtifactHealthTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dinf_artifact_health_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_onnx_pipeline_op_only_passes_without_ort(self) -> None:
        pipeline = self.tmp / "pipeline.json"
        pipeline.write_text(
            json.dumps(
                {
                    "format": "dart_inference.onnx_pipeline.v1",
                    "stages": [{"name": "merge", "op": "scatter_embeddings"}],
                }
            ),
            encoding="utf-8",
        )

        result = validate_artifact(engine="onnx", artifact=pipeline)

        self.assertTrue(result["passed"])
        self.assertEqual(result["checks"][0]["op"], "scatter_embeddings")

    def test_onnx_pipeline_invalid_stage_fails_before_ort(self) -> None:
        pipeline = self.tmp / "pipeline.json"
        pipeline.write_text(
            json.dumps(
                {
                    "format": "dart_inference.onnx_pipeline.v1",
                    "stages": [{"name": "decoder"}],
                }
            ),
            encoding="utf-8",
        )

        result = validate_artifact(engine="onnx", artifact=pipeline)

        self.assertFalse(result["passed"])
        self.assertEqual(result["checks"][0]["state"], "invalid")

    def test_coreml_health_uses_dart_runtime_load_probe(self) -> None:
        artifact = self.tmp / "model.mlmodelc"
        payload = {
            "passed": True,
            "device_profile": {
                "runtime_diagnostics": {"engine": "coreml"},
            },
        }
        with mock.patch("artifact_health.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess(
                args=["dart"],
                returncode=0,
                stdout="Running build hooks...\n" + json.dumps(payload),
                stderr="",
            )

            result = validate_artifact(
                engine="coreml",
                artifact=artifact,
                platform="macos",
            )

        command = run.call_args.args[0]
        self.assertTrue(result["passed"])
        self.assertEqual(result["checks"][0]["state"], "loaded")
        self.assertIn("--health-check", command)
        self.assertIn("--engine", command)
        self.assertIn("coreml", command)

    def test_classify_litert_missing_optional_support_libraries(self) -> None:
        failure = _classify_dart_runtime_failure(
            engine="litert",
            stdout="StateError: TfLiteInterpreterCreate failed [no optional support libraries loaded; attempted [\"libtensorflowlite_flex_jni.so\"]]",
            stderr="",
        )

        self.assertEqual(
            failure["failure_class"],
            "missing_optional_support_libraries",
        )

    def test_classify_litert_multi_section_container(self) -> None:
        failure = _classify_dart_runtime_failure(
            engine="litert",
            stdout="StateError: LiteRT container has multiple TFLite sections (4).",
            stderr="",
        )

        self.assertEqual(failure["failure_class"], "section_index_required")

    def test_classify_litert_runtime_version_mismatch(self) -> None:
        failure = _classify_dart_runtime_failure(
            engine="litert",
            stdout="StateError: TfLiteInterpreterCreate failed [tflite_error: Op builtin_code out of range: 206.]",
            stderr="",
        )

        self.assertEqual(failure["failure_class"], "runtime_version_mismatch")


if __name__ == "__main__":
    unittest.main()
