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

from artifact_health import (
    _canonical_provider,
    _classify_dart_runtime_failure,
    _select_onnx_provider,
    validate_artifact,
)


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

    def test_canonical_provider_handles_android_ep_aliases(self) -> None:
        self.assertEqual(_canonical_provider("nnapi"), "NNAPIExecutionProvider")
        self.assertEqual(
            _canonical_provider("androidnnapi"),
            "NNAPIExecutionProvider",
        )
        self.assertEqual(_canonical_provider("xnnpack"), "XnnpackExecutionProvider")
        self.assertEqual(_canonical_provider("npu"), "QNNExecutionProvider")

    def test_select_onnx_provider_expands_generic_npu_order(self) -> None:
        selected = _select_onnx_provider(
            "npu",
            ["CPUExecutionProvider", "NNAPIExecutionProvider"],
        )

        self.assertEqual(selected["provider"], "NNAPIExecutionProvider")

    def test_select_onnx_provider_falls_back_to_cpu_with_diagnostics(self) -> None:
        selected = _select_onnx_provider("qnn", ["CPUExecutionProvider"])

        self.assertEqual(selected["provider"], "CPUExecutionProvider")
        self.assertEqual(
            selected["fallback"],
            {
                "requested": "QNNExecutionProvider",
                "reason": "requested_provider_unavailable",
            },
        )

    def test_onnx_probe_receives_require_provider(self) -> None:
        model = self.tmp / "model.onnx"
        model.write_bytes(b"onnx")
        payload = {"passed": True, "selected_provider": "NNAPIExecutionProvider"}
        with (
            mock.patch(
                "artifact_health._onnx_checks",
                return_value=[
                    {"name": "model", "kind": "model", "path": str(model)}
                ],
            ),
            mock.patch("artifact_health.subprocess.run") as run,
        ):
            run.return_value = subprocess.CompletedProcess(
                args=["python"],
                returncode=0,
                stdout=json.dumps(payload),
                stderr="",
            )

            result = validate_artifact(
                engine="onnx",
                artifact=model,
                provider="npu",
                require_provider=True,
            )

        command = run.call_args.args[0]
        self.assertTrue(result["passed"])
        self.assertIn("--require-provider", command)

    def test_litert_pipeline_validates_with_dart_runtime_probe(self) -> None:
        pipeline = self.tmp / "pipeline.json"
        pipeline.write_text(
            json.dumps(
                {
                    "format": "dart_mlx_ffi.litert_pipeline.v1",
                    "stages": [{"name": "merge", "op": "scatter_embeddings"}],
                }
            ),
            encoding="utf-8",
        )
        payload = {
            "passed": True,
            "device_profile": {
                "runtime_diagnostics": {"engine": "litert", "pipeline": True},
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
                artifact=pipeline,
                platform="android",
            )

        command = run.call_args.args[0]
        self.assertTrue(result["passed"])
        self.assertTrue(result["pipeline"])
        self.assertEqual(result["stages"][0]["op"], "scatter_embeddings")
        self.assertIn("--engine", command)
        self.assertIn("litert", command)

    def test_litert_pipeline_model_stage_requires_explicit_inputs(self) -> None:
        pipeline = self.tmp / "pipeline.json"
        pipeline.write_text(
            json.dumps(
                {
                    "format": "dart_mlx_ffi.litert_pipeline.v1",
                    "stages": [{"name": "decoder", "model": "decoder.tflite"}],
                }
            ),
            encoding="utf-8",
        )

        result = validate_artifact(
            engine="litert",
            artifact=pipeline,
            platform="android",
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["checks"][0]["state"], "invalid")
        self.assertIn("declare inputs", result["checks"][0]["reason"])

    def test_coreml_health_uses_dart_runtime_load_probe(self) -> None:
        artifact = self.tmp / "model.mlmodelc"
        payload = {
            "passed": True,
            "device_profile": {
                "runtime_diagnostics": {"engine": "coreml"},
            },
        }
        with (
            mock.patch(
                "artifact_health.prepare_runtime_environment",
                return_value=(
                    {"TEST_RUNTIME_ENV": "1"},
                    {"coreml_env": {"ready": True}},
                ),
            ) as prepare,
            mock.patch("artifact_health.subprocess.run") as run,
        ):
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
        call_env = run.call_args.kwargs["env"]
        self.assertTrue(result["passed"])
        self.assertEqual(result["checks"][0]["state"], "loaded")
        self.assertEqual(
            result["checks"][0]["runtime_env"],
            {"coreml_env": {"ready": True}},
        )
        self.assertIn("--health-check", command)
        self.assertIn("--engine", command)
        self.assertIn("coreml", command)
        self.assertEqual(call_env["TEST_RUNTIME_ENV"], "1")
        prepare.assert_called_once()

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
