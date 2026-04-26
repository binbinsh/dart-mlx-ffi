from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

RUNTIME_DIR = Path(__file__).resolve().parent
ROOT = RUNTIME_DIR.parents[1]
sys.path.insert(0, str(RUNTIME_DIR))

from convert_artifacts import ArtifactConverter


class ConvertArtifactFailureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dinf_convert_failure_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_allow_conversion_fail_records_blocker_overlay(self) -> None:
        out = self.tmp / "artifacts.yaml"

        result = self._converter(
            command=["false"],
            out=out,
            allow_conversion_fail=True,
        ).run()

        self.assertEqual(result["conversion_failed_count"], 1)
        artifact_map = yaml.safe_load(out.read_text(encoding="utf-8"))
        model = artifact_map["models"]["paddle_ocr_vl"]
        self.assertIn("android", model["blocked_platforms"])
        self.assertIn("conversion.log", model["blocked_platforms"]["android"])
        self.assertIn("conversion_record.json", model["blocked_platform_reports"]["android"])
        self.assertIn("conversion.log", model["blocked_platform_logs"]["android"])
        self.assertEqual(
            model["blocked_platform_failure_classes"]["android"],
            "conversion_failed",
        )
        record = self._record()
        self.assertEqual(record["state"], "conversion_failed")
        self.assertTrue(record["log_path"].endswith("conversion.log"))

    def test_preflight_skip_records_blocker_overlay(self) -> None:
        out = self.tmp / "artifacts.yaml"

        result = self._converter(
            command=["false"],
            out=out,
            allow_conversion_fail=True,
            min_free_gb=1_000_000,
        ).run()

        self.assertEqual(result["preflight_skipped_count"], 1)
        artifact_map = yaml.safe_load(out.read_text(encoding="utf-8"))
        model = artifact_map["models"]["paddle_ocr_vl"]
        reason = model["blocked_platforms"]["android"]
        self.assertIn("free space", reason)
        self.assertIn("conversion_record.json", model["blocked_platform_reports"]["android"])
        self.assertNotIn("blocked_platform_logs", model)
        record = self._record()
        self.assertEqual(record["state"], "preflight_skipped")
        self.assertGreater(record["min_free_gb"], record["free_gb"])

    def test_engine_failure_does_not_block_existing_platform_fallback(self) -> None:
        out = self.tmp / "artifacts.yaml"
        base = self.tmp / "base.yaml"
        base.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "models": {
                        "paddle_ocr_vl": {
                            "source_model": "PaddlePaddle/PaddleOCR-VL-1.5",
                            "task": "vlm",
                            "platforms": {
                                "android": {
                                    "engine": "onnx",
                                    "artifact": "model.pipeline.json",
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

        result = self._converter(
            command=["false"],
            out=out,
            allow_conversion_fail=True,
            base_artifacts=base,
        ).run()

        self.assertEqual(result["conversion_failed_count"], 1)
        artifact_map = yaml.safe_load(out.read_text(encoding="utf-8"))
        model = artifact_map["models"]["paddle_ocr_vl"]
        self.assertNotIn("blocked_platforms", model)
        self.assertIn("conversion.log", model["blocked_engines"]["android"]["litert"])
        self.assertIn(
            "conversion_record.json",
            model["blocked_engine_reports"]["android"]["litert"],
        )
        self.assertEqual(
            model["blocked_engine_failure_classes"]["android"]["litert"],
            "conversion_failed",
        )

    def test_timeout_records_conversion_failure(self) -> None:
        out = self.tmp / "artifacts.yaml"
        command = [sys.executable, "-c", "import time; time.sleep(2)"]

        result = self._converter(
            command=command,
            out=out,
            allow_conversion_fail=True,
            timeout_seconds=1,
        ).run()

        self.assertEqual(result["conversion_failed_count"], 1)
        record = self._record()
        self.assertEqual(record["state"], "conversion_failed")
        self.assertTrue(record["timed_out"])
        self.assertEqual(record["timeout_seconds"], 1)
        self.assertIn("timed out after 1s", record["reason"])

    def test_failure_classifies_transformers_api_mismatch(self) -> None:
        out = self.tmp / "artifacts.yaml"
        command = [
            sys.executable,
            "-c",
            "import sys; print(\"ImportError: cannot import name 'check_model_inputs' from 'transformers.utils.generic'\"); sys.exit(1)",
        ]

        result = self._converter(
            command=command,
            out=out,
            allow_conversion_fail=True,
        ).run()

        self.assertEqual(result["conversion_failed_count"], 1)
        record = self._record()
        self.assertEqual(record["failure_class"], "transformers_api_mismatch")
        self.assertIn("Transformers API", record["failure_reason"])
        artifact_map = yaml.safe_load(out.read_text(encoding="utf-8"))
        model = artifact_map["models"]["paddle_ocr_vl"]
        self.assertEqual(
            model["blocked_platform_failure_classes"]["android"],
            "transformers_api_mismatch",
        )
        self.assertIn("Transformers API", model["blocked_platform_failure_reasons"]["android"])

    def test_failure_classifies_onnx2tf_parameter_replacement_required(self) -> None:
        out = self.tmp / "artifacts.yaml"
        command = [
            sys.executable,
            "-c",
            "import sys; print('ERROR: Read this and deal with it. https://github.com/PINTO0309/onnx2tf#parameter-replacement'); sys.exit(1)",
        ]

        result = self._converter(
            command=command,
            out=out,
            allow_conversion_fail=True,
        ).run()

        self.assertEqual(result["conversion_failed_count"], 1)
        record = self._record()
        self.assertEqual(
            record["failure_class"],
            "onnx2tf_parameter_replacement_required",
        )

    def test_returncode_134_classifies_exporter_runtime_crash(self) -> None:
        out = self.tmp / "artifacts.yaml"
        command = [sys.executable, "-c", "import os; os._exit(134)"]

        result = self._converter(
            command=command,
            out=out,
            allow_conversion_fail=True,
        ).run()

        self.assertEqual(result["conversion_failed_count"], 1)
        record = self._record()
        self.assertEqual(record["returncode"], 134)
        self.assertEqual(record["failure_class"], "exporter_runtime_crash")

    def test_failure_classifies_onnx2tf_loop_unsupported(self) -> None:
        out = self.tmp / "artifacts.yaml"
        command = [
            sys.executable,
            "-c",
            "import sys; print(\"ModuleNotFoundError: No module named 'onnx2tf.ops.Loop'\"); print('ERROR: Loop OP is not yet implemented.'); sys.exit(1)",
        ]

        result = self._converter(
            command=command,
            out=out,
            allow_conversion_fail=True,
        ).run()

        self.assertEqual(result["conversion_failed_count"], 1)
        record = self._record()
        self.assertEqual(record["failure_class"], "onnx2tf_unsupported_operator_loop")

    def test_failure_classifies_onnx2tf_sequenceempty_bug(self) -> None:
        out = self.tmp / "artifacts.yaml"
        command = [
            sys.executable,
            "-c",
            "import sys; print(\"File '/tmp/onnx2tf/ops/SequenceEmpty.py'\"); print(\"TypeError: 'dict' object is not callable\"); sys.exit(1)",
        ]

        result = self._converter(
            command=command,
            out=out,
            allow_conversion_fail=True,
        ).run()

        self.assertEqual(result["conversion_failed_count"], 1)
        record = self._record()
        self.assertEqual(record["failure_class"], "onnx2tf_sequenceempty_bug")

    def test_failure_classifies_onnx2tf_unsqueeze_shape_bug(self) -> None:
        out = self.tmp / "artifacts.yaml"
        command = [
            sys.executable,
            "-c",
            (
                "import sys; "
                "print(\"File '/tmp/onnx2tf/ops/Unsqueeze.py', line 131\"); "
                "print(\"UnboundLocalError: cannot access local variable "
                "'input_tensor_shape' where it is not associated with a value\"); "
                "sys.exit(1)"
            ),
        ]

        result = self._converter(
            command=command,
            out=out,
            allow_conversion_fail=True,
        ).run()

        self.assertEqual(result["conversion_failed_count"], 1)
        record = self._record()
        self.assertEqual(record["failure_class"], "onnx2tf_unsqueeze_shape_bug")

    def test_failure_classifies_onnx2tf_concat_dtype_mismatch(self) -> None:
        out = self.tmp / "artifacts.yaml"
        command = [
            sys.executable,
            "-c",
            (
                "import sys; "
                "print(\"NotImplementedError: Concat input dtypes must be compatible "
                "in flatbuffer_direct.\"); "
                "sys.exit(1)"
            ),
        ]

        result = self._converter(
            command=command,
            out=out,
            allow_conversion_fail=True,
        ).run()

        self.assertEqual(result["conversion_failed_count"], 1)
        record = self._record()
        self.assertEqual(record["failure_class"], "onnx2tf_concat_dtype_mismatch")

    def _converter(
        self,
        *,
        command: list[str],
        out: Path,
        allow_conversion_fail: bool,
        min_free_gb: float = 0,
        base_artifacts: Path | None = None,
        timeout_seconds: int | None = None,
    ) -> ArtifactConverter:
        return ArtifactConverter(
            recipes=self._recipes(command, timeout_seconds=timeout_seconds),
            recipes_path=ROOT / "benchmark/runtime/conversion_recipes.yaml",
            base_artifacts=base_artifacts,
            out_path=out,
            output_root=self.tmp / "converted",
            tools_root=self.tmp / "tools",
            model_filter={"paddle_ocr_vl"},
            engine_filter={"litert"},
            platform_filter=set(),
            dry_run=False,
            reuse_existing=False,
            overwrite=True,
            fetch_tools=False,
            artifact_health_check="none",
            allow_health_fail=False,
            allow_conversion_fail=allow_conversion_fail,
            min_free_gb=min_free_gb,
        )

    def _record(self) -> dict:
        return json.loads(
            (
                self.tmp
                / "converted"
                / "paddle_ocr_vl"
                / "litert"
                / "conversion_record.json"
            ).read_text(encoding="utf-8")
        )

    def _recipes(
        self,
        command: list[str],
        *,
        timeout_seconds: int | None = None,
    ) -> dict:
        return {
            "version": 1,
            "output_root": str(self.tmp / "converted"),
            "models": {
                "paddle_ocr_vl": {
                    "source_model": "PaddlePaddle/PaddleOCR-VL-1.5",
                    "task": "vlm",
                    "recipes": {
                        "litert": {
                            "engine": "litert",
                            "exporter": "test",
                            "platforms": ["android"],
                            "artifact_candidates": ["*.tflite"],
                            "command": command,
                            **(
                                {"timeout_seconds": timeout_seconds}
                                if timeout_seconds is not None
                                else {}
                            ),
                        }
                    },
                }
            },
        }


if __name__ == "__main__":
    unittest.main()
