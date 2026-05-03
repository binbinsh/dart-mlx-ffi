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

    def test_recipe_preflight_blocker_records_failure_metadata(self) -> None:
        out = self.tmp / "artifacts.yaml"

        result = self._converter(
            command=[],
            out=out,
            allow_conversion_fail=True,
            recipe_extra={
                "preflight_blocked": True,
                "preflight_failure_class": "coreml_exporter_missing_for_vlm",
                "preflight_failure_reason": (
                    "No production Core ML converter is wired for this VLM bundle."
                ),
            },
        ).run()

        self.assertEqual(result["preflight_skipped_count"], 1)
        artifact_map = yaml.safe_load(out.read_text(encoding="utf-8"))
        model = artifact_map["models"]["paddle_ocr_vl"]
        self.assertIn(
            "No production Core ML converter",
            model["blocked_platforms"]["android"],
        )
        self.assertEqual(
            model["blocked_platform_failure_classes"]["android"],
            "coreml_exporter_missing_for_vlm",
        )
        self.assertIn(
            "VLM bundle",
            model["blocked_platform_failure_reasons"]["android"],
        )
        record = self._record()
        self.assertEqual(record["state"], "preflight_skipped")
        self.assertTrue(record["preflight_blocked"])
        self.assertEqual(
            record["failure_class"],
            "coreml_exporter_missing_for_vlm",
        )

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
        self.assertEqual(record["failure_class"], "conversion_timeout")
        self.assertEqual(record["failure_reason"], "Converter exceeded timeout.")
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

    def test_failure_classifies_dynamic_module_file_missing(self) -> None:
        out = self.tmp / "artifacts.yaml"
        command = [
            sys.executable,
            "-c",
            (
                "import sys; "
                "print(\"  File '/tmp/transformers/utils/hub.py', line 583, in cached_files\"); "
                "print(\"OSError: repo does not appear to have a file named configuration_bailingmm.py\"); "
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
        self.assertEqual(record["failure_class"], "dynamic_module_file_missing")

    def test_failure_classifies_dynamic_module_dependency_missing(self) -> None:
        out = self.tmp / "artifacts.yaml"
        command = [
            sys.executable,
            "-c",
            (
                "import sys; "
                "print(\"ImportError: This modeling file requires the following packages that were not found in your environment: configuration_audio, configuration_bailing_talker\"); "
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
        self.assertEqual(
            record["failure_class"],
            "dynamic_module_dependency_missing",
        )

    def test_failure_classifies_gated_repo_access_denied(self) -> None:
        out = self.tmp / "artifacts.yaml"
        command = [
            sys.executable,
            "-c",
            (
                "import sys; "
                "print('OSError: You are trying to access a gated repo.'); "
                "print('401 Client Error. Access to model is restricted.'); "
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
        self.assertEqual(record["failure_class"], "gated_repo_access_denied")

    def test_failure_classifies_missing_python_dependency(self) -> None:
        out = self.tmp / "artifacts.yaml"
        command = [
            sys.executable,
            "-c",
            (
                "import sys; "
                "print('ImportError: mamba-ssm is required by the Mamba model but cannot be imported'); "
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
        self.assertEqual(record["failure_class"], "converter_dependency_missing")

    def test_failure_classifies_dependency_build_failure(self) -> None:
        out = self.tmp / "artifacts.yaml"
        command = [
            sys.executable,
            "-c",
            (
                "import sys; "
                "print('hint: This usually indicates a problem with the package or the build environment.'); "
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
        self.assertEqual(record["failure_class"], "converter_dependency_build_failed")

    def test_failure_classifies_cuda_toolchain_required(self) -> None:
        out = self.tmp / "artifacts.yaml"
        command = [
            sys.executable,
            "-c",
            (
                "import sys; "
                "print('UserWarning: mamba_ssm was requested, but nvcc was not found.'); "
                "print('hint: This usually indicates a problem with the package or the build environment.'); "
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
        self.assertEqual(
            record["failure_class"],
            "converter_dependency_requires_cuda_toolchain",
        )
        self.assertIn("CUDA/NVCC", record["failure_reason"])

    def test_failure_classifies_unrecognized_automodel_configuration(self) -> None:
        out = self.tmp / "artifacts.yaml"
        command = [
            sys.executable,
            "-c",
            (
                "import sys; "
                "print(\"ValueError: Unrecognized configuration class <class 'x.BailingMMConfig'> for this kind of AutoModel: AutoModelForTextToSpectrogram.\"); "
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
        self.assertEqual(record["failure_class"], "model_architecture_unsupported")

    def test_failure_classifies_minicpm_vlm_mapping_gap(self) -> None:
        out = self.tmp / "artifacts.yaml"
        command = [
            sys.executable,
            "-c",
            (
                "import sys; "
                "print(\"ValueError: Unrecognized configuration class "
                "<class 'configuration_minicpmo.MiniCPMOConfig'> for this kind "
                "of AutoModel: AutoModelForVision2Seq.\"); "
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
        self.assertEqual(record["failure_class"], "vlm_automodel_mapping_missing")

    def test_failure_classifies_unrecognized_transformers_architecture(self) -> None:
        out = self.tmp / "artifacts.yaml"
        command = [
            sys.executable,
            "-c",
            (
                "import sys; "
                "print(\"ValueError: The checkpoint you are trying to load has model type `qwen3_5` but Transformers does not recognize this architecture.\"); "
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
        self.assertEqual(record["failure_class"], "model_architecture_unsupported")

    def test_failure_classifies_export_task_unsupported(self) -> None:
        out = self.tmp / "artifacts.yaml"
        command = [
            sys.executable,
            "-c",
            (
                "import sys; "
                "print('ValueError: only supports the tasks text-generation for gemma3. Please use a supported task.'); "
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
        self.assertEqual(record["failure_class"], "export_task_unsupported")

    def test_failure_classifies_gemma3_vlm_exporter_gap(self) -> None:
        out = self.tmp / "artifacts.yaml"
        command = [
            sys.executable,
            "-c",
            (
                "import sys; "
                "print('ValueError: Asked to export a gemma3 model for the task "
                "image-to-text, but the Optimum ONNX exporter only supports the "
                "tasks feature-extraction, text-generation for gemma3.'); "
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
        self.assertEqual(
            record["failure_class"],
            "vlm_onnx_exporter_missing_for_architecture",
        )

    def test_failure_classifies_unsupported_task_exception(self) -> None:
        out = self.tmp / "artifacts.yaml"
        command = [
            sys.executable,
            "-c",
            (
                "import sys; "
                "print('ValueError: Unsupported task: text_to_audio'); "
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
        self.assertEqual(record["failure_class"], "export_task_unsupported")

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

    def test_failure_classifies_onnx2tf_attempt_timeout(self) -> None:
        out = self.tmp / "artifacts.yaml"
        command = [
            sys.executable,
            "-c",
            (
                "import sys; "
                "print('ERROR: onnx2tf attempt timed out after 900s.'); "
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
        self.assertEqual(record["failure_class"], "onnx2tf_attempt_timeout")

    def test_failure_classifies_onnx2tf_sequence_binding_bug(self) -> None:
        out = self.tmp / "artifacts.yaml"
        command = [
            sys.executable,
            "-c",
            (
                "import sys; "
                "print(\"File '/tmp/onnx2tf/ops/SequenceAt.py', line 79\"); "
                "print(\"KeyError: '/SplitToSequence_output_0'\"); "
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
        self.assertEqual(record["failure_class"], "onnx2tf_sequence_binding_bug")

    def test_failure_classifies_onnx_invalid_graph_topology(self) -> None:
        out = self.tmp / "artifacts.yaml"
        command = [
            sys.executable,
            "-c",
            (
                "import sys; "
                "print(\"Nodes in a graph must be topologically sorted\"); "
                "print(\"input '/SplitToSequence_output_0' of node /SequenceAt\"); "
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
        self.assertEqual(record["failure_class"], "onnx_invalid_graph_topology")

    def test_failure_classifies_invalid_subgraph_constant_binding(self) -> None:
        out = self.tmp / "artifacts.yaml"
        command = [
            sys.executable,
            "-c",
            (
                "import sys; "
                "print('node optimum____if'); "
                "print('Nodes in a graph must be topologically sorted'); "
                "print(\"input '/decoder/Constant_14_output_0' is not output of any previous nodes\"); "
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
        self.assertEqual(
            record["failure_class"],
            "onnx_invalid_subgraph_constant_binding",
        )
        self.assertIn("re-export", record["failure_reason"])

    def test_existing_failed_record_is_reclassified_from_log(self) -> None:
        out = self.tmp / "artifacts.yaml"
        record_dir = self.tmp / "converted" / "paddle_ocr_vl" / "litert"
        record_dir.mkdir(parents=True)
        (record_dir / "conversion.log").write_text(
            "UserWarning: mamba_ssm was requested, but nvcc was not found.\n"
            "hint: This usually indicates a problem with the package or the build environment.\n",
            encoding="utf-8",
        )
        (record_dir / "conversion_record.json").write_text(
            json.dumps(
                {
                    "model_id": "paddle_ocr_vl",
                    "engine": "litert",
                    "source_model": "PaddlePaddle/PaddleOCR-VL-1.5",
                    "task": "vlm",
                    "platforms": ["android"],
                    "state": "conversion_failed",
                    "returncode": 1,
                    "reason": "old reason",
                    "log_path": str(record_dir / "conversion.log"),
                    "failure_class": "converter_dependency_build_failed",
                    "report_path": str(record_dir / "conversion_record.json"),
                }
            ),
            encoding="utf-8",
        )

        result = ArtifactConverter(
            recipes=self._recipes(["false"]),
            recipes_path=ROOT / "benchmark/runtime/conversion_recipes.yaml",
            base_artifacts=None,
            out_path=out,
            output_root=self.tmp / "converted",
            tools_root=self.tmp / "tools",
            model_filter={"paddle_ocr_vl"},
            engine_filter={"litert"},
            platform_filter={"ios"},
            dry_run=False,
            reuse_existing=False,
            overwrite=False,
            fetch_tools=False,
            artifact_health_check="none",
            allow_health_fail=False,
            allow_conversion_fail=True,
            min_free_gb=0,
        ).run()

        self.assertEqual(result["imported_record_count"], 1)
        artifact_map = yaml.safe_load(out.read_text(encoding="utf-8"))
        model = artifact_map["models"]["paddle_ocr_vl"]
        self.assertEqual(
            model["blocked_platform_failure_classes"]["android"],
            "converter_dependency_requires_cuda_toolchain",
        )

    def _converter(
        self,
        *,
        command: list[str],
        out: Path,
        allow_conversion_fail: bool,
        min_free_gb: float = 0,
        base_artifacts: Path | None = None,
        timeout_seconds: int | None = None,
        recipe_extra: dict | None = None,
    ) -> ArtifactConverter:
        return ArtifactConverter(
            recipes=self._recipes(
                command,
                timeout_seconds=timeout_seconds,
                recipe_extra=recipe_extra,
            ),
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
        recipe_extra: dict | None = None,
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
                            **(recipe_extra or {}),
                        }
                    },
                }
            },
        }


if __name__ == "__main__":
    unittest.main()
