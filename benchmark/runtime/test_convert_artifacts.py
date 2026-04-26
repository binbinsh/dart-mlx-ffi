from __future__ import annotations

import json
import shutil
import subprocess
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

from audit import audit
from compare import compare_device_profile
from convert_artifacts import ArtifactConverter, _find_artifact, _normalized_extra_args
from engine_gap_report import build_report as build_engine_gap_report
from ort_env import OrtEnvironment
from prepare_inputs import _align_onnx_inputs, _payload
from promote import build_promotion_patch
from resolve_hf_artifacts import HuggingFaceArtifactResolver
from run_all import build_plan


class ConvertArtifactsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dinf_convert_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_find_artifact_accepts_coreml_bundles(self) -> None:
        out = self.tmp / "out"
        (out / "model.mlpackage").mkdir(parents=True)
        (out / "nested" / "model.mlmodelc").mkdir(parents=True)
        (out / "notes.txt").write_text("not an artifact", encoding="utf-8")

        self.assertEqual(
            _find_artifact(out, ["*.mlpackage"]),
            out / "model.mlpackage",
        )
        self.assertEqual(
            _find_artifact(out, ["**/*.mlmodelc"]),
            out / "nested" / "model.mlmodelc",
        )

    def test_find_artifact_accepts_coreml_pipeline_specs(self) -> None:
        out = self.tmp / "out"
        out.mkdir()
        (out / "conversion_report.json").write_text("{}", encoding="utf-8")
        (out / "pipeline.json").write_text(
            json.dumps(
                {
                    "format": "dart_inference.coreml_pipeline.v1",
                    "stages": [],
                }
            ),
            encoding="utf-8",
        )

        self.assertEqual(
            _find_artifact(out, ["*.json"]),
            out / "pipeline.json",
        )

    def test_coreml_llm_extra_args_drop_unsupported_remote_code_flag(self) -> None:
        kept, ignored = _normalized_extra_args(
            "coreml-llm",
            ["--trust-remote-code", "--other"],
        )

        self.assertEqual(kept, ["--other"])
        self.assertEqual(ignored, ["--trust-remote-code"])

    def test_reuse_existing_writes_converted_overlay(self) -> None:
        recipes = {
            "version": 1,
            "output_root": str(self.tmp / "converted"),
            "models": {
                "glm4_7_flash": {
                    "source_model": "zai-org/GLM-4.7-Flash",
                    "task": "text",
                    "recipes": {
                        "coreml": {
                            "engine": "coreml",
                            "exporter": "test",
                            "platforms": ["ios", "macos"],
                            "artifact_candidates": ["*.mlpackage"],
                            "command": ["false"],
                        }
                    },
                }
            },
        }
        artifact = self.tmp / "converted" / "glm4_7_flash" / "coreml" / "model.mlpackage"
        artifact.mkdir(parents=True)
        out = self.tmp / "artifacts.yaml"

        result = ArtifactConverter(
            recipes=recipes,
            recipes_path=ROOT / "benchmark/runtime/conversion_recipes.yaml",
            base_artifacts=None,
            out_path=out,
            output_root=self.tmp / "converted",
            tools_root=self.tmp / "tools",
            model_filter={"glm4_7_flash"},
            engine_filter={"coreml"},
            platform_filter=set(),
            dry_run=False,
            reuse_existing=True,
            overwrite=False,
            fetch_tools=False,
            artifact_health_check="none",
            allow_health_fail=False,
            allow_conversion_fail=False,
            min_free_gb=0,
        ).run()

        self.assertEqual(result["reused_count"], 1)
        artifact_map = yaml.safe_load(out.read_text(encoding="utf-8"))
        cell = artifact_map["models"]["glm4_7_flash"]["platforms"]["ios"]
        self.assertEqual(cell["artifact_source"], "converted")
        self.assertTrue(cell["unblock_platform"])
        self.assertEqual(cell["engine"], "coreml")

    def test_reuse_existing_health_check_records_report_before_unblocking(self) -> None:
        recipes = {
            "version": 1,
            "output_root": str(self.tmp / "converted"),
            "models": {
                "glm4_7_flash": {
                    "source_model": "zai-org/GLM-4.7-Flash",
                    "task": "text",
                    "recipes": {
                        "litert": {
                            "engine": "litert",
                            "exporter": "test",
                            "platforms": ["android"],
                            "delegate_by_platform": {"android": "xnnpack"},
                            "artifact_candidates": ["*.tflite"],
                            "command": ["false"],
                        }
                    },
                }
            },
        }
        artifact = self.tmp / "converted" / "glm4_7_flash" / "litert" / "model.tflite"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"TFL3")
        out = self.tmp / "artifacts.yaml"
        with mock.patch("convert_artifacts.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess(
                args=["python"],
                returncode=0,
            )
            result = ArtifactConverter(
                recipes=recipes,
                recipes_path=ROOT / "benchmark/runtime/conversion_recipes.yaml",
                base_artifacts=None,
                out_path=out,
                output_root=self.tmp / "converted",
                tools_root=self.tmp / "tools",
                model_filter={"glm4_7_flash"},
                engine_filter={"litert"},
                platform_filter=set(),
                dry_run=False,
                reuse_existing=True,
                overwrite=False,
                fetch_tools=False,
                artifact_health_check="run",
                allow_health_fail=False,
                allow_conversion_fail=False,
                min_free_gb=0,
            ).run()

        command = run.call_args.args[0]
        self.assertEqual(result["reused_count"], 1)
        self.assertIn("artifact_health.py", " ".join(command))
        artifact_map = yaml.safe_load(out.read_text(encoding="utf-8"))
        cell = artifact_map["models"]["glm4_7_flash"]["platforms"]["android"]
        self.assertTrue(cell["artifact_health_passed"])
        self.assertIn("artifact_health_report", cell)
        self.assertEqual(cell["delegate"], "xnnpack")

    def test_converted_overlay_unblocks_run_plan(self) -> None:
        artifact = self.tmp / "converted" / "glm4_7_flash" / "litert" / "model.tflite"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"TFL3")
        artifact_map = {
            "version": 1,
            "defaults": {
                "input_json": "benchmark/runtime/fixtures/tiny_input.json",
                "platforms": {"android": {"delegate": "xnnpack"}},
            },
            "models": {
                "glm4_7_flash": {
                    "source_model": "zai-org/GLM-4.7-Flash",
                    "task": "text",
                    "artifact_coverage": "converted",
                    "platforms": {
                        "android": {
                            "engine": "litert",
                            "artifact": str(artifact),
                            "artifact_source": "converted",
                            "unblock_platform": True,
                        }
                    },
                }
            },
        }
        artifacts_path = self.tmp / "artifacts.yaml"
        artifacts_path.write_text(
            yaml.safe_dump(artifact_map, sort_keys=False),
            encoding="utf-8",
        )

        plan = build_plan(
            Namespace(
                config=ROOT / "benchmark/runtime/models.yaml",
                artifacts=artifacts_path,
                model_id="glm4_7_flash",
                platform="android",
                engine="litert",
                out_root=self.tmp / "out",
                plan_out=None,
                run=False,
                allow_fail=True,
                dry_run=False,
                path_check="always",
                execution_check="none",
            )
        )

        self.assertEqual(plan["blocked_count"], 0)
        self.assertEqual(plan["ready_count"], 1)
        self.assertEqual(plan["cells"][0]["state"], "ready")
        self.assertEqual(plan["cells"][0]["artifact_coverage"], "converted")

    def test_required_fixture_defaults_make_vlm_plan_ready(self) -> None:
        artifact = self.tmp / "model.mlpackage"
        artifact.mkdir()
        artifact_map = {
            "version": 1,
            "defaults": {
                "input_json": "benchmark/runtime/fixtures/tiny_input.json",
                "prompt_file": "benchmark/runtime/fixtures/text_prompt.txt",
                "image_file": "benchmark/runtime/fixtures/image.png",
                "audio_file": "benchmark/runtime/fixtures/audio.wav",
                "platforms": {"macos": {"coreml_mode": "decode"}},
            },
            "models": {
                "qwen3_vl": {
                    "task": "vlm",
                    "required_fixtures": ["text_prompt", "image"],
                    "platforms": {
                        "macos": {
                            "engine": "coreml",
                            "artifact": str(artifact),
                            "baseline_engine": "coreml",
                        }
                    },
                }
            },
        }
        artifacts_path = self.tmp / "artifacts.yaml"
        artifacts_path.write_text(
            yaml.safe_dump(artifact_map, sort_keys=False),
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
                run=False,
                allow_fail=True,
                dry_run=False,
                path_check="always",
                execution_check="none",
            )
        )

        self.assertEqual(plan["blocked_count"], 0)
        self.assertEqual(plan["ready_count"], 1)

    def test_required_fixture_missing_blocks_plan(self) -> None:
        artifact = self.tmp / "model.mlpackage"
        artifact.mkdir()
        artifact_map = {
            "version": 1,
            "defaults": {
                "input_json": "benchmark/runtime/fixtures/tiny_input.json",
                "prompt_file": "benchmark/runtime/fixtures/text_prompt.txt",
            },
            "models": {
                "qwen3_vl": {
                    "task": "vlm",
                    "required_fixtures": ["text_prompt", "image"],
                    "platforms": {
                        "macos": {
                            "engine": "coreml",
                            "artifact": str(artifact),
                            "baseline_engine": "coreml",
                        }
                    },
                }
            },
        }
        artifacts_path = self.tmp / "artifacts.yaml"
        artifacts_path.write_text(
            yaml.safe_dump(artifact_map, sort_keys=False),
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
                run=False,
                allow_fail=True,
                dry_run=False,
                path_check="none",
                execution_check="none",
            )
        )

        self.assertEqual(plan["ready_count"], 0)
        self.assertEqual(plan["blocked_count"], 1)
        self.assertIn("Missing required image fixture", plan["cells"][0]["reasons"])

    def test_input_sidecar_missing_blocks_plan(self) -> None:
        artifact = self.tmp / "model.onnx"
        artifact.write_bytes(b"onnx")
        input_json = self.tmp / "input.json"
        input_json.write_text(
            """
{
  "inputs": {
    "image": {
      "dtype": "uint8",
      "shape": [3],
      "file": "missing.bin"
    }
  }
}
""".strip()
            + "\n",
            encoding="utf-8",
        )
        artifact_map = {
            "version": 1,
            "defaults": {"input_json": str(input_json)},
            "models": {
                "qwen3_5": {
                    "task": "tensor",
                    "platforms": {
                        "linux": {
                            "engine": "onnx",
                            "artifact": str(artifact),
                            "baseline_engine": "onnx",
                        }
                    },
                }
            },
        }
        artifacts_path = self.tmp / "artifacts.yaml"
        artifacts_path.write_text(
            yaml.safe_dump(artifact_map, sort_keys=False),
            encoding="utf-8",
        )

        plan = build_plan(
            Namespace(
                config=ROOT / "benchmark/runtime/models.yaml",
                artifacts=artifacts_path,
                model_id="qwen3_5",
                platform="linux",
                engine="onnx",
                out_root=self.tmp / "out",
                plan_out=None,
                run=False,
                allow_fail=True,
                dry_run=False,
                path_check="always",
                execution_check="none",
            )
        )

        self.assertEqual(plan["ready_count"], 0)
        self.assertEqual(plan["blocked_count"], 1)
        self.assertIn("Missing input sidecar for image", plan["cells"][0]["reasons"][0])

    def test_prepare_inputs_plan_does_not_require_existing_input_json(self) -> None:
        artifact = self.tmp / "model.onnx"
        artifact.write_bytes(b"onnx")
        artifact_map = {
            "version": 1,
            "defaults": {"input_json": str(self.tmp / "generated.json")},
            "models": {
                "qwen3_5": {
                    "source_model": "hf-internal-testing/tiny-random-gpt2",
                    "task": "text",
                    "platforms": {
                        "linux": {
                            "engine": "onnx",
                            "artifact": str(artifact),
                            "baseline_engine": "onnx",
                        }
                    },
                }
            },
        }
        artifacts_path = self.tmp / "artifacts.yaml"
        artifacts_path.write_text(
            yaml.safe_dump(artifact_map, sort_keys=False),
            encoding="utf-8",
        )

        plan = build_plan(
            Namespace(
                config=ROOT / "benchmark/runtime/models.yaml",
                artifacts=artifacts_path,
                model_id="qwen3_5",
                platform="linux",
                engine="onnx",
                out_root=self.tmp / "out",
                plan_out=None,
                run=True,
                prepare_inputs=True,
                allow_fail=True,
                dry_run=False,
                path_check="always",
                execution_check="none",
            )
        )

        self.assertEqual(plan["blocked_count"], 0)
        self.assertEqual(plan["ready_count"], 1)
        self.assertTrue(plan["cells"][0]["prepare_input_command"])
        self.assertIn("--onnx-artifact", plan["cells"][0]["prepare_input_command"])

    def test_prepare_inputs_plan_passes_onnx_pipeline_json(self) -> None:
        artifact = self.tmp / "pipeline.json"
        artifact.write_text(
            json.dumps({"format": "dart_inference.onnx_pipeline.v1", "stages": []}),
            encoding="utf-8",
        )
        artifact_map = {
            "version": 1,
            "models": {
                "qwen3_5": {
                    "source_model": "hf-internal-testing/tiny-random-gpt2",
                    "task": "text",
                    "platforms": {
                        "linux": {
                            "engine": "onnx",
                            "artifact": str(artifact),
                            "baseline_engine": "onnx",
                        }
                    },
                }
            },
        }
        artifacts_path = self.tmp / "artifacts.yaml"
        artifacts_path.write_text(
            yaml.safe_dump(artifact_map, sort_keys=False),
            encoding="utf-8",
        )

        plan = build_plan(
            Namespace(
                config=ROOT / "benchmark/runtime/models.yaml",
                artifacts=artifacts_path,
                model_id="qwen3_5",
                platform="linux",
                engine="onnx",
                out_root=self.tmp / "out",
                plan_out=None,
                run=True,
                prepare_inputs=True,
                allow_fail=True,
                dry_run=False,
                path_check="always",
                execution_check="none",
            )
        )

        command = plan["cells"][0]["prepare_input_command"]
        self.assertIn("--onnx-artifact", command)
        self.assertIn(str(artifact), command)

    def test_vlm_prepare_inputs_uses_group_and_does_not_truncate_by_default(self) -> None:
        artifact = self.tmp / "pipeline.json"
        artifact.write_text(
            json.dumps({"format": "dart_inference.onnx_pipeline.v1", "stages": []}),
            encoding="utf-8",
        )
        artifact_map = {
            "version": 1,
            "defaults": {"max_tokens": 64},
            "models": {
                "paddle_ocr_vl": {
                    "source_model": "PaddlePaddle/PaddleOCR-VL-1.5",
                    "task": "vlm",
                    "platforms": {
                        "linux": {
                            "engine": "onnx",
                            "artifact": str(artifact),
                            "baseline_engine": "onnx",
                        }
                    },
                }
            },
        }
        artifacts_path = self.tmp / "artifacts.yaml"
        artifacts_path.write_text(
            yaml.safe_dump(artifact_map, sort_keys=False),
            encoding="utf-8",
        )

        plan = build_plan(
            Namespace(
                config=ROOT / "benchmark/runtime/models.yaml",
                artifacts=artifacts_path,
                model_id="paddle_ocr_vl",
                platform="linux",
                engine="onnx",
                out_root=self.tmp / "out",
                plan_out=None,
                run=True,
                prepare_inputs=True,
                allow_fail=True,
                dry_run=False,
                path_check="always",
                execution_check="none",
            )
        )

        command = plan["cells"][0]["prepare_input_command"]
        self.assertEqual(command[:4], ["uv", "run", "--group", "vlm-prepare"])
        self.assertNotIn("--max-length", command)

    def test_vlm_auto_prepare_inputs_when_tiny_fixture_is_selected(self) -> None:
        artifact = self.tmp / "pipeline.json"
        artifact.write_text(
            json.dumps({"format": "dart_inference.onnx_pipeline.v1", "stages": []}),
            encoding="utf-8",
        )
        artifact_map = {
            "version": 1,
            "defaults": {"input_json": "benchmark/runtime/fixtures/tiny_input.json"},
            "models": {
                "paddle_ocr_vl": {
                    "source_model": "PaddlePaddle/PaddleOCR-VL-1.5",
                    "task": "vlm",
                    "platforms": {"linux": {"engine": "onnx", "artifact": str(artifact)}},
                }
            },
        }
        artifacts_path = self.tmp / "artifacts.yaml"
        artifacts_path.write_text(
            yaml.safe_dump(artifact_map, sort_keys=False),
            encoding="utf-8",
        )
        plan = build_plan(
            Namespace(
                config=ROOT / "benchmark/runtime/models.yaml",
                artifacts=artifacts_path,
                model_id="paddle_ocr_vl",
                platform="linux",
                engine="onnx",
                out_root=self.tmp / "out",
                plan_out=None,
                run=True,
                prepare_inputs=False,
                allow_fail=True,
                dry_run=False,
                path_check="always",
                execution_check="none",
            )
        )
        self.assertTrue(plan["cells"][0]["prepare_input_command"])
        self.assertTrue(plan["cells"][0]["command"])

    def test_run_plan_includes_artifact_health_command_for_local_onnx(self) -> None:
        artifact = self.tmp / "model.onnx"
        artifact.write_bytes(b"onnx")
        artifact_map = {
            "version": 1,
            "models": {
                "qwen3_5": {
                    "source_model": "hf-internal-testing/tiny-random-gpt2",
                    "task": "text",
                    "platforms": {
                        "linux": {
                            "engine": "onnx",
                            "artifact": str(artifact),
                            "baseline_engine": "onnx",
                        }
                    },
                }
            },
        }
        artifacts_path = self.tmp / "artifacts.yaml"
        artifacts_path.write_text(
            yaml.safe_dump(artifact_map, sort_keys=False),
            encoding="utf-8",
        )

        plan = build_plan(
            Namespace(
                config=ROOT / "benchmark/runtime/models.yaml",
                artifacts=artifacts_path,
                model_id="qwen3_5",
                platform="linux",
                engine="onnx",
                out_root=self.tmp / "out",
                plan_out=None,
                run=True,
                prepare_inputs=False,
                allow_fail=True,
                dry_run=False,
                path_check="always",
                execution_check="none",
                artifact_health_check="always",
            )
        )

        command = plan["cells"][0]["artifact_health_command"]
        self.assertEqual(command[:4], ["uv", "run", "--group", "onnx-convert"])
        self.assertTrue(any(item.endswith("artifact_health.py") for item in command))
        self.assertIn("--platform", command)
        self.assertIn("linux", command)
        self.assertIn("--artifact", command)
        self.assertIn(str(artifact), command)
        self.assertTrue(
            any(
                str(item).startswith(str(self.tmp / "out" / "_artifact_health"))
                for item in command
            )
        )


class ConverterCommandSmokeTest(unittest.TestCase):
    def test_onnx_exporter_command_is_available(self) -> None:
        completed = subprocess.run(
            [
                "uv",
                "run",
                "--group",
                "onnx-convert",
                "--with",
                "transformers<5",
                "--with",
                "torch>=2.11.0",
                "--with",
                "torchvision>=0.26.0",
                "optimum-cli",
                "export",
                "onnx",
                "--help",
            ],
            cwd=ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("usage: optimum-cli export onnx", completed.stdout)


class PrepareInputsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dinf_prepare_inputs_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_payload_writes_large_tensor_sidecar(self) -> None:
        import numpy as np

        out = self.tmp / "input.json"
        payload = _payload(
            {"pixel_values": np.arange(8, dtype=np.float32).reshape(1, 2, 4)},
            out=out,
            metadata={"model_id": "tiny"},
            sidecar_threshold=4,
        )

        spec = payload["inputs"]["pixel_values"]
        self.assertEqual(spec["dtype"], "float32")
        self.assertEqual(spec["shape"], [1, 2, 4])
        self.assertEqual(spec["file"], "input/pixel_values.float32.bin")
        self.assertTrue((self.tmp / spec["file"]).exists())

    def test_payload_keeps_small_tensor_inline(self) -> None:
        import numpy as np

        payload = _payload(
            {"input_ids": np.array([[1, 2]], dtype=np.int64)},
            out=self.tmp / "input.json",
            metadata={},
            sidecar_threshold=4,
        )

        spec = payload["inputs"]["input_ids"]
        self.assertEqual(spec["dtype"], "int64")
        self.assertEqual(spec["values"], [[1, 2]])

    def test_payload_casts_unsupported_integer_sidecar_dtype(self) -> None:
        import numpy as np

        out = self.tmp / "input.json"
        payload = _payload(
            {"ids": np.arange(8, dtype=np.int16)},
            out=out,
            metadata={},
            sidecar_threshold=4,
        )

        spec = payload["inputs"]["ids"]
        sidecar = self.tmp / spec["file"]
        self.assertEqual(spec["dtype"], "int64")
        self.assertEqual(sidecar.stat().st_size, 8 * 8)

    def test_align_onnx_pipeline_derives_scatter_and_bool_inputs(self) -> None:
        try:
            import numpy as np
            import onnx
            from onnx import TensorProto, helper
        except ImportError:
            self.skipTest("onnx is not installed")

        decoder = self.tmp / "decoder.onnx"
        graph = helper.make_graph(
            [helper.make_node("Identity", ["inputs_embeds"], ["logits"])],
            "decoder",
            [
                helper.make_tensor_value_info(
                    "inputs_embeds",
                    TensorProto.FLOAT,
                    ["batch_size", "sequence_length", 4],
                ),
                helper.make_tensor_value_info(
                    "use_cache_branch",
                    TensorProto.BOOL,
                    [1],
                ),
            ],
            [
                helper.make_tensor_value_info(
                    "logits",
                    TensorProto.FLOAT,
                    ["batch_size", "sequence_length", 4],
                )
            ],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
        model.ir_version = 10
        onnx.save(model, decoder)
        pipeline = self.tmp / "pipeline.json"
        pipeline.write_text(
            json.dumps(
                {
                    "format": "dart_inference.onnx_pipeline.v1",
                    "stages": [
                        {
                            "name": "merge",
                            "op": "scatter_embeddings",
                            "outputs": {"output": "inputs_embeds"},
                        },
                        {"name": "decoder", "model": str(decoder)},
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        aligned = _align_onnx_inputs(
            {"mm_token_type_ids": np.array([[0, 1, 1, 0]], dtype=np.int32)},
            pipeline,
            batch_size=1,
            seq_length=4,
            past_length=0,
        )

        self.assertEqual(aligned["image_token_indices"].tolist(), [1, 2])
        self.assertEqual(aligned["use_cache_branch"].dtype, np.bool_)
        self.assertEqual(aligned["use_cache_branch"].tolist(), [False])
        self.assertNotIn("inputs_embeds", aligned)


class OrtEnvTest(unittest.TestCase):
    def test_ort_environment_formats_build_variables(self) -> None:
        env = OrtEnvironment(
            version="1.25.0",
            include_dir=Path("/tmp/include"),
            library=Path("/tmp/libonnxruntime.dylib"),
            runtime_library=Path("/tmp/libonnxruntime.dylib"),
            package_root=Path("/tmp/onnxruntime"),
        )

        values = env.to_env()
        self.assertEqual(values["DART_INFERENCE_ENABLE_ORT"], "1")
        self.assertEqual(values["DART_INFERENCE_ORT_INCLUDE_DIR"], "/tmp/include")
        self.assertEqual(values["DART_INFERENCE_ORT_LIBRARY"], "/tmp/libonnxruntime.dylib")
        self.assertTrue(env.ready)


class EngineGapTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dinf_resolver_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_resolver_reports_preferred_engine_fallbacks(self) -> None:
        catalog = _fallback_catalog()
        resolver = HuggingFaceArtifactResolver(
            catalog=catalog,
            catalog_path=ROOT / "benchmark/runtime/hf_artifacts.yaml",
            out_path=ROOT / "benchmark/runtime/artifacts.local.yaml",
            cache_root=None,
            model_filter={"paddle_ocr_vl"},
            platform_filter=set(),
            engine_filter=set(),
            local_files_only=True,
            allow_missing=True,
        )

        plan = resolver.plan()
        by_platform = {cell["platform"]: cell for cell in plan["cells"]}

        self.assertEqual(by_platform["ios"]["engine"], "mlx")
        self.assertEqual(by_platform["ios"]["fallback_from"], ["coreml"])
        self.assertEqual(by_platform["android"]["engine"], "onnx")
        self.assertEqual(by_platform["android"]["fallback_from"], ["litert"])

    def test_resolver_preserves_component_artifacts(self) -> None:
        catalog = _fallback_catalog()
        resolver = HuggingFaceArtifactResolver(
            catalog=catalog,
            catalog_path=ROOT / "benchmark/runtime/hf_artifacts.yaml",
            out_path=ROOT / "benchmark/runtime/artifacts.local.yaml",
            cache_root=None,
            model_filter={"paddle_ocr_vl"},
            platform_filter={"linux"},
            engine_filter=set(),
            local_files_only=True,
            allow_missing=True,
        )

        plan = resolver.plan()

        cell = plan["cells"][0]
        self.assertEqual(cell["engine"], "onnx")
        self.assertEqual(
            cell["component_artifacts"]["vision_encoder"],
            "onnx/vision_encoder.onnx",
        )

    def test_resolver_writes_pipeline_specs_from_components(self) -> None:
        artifact_dir = self.tmp / "snapshots" / "demo"
        (artifact_dir / "onnx").mkdir(parents=True)
        for name in ("embed.onnx", "decoder.onnx"):
            (artifact_dir / "onnx" / name).write_bytes(b"onnx")
        catalog = {
            "support_policy": {"production_requires": {"platforms": ["linux"]}},
            "engine_platforms": {"onnx": ["linux"]},
            "engine_order": {"linux": ["onnx"]},
            "models": {
                "tiny_pipeline": {
                    "family": "Tiny",
                    "source_model": "acme/tiny",
                    "artifacts": {
                        "onnx": {
                            "repo": "acme/tiny",
                            "artifact": "onnx/decoder.onnx",
                            "component_artifacts": {
                                "embed": "onnx/embed.onnx",
                                "decoder": "onnx/decoder.onnx",
                            },
                            "pipeline": {
                                "format": "dart_inference.onnx_pipeline.v1",
                                "stages": [
                                    {
                                        "name": "embed",
                                        "model": "{component:embed}",
                                        "outputs": {"output": "hidden"},
                                    },
                                    {
                                        "name": "decoder",
                                        "model": "{component:decoder}",
                                        "inputs": {"input": "hidden"},
                                    },
                                ],
                                "outputs": ["logits"],
                            },
                        }
                    },
                }
            },
        }
        resolver = _LocalResolver(
            catalog=catalog,
            catalog_path=ROOT / "benchmark/runtime/hf_artifacts.yaml",
            out_path=self.tmp / "artifacts.yaml",
            cache_root=self.tmp / "cache",
            model_filter={"tiny_pipeline"},
            platform_filter={"linux"},
            engine_filter=set(),
            local_files_only=True,
            allow_missing=True,
            snapshot_root=artifact_dir,
        )

        artifact_map = resolver.resolve()

        cell = artifact_map["models"]["tiny_pipeline"]["platforms"]["linux"]
        pipeline_path = Path(cell["artifact"])
        self.assertEqual(pipeline_path.suffix, ".json")
        self.assertTrue(pipeline_path.exists())
        payload = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["format"], "dart_inference.onnx_pipeline.v1")
        self.assertEqual(
            payload["stages"][0]["model"],
            str(artifact_dir / "onnx" / "embed.onnx"),
        )
        self.assertEqual(payload["stages"][1]["inputs"]["input"], "hidden")
        self.assertEqual(
            cell["component_artifacts_resolved"]["decoder"],
            str(artifact_dir / "onnx" / "decoder.onnx"),
        )

    def test_engine_gap_report_includes_conversion_commands(self) -> None:
        report = build_engine_gap_report(
            _fallback_catalog(),
            recipes={
                "models": {
                    "paddle_ocr_vl": {
                        "recipes": {
                            "coreml": {"preset": "coreml_coremlllm"},
                            "litert": {"preset": "litert_hf_vlm"},
                        }
                    }
                }
            },
            catalog_path=ROOT / "benchmark/runtime/hf_artifacts.yaml",
            recipes_path=ROOT / "benchmark/runtime/conversion_recipes.yaml",
            model_filter={"paddle_ocr_vl"},
            platform_filter=set(),
            engine_filter=set(),
        )
        by_platform = {cell["platform"]: cell for cell in report["cells"]}

        self.assertEqual(report["fallback_ready_count"], 2)
        self.assertEqual(by_platform["ios"]["selected_engine"], "mlx")
        self.assertEqual(by_platform["ios"]["missing_preferred_engines"], ["coreml"])
        self.assertIn("coreml", by_platform["ios"]["conversion"])
        self.assertEqual(by_platform["android"]["selected_engine"], "onnx")
        self.assertEqual(
            by_platform["android"]["missing_preferred_engines"],
            ["litert"],
        )
        self.assertIn("litert", by_platform["android"]["conversion"])


class _LocalResolver(HuggingFaceArtifactResolver):
    def __init__(self, *args, snapshot_root: Path, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._snapshot_root = snapshot_root

    def _download(self, artifact: dict) -> Path:
        artifact_name = str(artifact.get("artifact") or ".")
        if artifact_name == ".":
            return self._snapshot_root
        return self._snapshot_root / artifact_name


class PromotionFallbackGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dinf_promotion_gate_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_fallback_artifacts_block_audit_and_promotion(self) -> None:
        config_path = self.tmp / "models.yaml"
        artifacts_path = self.tmp / "artifacts.yaml"
        out_root = self.tmp / "out"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "support_policy": {
                        "production_requires": {
                            "platforms": ["ios", "android"],
                        }
                    },
                    "first_wave": [
                        {
                            "id": "paddle_ocr_vl",
                            "family": "PaddleOCR-VL",
                            "support_level": "staging",
                        }
                    ],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        artifacts_path.write_text(
            yaml.safe_dump(
                {
                    "models": {
                        "paddle_ocr_vl": {
                            "platforms": {
                                "ios": {
                                    "engine": "mlx",
                                    "artifact": "hf://example/mlx",
                                    "fallback_from": ["coreml"],
                                },
                                "android": {
                                    "engine": "onnx",
                                    "artifact": "hf://example/onnx",
                                    "fallback_from": ["litert"],
                                },
                            }
                        }
                    }
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        _write_passing_verdict(out_root, "paddle_ocr_vl", "ios", "mlx")
        _write_passing_verdict(out_root, "paddle_ocr_vl", "android", "onnx")

        audit_payload = audit(
            config_path=config_path,
            out_root=out_root,
            artifacts_path=artifacts_path,
            model_id="paddle_ocr_vl",
        )
        model_audit = audit_payload["models"][0]
        self.assertFalse(model_audit["production_ready"])
        self.assertEqual(model_audit["blocked_count"], 2)
        self.assertIn(
            "artifact_fallback",
            model_audit["platforms"][0]["failed_checks"][0],
        )

        patch = build_promotion_patch(
            config_path=config_path,
            out_root=out_root,
            artifacts_path=artifacts_path,
            model_id="paddle_ocr_vl",
        )
        model_patch = patch["models"][0]
        self.assertEqual(model_patch["supportLevel"], "staging")
        self.assertFalse(model_patch["productionReady"])
        self.assertIn("coreml", model_patch["blockedPlatforms"]["ios"])
        self.assertIn("litert", model_patch["blockedPlatforms"]["android"])


class CompareDeviceProfileTest(unittest.TestCase):
    def test_onnx_pipeline_provider_checks_all_stages(self) -> None:
        result = compare_device_profile(
            {
                "engine": "onnx",
                "device_profile": {
                    "runtime_diagnostics": {
                        "pipeline": True,
                        "stages": [
                            {"diagnostics": {"provider": "CPUExecutionProvider"}},
                            {"diagnostics": {"provider": "CPUExecutionProvider"}},
                        ],
                    }
                },
            },
            {
                "require_device_profile": True,
                "required_provider": "cpu",
            },
        )

        self.assertTrue(result["passed"])


def _fallback_catalog() -> dict:
    return {
        "support_policy": {
            "production_requires": {"platforms": ["ios", "android"]}
        },
        "engine_platforms": {
            "mlx": ["ios", "macos"],
            "coreml": ["ios", "macos"],
            "onnx": ["ios", "macos", "windows", "linux", "android"],
            "litert": ["android"],
        },
        "engine_order": {
            "ios": ["coreml", "mlx", "onnx"],
            "android": ["litert", "onnx"],
        },
        "models": {
            "paddle_ocr_vl": {
                "family": "PaddleOCR-VL",
                "source_model": "PaddlePaddle/PaddleOCR-VL-1.5",
                "artifacts": {
                    "mlx": {
                        "repo": "mlx-community/PaddleOCR-VL-1.5-8bit",
                        "artifact": ".",
                    },
                    "onnx": {
                        "repo": "lbm364dl/PaddleOCR-VL-1.5-ONNX",
                        "artifact": "onnx/decoder_model_merged.onnx",
                        "component_artifacts": {
                            "embed_tokens": "onnx/embed_tokens.onnx",
                            "vision_encoder": "onnx/vision_encoder.onnx",
                            "decoder": "onnx/decoder_model_merged.onnx",
                        },
                    },
                },
            }
        },
    }


def _write_passing_verdict(
    out_root: Path,
    model_id: str,
    platform: str,
    engine: str,
) -> None:
    path = out_root / model_id / platform / "verdict.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json_dump(
            {
                "verdict": {
                    "passed": True,
                    "correctness": {"passed": True, "checks": []},
                    "speed": {"passed": True, "checks": []},
                    "peak_memory": {"passed": True, "checks": []},
                    "device_profile": {"passed": True, "checks": []},
                },
                "candidate": {
                    "model_id": model_id,
                    "platform": platform,
                    "engine": engine,
                    "metrics": {"peak_memory_bytes": 100},
                    "device_profile": {},
                },
                "baseline": {
                    "model_id": model_id,
                    "platform": platform,
                    "engine": engine,
                    "metrics": {"peak_memory_bytes": 100},
                },
            }
        ),
        encoding="utf-8",
    )


def json_dump(value: dict) -> str:
    import json

    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    unittest.main()
