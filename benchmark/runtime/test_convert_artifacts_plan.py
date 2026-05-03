from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from typing import Any
from unittest import mock

import yaml

RUNTIME_DIR = Path(__file__).resolve().parent
ROOT = RUNTIME_DIR.parents[1]
sys.path.insert(0, str(RUNTIME_DIR))

from audit import audit
from compare import compare_device_profile
from convert_artifacts import (
    ArtifactConverter,
    _converter_cache_env,
    _expand_command,
    _find_artifact,
    _normalized_extra_args,
    _normalized_with_packages,
)
from engine_gap_report import build_report as build_engine_gap_report
from ort_env import OrtEnvironment
from prepare_inputs import _align_onnx_inputs, _payload
from promote import build_promotion_patch
from resolve_hf_artifacts import HuggingFaceArtifactResolver
from run_all import build_plan
from convert_artifacts_test_support import _fallback_catalog


class ConvertArtifactsPlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dmf_convert_plan_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_converted_overlay_unblocks_run_plan(self) -> None:
        artifact = self.tmp / "converted" / "qwen3_6_27b" / "litert" / "model.tflite"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"TFL3")
        artifact_map = {
            "version": 1,
            "defaults": {
                "input_json": "benchmark/runtime/fixtures/tiny_input.json",
                "platforms": {"android": {"delegate": "xnnpack"}},
            },
            "models": {
                "qwen3_6_27b": {
                    "source_model": "Qwen/Qwen3.6-27B",
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
                model_id="qwen3_6_27b",
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

    def test_blocked_engine_blocks_run_plan(self) -> None:
        artifact_map = {
            "version": 1,
            "defaults": {
                "input_json": "benchmark/runtime/fixtures/tiny_input.json",
            },
            "models": {
                "qwen2_5": {
                    "source_model": "Qwen/Qwen2.5-0.5B-Instruct",
                    "task": "text",
                    "blocked_engines": {
                        "android": {
                            "onnx": "ONNX artifact is component-only.",
                        }
                    },
                    "platforms": {
                        "android": {
                            "engine": "onnx",
                            "artifact": "hf://onnx-community/Qwen2.5-0.5B-Instruct/onnx/model_q4f16.onnx",
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
                model_id="qwen2_5",
                platform="android",
                engine="onnx",
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
        self.assertIn("component-only", plan["cells"][0]["reasons"][0])

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
            json.dumps({"format": "dart_mlx_ffi.onnx_pipeline.v1", "stages": []}),
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
            json.dumps({"format": "dart_mlx_ffi.onnx_pipeline.v1", "stages": []}),
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
            json.dumps({"format": "dart_mlx_ffi.onnx_pipeline.v1", "stages": []}),
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
        env = {
            key: value
            for key, value in os.environ.items()
            if key
            not in {
                "VIRTUAL_ENV",
                "PYTHONPATH",
                "PYTHONHOME",
                "CONDA_PREFIX",
                "__PYVENV_LAUNCHER__",
                "PYTHONEXECUTABLE",
            }
        }
        completed = subprocess.run(
            [
                "uvx",
                "--from",
                "optimum[onnxruntime]==2.1.0",
                "--with",
                "accelerate>=1.13.0",
                "--with",
                "onnx>=1.21.0",
                "--with",
                "onnxruntime>=1.25.0",
                "--with",
                "transformers>=4.57.0,<4.58.0",
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
            env=env,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("usage: optimum-cli export onnx", completed.stdout)


class PrepareInputsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dmf_prepare_inputs_test_"))

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
                    "format": "dart_mlx_ffi.onnx_pipeline.v1",
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
        self.assertEqual(values["DART_MLX_ENABLE_ORT"], "1")
        self.assertEqual(values["DART_MLX_ORT_INCLUDE_DIR"], "/tmp/include")
        self.assertEqual(values["DART_MLX_ORT_LIBRARY"], "/tmp/libonnxruntime.dylib")
        self.assertTrue(env.ready)


class EngineGapTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dmf_resolver_test_"))

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

    def test_resolver_skips_blocked_engines(self) -> None:
        catalog = _fallback_catalog()
        catalog["models"]["paddle_ocr_vl"]["blocked_engines"] = {
            "android": {
                "onnx": "ONNX component bundle is not a full Android runtime.",
            }
        }
        resolver = HuggingFaceArtifactResolver(
            catalog=catalog,
            catalog_path=ROOT / "benchmark/runtime/hf_artifacts.yaml",
            out_path=ROOT / "benchmark/runtime/artifacts.local.yaml",
            cache_root=None,
            model_filter={"paddle_ocr_vl"},
            platform_filter={"android"},
            engine_filter=set(),
            local_files_only=True,
            allow_missing=True,
        )

        plan = resolver.plan()

        cell = plan["cells"][0]
        self.assertEqual(cell["state"], "missing")

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
                                "format": "dart_mlx_ffi.onnx_pipeline.v1",
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
        self.assertEqual(payload["format"], "dart_mlx_ffi.onnx_pipeline.v1")
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
            artifacts={},
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

    def test_engine_gap_report_blocks_component_only_engine(self) -> None:
        catalog = _fallback_catalog()
        catalog["models"]["paddle_ocr_vl"]["blocked_engines"] = {
            "android": {
                "onnx": "ONNX component bundle is not a full Android runtime.",
            }
        }
        catalog["models"]["paddle_ocr_vl"]["blocked_engine_failure_classes"] = {
            "android": {
                "onnx": "hf_component_only_artifact",
            }
        }
        catalog["models"]["paddle_ocr_vl"]["blocked_engine_failure_reasons"] = {
            "android": {
                "onnx": "Only component sidecars are available.",
            }
        }

        report = build_engine_gap_report(
            catalog,
            artifacts={},
            recipes={
                "models": {
                    "paddle_ocr_vl": {
                        "recipes": {
                            "litert": {"preset": "litert_hf_vlm"},
                            "onnx": {"preset": "onnx_vlm"},
                        }
                    }
                }
            },
            catalog_path=ROOT / "benchmark/runtime/hf_artifacts.yaml",
            recipes_path=ROOT / "benchmark/runtime/conversion_recipes.yaml",
            model_filter={"paddle_ocr_vl"},
            platform_filter={"android"},
            engine_filter=set(),
        )

        cell = report["cells"][0]
        self.assertEqual(cell["state"], "blocked")
        self.assertIn("onnx", cell["blocked_engines"])
        self.assertEqual(
            cell["blocked_engine_failure_classes"]["onnx"],
            "hf_component_only_artifact",
        )
        self.assertIn("component sidecars", cell["blocked_engine_failure_reasons"]["onnx"])
        self.assertIn("onnx", cell["conversion"])

    def test_engine_gap_report_uses_converted_artifact_overlay(self) -> None:
        report = build_engine_gap_report(
            _fallback_catalog(),
            artifacts={
                "models": {
                    "paddle_ocr_vl": {
                        "platforms": {
                            "ios": {
                                "engine": "coreml",
                                "artifact": "benchmark/artifacts/converted/paddle/coreml/pipeline.json",
                                "artifact_source": "converted",
                                "source_uri": "converted://paddle_ocr_vl/coreml",
                            },
                            "android": {
                                "engine": "litert",
                                "artifact": "benchmark/artifacts/converted/paddle/litert/model.tflite",
                                "artifact_source": "converted",
                                "source_uri": "converted://paddle_ocr_vl/litert",
                            },
                        }
                    }
                }
            },
            recipes={},
            catalog_path=ROOT / "benchmark/runtime/hf_artifacts.yaml",
            artifacts_path=ROOT / "benchmark/runtime/artifacts.converted.yaml",
            recipes_path=ROOT / "benchmark/runtime/conversion_recipes.yaml",
            model_filter={"paddle_ocr_vl"},
            platform_filter=set(),
            engine_filter=set(),
        )
        by_platform = {cell["platform"]: cell for cell in report["cells"]}

        self.assertEqual(report["preferred_ready_count"], 2)
        self.assertEqual(report["fallback_ready_count"], 0)
        self.assertEqual(by_platform["ios"]["selected_engine"], "coreml")
        self.assertEqual(by_platform["ios"]["source_uri"], "converted://paddle_ocr_vl/coreml")
        self.assertEqual(by_platform["android"]["selected_engine"], "litert")

    def test_engine_gap_report_includes_blocked_conversion_remediation(self) -> None:
        catalog = _fallback_catalog()
        model = catalog["models"]["paddle_ocr_vl"]
        model["blocked_platforms"] = {"android": "LiteRT conversion failed."}
        model["blocked_platform_failure_classes"] = {
            "android": "onnx2tf_attempt_timeout"
        }
        model["blocked_platform_reports"] = {
            "android": "benchmark/artifacts_local/converted/paddle/litert/conversion_record.json"
        }

        report = build_engine_gap_report(
            catalog,
            artifacts={},
            recipes={
                "models": {
                    "paddle_ocr_vl": {
                        "recipes": {
                            "litert": {"preset": "litert_hf_vlm"},
                            "onnx": {"preset": "onnx_vlm"},
                        }
                    }
                }
            },
            catalog_path=ROOT / "benchmark/runtime/hf_artifacts.yaml",
            recipes_path=ROOT / "benchmark/runtime/conversion_recipes.yaml",
            model_filter={"paddle_ocr_vl"},
            platform_filter={"android"},
            engine_filter=set(),
        )
        cell = report["cells"][0]

        self.assertEqual(cell["state"], "blocked")
        self.assertEqual(cell["failure_class"], "onnx2tf_attempt_timeout")
        self.assertIn("conversion_record.json", cell["report"])
        self.assertIn("litert", cell["conversion"])
        self.assertIn("onnx", cell["conversion"])


class _LocalResolver(HuggingFaceArtifactResolver):
    def __init__(self, *args, snapshot_root: Path, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._snapshot_root = snapshot_root

    def _download(self, artifact: dict) -> Path:
        artifact_name = str(artifact.get("artifact") or ".")
        if artifact_name == ".":
            return self._snapshot_root
        return self._snapshot_root / artifact_name
