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


class ConvertArtifactsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dmf_convert_test_"))

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
                    "format": "dart_mlx_ffi.coreml_pipeline.v1",
                    "stages": [],
                }
            ),
            encoding="utf-8",
        )

        self.assertEqual(
            _find_artifact(out, ["*.json"]),
            out / "pipeline.json",
        )

    def test_find_artifact_accepts_litert_pipeline_specs(self) -> None:
        out = self.tmp / "out"
        out.mkdir()
        (out / "pipeline.json").write_text(
            json.dumps(
                {
                    "format": "dart_mlx_ffi.litert_pipeline.v1",
                    "stages": [],
                }
            ),
            encoding="utf-8",
        )

        self.assertEqual(
            _find_artifact(out, ["*.json"]),
            out / "pipeline.json",
        )

    def test_find_artifact_ignores_patched_source_sidecars(self) -> None:
        out = self.tmp / "out"
        (out / "_patched_source" / "source_model").mkdir(parents=True)
        (out / "_patched_source" / "source_model" / "campplus.onnx").write_bytes(
            b"onnx"
        )
        (out / "exported").mkdir(parents=True)
        (out / "exported" / "model.onnx").write_bytes(b"onnx")

        self.assertEqual(
            _find_artifact(out, ["**/*.onnx"]),
            out / "exported" / "model.onnx",
        )

    def test_converter_cache_env_defaults_to_local_benchmark_cache(self) -> None:
        with (
            mock.patch("convert_artifacts_support.ROOT", self.tmp),
            mock.patch.dict(
                os.environ,
                {
                    "UV_CACHE_DIR": "",
                    "HF_HOME": "",
                    "XDG_CACHE_HOME": "",
                    "TOKENIZERS_PARALLELISM": "",
                    "HF_HUB_DISABLE_XET": "",
                },
                clear=False,
            ),
        ):
            env = _converter_cache_env()

        self.assertEqual(
            env["UV_CACHE_DIR"],
            str(self.tmp / "benchmark" / ".uv_cache"),
        )
        self.assertEqual(
            env["HF_HOME"],
            str(self.tmp / "benchmark" / ".hf_home"),
        )
        self.assertEqual(
            env["XDG_CACHE_HOME"],
            str(self.tmp / "benchmark" / ".cache"),
        )
        self.assertEqual(env["TOKENIZERS_PARALLELISM"], "false")
        self.assertEqual(env["HF_HUB_DISABLE_XET"], "1")

    def test_converter_cache_env_respects_existing_environment(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "UV_CACHE_DIR": "/tmp/uv-cache",
                "HF_HOME": "/tmp/hf-home",
                "XDG_CACHE_HOME": "/tmp/xdg-cache",
                "TOKENIZERS_PARALLELISM": "true",
                "HF_HUB_DISABLE_XET": "0",
            },
            clear=False,
        ):
            env = _converter_cache_env()

        self.assertNotIn("UV_CACHE_DIR", env)
        self.assertNotIn("HF_HOME", env)
        self.assertNotIn("XDG_CACHE_HOME", env)
        self.assertNotIn("TOKENIZERS_PARALLELISM", env)
        self.assertNotIn("HF_HUB_DISABLE_XET", env)

    def test_timeout_seconds_override_replaces_recipe_timeout(self) -> None:
        recipes = {
            "version": 1,
            "output_root": str(self.tmp / "converted"),
            "models": {
                "qwen3_5": {
                    "source_model": "Qwen/Qwen3.5-0.8B",
                    "task": "text",
                    "recipes": {
                        "onnx": {
                            "engine": "onnx",
                            "exporter": "test",
                            "platforms": ["linux"],
                            "artifact_candidates": ["*.onnx"],
                            "timeout_seconds": 900,
                            "command": ["uv", "run", "python", "-c", "print('ok')"],
                        }
                    },
                }
            },
        }
        out = self.tmp / "artifacts.yaml"

        result = ArtifactConverter(
            recipes=recipes,
            recipes_path=ROOT / "benchmark/runtime/conversion_recipes.yaml",
            base_artifacts=None,
            out_path=out,
            output_root=self.tmp / "converted",
            tools_root=self.tmp / "tools",
            model_filter={"qwen3_5"},
            engine_filter={"onnx"},
            platform_filter=set(),
            dry_run=True,
            reuse_existing=False,
            overwrite=False,
            fetch_tools=False,
            artifact_health_check="none",
            allow_health_fail=False,
            allow_conversion_fail=False,
            min_free_gb=0,
            timeout_seconds_override=7,
        ).run()

        record = result["records"][0]
        self.assertEqual(record["timeout_seconds"], 7)

    def test_converter_run_env_strips_virtualenv_related_variables(self) -> None:
        recipes = {
            "version": 1,
            "output_root": str(self.tmp / "converted"),
            "models": {
                "qwen3_5": {
                    "source_model": "Qwen/Qwen3.5-0.8B",
                    "task": "text",
                    "recipes": {
                        "onnx": {
                            "engine": "onnx",
                            "exporter": "test",
                            "platforms": ["linux"],
                            "artifact_candidates": ["*.onnx"],
                            "command": ["uv", "run", "python", "-c", "print('ok')"],
                        }
                    },
                }
            },
        }
        out = self.tmp / "artifacts.yaml"
        captured_env: dict[str, str] = {}
        artifact = self.tmp / "converted" / "qwen3_5" / "onnx" / "model.onnx"

        def fake_run(
            cmd: list[str],
            *,
            cwd: Path,
            env: dict[str, str],
            stdout: Any,
            stderr: Any,
            timeout: int | None,
            check: bool,
        ) -> subprocess.CompletedProcess[str]:
            del cwd, stdout, stderr, timeout, check
            captured_env.update(env)
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(b"onnx")
            return subprocess.CompletedProcess(cmd, 0)

        with (
            mock.patch.dict(
                os.environ,
                {
                    "VIRTUAL_ENV": "/tmp/fake-venv",
                    "PYTHONPATH": "/tmp/fake-pythonpath",
                    "PYTHONHOME": "/tmp/fake-pythonhome",
                    "CONDA_PREFIX": "/tmp/fake-conda",
                    "__PYVENV_LAUNCHER__": "/tmp/fake-launcher",
                    "PYTHONEXECUTABLE": "/tmp/fake-python",
                },
                clear=False,
            ),
            mock.patch("convert_artifacts.subprocess.run", side_effect=fake_run),
        ):
            result = ArtifactConverter(
                recipes=recipes,
                recipes_path=ROOT / "benchmark/runtime/conversion_recipes.yaml",
                base_artifacts=None,
                out_path=out,
                output_root=self.tmp / "converted",
                tools_root=self.tmp / "tools",
                model_filter={"qwen3_5"},
                engine_filter={"onnx"},
                platform_filter=set(),
                dry_run=False,
                reuse_existing=False,
                overwrite=True,
                fetch_tools=False,
                artifact_health_check="none",
                allow_health_fail=False,
                allow_conversion_fail=False,
                min_free_gb=0,
            ).run()

        self.assertEqual(result["converted_count"], 1)
        self.assertNotIn("VIRTUAL_ENV", captured_env)
        self.assertNotIn("PYTHONPATH", captured_env)
        self.assertNotIn("PYTHONHOME", captured_env)
        self.assertNotIn("CONDA_PREFIX", captured_env)
        self.assertNotIn("__PYVENV_LAUNCHER__", captured_env)
        self.assertNotIn("PYTHONEXECUTABLE", captured_env)

    def test_coreml_llm_extra_args_drop_unsupported_remote_code_flag(self) -> None:
        kept, ignored = _normalized_extra_args(
            "coreml-llm",
            ["--trust-remote-code", "--other"],
        )

        self.assertEqual(kept, ["--other"])
        self.assertEqual(ignored, ["--trust-remote-code"])

    def test_expand_command_supports_list_placeholder(self) -> None:
        command = _expand_command(
            ["uv", "run", "{with_args}", "tool", "{extra_args}"],
            {
                "with_args": ["--with", "pkg-a", "--with", "pkg-b"],
                "extra_args": ["--foo", "bar"],
            },
        )

        self.assertEqual(
            command,
            [
                "uv",
                "run",
                "--with",
                "pkg-a",
                "--with",
                "pkg-b",
                "tool",
                "--foo",
                "bar",
            ],
        )

    def test_normalized_with_packages_deduplicates_and_ignores_empty(self) -> None:
        self.assertEqual(
            _normalized_with_packages(["", " mamba-ssm>=2.2.4 ", "mamba-ssm>=2.2.4"]),
            ["mamba-ssm>=2.2.4"],
        )
        self.assertEqual(
            _normalized_with_packages(" causal-conv1d>=1.4.0 "),
            ["causal-conv1d>=1.4.0"],
        )

    def test_reuse_existing_writes_converted_overlay(self) -> None:
        recipes = {
            "version": 1,
            "output_root": str(self.tmp / "converted"),
            "models": {
                "qwen3_6_27b": {
                    "source_model": "Qwen/Qwen3.6-27B",
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
        artifact = self.tmp / "converted" / "qwen3_6_27b" / "coreml" / "model.mlpackage"
        artifact.mkdir(parents=True)
        out = self.tmp / "artifacts.yaml"

        result = ArtifactConverter(
            recipes=recipes,
            recipes_path=ROOT / "benchmark/runtime/conversion_recipes.yaml",
            base_artifacts=None,
            out_path=out,
            output_root=self.tmp / "converted",
            tools_root=self.tmp / "tools",
            model_filter={"qwen3_6_27b"},
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
        cell = artifact_map["models"]["qwen3_6_27b"]["platforms"]["ios"]
        self.assertEqual(cell["artifact_source"], "converted")
        self.assertTrue(cell["unblock_platform"])
        self.assertEqual(cell["engine"], "coreml")

    def test_seeded_catalog_keeps_fallback_platform_when_conversion_fails(self) -> None:
        catalog_path = self.tmp / "catalog.yaml"
        catalog_path.write_text(
            yaml.safe_dump(
                {
                    "defaults": {},
                    "models": {
                        "kitten_tts": {
                            "source_model": "KittenML/kitten-tts-mini-0.8",
                            "task": "tts",
                            "artifact_coverage": "partial",
                            "platforms": {
                                "android": {
                                    "engine": "onnx",
                                    "artifact": "hf://onnx-community/KittenTTS-Mini-v0.8-ONNX/onnx/model.onnx",
                                    "artifact_source": "huggingface",
                                }
                            },
                        }
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        recipes = {
            "version": 1,
            "output_root": str(self.tmp / "converted"),
            "source_artifact_catalog": str(catalog_path),
            "seed_models_from_catalog": True,
            "models": {
                "kitten_tts": {
                    "source_model": "KittenML/kitten-tts-mini-0.8",
                    "task": "tts",
                    "recipes": {
                        "litert": {
                            "engine": "litert",
                            "exporter": "test",
                            "platforms": ["android"],
                            "artifact_candidates": ["*.tflite"],
                            "command": [sys.executable, "-c", "import sys; sys.exit(1)"],
                        }
                    },
                }
            },
        }
        out = self.tmp / "artifacts.yaml"

        result = ArtifactConverter(
            recipes=recipes,
            recipes_path=ROOT / "benchmark/runtime/conversion_recipes.yaml",
            base_artifacts=None,
            out_path=out,
            output_root=self.tmp / "converted",
            tools_root=self.tmp / "tools",
            model_filter={"kitten_tts"},
            engine_filter={"litert"},
            platform_filter=set(),
            dry_run=False,
            reuse_existing=False,
            overwrite=True,
            fetch_tools=False,
            artifact_health_check="none",
            allow_health_fail=False,
            allow_conversion_fail=True,
            min_free_gb=0,
        ).run()

        self.assertEqual(result["conversion_failed_count"], 1)
        artifact_map = yaml.safe_load(out.read_text(encoding="utf-8"))
        model = artifact_map["models"]["kitten_tts"]
        self.assertEqual(model["platforms"]["android"]["engine"], "onnx")
        self.assertNotIn("blocked_platforms", model)
        self.assertIn("android", model["blocked_engines"])
        self.assertIn("litert", model["blocked_engines"]["android"])

    def test_seed_models_from_source_catalog_builds_android_onnx_fallback(self) -> None:
        catalog_path = self.tmp / "catalog.yaml"
        catalog_path.write_text(
            yaml.safe_dump(
                {
                    "support_policy": {
                        "production_requires": {"platforms": ["android"]},
                    },
                    "defaults": {
                        "platforms": {"android": {"delegate": "xnnpack"}},
                    },
                    "engine_platforms": {
                        "litert": ["android"],
                        "onnx": ["android"],
                    },
                    "engine_order": {"android": ["litert", "onnx"]},
                    "models": {
                        "kitten_tts": {
                            "source_model": "KittenML/kitten-tts-mini-0.8",
                            "task": "tts",
                            "artifact_coverage": "partial",
                            "artifacts": {
                                "onnx": {
                                    "repo": "onnx-community/KittenTTS-Mini-v0.8-ONNX",
                                    "artifact": "onnx/model.onnx",
                                }
                            },
                        }
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        recipes = {
            "version": 1,
            "output_root": str(self.tmp / "converted"),
            "source_artifact_catalog": str(catalog_path),
            "seed_models_from_catalog": True,
            "models": {},
        }
        converter = ArtifactConverter(
            recipes=recipes,
            recipes_path=ROOT / "benchmark/runtime/conversion_recipes.yaml",
            base_artifacts=None,
            out_path=self.tmp / "out.yaml",
            output_root=self.tmp / "converted",
            tools_root=self.tmp / "tools",
            model_filter=set(),
            engine_filter=set(),
            platform_filter=set(),
            dry_run=True,
            reuse_existing=False,
            overwrite=False,
            fetch_tools=False,
            artifact_health_check="none",
            allow_health_fail=False,
            allow_conversion_fail=True,
            min_free_gb=0,
        )

        artifact_map = converter._base_artifact_map()
        self.assertTrue(str(artifact_map["source_catalog"]).endswith("catalog.yaml"))
        android = artifact_map["models"]["kitten_tts"]["platforms"]["android"]
        self.assertEqual(android["engine"], "onnx")
        self.assertEqual(android["artifact_source"], "huggingface")
        self.assertEqual(android["artifact"], "hf://onnx-community/KittenTTS-Mini-v0.8-ONNX/onnx/model.onnx")
        self.assertEqual(android["fallback_from"], ["litert"])

    def test_existing_converted_record_overlays_seeded_platform(self) -> None:
        catalog_path = self.tmp / "catalog.yaml"
        catalog_path.write_text(
            yaml.safe_dump(
                {
                    "support_policy": {
                        "production_requires": {"platforms": ["ios"]},
                    },
                    "engine_platforms": {
                        "coreml": ["ios"],
                        "onnx": ["ios"],
                    },
                    "engine_order": {"ios": ["coreml", "onnx"]},
                    "models": {
                        "paddle_ocr_vl": {
                            "source_model": "PaddlePaddle/PaddleOCR-VL-1.5",
                            "task": "vlm",
                            "artifacts": {
                                "onnx": {
                                    "repo": "lbm364dl/PaddleOCR-VL-1.5-ONNX",
                                    "artifact": "onnx/decoder_model_merged.onnx",
                                }
                            },
                        }
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        output_root = self.tmp / "converted"
        report = output_root / "paddle_ocr_vl" / "coreml" / "conversion_record.json"
        artifact = report.parent / "pipeline.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(
            json.dumps(
                {
                    "format": "dart_mlx_ffi.coreml_pipeline.v1",
                    "stages": [],
                }
            ),
            encoding="utf-8",
        )
        report.write_text(
            json.dumps(
                {
                    "model_id": "paddle_ocr_vl",
                    "engine": "coreml",
                    "task": "vlm",
                    "source_model": "PaddlePaddle/PaddleOCR-VL-1.5",
                    "platforms": ["ios"],
                    "state": "reused",
                    "returncode": 0,
                    "artifact": str(artifact),
                    "report_path": str(report),
                }
            ),
            encoding="utf-8",
        )
        recipes = {
            "version": 1,
            "output_root": str(output_root),
            "source_artifact_catalog": str(catalog_path),
            "seed_models_from_catalog": True,
            "models": {},
        }
        out = self.tmp / "artifacts.yaml"

        result = ArtifactConverter(
            recipes=recipes,
            recipes_path=ROOT / "benchmark/runtime/conversion_recipes.yaml",
            base_artifacts=None,
            out_path=out,
            output_root=output_root,
            tools_root=self.tmp / "tools",
            model_filter=set(),
            engine_filter=set(),
            platform_filter=set(),
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
        ios = artifact_map["models"]["paddle_ocr_vl"]["platforms"]["ios"]
        self.assertEqual(ios["engine"], "coreml")
        self.assertEqual(ios["artifact_source"], "converted")
        self.assertTrue(ios["artifact"].endswith("pipeline.json"))
        self.assertTrue(ios["unblock_platform"])

    def test_existing_failed_record_marks_engine_blocker_on_fallback_platform(self) -> None:
        catalog_path = self.tmp / "catalog.yaml"
        catalog_path.write_text(
            yaml.safe_dump(
                {
                    "support_policy": {
                        "production_requires": {"platforms": ["android"]},
                    },
                    "defaults": {
                        "platforms": {"android": {"delegate": "xnnpack"}},
                    },
                    "engine_platforms": {
                        "litert": ["android"],
                        "onnx": ["android"],
                    },
                    "engine_order": {"android": ["litert", "onnx"]},
                    "models": {
                        "kitten_tts": {
                            "source_model": "KittenML/kitten-tts-mini-0.8",
                            "task": "tts",
                            "artifacts": {
                                "onnx": {
                                    "repo": "onnx-community/KittenTTS-Mini-v0.8-ONNX",
                                    "artifact": "onnx/model.onnx",
                                }
                            },
                        }
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        output_root = self.tmp / "converted"
        report = output_root / "kitten_tts" / "litert" / "conversion_record.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(
                {
                    "model_id": "kitten_tts",
                    "engine": "litert",
                    "task": "tts",
                    "source_model": "KittenML/kitten-tts-mini-0.8",
                    "platforms": ["android"],
                    "state": "conversion_failed",
                    "returncode": 1,
                    "reason": "LiteRT conversion failed.",
                    "report_path": str(report),
                    "log_path": "benchmark/artifacts/converted/kitten_tts/litert/conversion.log",
                }
            ),
            encoding="utf-8",
        )
        recipes = {
            "version": 1,
            "output_root": str(output_root),
            "source_artifact_catalog": str(catalog_path),
            "seed_models_from_catalog": True,
            "models": {},
        }
        out = self.tmp / "artifacts.yaml"

        result = ArtifactConverter(
            recipes=recipes,
            recipes_path=ROOT / "benchmark/runtime/conversion_recipes.yaml",
            base_artifacts=None,
            out_path=out,
            output_root=output_root,
            tools_root=self.tmp / "tools",
            model_filter=set(),
            engine_filter=set(),
            platform_filter=set(),
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
        model = artifact_map["models"]["kitten_tts"]
        self.assertEqual(model["platforms"]["android"]["engine"], "onnx")
        self.assertEqual(
            model["blocked_engines"]["android"]["litert"],
            "LiteRT conversion failed.",
        )
        self.assertIn(
            "conversion_record.json",
            model["blocked_engine_reports"]["android"]["litert"],
        )

    def test_existing_preflight_record_is_not_imported_as_blocker(self) -> None:
        catalog_path = self.tmp / "catalog.yaml"
        catalog_path.write_text(
            yaml.safe_dump(
                {
                    "support_policy": {
                        "production_requires": {"platforms": ["android"]},
                    },
                    "engine_platforms": {
                        "litert": ["android"],
                        "onnx": ["android"],
                    },
                    "engine_order": {"android": ["litert", "onnx"]},
                    "models": {
                        "kitten_tts": {
                            "source_model": "KittenML/kitten-tts-mini-0.8",
                            "task": "tts",
                            "artifacts": {
                                "onnx": {
                                    "repo": "onnx-community/KittenTTS-Mini-v0.8-ONNX",
                                    "artifact": "onnx/model.onnx",
                                }
                            },
                        }
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        output_root = self.tmp / "converted"
        report = output_root / "kitten_tts" / "litert" / "conversion_record.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(
                {
                    "model_id": "kitten_tts",
                    "engine": "litert",
                    "task": "tts",
                    "source_model": "KittenML/kitten-tts-mini-0.8",
                    "platforms": ["android"],
                    "state": "preflight_skipped",
                    "returncode": 0,
                    "reason": "Skipped by min-free-gb.",
                    "report_path": str(report),
                }
            ),
            encoding="utf-8",
        )
        recipes = {
            "version": 1,
            "output_root": str(output_root),
            "source_artifact_catalog": str(catalog_path),
            "seed_models_from_catalog": True,
            "models": {},
        }
        out = self.tmp / "artifacts.yaml"

        result = ArtifactConverter(
            recipes=recipes,
            recipes_path=ROOT / "benchmark/runtime/conversion_recipes.yaml",
            base_artifacts=None,
            out_path=out,
            output_root=output_root,
            tools_root=self.tmp / "tools",
            model_filter=set(),
            engine_filter=set(),
            platform_filter=set(),
            dry_run=False,
            reuse_existing=False,
            overwrite=False,
            fetch_tools=False,
            artifact_health_check="none",
            allow_health_fail=False,
            allow_conversion_fail=True,
            min_free_gb=0,
        ).run()

        self.assertEqual(result["imported_record_count"], 0)
        artifact_map = yaml.safe_load(out.read_text(encoding="utf-8"))
        model = artifact_map["models"]["kitten_tts"]
        self.assertEqual(model["platforms"]["android"]["engine"], "onnx")
        self.assertNotIn("blocked_engines", model)
        self.assertNotIn("blocked_platforms", model)

    def test_existing_records_for_removed_models_are_not_imported(self) -> None:
        catalog_path = self.tmp / "catalog.yaml"
        catalog_path.write_text(
            yaml.safe_dump(
                {
                    "support_policy": {
                        "production_requires": {"platforms": ["windows"]},
                    },
                    "engine_platforms": {"onnx": ["windows"]},
                    "engine_order": {"windows": ["onnx"]},
                    "models": {
                        "translategemma_4b_it": {
                            "source_model": "google/translategemma-4b-it",
                            "task": "text",
                        }
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        output_root = self.tmp / "converted"
        report = output_root / "translategemma_27b_it" / "onnx" / "conversion_record.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(
                {
                    "model_id": "translategemma_27b_it",
                    "engine": "onnx",
                    "task": "text",
                    "source_model": "google/translategemma-27b-it",
                    "platforms": ["windows"],
                    "state": "conversion_failed",
                    "returncode": 1,
                    "reason": "stale model failure",
                    "report_path": str(report),
                }
            ),
            encoding="utf-8",
        )
        out = self.tmp / "artifacts.yaml"

        result = ArtifactConverter(
            recipes={
                "version": 1,
                "output_root": str(output_root),
                "source_artifact_catalog": str(catalog_path),
                "seed_models_from_catalog": True,
                "models": {},
            },
            recipes_path=ROOT / "benchmark/runtime/conversion_recipes.yaml",
            base_artifacts=None,
            out_path=out,
            output_root=output_root,
            tools_root=self.tmp / "tools",
            model_filter=set(),
            engine_filter=set(),
            platform_filter=set(),
            dry_run=False,
            reuse_existing=False,
            overwrite=False,
            fetch_tools=False,
            artifact_health_check="none",
            allow_health_fail=False,
            allow_conversion_fail=True,
            min_free_gb=0,
        ).run()

        self.assertEqual(result["imported_record_count"], 0)
        artifact_map = yaml.safe_load(out.read_text(encoding="utf-8"))
        self.assertIn("translategemma_4b_it", artifact_map["models"])
        self.assertNotIn("translategemma_27b_it", artifact_map["models"])

    def test_existing_recipe_preflight_record_is_imported_as_engine_blocker(self) -> None:
        output_root = self.tmp / "converted"
        report = output_root / "minicpm_o_4_5" / "coreml" / "conversion_record.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(
                {
                    "model_id": "minicpm_o_4_5",
                    "engine": "coreml",
                    "task": "vlm",
                    "source_model": "openbmb/MiniCPM-o-4_5",
                    "platforms": ["ios"],
                    "state": "preflight_skipped",
                    "returncode": 0,
                    "reason": "Skipped by recipe preflight.",
                    "preflight_blocked": True,
                    "failure_class": "coreml_exporter_missing_for_vlm",
                    "failure_reason": "No Core ML VLM exporter is wired.",
                    "report_path": str(report),
                }
            ),
            encoding="utf-8",
        )
        base = self.tmp / "base.yaml"
        base.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "models": {
                        "minicpm_o_4_5": {
                            "source_model": "openbmb/MiniCPM-o-4_5",
                            "task": "vlm",
                            "platforms": {
                                "ios": {
                                    "engine": "mlx",
                                    "artifact": "hf://mlx-community/MiniCPM-o-4_5-4bit/.",
                                    "fallback_from": ["coreml"],
                                }
                            },
                        }
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        out = self.tmp / "artifacts.yaml"

        result = ArtifactConverter(
            recipes={"version": 1, "output_root": str(output_root), "models": {}},
            recipes_path=ROOT / "benchmark/runtime/conversion_recipes.yaml",
            base_artifacts=base,
            out_path=out,
            output_root=output_root,
            tools_root=self.tmp / "tools",
            model_filter=set(),
            engine_filter=set(),
            platform_filter=set(),
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
        model = artifact_map["models"]["minicpm_o_4_5"]
        self.assertEqual(model["platforms"]["ios"]["engine"], "mlx")
        self.assertEqual(
            model["blocked_engine_failure_classes"]["ios"]["coreml"],
            "coreml_exporter_missing_for_vlm",
        )

    def test_reuse_existing_health_check_records_report_before_unblocking(self) -> None:
        recipes = {
            "version": 1,
            "output_root": str(self.tmp / "converted"),
            "models": {
                "qwen3_6_27b": {
                    "source_model": "Qwen/Qwen3.6-27B",
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
        artifact = self.tmp / "converted" / "qwen3_6_27b" / "litert" / "model.tflite"
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
                model_filter={"qwen3_6_27b"},
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
        cell = artifact_map["models"]["qwen3_6_27b"]["platforms"]["android"]
        self.assertTrue(cell["artifact_health_passed"])
        self.assertIn("artifact_health_report", cell)
        self.assertEqual(cell["delegate"], "xnnpack")

    def test_successful_reuse_clears_stale_engine_and_platform_blockers(self) -> None:
        recipes = {
            "version": 1,
            "output_root": str(self.tmp / "converted"),
            "models": {
                "kitten_tts": {
                    "source_model": "KittenML/kitten-tts-mini-0.8",
                    "task": "tts",
                    "recipes": {
                        "litert": {
                            "engine": "litert",
                            "exporter": "test",
                            "platforms": ["android"],
                            "delegate_by_platform": {"android": "xnnpack"},
                            "artifact_candidates": ["*.tflite"],
                            "command": [sys.executable, "-c", "raise SystemExit(1)"],
                        }
                    },
                }
            },
        }
        artifact = self.tmp / "converted" / "kitten_tts" / "litert" / "model.tflite"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"TFL3")
        base_artifacts = self.tmp / "base.yaml"
        base_artifacts.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "models": {
                        "kitten_tts": {
                            "source_model": "KittenML/kitten-tts-mini-0.8",
                            "task": "tts",
                            "artifact_coverage": "partial",
                            "platforms": {
                                "android": {
                                    "engine": "onnx",
                                    "artifact": "hf://onnx-community/KittenTTS-Mini-v0.8-ONNX/onnx/model.onnx",
                                    "provider": "cpu",
                                }
                            },
                            "blocked_engines": {
                                "android": {"litert": "LiteRT conversion failed."}
                            },
                            "blocked_engine_reports": {
                                "android": {
                                    "litert": "benchmark/artifacts_local/converted/kitten_tts/litert/conversion_record.json"
                                }
                            },
                            "blocked_platforms": {
                                "android": "Android conversion blocked."
                            },
                            "blocked_platform_reports": {
                                "android": "benchmark/artifacts_local/converted/kitten_tts/litert/conversion_record.json"
                            },
                        }
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        out = self.tmp / "artifacts.yaml"

        result = ArtifactConverter(
            recipes=recipes,
            recipes_path=ROOT / "benchmark/runtime/conversion_recipes.yaml",
            base_artifacts=base_artifacts,
            out_path=out,
            output_root=self.tmp / "converted",
            tools_root=self.tmp / "tools",
            model_filter={"kitten_tts"},
            engine_filter={"litert"},
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
        model = artifact_map["models"]["kitten_tts"]
        self.assertEqual(model["platforms"]["android"]["engine"], "litert")
        self.assertEqual(model["artifact_coverage"], "converted")
        self.assertNotIn("blocked_engines", model)
        self.assertNotIn("blocked_engine_reports", model)
        self.assertNotIn("blocked_platforms", model)
        self.assertNotIn("blocked_platform_reports", model)

    def test_failed_conversion_clears_stale_same_engine_converted_cell(self) -> None:
        recipes = {
            "version": 1,
            "output_root": str(self.tmp / "converted"),
            "models": {
                "kitten_tts": {
                    "source_model": "KittenML/kitten-tts-mini-0.8",
                    "task": "tts",
                    "recipes": {
                        "litert": {
                            "engine": "litert",
                            "exporter": "test",
                            "platforms": ["android"],
                            "artifact_candidates": ["*.tflite"],
                            "command": [sys.executable, "-c", "raise SystemExit(1)"],
                        }
                    },
                }
            },
        }
        base_artifacts = self.tmp / "base.yaml"
        base_artifacts.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "models": {
                        "kitten_tts": {
                            "source_model": "KittenML/kitten-tts-mini-0.8",
                            "task": "tts",
                            "artifact_coverage": "converted",
                            "platforms": {
                                "android": {
                                    "engine": "litert",
                                    "artifact": "benchmark/artifacts_local/converted/kitten_tts/litert/model.tflite",
                                    "artifact_source": "converted",
                                    "source_uri": "converted://kitten_tts/litert",
                                }
                            },
                        }
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        out = self.tmp / "artifacts.yaml"

        result = ArtifactConverter(
            recipes=recipes,
            recipes_path=ROOT / "benchmark/runtime/conversion_recipes.yaml",
            base_artifacts=base_artifacts,
            out_path=out,
            output_root=self.tmp / "converted",
            tools_root=self.tmp / "tools",
            model_filter={"kitten_tts"},
            engine_filter={"litert"},
            platform_filter=set(),
            dry_run=False,
            reuse_existing=False,
            overwrite=False,
            fetch_tools=False,
            artifact_health_check="none",
            allow_health_fail=False,
            allow_conversion_fail=True,
            min_free_gb=0,
        ).run()

        self.assertEqual(result["conversion_failed_count"], 1)
        artifact_map = yaml.safe_load(out.read_text(encoding="utf-8"))
        model = artifact_map["models"]["kitten_tts"]
        self.assertNotIn("platforms", model)
        self.assertIn("blocked_platforms", model)
        self.assertIn("android", model["blocked_platforms"])
        self.assertNotIn("blocked_engines", model)
