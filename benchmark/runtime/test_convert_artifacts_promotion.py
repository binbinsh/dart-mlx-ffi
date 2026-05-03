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
from convert_artifacts_test_support import _write_passing_verdict, json_dump


class PromotionFallbackGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dmf_promotion_gate_test_"))

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

    def test_promotion_patch_contains_platform_artifacts(self) -> None:
        config_path = self.tmp / "models.yaml"
        artifacts_path = self.tmp / "artifacts.yaml"
        out_root = self.tmp / "out"
        coreml_pipeline = self.tmp / "coreml_pipeline.json"
        coreml_pipeline.write_text(
            json_dump(
                {
                    "format": "dart_mlx_ffi.coreml_pipeline.v1",
                    "stages": [],
                }
            ),
            encoding="utf-8",
        )
        litert_model = self.tmp / "model.tflite"
        litert_model.write_bytes(b"tflite")

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
                                    "engine": "coreml",
                                    "artifact": str(coreml_pipeline),
                                    "source_uri": "converted://paddle_ocr_vl/coreml",
                                    "artifact_source": "converted",
                                },
                                "android": {
                                    "engine": "litert",
                                    "artifact": str(litert_model),
                                    "source_uri": "converted://paddle_ocr_vl/litert",
                                    "artifact_source": "converted",
                                },
                            }
                        }
                    }
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        _write_passing_verdict(out_root, "paddle_ocr_vl", "ios", "coreml")
        _write_passing_verdict(out_root, "paddle_ocr_vl", "android", "litert")

        patch = build_promotion_patch(
            config_path=config_path,
            out_root=out_root,
            artifacts_path=artifacts_path,
            model_id="paddle_ocr_vl",
        )
        model_patch = patch["models"][0]
        self.assertEqual(model_patch["supportLevel"], "production")
        artifacts = model_patch["platformArtifacts"]
        self.assertEqual(
            artifacts["coreml"]["path"],
            str(coreml_pipeline),
        )
        self.assertEqual(artifacts["coreml"]["format"], "coreml-pipeline")
        self.assertEqual(artifacts["litert"]["path"], str(litert_model))
        self.assertEqual(artifacts["litert"]["format"], "tflite")
        ios_status = model_patch["validationStatus"]["ios"]
        self.assertTrue(ios_status["identityPassed"])
        self.assertEqual(ios_status["endToEndRatio"], 1.0)
        self.assertEqual(ios_status["iterationCount"], 5)
        self.assertEqual(ios_status["warmupCount"], 1)
        self.assertEqual(ios_status["latencyMs"]["sampleCount"], 5)
        self.assertEqual(ios_status["latencyMs"]["mean"], 10.0)
        self.assertEqual(ios_status["runConfig"]["iters"], 5)
        self.assertEqual(ios_status["inputSignature"]["digest"], "same-input")

    def test_promotion_requires_identity_section(self) -> None:
        config_path = self.tmp / "models.yaml"
        artifacts_path = self.tmp / "artifacts.yaml"
        out_root = self.tmp / "out"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "support_policy": {
                        "production_requires": {"platforms": ["linux"]}
                    },
                    "first_wave": [
                        {
                            "id": "qwen2_5",
                            "family": "Qwen2.5",
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
                        "qwen2_5": {
                            "platforms": {
                                "linux": {
                                    "engine": "onnx",
                                    "artifact": "hf://example/model.onnx",
                                }
                            }
                        }
                    }
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        verdict = out_root / "qwen2_5" / "linux" / "verdict.json"
        verdict.parent.mkdir(parents=True, exist_ok=True)
        verdict.write_text(
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
                        "model_id": "qwen2_5",
                        "platform": "linux",
                        "engine": "onnx",
                    },
                }
            ),
            encoding="utf-8",
        )

        patch = build_promotion_patch(
            config_path=config_path,
            out_root=out_root,
            artifacts_path=artifacts_path,
            model_id="qwen2_5",
        )
        audit_payload = audit(
            config_path=config_path,
            out_root=out_root,
            artifacts_path=artifacts_path,
            model_id="qwen2_5",
        )

        model_patch = patch["models"][0]
        self.assertEqual(model_patch["supportLevel"], "staging")
        linux_status = model_patch["validationStatus"]["linux"]
        self.assertFalse(linux_status["identityPassed"])
        self.assertIn("Failed verdicts: linux", model_patch["notes"])
        audit_model = audit_payload["models"][0]
        self.assertFalse(audit_model["production_ready"])
        self.assertFalse(audit_model["platforms"][0]["gates"]["identity"])


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
