from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


RUNTIME_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNTIME_DIR / "converters"))

import ort_genai_builder_export  # noqa: E402


class OrtGenAiBuilderExportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dmf_ort_genai_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_build_builder_command_uses_local_source_when_patched(self) -> None:
        cmd = ort_genai_builder_export._build_builder_command(
            source=self.tmp / "source",
            output_dir=self.tmp / "out",
            precision="int4",
            execution_provider="cpu",
            cache_dir=self.tmp / "cache",
            extra_options=["filename=model.onnx"],
            passthrough=["--log_level", "1"],
        )

        self.assertIn("onnxruntime_genai.models.builder", cmd)
        self.assertIn("-i", cmd)
        self.assertIn(str(self.tmp / "source"), cmd)
        self.assertNotIn("Qwen/Qwen3.6-27B", cmd)
        self.assertIn("--extra_options", cmd)
        self.assertIn("filename=model.onnx", cmd)
        self.assertEqual(cmd[-2:], ["--log_level", "1"])

    def test_build_builder_command_uses_hf_model_without_patch(self) -> None:
        cmd = ort_genai_builder_export._build_builder_command(
            source="Qwen/Qwen3.6-27B",
            output_dir=self.tmp / "out",
            precision="int4",
            execution_provider="cpu",
            cache_dir=self.tmp / "cache",
            extra_options=[],
            passthrough=[],
        )

        self.assertIn("-m", cmd)
        self.assertIn("Qwen/Qwen3.6-27B", cmd)
        self.assertNotIn("-i", cmd)

    def test_prepare_source_patches_model_type(self) -> None:
        snapshot = self.tmp / "snapshot"

        def fake_snapshot_download_with_fallback(**kwargs: object) -> str:
            local_dir = Path(str(kwargs["local_dir"]))
            local_dir.mkdir(parents=True)
            (local_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3_5",
                        "text_config": {
                            "hidden_size": 5120,
                            "eos_token_id": 151643,
                            "pad_token_id": None,
                            "rope_parameters": {
                                "mrope_section": [11, 11, 10],
                                "partial_rotary_factor": 0.25,
                                "rope_type": "default",
                                "rope_theta": 10000000,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            return str(snapshot)

        with mock.patch.object(
            ort_genai_builder_export,
            "snapshot_download_with_fallback",
            side_effect=fake_snapshot_download_with_fallback,
        ):
            source = ort_genai_builder_export._prepare_source(
                model="Qwen/Qwen3.6-27B",
                output_dir=self.tmp / "out",
                revision=None,
                model_type_from="qwen3_5",
                model_type_to="qwen3",
                endpoint="https://hf-mirror.com",
                fallback_endpoint=None,
                allow_patterns=["config.json", " "],
            )

        config = json.loads((Path(source) / "config.json").read_text(encoding="utf-8"))
        report = json.loads(
            (self.tmp / "out" / "source_patch_report.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["model_type"], "qwen3")
        self.assertEqual(config["hidden_size"], 5120)
        self.assertEqual(config["pad_token_id"], 151643)
        self.assertNotIn("pad_token_id", config["text_config"])
        self.assertEqual(config["rope_scaling"]["mrope_section"], [11, 11, 10])
        self.assertEqual(config["rope_scaling"]["rope_type"], "default")
        self.assertNotIn("type", config["rope_scaling"])
        self.assertEqual(config["rope_theta"], 10000000)
        self.assertEqual(config["partial_rotary_factor"], 0.25)
        self.assertIn("hidden_size", report["flattened_text_config_keys"])
        self.assertEqual(report["dropped_null_text_config_keys"], ["pad_token_id"])
        self.assertEqual(report["patched_model_type"], "qwen3")
        self.assertEqual(report["original_model_type"], "qwen3_5")
        self.assertTrue(report["rope_parameters_to_rope_scaling"])

    def test_prepare_source_downloads_allow_patterns_without_model_type_patch(
        self,
    ) -> None:
        def fake_snapshot_download_with_fallback(**kwargs: object) -> str:
            local_dir = Path(str(kwargs["local_dir"]))
            local_dir.mkdir(parents=True)
            (local_dir / "config.json").write_text(
                json.dumps({"model_type": "gemma3"}),
                encoding="utf-8",
            )
            self.assertEqual(kwargs["allow_patterns"], ["config.json"])
            return str(local_dir)

        with mock.patch.object(
            ort_genai_builder_export,
            "snapshot_download_with_fallback",
            side_effect=fake_snapshot_download_with_fallback,
        ):
            source = ort_genai_builder_export._prepare_source(
                model="aisingapore/Gemma-SEA-LION-v4-4B-VL",
                output_dir=self.tmp / "out",
                revision=None,
                model_type_from=None,
                model_type_to=None,
                endpoint="https://hf-mirror.com",
                fallback_endpoint=None,
                allow_patterns=["config.json"],
            )

        report = json.loads(
            (self.tmp / "out" / "source_patch_report.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(source, self.tmp / "out" / "_patched_source" / "source_model")
        self.assertEqual(report["source_model"], "aisingapore/Gemma-SEA-LION-v4-4B-VL")
        self.assertNotIn("patched_model_type", report)

    def test_clean_patterns_returns_none_for_empty_values(self) -> None:
        self.assertIsNone(ort_genai_builder_export._clean_patterns(["", " "]))
        self.assertEqual(
            ort_genai_builder_export._clean_patterns(["config.json", " tokenizer* "]),
            ["config.json", "tokenizer*"],
        )


if __name__ == "__main__":
    unittest.main()
