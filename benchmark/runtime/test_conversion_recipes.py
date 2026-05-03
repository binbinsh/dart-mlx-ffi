from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml

RUNTIME_DIR = Path(__file__).resolve().parent
ROOT = RUNTIME_DIR.parents[1]
sys.path.insert(0, str(RUNTIME_DIR))


class ConversionRecipesTest(unittest.TestCase):
    def test_recipes_seed_models_from_catalog_enabled(self) -> None:
        recipes = yaml.safe_load(
            (ROOT / "benchmark/runtime/conversion_recipes.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(bool(recipes.get("seed_models_from_catalog")))

    def test_onnx_vlm_uses_optimum_supported_task_name(self) -> None:
        recipes = yaml.safe_load(
            (ROOT / "benchmark/runtime/conversion_recipes.yaml").read_text(
                encoding="utf-8"
            )
        )
        onnx_vlm = ((recipes.get("presets") or {}).get("onnx_vlm") or {})
        self.assertEqual(onnx_vlm.get("export_task"), "image-to-text")

    def test_onnx_presets_pin_transformers_457_and_support_extra_with(self) -> None:
        recipes = yaml.safe_load(
            (ROOT / "benchmark/runtime/conversion_recipes.yaml").read_text(
                encoding="utf-8"
            )
        )
        presets = recipes.get("presets") or {}
        for name in [
            "onnx_text_generation",
            "onnx_text_generation_model_type_patch",
            "onnx_ort_genai_text_generation_model_type_patch",
            "onnx_vlm",
            "onnx_text_to_audio",
            "onnx_ming_omni_text_to_audio",
        ]:
            preset = (presets.get(name) or {})
            command = [str(item) for item in (preset.get("command") or [])]
            self.assertIn("transformers>=4.57.0,<4.58.0", command)
            if name == "onnx_ort_genai_text_generation_model_type_patch":
                self.assertIn("onnxruntime-genai>=0.9.0", command)
                self.assertIn("onnx-ir", command)
                self.assertIn("torch>=2.7.0", command)
                self.assertIn("ort_genai_builder_export.py", " ".join(command))
            else:
                self.assertIn("optimum[onnxruntime]==2.1.0", command)
            self.assertIn("--from", command)
            self.assertEqual(command[0], "uvx")
            self.assertIn("{with_args}", command)

    def test_onnx_model_type_patch_preset_uses_patch_script(self) -> None:
        recipes = yaml.safe_load(
            (ROOT / "benchmark/runtime/conversion_recipes.yaml").read_text(
                encoding="utf-8"
            )
        )
        preset = ((recipes.get("presets") or {}).get("onnx_text_generation_model_type_patch") or {})
        command = [str(item) for item in (preset.get("command") or [])]
        self.assertIn("model_type_patched_onnx_export.py", " ".join(command))
        self.assertIn("--model-type-to", command)
        self.assertIn("{model_type_patch}", command)
        self.assertIn("--model-type-from", command)
        self.assertIn("{model_type_expected}", command)

    def test_litert_model_type_patch_preset_uses_patch_script(self) -> None:
        recipes = yaml.safe_load(
            (ROOT / "benchmark/runtime/conversion_recipes.yaml").read_text(
                encoding="utf-8"
            )
        )
        preset = (
            (recipes.get("presets") or {}).get(
                "litert_hf_text_generation_model_type_patch"
            )
            or {}
        )
        command = [str(item) for item in (preset.get("command") or [])]
        self.assertIn("litert_hf_export.py", " ".join(command))
        self.assertIn("--model-type-to", command)
        self.assertIn("{model_type_patch}", command)
        self.assertIn("--model-type-from", command)
        self.assertIn("{model_type_expected}", command)
        self.assertEqual(preset.get("timeout_seconds"), 21600)

    def test_litert_onnx2tf_preset_exists(self) -> None:
        recipes = yaml.safe_load(
            (ROOT / "benchmark/runtime/conversion_recipes.yaml").read_text(
                encoding="utf-8"
            )
        )
        preset = ((recipes.get("presets") or {}).get("litert_onnx2tf") or {})
        command = [str(item) for item in preset.get("command") or []]
        self.assertEqual(preset.get("engine"), "litert")
        self.assertIn("onnx_to_litert.py", " ".join(command))
        self.assertIn("--attempt-timeout-seconds", command)

    def test_litert_onnx_pipeline_preset_exists(self) -> None:
        recipes = yaml.safe_load(
            (ROOT / "benchmark/runtime/conversion_recipes.yaml").read_text(
                encoding="utf-8"
            )
        )
        preset = ((recipes.get("presets") or {}).get("litert_onnx_pipeline2tf") or {})
        command = [str(item) for item in preset.get("command") or []]
        self.assertEqual(preset.get("engine"), "litert")
        self.assertIn("onnx_pipeline_to_litert.py", " ".join(command))
        self.assertIn("--catalog", command)
        self.assertIn("--model-id", command)
        self.assertIn("pipeline.json", preset.get("artifact_candidates") or [])

    def test_paddle_litert_uses_onnx_component_pipeline_converter(self) -> None:
        recipes = yaml.safe_load(
            (ROOT / "benchmark/runtime/conversion_recipes.yaml").read_text(
                encoding="utf-8"
            )
        )
        litert = (
            ((recipes.get("models") or {}).get("paddle_ocr_vl") or {})
            .get("recipes", {})
            .get("litert")
            or {}
        )
        self.assertEqual(litert.get("preset"), "litert_onnx_pipeline2tf")

    def test_qwen3_asr_onnx_uses_model_level_runner(self) -> None:
        recipes = yaml.safe_load(
            (ROOT / "benchmark/runtime/conversion_recipes.yaml").read_text(
                encoding="utf-8"
            )
        )
        qwen = ((recipes.get("models") or {}).get("qwen3_asr") or {}).get(
            "recipes"
        ) or {}
        self.assertTrue((qwen.get("coreml") or {}).get("preflight_blocked"))
        self.assertEqual(
            (qwen.get("coreml") or {}).get("preflight_failure_class"),
            "coreml_ane_compile_failed",
        )
        self.assertFalse((qwen.get("onnx") or {}).get("preflight_blocked", False))
        self.assertEqual(
            ((qwen.get("onnx") or {}).get("provider_by_platform") or {}).get("android"),
            "npu",
        )
        self.assertEqual(
            (qwen.get("onnx") or {}).get("runner"),
            "Qwen3AsrNativeRunner.loadOnnxBundle",
        )
        self.assertFalse((qwen.get("litert") or {}).get("preflight_blocked", False))
        self.assertEqual(
            (qwen.get("litert") or {}).get("runner"),
            "Qwen3AsrNativeRunner.loadLiteRtBundle",
        )
        self.assertEqual(
            (qwen.get("litert") or {}).get("preset"),
            "litert_qwen3_asr_onnx_components",
        )
        presets = recipes.get("presets") or {}
        command = [
            str(item)
            for item in (
                (presets.get("litert_qwen3_asr_onnx_components") or {}).get(
                    "command"
                )
                or []
            )
        ]
        self.assertIn("qwen3_asr_onnx_to_litert.py", " ".join(command))

    def test_android_fallback_models_have_litert_conversion_recipes(self) -> None:
        recipes = yaml.safe_load(
            (ROOT / "benchmark/runtime/conversion_recipes.yaml").read_text(
                encoding="utf-8"
            )
        )
        models = recipes.get("models") or {}
        silero = ((models.get("silero_vad") or {}).get("recipes") or {}).get("litert")
        kitten = ((models.get("kitten_tts") or {}).get("recipes") or {}).get("litert")
        self.assertEqual((silero or {}).get("preset"), "litert_onnx2tf")
        self.assertEqual((kitten or {}).get("preset"), "litert_onnx2tf")

    def test_onnx2tf_fallback_recipes_define_source_candidates(self) -> None:
        recipes = yaml.safe_load(
            (ROOT / "benchmark/runtime/conversion_recipes.yaml").read_text(
                encoding="utf-8"
            )
        )
        models = recipes.get("models") or {}
        silero = ((models.get("silero_vad") or {}).get("recipes") or {}).get("litert")
        kitten = ((models.get("kitten_tts") or {}).get("recipes") or {}).get("litert")
        silero_args = [str(item) for item in ((silero or {}).get("extra_args") or [])]
        kitten_args = [str(item) for item in ((kitten or {}).get("extra_args") or [])]
        self.assertIn("--source-candidate", silero_args)
        self.assertIn("--source-candidate", kitten_args)

    def test_onnx2tf_problematic_models_use_shorter_attempt_timeout(self) -> None:
        recipes = yaml.safe_load(
            (ROOT / "benchmark/runtime/conversion_recipes.yaml").read_text(
                encoding="utf-8"
            )
        )
        models = recipes.get("models") or {}
        silero = ((models.get("silero_vad") or {}).get("recipes") or {}).get("litert")
        kitten = ((models.get("kitten_tts") or {}).get("recipes") or {}).get("litert")
        self.assertEqual((silero or {}).get("attempt_timeout_seconds"), 180)
        self.assertEqual((kitten or {}).get("attempt_timeout_seconds"), 180)

    def test_ming_omni_uses_patched_source_exporters(self) -> None:
        recipes = yaml.safe_load(
            (ROOT / "benchmark/runtime/conversion_recipes.yaml").read_text(
                encoding="utf-8"
            )
        )
        presets = recipes.get("presets") or {}
        models = recipes.get("models") or {}
        ming = (models.get("ming_omni_tts_0_5b") or {}).get("recipes") or {}
        self.assertEqual((ming.get("onnx") or {}).get("preset"), "onnx_ming_omni_text_to_audio")
        self.assertEqual(
            (ming.get("litert") or {}).get("preset"),
            "litert_ming_omni_text_to_audio",
        )

        onnx_command = [str(item) for item in ((presets.get("onnx_ming_omni_text_to_audio") or {}).get("command") or [])]
        litert_command = [str(item) for item in ((presets.get("litert_ming_omni_text_to_audio") or {}).get("command") or [])]
        self.assertIn("ming_omni_onnx_export.py", " ".join(onnx_command))
        self.assertIn("ming_omni_litert_export.py", " ".join(litert_command))
        self.assertIn("--export-small-litert-components", litert_command)
        self.assertIn("--export-audio-decoder-litert", litert_command)

    def test_qwen_36_onnx_recipe_applies_model_type_patch(self) -> None:
        recipes = yaml.safe_load(
            (ROOT / "benchmark/runtime/conversion_recipes.yaml").read_text(
                encoding="utf-8"
            )
        )
        models = recipes.get("models") or {}
        qwen = ((models.get("qwen3_6_27b") or {}).get("recipes") or {}).get("onnx") or {}
        qwen_args = [str(item) for item in (qwen.get("extra_args") or [])]
        self.assertEqual(
            qwen.get("preset"),
            "onnx_ort_genai_text_generation_model_type_patch",
        )
        self.assertEqual(qwen.get("model_type_expected"), "qwen3_5")
        self.assertEqual(qwen.get("model_type_patch"), "qwen3")
        self.assertEqual(qwen.get("timeout_seconds"), 21600)
        self.assertIn("--extra-option", qwen_args)
        self.assertIn("filename=model.onnx", qwen_args)

    def test_qwen_36_litert_recipe_applies_model_type_patch(self) -> None:
        recipes = yaml.safe_load(
            (ROOT / "benchmark/runtime/conversion_recipes.yaml").read_text(
                encoding="utf-8"
            )
        )
        models = recipes.get("models") or {}
        qwen = (
            ((models.get("qwen3_6_27b") or {}).get("recipes") or {}).get("litert")
            or {}
        )
        self.assertEqual(qwen.get("preset"), "litert_hf_text_generation_model_type_patch")
        self.assertEqual(qwen.get("model_type_expected"), "qwen3_5")
        self.assertEqual(qwen.get("model_type_patch"), "qwen3")
        self.assertEqual(qwen.get("timeout_seconds"), 21600)

    def test_minicpm_litert_uses_remote_causal_lm_override(self) -> None:
        recipes = yaml.safe_load(
            (ROOT / "benchmark/runtime/conversion_recipes.yaml").read_text(
                encoding="utf-8"
            )
        )
        models = recipes.get("models") or {}
        litert = (
            ((models.get("minicpm_o_4_5") or {}).get("recipes") or {}).get("litert")
            or {}
        )
        args = [str(item) for item in (litert.get("extra_args") or [])]
        self.assertIn("--auto-model-override", args)
        self.assertIn("AutoModelForCausalLM", args)

    def test_blocked_hf_catalog_platforms_have_failure_metadata(self) -> None:
        catalog = yaml.safe_load(
            (ROOT / "benchmark/runtime/hf_artifacts.yaml").read_text(
                encoding="utf-8"
            )
        )
        models = catalog.get("models") or {}
        for model_id, model in models.items():
            blocked = model.get("blocked_platforms") or {}
            if not blocked:
                continue
            classes = model.get("blocked_platform_failure_classes") or {}
            reasons = model.get("blocked_platform_failure_reasons") or {}
            for platform in blocked:
                self.assertIn(platform, classes, model_id)
                self.assertIn(platform, reasons, model_id)
                self.assertTrue(str(classes[platform]).strip(), model_id)
                self.assertTrue(str(reasons[platform]).strip(), model_id)

    def test_blocked_runtime_models_have_failure_metadata(self) -> None:
        config = yaml.safe_load(
            (ROOT / "benchmark/runtime/models.yaml").read_text(encoding="utf-8")
        )
        for model in config.get("models") or []:
            blocked = model.get("blocked_platforms") or {}
            if not blocked:
                continue
            classes = model.get("blocked_platform_failure_classes") or {}
            reasons = model.get("blocked_platform_failure_reasons") or {}
            for platform in blocked:
                self.assertIn(platform, classes, model.get("id"))
                self.assertIn(platform, reasons, model.get("id"))
                self.assertTrue(str(classes[platform]).strip(), model.get("id"))
                self.assertTrue(str(reasons[platform]).strip(), model.get("id"))


if __name__ == "__main__":
    unittest.main()
