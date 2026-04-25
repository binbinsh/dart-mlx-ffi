from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml

RUNTIME_DIR = Path(__file__).resolve().parent
ROOT = RUNTIME_DIR.parents[1]
sys.path.insert(0, str(RUNTIME_DIR))


class ConversionRecipesTest(unittest.TestCase):
    def test_onnx_vlm_uses_optimum_supported_task_name(self) -> None:
        recipes = yaml.safe_load(
            (ROOT / "benchmark/runtime/conversion_recipes.yaml").read_text(
                encoding="utf-8"
            )
        )
        onnx_vlm = ((recipes.get("presets") or {}).get("onnx_vlm") or {})
        self.assertEqual(onnx_vlm.get("export_task"), "image-to-text")

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


if __name__ == "__main__":
    unittest.main()
