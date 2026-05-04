"""Tests for ``patch_decoder_config.patch_config``."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parent / "patch_decoder_config.py"
    spec = importlib.util.spec_from_file_location("patch_decoder_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_module()


SOURCE_CONFIG: dict = {
    "architectures": ["PaddleOCRVLForConditionalGeneration"],
    "model_type": "paddleocr_vl",
    "auto_map": {
        "AutoConfig": "configuration_paddleocr_vl.PaddleOCRVLConfig",
        "AutoModel": "modeling_paddleocr_vl.PaddleOCRVLForConditionalGeneration",
    },
    "hidden_size": 1024,
    "vision_config": {"hidden_size": 1152},
    "image_token_id": 100295,
    "video_token_id": 101307,
    "vision_start_token_id": 101305,
    "vision_end_token_id": 101306,
    "vocab_size": 103424,
}


class PatchDecoderConfigTest(unittest.TestCase):
    def _write_source(self, td: str) -> Path:
        path = Path(td) / "config.json"
        path.write_text(json.dumps(SOURCE_CONFIG, indent=2), encoding="utf-8")
        return path

    def test_patch_replaces_architecture_and_strips_multimodal_fields(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = self._write_source(td)
            result = mod.patch_config(cfg)
            self.assertEqual(result["architectures"], ["Ernie4_5ForCausalLM"])
            self.assertEqual(result["model_type"], "ernie4_5")
            for key in mod.MULTIMODAL_ONLY_KEYS:
                self.assertNotIn(key, result)
            self.assertNotIn("auto_map", result)
            # Decoder-only fields preserved.
            self.assertEqual(result["hidden_size"], 1024)
            self.assertEqual(result["vocab_size"], 103424)

            on_disk = json.loads(cfg.read_text(encoding="utf-8"))
            self.assertEqual(on_disk["architectures"], ["Ernie4_5ForCausalLM"])

    def test_patch_keep_auto_map_and_custom_arch(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = self._write_source(td)
            result = mod.patch_config(
                cfg,
                architecture="ErnieForCausalLM",
                model_type=None,
                drop_auto_map=False,
            )
            self.assertEqual(result["architectures"], ["ErnieForCausalLM"])
            # model_type left untouched when None passed.
            self.assertEqual(result["model_type"], "paddleocr_vl")
            self.assertIn("auto_map", result)

    def test_patch_missing_config_raises(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                mod.patch_config(Path(td) / "nope.json")


if __name__ == "__main__":
    unittest.main()
