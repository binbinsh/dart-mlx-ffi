from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

RUNTIME_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNTIME_DIR / "converters"))

from model_type_patched_onnx_export import _patch_model_type


class ModelTypePatchedOnnxExportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dmf_model_type_patch_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_patch_model_type_updates_config(self) -> None:
        config = self.tmp / "config.json"
        config.write_text(
            json.dumps({"model_type": "qwen3_5", "hidden_size": 4096}),
            encoding="utf-8",
        )
        report = _patch_model_type(
            config_path=config,
            expected="qwen3_5",
            patched="qwen3",
        )

        self.assertEqual(report["original_model_type"], "qwen3_5")
        self.assertEqual(report["patched_model_type"], "qwen3")
        decoded = json.loads(config.read_text(encoding="utf-8"))
        self.assertEqual(decoded["model_type"], "qwen3")
        self.assertEqual(decoded["hidden_size"], 4096)

    def test_patch_model_type_rejects_unexpected_type(self) -> None:
        config = self.tmp / "config.json"
        config.write_text(json.dumps({"model_type": "glm4"}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Expected model_type"):
            _patch_model_type(
                config_path=config,
                expected="glm4_moe_lite",
                patched="glm4_moe",
            )


if __name__ == "__main__":
    unittest.main()
