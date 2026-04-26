from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

RUNTIME_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNTIME_DIR))

from prepare_inputs import _align_coreml_inputs


class PrepareInputsCoreMlAlignmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dinf_prepare_inputs_coreml_"))
        self.artifact = self.tmp / "pipeline.json"
        self.artifact.write_text(
            json.dumps({"format": "dart_inference.coreml_pipeline.v1", "stages": []}),
            encoding="utf-8",
        )
        sample = {
            "inputs": {
                "input_ids": {"dtype": "int32", "shape": [1, 160], "values": [0] * 160},
                "attention_mask": {
                    "dtype": "int32",
                    "shape": [1, 160],
                    "values": [1] * 160,
                },
                "position_ids": {
                    "dtype": "int32",
                    "shape": [3, 1, 160],
                    "values": [0] * (3 * 160),
                },
                "pixel_values": {"dtype": "float32", "shape": [576, 3, 14, 14]},
                "image_token_indices": {
                    "dtype": "int32",
                    "shape": [144],
                    "values": list(range(144)),
                },
            }
        }
        (self.tmp / "sample_input.json").write_text(
            json.dumps(sample),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_aligns_coreml_inputs_to_sample_shapes(self) -> None:
        input_ids = np.arange(1, 159, dtype=np.int64).reshape(1, 158)
        attention = np.ones((1, 158), dtype=np.int64)
        position_ids = np.tile(np.arange(158, dtype=np.int64), (3, 1, 1))
        pixel_values = np.zeros((1, 576, 3, 14, 14), dtype=np.float32)
        tensors = {
            "input_ids": input_ids,
            "attention_mask": attention,
            "position_ids": position_ids,
            "pixel_values": pixel_values,
            "image_token_indices": np.arange(144, dtype=np.int64),
        }

        aligned = _align_coreml_inputs(tensors, self.artifact)

        self.assertEqual(aligned["input_ids"].shape, (1, 160))
        self.assertEqual(aligned["attention_mask"].shape, (1, 160))
        self.assertEqual(aligned["position_ids"].shape, (3, 1, 160))
        self.assertEqual(aligned["pixel_values"].shape, (576, 3, 14, 14))
        self.assertEqual(aligned["image_token_indices"].shape, (144,))
        self.assertEqual(aligned["input_ids"].dtype, np.int32)
        self.assertEqual(aligned["attention_mask"].dtype, np.int32)
        np.testing.assert_array_equal(aligned["input_ids"][0, :158], input_ids[0])
        self.assertEqual(int(aligned["input_ids"][0, 158]), 0)
        self.assertEqual(int(aligned["input_ids"][0, 159]), 0)

    def test_uses_sample_values_for_missing_inputs(self) -> None:
        tensors = {"input_ids": np.arange(1, 161, dtype=np.int64).reshape(1, 160)}
        aligned = _align_coreml_inputs(tensors, self.artifact)
        self.assertIn("image_token_indices", aligned)
        np.testing.assert_array_equal(
            aligned["image_token_indices"][:4],
            np.array([0, 1, 2, 3], dtype=np.int32),
        )


if __name__ == "__main__":
    unittest.main()
