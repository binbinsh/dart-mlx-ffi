from __future__ import annotations

import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

RUNTIME_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNTIME_DIR))

from prepare_inputs import _processor_tensors


class PrepareInputsProcessorFallbackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dinf_prepare_inputs_processor_"))
        self.image = self.tmp / "image.png"
        self.image.write_bytes(b"png")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_processor_falls_back_to_pt_when_fast_numpy_is_rejected(self) -> None:
        class FakeImage:
            def convert(self, _: str) -> "FakeImage":
                return self

        class FakeImageModule:
            @staticmethod
            def open(_: Path) -> FakeImage:
                return FakeImage()

        class FakeProcessor:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def __call__(self, *_, **kwargs):
                tensor_type = kwargs.get("return_tensors")
                self.calls.append(str(tensor_type))
                if tensor_type == "np":
                    raise ValueError(
                        "Only returning PyTorch tensors is currently supported."
                    )
                return {
                    "input_ids": np.array([[1, 2]], dtype=np.int64),
                    "attention_mask": np.array([[1, 1]], dtype=np.int64),
                    "pixel_values": np.zeros((1, 3, 2, 2), dtype=np.float32),
                }

        class FakeAutoProcessor:
            kwargs_seen: dict[str, object] = {}
            processor = FakeProcessor()

            @classmethod
            def from_pretrained(cls, *_args, **kwargs):
                cls.kwargs_seen = dict(kwargs)
                return cls.processor

        fake_pil = types.ModuleType("PIL")
        fake_pil.Image = FakeImageModule
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoProcessor = FakeAutoProcessor

        with mock.patch.dict(
            sys.modules,
            {"PIL": fake_pil, "transformers": fake_transformers},
        ):
            tensors = _processor_tensors(
                "demo/model",
                text="hello",
                image_file=self.image,
                trust_remote_code=False,
                max_length=None,
            )

        self.assertEqual(FakeAutoProcessor.kwargs_seen.get("use_fast"), False)
        self.assertEqual(FakeAutoProcessor.processor.calls, ["np", "pt"])
        self.assertIn("input_ids", tensors)
        self.assertIn("pixel_values", tensors)


if __name__ == "__main__":
    unittest.main()
