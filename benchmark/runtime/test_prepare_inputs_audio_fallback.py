from __future__ import annotations

import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

RUNTIME_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNTIME_DIR))

import prepare_inputs
from prepare_inputs import (
    _align_litert_inputs,
    _align_onnx_inputs,
    _prepare_out_path,
    _prepare_tensors_with_audio_fallback,
)


class PrepareInputsAudioFallbackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dmf_prepare_inputs_audio_"))
        self.audio = self.tmp / "audio.wav"
        waveform = np.linspace(-0.2, 0.2, 320, dtype=np.float32)
        sf.write(self.audio, waveform, samplerate=16000)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_audio_prepare_falls_back_when_processor_unavailable(self) -> None:
        with mock.patch(
            "prepare_inputs._audio_tensors",
            side_effect=OSError("processor unavailable"),
        ):
            tensors, fallback = _prepare_tensors_with_audio_fallback(
                source_model="snakers4/silero-vad",
                task="vad",
                prompt="hello",
                embedding_query="hello",
                image_file=None,
                audio_file=self.audio,
                trust_remote_code=False,
                max_length=None,
            )

        self.assertIsNotNone(fallback)
        self.assertIn("OSError", str(fallback))
        self.assertIn("input_values", tensors)
        self.assertIn("sr", tensors)
        self.assertEqual(tensors["input_values"].shape[0], 1)
        self.assertEqual(tensors["input_values"].shape[1], 512)
        self.assertEqual(tensors["sr"].dtype, np.int64)

    def test_qwen3_asr_uses_raw_waveform_for_model_level_runner(self) -> None:
        with mock.patch("prepare_inputs._audio_tensors") as audio_tensors:
            tensors, fallback = _prepare_tensors_with_audio_fallback(
                source_model="Qwen/Qwen3-ASR-1.7B",
                task="audio",
                prompt="hello",
                embedding_query="hello",
                image_file=None,
                audio_file=self.audio,
                trust_remote_code=False,
                max_length=None,
            )

        audio_tensors.assert_not_called()
        self.assertIsNone(fallback)
        self.assertIn("audio", tensors)
        self.assertEqual(tensors["audio"].shape[0], 1)

    def test_align_litert_inputs_matches_serving_default_names(self) -> None:
        model = self.tmp / "model.tflite"
        model.write_bytes(b"litert")
        tensors = {
            "input_values": np.arange(16, dtype=np.float32),
            "state": np.ones((2, 128), dtype=np.float32),
            "sampling_rate": np.asarray([16000], dtype=np.int64),
        }

        class FakeInterpreter:
            def __init__(self, model_path: str) -> None:
                self.model_path = model_path

            def allocate_tensors(self) -> None:
                return None

            def get_input_details(self) -> list[dict[str, object]]:
                return [
                    {
                        "name": "serving_default_input:0",
                        "dtype": np.float32,
                        "shape": np.asarray([1, 16], dtype=np.int32),
                    },
                    {
                        "name": "serving_default_state:0",
                        "dtype": np.float32,
                        "shape": np.asarray([2, 128, 1], dtype=np.int32),
                    },
                    {
                        "name": "serving_default_sr:0",
                        "dtype": np.int64,
                        "shape": np.asarray([1], dtype=np.int32),
                    },
                ]

        fake_tf = types.SimpleNamespace(lite=types.SimpleNamespace(Interpreter=FakeInterpreter))
        with mock.patch.dict(sys.modules, {"tensorflow": fake_tf}):
            aligned = _align_litert_inputs(tensors, model)

        self.assertIn("serving_default_input:0", aligned)
        self.assertEqual(aligned["serving_default_input:0"].shape, (1, 16))
        self.assertEqual(aligned["serving_default_state:0"].shape, (2, 128, 1))
        self.assertEqual(aligned["serving_default_sr:0"].shape, (1,))

    def test_align_onnx_inputs_drops_aliases_and_matches_audio_names(self) -> None:
        model = self.tmp / "model.onnx"
        model.write_bytes(b"onnx")
        tensors = {
            "input_values": np.arange(16, dtype=np.float32).reshape(1, 16),
            "state": np.ones((2, 1, 128), dtype=np.float32),
            "sampling_rate": np.asarray([16000], dtype=np.int64),
        }

        fake_onnx = types.SimpleNamespace(
            load=lambda *_, **__: _FakeOnnxModel(
                [
                    _FakeOnnxInput("input", 1, [0, 0]),
                    _FakeOnnxInput("state", 1, [2, 0, 128]),
                    _FakeOnnxInput("sr", 7, []),
                ]
            )
        )
        with mock.patch.dict(sys.modules, {"onnx": fake_onnx}):
            aligned = _align_onnx_inputs(
                tensors,
                model,
                batch_size=1,
                seq_length=8,
                past_length=0,
            )

        self.assertEqual(set(aligned), {"input", "state", "sr"})
        self.assertEqual(aligned["input"].shape, (1, 16))
        self.assertEqual(aligned["state"].shape, (2, 1, 128))
        self.assertEqual(aligned["sr"].shape, ())

    def test_prepare_out_path_falls_back_from_broken_symlink_parent(self) -> None:
        root = self.tmp
        requested_parent = root / "benchmark" / "out"
        broken_target = root / "missing" / "out_target"
        requested_parent.parent.mkdir(parents=True, exist_ok=True)
        requested_parent.symlink_to(broken_target)
        requested = requested_parent / "runtime" / "silero_vad" / "android" / "input.json"
        with mock.patch.object(prepare_inputs, "ROOT", root):
            resolved, meta = _prepare_out_path(requested)
        self.assertEqual(meta["mode"], "fallback")
        self.assertTrue(str(resolved).startswith(str(root / "benchmark" / "out_local")))

    def test_prepare_out_path_relative_request_preserves_runtime_layout(self) -> None:
        root = self.tmp
        requested_parent = root / "benchmark" / "out"
        broken_target = root / "missing" / "out_target"
        requested_parent.parent.mkdir(parents=True, exist_ok=True)
        requested_parent.symlink_to(broken_target)
        requested = Path("benchmark/out/runtime/silero_vad/android/input.json")
        with mock.patch.object(prepare_inputs, "ROOT", root):
            resolved, meta = _prepare_out_path(requested)
        self.assertEqual(meta["mode"], "fallback")
        self.assertEqual(
            resolved,
            root / "benchmark" / "out_local" / "runtime" / "silero_vad" / "android" / "input.json",
        )


class _FakeOnnxModel:
    def __init__(self, inputs: list[object]) -> None:
        self.graph = types.SimpleNamespace(input=inputs)


class _FakeOnnxInput:
    def __init__(self, name: str, elem_type: int, dims: list[int]) -> None:
        self.name = name
        shape = types.SimpleNamespace(dim=[_FakeOnnxDim(dim) for dim in dims])
        tensor_type = types.SimpleNamespace(elem_type=elem_type, shape=shape)
        self.type = types.SimpleNamespace(tensor_type=tensor_type)


class _FakeOnnxDim:
    def __init__(self, dim_value: int) -> None:
        self.dim_value = dim_value
        self.dim_param = ""


if __name__ == "__main__":
    unittest.main()
