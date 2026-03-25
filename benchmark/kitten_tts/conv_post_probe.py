from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

try:
    from ..common import add_vendor_to_path, resolve_model_path
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common import add_vendor_to_path, resolve_model_path

add_vendor_to_path("mlx-audio")

import mlx.core as mx
from mlx_audio.tts.utils import get_model_path, load
import mlx_audio.tts.models.kitten_tts.istftnet as istft

try:
    from .mlx_audio_compare import _configure_espeak
    from .mlx_audio_models import TTS_TEXT, pick_voice
    from .mlx_audio_export import (
        build_alignment,
        build_pred_dur,
        forward_front_base,
        forward_front_tail,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mlx_audio_compare import _configure_espeak
    from mlx_audio_models import TTS_TEXT, pick_voice
    from mlx_audio_export import (
        build_alignment,
        build_pred_dur,
        forward_front_base,
        forward_front_tail,
    )

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "private_ane" / "shared" / "benchmark"))

from ane_private_mlprogram_bench import bench_one, build_helper


def _capture_conv_post_input(model, input_ids, ref_s) -> np.ndarray:
    target = model.decoder.generator.conv_post
    captured: dict[str, np.ndarray] = {}
    original = istft.ConvWeighted.__call__

    def wrapped(self, x, conv):
        if self is target and "input" not in captured:
            captured["input"] = np.array(
                x.astype(mx.float32).tolist(),
                dtype=np.float32,
            )
        return original(self, x, conv)

    istft.ConvWeighted.__call__ = wrapped
    try:
        mx.random.seed(0)
        out = model(input_ids, ref_s, return_output=True)
        mx.eval(out.audio, out.pred_dur)
        mx.synchronize()
    finally:
        istft.ConvWeighted.__call__ = original
    return captured["input"]


def _weight_norm_kernel(conv_weighted) -> tuple[np.ndarray, np.ndarray]:
    weight_g = np.array(
        conv_weighted.weight_g.astype(mx.float32).tolist(),
        dtype=np.float32,
    )
    weight_v = np.array(
        conv_weighted.weight_v.astype(mx.float32).tolist(),
        dtype=np.float32,
    )
    bias = np.array(conv_weighted.bias.astype(mx.float32).tolist(), dtype=np.float32)
    denom = np.sqrt(np.sum(np.square(weight_v), axis=(1, 2), keepdims=True) + 1e-7)
    kernel = (weight_v / denom) * weight_g
    kernel = np.transpose(kernel, (0, 2, 1)).reshape(
        kernel.shape[0],
        kernel.shape[2],
        1,
        kernel.shape[1],
    )
    return kernel.astype(np.float32), bias


def main() -> None:
    _configure_espeak()
    snapshot_path = resolve_model_path(
        "mlx-community/kitten-tts-nano-0.8-6bit",
        get_model_path,
    )
    model = load(str(snapshot_path), lazy=False)
    selected_voice = pick_voice(model, requested="expr-voice-5-m")
    input_ids, ref_s, _ = model._prepare_inputs(TTS_TEXT, selected_voice, 1.0, True)

    conv_post_input = _capture_conv_post_input(model, input_ids, ref_s)
    conv_post = model.decoder.generator.conv_post

    for _ in range(5):
        out = conv_post(mx.array(conv_post_input, dtype=mx.float32), mx.conv1d)
        mx.eval(out)
        mx.synchronize()

    started = time.perf_counter()
    for _ in range(20):
        out = conv_post(mx.array(conv_post_input, dtype=mx.float32), mx.conv1d)
        mx.eval(out)
        mx.synchronize()
    mlx_per_iter_ms = (time.perf_counter() - started) * 1000.0 / 20.0

    kernel, bias = _weight_norm_kernel(conv_post)
    conv_post_input_ane = np.transpose(conv_post_input, (0, 2, 1)).reshape(
        1,
        conv_post_input.shape[2],
        1,
        conv_post_input.shape[1],
    )

    work_dir = Path(tempfile.mkdtemp(prefix="kitten_conv_post_probe_"))
    try:
        helper = build_helper(work_dir)

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=conv_post_input_ane.shape, dtype=types.fp32),
            ]
        )
        def prog(x):
            y = mb.conv(
                x=x,
                weight=kernel,
                pad_type="custom",
                pad=[0, 0, 3, 3],
                strides=[1, 1],
                dilations=[1, 1],
                groups=1,
            )
            return mb.add(x=y, y=bias.reshape(1, bias.shape[0], 1, 1))

        model_ct = ct.convert(prog, convert_to="mlprogram")
        package = work_dir / "conv_post.mlpackage"
        model_ct.save(str(package))
        compiled = Path(ct.models.utils.compile_model(str(package)))
        ane = bench_one(
            helper,
            compiled,
            channels=conv_post_input_ane.shape[1],
            spatial=conv_post_input_ane.shape[3],
            iters=20,
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    report = {
        "runtime": "kitten_tts_conv_post_probe",
        "snapshot_path": str(snapshot_path),
        "voice": selected_voice,
        "input_shape": list(conv_post_input.shape),
        "kernel_shape": list(conv_post.weight_v.shape),
        "mlx_per_iter_ms": mlx_per_iter_ms,
        "ane": ane,
    }
    print(json.dumps(report))


if __name__ == "__main__":
    main()
