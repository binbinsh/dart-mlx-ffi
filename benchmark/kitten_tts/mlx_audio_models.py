from __future__ import annotations

import sys
from pathlib import Path
import time
from typing import Any

try:
    from ..common import add_vendor_to_path, cleanup_mlx, preview, resolve_model_path
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common import add_vendor_to_path, cleanup_mlx, preview, resolve_model_path

add_vendor_to_path("mlx-audio")

import mlx.core as mx

TTS_TEXT = (
    "MLX on Apple Silicon makes local speech synthesis practical, fast, and "
    "memory efficient."
)

MODEL_SPECS = [
    {
        "name": "kitten_tts_nano_08",
        "model_id": "mlx-community/kitten-tts-nano-0.8-6bit",
        "backend": "mlx_audio",
        "kind": "audio",
    },
]


def benchmark_model(
    spec: dict[str, Any],
    *,
    warmup: int,
    iters: int,
    voice: str | None,
) -> dict[str, Any]:
    _configure_espeak()
    from mlx_audio.tts.utils import get_model_path, load

    model_path = resolve_model_path(spec["model_id"], get_model_path)
    model = load(str(model_path), lazy=False)
    selected_voice = pick_voice(model, requested=voice)

    def forward() -> mx.array:
        mx.random.seed(0)
        parts = [
            segment.audio.astype(mx.float32)
            for segment in model.generate(
                TTS_TEXT,
                voice=selected_voice,
                clean_text=True,
                chunk_size=400,
            )
        ]
        if not parts:
            raise RuntimeError("KittenTTS produced no audio segments.")
        audio = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)
        mx.eval(audio)
        mx.synchronize()
        return audio

    try:
        for _ in range(warmup):
            _ = forward()

        mx.reset_peak_memory()
        before_peak = int(mx.get_peak_memory())
        started = time.perf_counter()
        last = None
        for _ in range(iters):
            last = forward()
        total_ms = (time.perf_counter() - started) * 1000.0
        assert last is not None
        output_values = [float(v) for v in last[:16].tolist()]
        after_peak = int(mx.get_peak_memory())
        samples = int(last.shape[0])
        return {
            "name": spec["name"],
            "model_id": spec["model_id"],
            "backend": spec["backend"],
            "kind": spec["kind"],
            "snapshot_path": str(model_path.resolve()),
            "input_text": TTS_TEXT,
            "voice": selected_voice,
            "output_kind": "audio",
            "output_shape": [samples],
            "output_preview": preview(output_values),
            "samples": samples,
            "sample_rate": int(model.sample_rate),
            "duration_seconds": samples / float(model.sample_rate),
            "total_ms": total_ms,
            "per_iter_ms": total_ms / iters,
            "peak_bytes_delta": after_peak - before_peak,
        }
    finally:
        del model
        cleanup_mlx(mx)


def pick_voice(model: Any, requested: str | None) -> str:
    candidates = []
    if requested:
        candidates.append(requested)
    candidates.append("expr-voice-5-m")
    candidates.extend(sorted(getattr(model, "voice_aliases", {}).keys()))
    candidates.extend(sorted(getattr(model, "voices", {}).keys()))

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate in model.voice_aliases or candidate in model.voices:
            return candidate
    raise RuntimeError("No usable KittenTTS voice was found in voices.npz.")


def _configure_espeak() -> None:
    import espeakng_loader
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    espeakng_loader.make_library_available()
    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
