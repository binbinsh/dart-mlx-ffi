"""End-to-end mel parity check: updated Slaney-based algorithm vs moona3k reference.

Usage:
    uv run --no-project \
        --with "mlx-qwen3-asr @ git+https://github.com/moona3k/mlx-qwen3-asr.git" \
        --with numpy \
        python benchmark/qwen3_asr_mel_e2e_parity.py
"""

from __future__ import annotations

import json
import math
import sys

import mlx.core as mx
import numpy as np


SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 400
HOP_LENGTH = 160

_LN_6_4 = math.log(6.4)


def slaney_hz_to_mel(hz: float) -> float:
    if hz < 1000.0:
        return 3.0 * hz / 200.0
    return 15.0 + 27.0 * math.log(hz / 1000.0) / _LN_6_4


def slaney_mel_to_hz(mel: float) -> float:
    if mel < 15.0:
        return 200.0 * mel / 3.0
    return 1000.0 * math.exp((mel - 15.0) * _LN_6_4 / 27.0)


def our_mel_filterbank() -> np.ndarray:
    """Slaney mel filterbank with slaney normalization (matches Dart mel.dart)."""
    bins = N_FFT // 2 + 1
    out = np.zeros((N_MELS, bins), dtype=np.float64)
    mel_min = slaney_hz_to_mel(0.0)
    mel_max = slaney_hz_to_mel(SAMPLE_RATE / 2.0)
    mel_points = [
        mel_min + (mel_max - mel_min) * i / (N_MELS + 1) for i in range(N_MELS + 2)
    ]
    hz_points = [slaney_mel_to_hz(m) for m in mel_points]
    fft_freqs = [SAMPLE_RATE * i / N_FFT for i in range(bins)]

    for m in range(N_MELS):
        lower = hz_points[m]
        center = hz_points[m + 1]
        upper = hz_points[m + 2]
        enorm = 2.0 / (upper - lower) if upper > lower else 0.0
        for b in range(bins):
            freq = fft_freqs[b]
            left = (freq - lower) / (center - lower) if center > lower else 0.0
            right = (upper - freq) / (upper - center) if upper > center else 0.0
            out[m, b] = max(0.0, min(left, right)) * enorm

    return out.astype(np.float32)


def our_log_mel_raw(audio_np: np.ndarray) -> np.ndarray:
    """Full mel spectrogram in (128, frames) layout for comparison."""
    fb = our_mel_filterbank()
    padded = np.pad(audio_np, N_FFT // 2, mode="reflect")
    padded_len = len(padded)
    frame_count = max(1, 1 + (padded_len - N_FFT) // HOP_LENGTH)
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)
    starts = np.arange(frame_count) * HOP_LENGTH
    offsets = np.arange(N_FFT)
    indices = starts[:, None] + offsets[None, :]
    frames = padded[indices]
    windowed = frames * window[None, :]
    spectrum = np.fft.rfft(windowed, axis=1)
    power = np.abs(spectrum) ** 2
    mel = fb @ power.T
    mel = mel[:, :-1]
    log_mel = np.log10(np.maximum(mel, 1e-10))
    max_val = log_mel.max()
    log_mel = np.maximum(log_mel, max_val - 8.0)
    log_mel = (log_mel + 4.0) / 4.0
    return log_mel  # (128, frames)


def ref_log_mel(audio_np: np.ndarray) -> np.ndarray:
    from mlx_qwen3_asr.audio import log_mel_spectrogram

    audio_mx = mx.array(audio_np)
    mel_mx = log_mel_spectrogram(audio_mx)
    mx.eval(mel_mx)
    return np.array(mel_mx)


def ref_filterbank() -> np.ndarray:
    import os
    from mlx_qwen3_asr import audio as audio_mod

    mod_dir = os.path.dirname(audio_mod.__file__)
    npz_path = os.path.join(mod_dir, "assets", "mel_filters.npz")
    return np.load(npz_path)["mel_128"]


def compare(name: str, a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        return {
            "name": name,
            "parity": "FAIL",
            "reason": f"shape: {a.shape} vs {b.shape}",
        }
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {
        "name": name,
        "shape": list(a.shape),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "parity": "PASS" if diff.max() < 1e-4 else "FAIL",
    }


def main() -> None:
    np.random.seed(42)
    results = {}

    # Filterbank parity
    results["filterbank"] = compare(
        "filterbank", our_mel_filterbank(), ref_filterbank()
    )

    # Mel spectrogram parity on various signals
    for label, audio in [
        ("random_1s", np.random.randn(SAMPLE_RATE).astype(np.float32) * 0.1),
        ("random_2s", np.random.randn(SAMPLE_RATE * 2).astype(np.float32) * 0.1),
        ("random_3s", np.random.randn(SAMPLE_RATE * 3).astype(np.float32) * 0.1),
        (
            "sine_440hz_2s",
            (
                0.5 * np.sin(2 * np.pi * 440 * np.arange(SAMPLE_RATE * 2) / SAMPLE_RATE)
            ).astype(np.float32),
        ),
        ("short_50ms", np.random.randn(800).astype(np.float32) * 0.1),
    ]:
        our = our_log_mel_raw(audio)
        ref = ref_log_mel(audio)
        results[f"mel_{label}"] = compare(f"mel_{label}", our, ref)

    json.dump(results, sys.stdout, indent=2)
    print()

    # Summary
    all_pass = all(r.get("parity") == "PASS" for r in results.values())
    print(
        f"\n{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}", file=sys.stderr
    )
    for name, r in results.items():
        status = r.get("parity", "?")
        diff = r.get("max_abs_diff", "N/A")
        print(f"  {status} {name}: max_diff={diff}", file=sys.stderr)


if __name__ == "__main__":
    main()
