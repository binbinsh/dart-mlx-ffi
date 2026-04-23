#!/usr/bin/env python3
"""Generate a compact probe report for Python mlx-audio Qwen3-TTS."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from mlx_audio.tts.utils import load_model
from mlx_audio.utils import load_audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Python Qwen3-TTS output")
    parser.add_argument("model_path", help="Local Qwen3-TTS bundle path")
    parser.add_argument("audio_path", help="Reference WAV path")
    parser.add_argument("ref_text", help="Reference transcript")
    parser.add_argument("text", help="Target text")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = load_model(model_path=Path(args.model_path).expanduser().resolve())
    ref_audio = load_audio(str(Path(args.audio_path).expanduser().resolve()), sample_rate=model.sample_rate)

    start = time.perf_counter()
    chunks = []
    for result in model.generate(
        text=args.text,
        ref_audio=ref_audio,
        ref_text=args.ref_text,
        stream=True,
        streaming_interval=2.0,
        temperature=0.0,
        top_k=50,
        top_p=1.0,
        repetition_penalty=1.5,
    ):
        chunks.append(np.asarray(result.audio, dtype=np.float32).reshape(-1))
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    audio = np.concatenate(chunks) if chunks else np.zeros((0,), dtype=np.float32)
    print(
        json.dumps(
            {
                "sample_rate": int(model.sample_rate),
                "samples": int(audio.shape[0]),
                "elapsed_ms": elapsed_ms,
                "sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
                "head": [f"{value:.8f}" for value in audio[:16]],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
