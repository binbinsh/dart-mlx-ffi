#!/usr/bin/env python3
"""Prepare cached Qwen3-TTS reference features.

This uses the official mlx-audio runtime to compute the two reference-side
artifacts the Dart runtime needs for ICL voice cloning parity:

1. speaker embedding (x-vector style prefix conditioning)
2. reference codec tokens from the speech tokenizer encoder

Output format is a compact JSON file so Dart can load it without extra Python
dependencies at runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
from mlx_audio.tts.utils import load_model
from mlx_audio.utils import load_audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a cached Qwen3-TTS reference")
    parser.add_argument("model_path", help="Local Qwen3-TTS bundle path")
    parser.add_argument("audio_path", help="Reference WAV path")
    parser.add_argument("ref_text", help="Reference transcript")
    parser.add_argument("output_path", help="Output JSON path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = Path(args.model_path).expanduser().resolve()
    audio_path = Path(args.audio_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()

    model = load_model(model_path=model_path)
    audio = load_audio(str(audio_path), sample_rate=model.sample_rate)

    speaker = model.extract_speaker_embedding(audio)
    if audio.ndim == 1:
        audio_for_codes = audio[None, None, :]
    elif audio.ndim == 2:
        audio_for_codes = audio[None, :, :]
    else:
        audio_for_codes = audio
    ref_codes = model.speech_tokenizer.encode(audio_for_codes)
    mx.eval(speaker, ref_codes)

    speaker_values = speaker.reshape((-1,)).tolist()
    code_values = ref_codes[0].tolist()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "model_path": str(model_path),
                "audio_path": str(audio_path),
                "ref_text": args.ref_text,
                "speaker_embedding": speaker_values,
                "ref_codes": code_values,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
