#!/usr/bin/env python3
"""Decode Qwen3-TTS frames through Python's streaming decoder."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import mlx.core as mx
from mlx_audio.tts.utils import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode Qwen3-TTS frames via Python decoder")
    parser.add_argument("model_path", help="Local Qwen3-TTS bundle path")
    parser.add_argument("frames_json", help="Path to JSON file containing frames")
    parser.add_argument("--pcm-out", help="Optional raw float32 PCM output path")
    parser.add_argument(
        "--mode",
        choices=("stream", "chunked"),
        default="stream",
        help="Decoder path to use",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = load_model(model_path=Path(args.model_path).expanduser().resolve())
    payload = json.loads(Path(args.frames_json).expanduser().resolve().read_text(encoding="utf-8"))
    frames = payload.get("frames", [])
    if not isinstance(frames, list):
        raise SystemExit("frames_json is missing a frames list")
    if not frames:
        audio = np.zeros((0,), dtype=np.float32)
    else:
        time = len(frames)
        groups = len(frames[0])
        flat = [int(v) for frame in frames for v in frame]
        codes = mx.array(flat, dtype=mx.int32).reshape((1, time, groups))
        decoder = model.speech_tokenizer.decoder
        if args.mode == "chunked":
            audio = np.asarray(model.speech_tokenizer.decode(codes)[0][0], dtype=np.float32).reshape(-1)
        else:
            decoder.reset_streaming_state()
            chunk = decoder.streaming_step(mx.transpose(codes, (0, 2, 1))).squeeze(1)[0]
            audio = np.asarray(chunk, dtype=np.float32).reshape(-1)
        mx.eval()

    if args.pcm_out:
        Path(args.pcm_out).expanduser().resolve().write_bytes(audio.tobytes())

    print(
        json.dumps(
            {
                "samples": int(audio.shape[0]),
                "sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
                "head": [f"{value:.8f}" for value in audio[:16]],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
