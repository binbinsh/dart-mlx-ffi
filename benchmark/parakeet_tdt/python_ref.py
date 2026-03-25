from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import mlx.core as mx
import numpy as np
from parakeet_mlx import from_pretrained


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--mel", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = json.loads(Path(args.meta).read_text())
    shape = tuple(meta["shape"])
    mel_np = np.fromfile(args.mel, dtype=np.float32).reshape(shape)
    mel = mx.array(mel_np)
    model = from_pretrained(args.model_id, dtype=mx.float32)

    for _ in range(args.warmup):
        model.generate(mel)[0].text.strip()

    started = time.perf_counter()
    text = ""
    for _ in range(args.iters):
        text = model.generate(mel)[0].text.strip()
    python_ms = (time.perf_counter() - started) * 1000.0 / args.iters

    lengths = mx.array([shape[1]], dtype=mx.int32)
    features, _ = model.encoder(mel, lengths)
    decoder_out, _ = model.decoder(None, None)
    decoder_out = decoder_out.astype(features.dtype)
    joint_out = model.joint(features[:, 0:1], decoder_out)
    token_logits = (
        joint_out[0, 0, :, : len(model.vocabulary) + 1]
        .astype(mx.float32)
        .reshape([-1])
        .tolist()[:16]
    )
    duration_logits = (
        joint_out[0, 0, :, len(model.vocabulary) + 1 :]
        .astype(mx.float32)
        .reshape([-1])
        .tolist()
    )

    print(
        json.dumps(
            {
                "text": text,
                "python_ms": python_ms,
                "token_preview": [float(v) for v in token_logits],
                "duration_logits": [float(v) for v in duration_logits],
            }
        )
    )


if __name__ == "__main__":
    main()
