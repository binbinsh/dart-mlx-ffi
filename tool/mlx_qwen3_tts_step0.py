#!/usr/bin/env python3
"""Dump Python MLX Qwen3-TTS step-0 diagnostics for parity comparison.

Outputs JSON with:
  - input_embeds_stats: shape, l2, head/tail
  - post_prefill_hidden_stats: shape, l2, head/tail of hidden[:, -1, :]
  - step0_logits_stats: l2, head/tail, top-k ids + values
  - step0_token: sampled id at temperature=0.0

Only the step-0 forward pass is evaluated; the loop does not run.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mlx.core as mx

from mlx_audio.tts.utils import load_model
from mlx_audio.utils import load_audio


def _stats(name: str, arr: mx.array, head: int = 8, tail: int = 8) -> dict:
    mx.eval(arr)
    flat = arr.reshape(-1)
    n = int(flat.shape[0])
    as_list = flat.tolist()
    sq = sum(float(v) * float(v) for v in as_list)
    l2 = math.sqrt(sq)
    return {
        "name": name,
        "shape": list(arr.shape),
        "count": n,
        "l2": l2,
        "head": [float(v) for v in as_list[:head]],
        "tail": [float(v) for v in as_list[-tail:]] if n > tail else [],
    }


def _topk(logits_row: list, k: int) -> list:
    indexed = list(enumerate(logits_row))
    indexed.sort(key=lambda iv: iv[1], reverse=True)
    return [{"id": int(i), "value": float(v)} for i, v in indexed[:k]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("audio_path")
    parser.add_argument("ref_text")
    parser.add_argument("text")
    parser.add_argument("output_path")
    args = parser.parse_args()

    model = load_model(model_path=Path(args.model_path).expanduser().resolve())
    ref_audio = load_audio(
        str(Path(args.audio_path).expanduser().resolve()),
        sample_rate=model.sample_rate,
    )

    input_embeds, trailing_text_hidden, tts_pad_embed, ref_codes = (
        model._prepare_icl_generation_inputs(  # noqa: SLF001
            text=args.text,
            ref_audio=ref_audio,
            ref_text=args.ref_text,
            language="auto",
        )
    )

    cache = model.talker.make_cache()
    logits, hidden = model.talker(input_embeds, cache=cache)
    mx.eval(logits, hidden)

    last_hidden = hidden[:, -1, :]
    last_logits = logits[:, -1, :]
    mx.eval(last_hidden, last_logits)

    config = model.config.talker_config
    eos_token_id = config.codec_eos_token_id
    suppress_tokens = [
        i for i in range(config.vocab_size - 1024, config.vocab_size) if i != eos_token_id
    ]
    suppress_idx = mx.array(suppress_tokens, dtype=mx.int32)
    suppressed_logits = mx.put_along_axis(
        last_logits,
        suppress_idx[None, :],
        mx.array(float("-inf"), last_logits.dtype),
        axis=-1,
    )
    step0_token = int(mx.argmax(suppressed_logits, axis=-1).tolist()[0])

    logits_row = suppressed_logits[0].tolist()

    payload = {
        "input_embeds_stats": _stats("input_embeds", input_embeds),
        "post_prefill_hidden_stats": _stats("hidden_last", last_hidden),
        "step0_logits_stats": _stats("logits_last", suppressed_logits),
        "step0_logits_topk": _topk(logits_row, 20),
        "step0_token": step0_token,
        "eos_token_id": int(eos_token_id),
        "vocab_size": int(config.vocab_size),
    }

    out = Path(args.output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
