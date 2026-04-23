#!/usr/bin/env python3
"""Dump greedy Qwen3-TTS ICL code frames for decoder parity checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
from mlx_audio.tts.utils import load_model
from mlx_audio.utils import load_audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump Qwen3-TTS greedy code frames")
    parser.add_argument("model_path", help="Local Qwen3-TTS bundle path")
    parser.add_argument("audio_path", help="Reference WAV path")
    parser.add_argument("ref_text", help="Reference transcript")
    parser.add_argument("text", help="Target text")
    parser.add_argument("output_path", help="Output JSON path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = load_model(model_path=Path(args.model_path).expanduser().resolve())
    ref_audio = load_audio(str(Path(args.audio_path).expanduser().resolve()), sample_rate=model.sample_rate)

    input_embeds, _trailing_text_hidden, tts_pad_embed, _ref_codes = model._prepare_icl_generation_inputs(  # noqa: SLF001
        text=args.text,
        ref_audio=ref_audio,
        ref_text=args.ref_text,
        language="auto",
    )
    cache = model.talker.make_cache()
    code_cache = model.talker.code_predictor.make_cache()
    generated_codes = []
    generated_token_ids = []
    config = model.config.talker_config
    eos_token_id = config.codec_eos_token_id
    suppress_tokens = [
        i for i in range(config.vocab_size - 1024, config.vocab_size) if i != eos_token_id
    ]
    target_token_count = len(model.tokenizer.encode(args.text))
    effective_max_tokens = min(4096, max(75, target_token_count * 6))
    for _ in range(effective_max_tokens):
        logits, hidden = model.talker(input_embeds, cache=cache)
        next_token = model._sample_token(  # noqa: SLF001
            logits,
            temperature=0.0,
            top_k=50,
            top_p=1.0,
            repetition_penalty=1.5,
            generated_tokens=(generated_token_ids if generated_token_ids else None),
            suppress_tokens=suppress_tokens,
            eos_token_id=eos_token_id,
        )
        if int(next_token[0, 0]) == eos_token_id:
            break
        generated_token_ids.append(int(next_token[0, 0]))
        code_tokens = [next_token]
        code_hidden = hidden[:, -1:, :]
        for c in code_cache:
            c.keys = None
            c.values = None
            c.offset = 0
        for code_idx in range(config.num_code_groups - 1):
            if code_idx == 0:
                code_0_embed = model.talker.get_input_embeddings()(next_token)
                code_input = mx.concatenate([code_hidden, code_0_embed], axis=1)
            else:
                code_input = model.talker.code_predictor.codec_embedding[code_idx - 1](code_tokens[-1])
            code_logits, code_cache, _ = model.talker.code_predictor(
                code_input,
                cache=code_cache,
                generation_step=code_idx,
            )
            next_code = model._sample_token(  # noqa: SLF001
                code_logits,
                temperature=0.0,
                top_k=50,
                top_p=1.0,
            )
            code_tokens.append(next_code)
        all_codes = mx.concatenate(code_tokens, axis=1)
        generated_codes.append(all_codes)
        codec_embed = model.talker.get_input_embeddings()(next_token)
        for index, code in enumerate(code_tokens[1:]):
            codec_embed = codec_embed + model.talker.code_predictor.codec_embedding[index](code)
        input_embeds = tts_pad_embed + codec_embed
    if generated_codes:
        stacked = mx.stack(generated_codes, axis=1)
        frames = stacked.tolist()[0]
        payload = {"frames": frames}
    else:
        payload = {"frames": []}

    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
