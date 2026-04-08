"""Qwen3-ASR speed benchmark (Python / moona3k).

Measures inference speed excluding model load time. Runs multiple iterations
for reliable timing data.

Usage:
    uv run --no-project \
        --with "mlx-qwen3-asr @ git+https://github.com/moona3k/mlx-qwen3-asr.git" \
        --with numpy \
        python vendors/dart-mlx-ffi/benchmark/qwen3_asr/speed_bench.py \
            --audio /tmp/speech_test_16k.wav [--iters 3]
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def load_model_selective_quant(model_path: str):
    """Load moona3k model with selective quantization and fixed lm_head."""
    from mlx_qwen3_asr.config import Qwen3ASRConfig
    from mlx_qwen3_asr.load_models import (
        _load_safetensors,
        _resolve_path,
        remap_weights,
    )
    from mlx_qwen3_asr.model import Qwen3ASRModel

    resolved = _resolve_path(model_path)
    with open(resolved / "config.json") as f:
        raw_config = json.load(f)
    config = Qwen3ASRConfig.from_dict(raw_config)
    quant_cfg = raw_config.get("quantization", {})
    bits = int(quant_cfg.get("bits", 8))
    group_size = int(quant_cfg.get("group_size", 64))

    weights = _load_safetensors(resolved)
    weights = remap_weights(weights)
    model = Qwen3ASRModel(config)
    nn.quantize(model.model, bits=bits, group_size=group_size)
    model.load_weights(list(weights.items()), strict=False)

    if config.text_config.tie_word_embeddings:
        embed = model.model.embed_tokens
        dequant_w = mx.dequantize(
            embed.weight,
            embed.scales,
            embed.biases,
            group_size=group_size,
            bits=bits,
        )
        model.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
        )
        model.lm_head.weight = dequant_w

    mx.eval(model.parameters())
    model.eval()
    return model, config, str(resolved)


def parse_wav(path: str) -> np.ndarray:
    """Parse 16-bit PCM WAV to float32."""
    with open(path, "rb") as f:
        data = f.read()
    pos = 12
    while pos + 8 <= len(data):
        cid = data[pos : pos + 4]
        csz = struct.unpack_from("<I", data, pos + 4)[0]
        if cid == b"data":
            raw = data[pos + 8 : pos + 8 + csz]
            return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        pos += 8 + csz + (csz % 2)
    raise ValueError("No data chunk")


def bench_one(model, config, resolved_path: str, audio_np: np.ndarray) -> dict:
    """Run one full inference and return timing breakdown."""
    from mlx_qwen3_asr.audio import log_mel_spectrogram
    from mlx_qwen3_asr.generate import GenerationConfig, generate
    from mlx_qwen3_asr.tokenizer import Tokenizer

    tokenizer = Tokenizer(resolved_path)

    # Mel
    t0 = time.perf_counter()
    audio_mx = mx.array(audio_np)
    mel = log_mel_spectrogram(audio_mx)
    mx.eval(mel)
    mel_ms = (time.perf_counter() - t0) * 1000

    # Encoder
    t0 = time.perf_counter()
    n_frames = mel.shape[1]
    mel_b = mel[None, :, :]
    feat_lens = mx.array([n_frames])
    enc_out, _ = model.audio_tower(mel_b, feat_lens)
    mx.eval(enc_out)
    enc_ms = (time.perf_counter() - t0) * 1000

    # Prompt
    n_audio = enc_out.shape[1]
    prompt = tokenizer.build_prompt_tokens(
        n_audio_tokens=n_audio,
        language=None,
        context="",
    )
    input_ids = mx.array([prompt])
    seq_len = input_ids.shape[1]
    positions = mx.arange(seq_len)[None, :]
    position_ids = mx.stack([positions, positions, positions], axis=1)

    # Prefill + decode
    t0 = time.perf_counter()
    cache = model.create_cache(max_seq_len=seq_len + 448)
    prefill_logits = model.prefill(
        input_ids=input_ids,
        audio_features=enc_out,
        position_ids=position_ids,
        cache=cache,
    )
    mx.eval(prefill_logits)
    prefill_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    gen_config = GenerationConfig(max_new_tokens=448, temperature=0.0)
    gen_tokens = generate(
        model=model,
        input_ids=input_ids,
        audio_features=enc_out,
        position_ids=position_ids,
        config=gen_config,
    )
    mx.eval(mx.array(gen_tokens))
    total_gen_ms = (time.perf_counter() - t0) * 1000

    text = tokenizer.decode(gen_tokens)
    n_tokens = len(gen_tokens)

    return {
        "mel_ms": mel_ms,
        "encoder_ms": enc_ms,
        "prefill_ms": prefill_ms,
        "generate_ms": total_gen_ms,
        "total_inference_ms": mel_ms + enc_ms + total_gen_ms,
        "n_tokens": n_tokens,
        "tokens_per_sec": n_tokens / (total_gen_ms / 1000) if total_gen_ms > 0 else 0,
        "text_preview": text[:80],
    }


def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR Python speed benchmark")
    parser.add_argument("--audio", required=True, help="16kHz mono WAV path")
    parser.add_argument("--model", default="mlx-community/Qwen3-ASR-0.6B-8bit")
    parser.add_argument("--iters", type=int, default=3, help="Number of iterations")
    args = parser.parse_args()

    audio_np = parse_wav(args.audio)
    duration_s = len(audio_np) / 16000
    print(f"Audio: {len(audio_np)} samples ({duration_s:.2f}s)", file=sys.stderr)

    print("Loading model...", file=sys.stderr)
    t0 = time.perf_counter()
    model, config, resolved = load_model_selective_quant(args.model)
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"Model loaded in {load_ms:.0f}ms", file=sys.stderr)

    # Warmup run
    print("Warmup run...", file=sys.stderr)
    _ = bench_one(model, config, resolved, audio_np)

    # Timed runs
    results = []
    for i in range(args.iters):
        print(f"Iteration {i + 1}/{args.iters}...", file=sys.stderr)
        r = bench_one(model, config, resolved, audio_np)
        results.append(r)
        print(
            f"  total={r['total_inference_ms']:.1f}ms  "
            f"mel={r['mel_ms']:.1f}ms  enc={r['encoder_ms']:.1f}ms  "
            f"gen={r['generate_ms']:.1f}ms  "
            f"tokens={r['n_tokens']}  "
            f"tok/s={r['tokens_per_sec']:.1f}",
            file=sys.stderr,
        )

    # Summary
    avg = {
        k: sum(r[k] for r in results) / len(results)
        for k in [
            "mel_ms",
            "encoder_ms",
            "prefill_ms",
            "generate_ms",
            "total_inference_ms",
            "tokens_per_sec",
        ]
    }
    avg["n_tokens"] = results[0]["n_tokens"]
    avg["audio_duration_s"] = duration_s
    avg["rtf"] = (avg["total_inference_ms"] / 1000) / duration_s
    avg["iters"] = args.iters
    avg["model_load_ms"] = load_ms
    avg["runtime"] = "python"
    avg["text_preview"] = results[0]["text_preview"]

    print(f"\n=== Python Benchmark Summary ({args.iters} iters) ===", file=sys.stderr)
    print(f"  Audio duration:    {duration_s:.2f}s", file=sys.stderr)
    print(f"  Model load:        {load_ms:.0f}ms", file=sys.stderr)
    print(f"  Mel:               {avg['mel_ms']:.1f}ms", file=sys.stderr)
    print(f"  Encoder:           {avg['encoder_ms']:.1f}ms", file=sys.stderr)
    print(f"  Prefill:           {avg['prefill_ms']:.1f}ms", file=sys.stderr)
    print(f"  Generate:          {avg['generate_ms']:.1f}ms", file=sys.stderr)
    print(f"  Total inference:   {avg['total_inference_ms']:.1f}ms", file=sys.stderr)
    print(f"  RTF:               {avg['rtf']:.4f}", file=sys.stderr)
    print(f"  Tokens:            {avg['n_tokens']}", file=sys.stderr)
    print(f"  Tokens/sec:        {avg['tokens_per_sec']:.1f}", file=sys.stderr)

    # JSON output to stdout
    print(json.dumps(avg, indent=2))


if __name__ == "__main__":
    main()
