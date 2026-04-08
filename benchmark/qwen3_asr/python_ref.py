"""Qwen3-ASR e2e reference: generate test WAV, run moona3k pipeline, save intermediates.

Saves to benchmark/qwen3_asr/ref_data/:
  - test_audio.bin      (float32 PCM, 16kHz mono)
  - mel_shape.json      (mel spectrogram shape)
  - mel.bin             (float32 mel spectrogram, row-major)
  - encoder_shape.json  (encoder output shape)
  - encoder.bin         (float32 encoder output, row-major)
  - text.txt            (final decoded text)
  - prompt_tokens.json  (prompt token IDs)
  - prefill_logits.json (prefill logits top-10 + first token)
  - gen_tokens.json     (all generated token IDs + decode)
  - metadata.json       (timing, config summary)

Usage:
    uv run --no-project \
        --with "mlx-qwen3-asr @ git+https://github.com/moona3k/mlx-qwen3-asr.git" \
        --with numpy \
        python vendors/dart-mlx-ffi/benchmark/qwen3_asr/python_ref.py \
            [--audio path/to/audio.wav]

Note: The mlx-community/Qwen3-ASR-0.6B-8bit snapshot has a partially
quantized layout (text decoder quantized, audio encoder float). moona3k's
load_model() calls nn.quantize() on everything which breaks. We load
manually with selective quantization.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def generate_test_audio(duration_s: float = 3.0, sr: int = 16000) -> np.ndarray:
    """Generate a test signal: spoken-frequency sine sweep + noise."""
    t = np.linspace(0, duration_s, int(sr * duration_s), dtype=np.float32)
    freq = 200 + 600 * t / duration_s
    phase = 2 * np.pi * np.cumsum(freq) / sr
    signal = 0.3 * np.sin(phase).astype(np.float32)
    np.random.seed(42)
    signal += 0.02 * np.random.randn(len(signal)).astype(np.float32)
    return signal


def write_wav(path: str, audio: np.ndarray, sr: int = 16000) -> None:
    """Write a 16-bit PCM WAV file."""
    pcm16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    data_bytes = pcm16.tobytes()
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(data_bytes)))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<I", sr))
        f.write(struct.pack("<I", sr * 2))
        f.write(struct.pack("<H", 2))
        f.write(struct.pack("<H", 16))
        f.write(b"data")
        f.write(struct.pack("<I", len(data_bytes)))
        f.write(data_bytes)


def save_float32_bin(path: str, arr: np.ndarray) -> None:
    """Save numpy array as raw float32 bytes."""
    arr.astype(np.float32).tofile(path)


def load_model_selective_quant(model_path: str):
    """Load moona3k model with selective quantization and fixed lm_head.

    The mlx-community/Qwen3-ASR-0.6B-8bit snapshot only quantizes the
    text decoder (model.*), not the audio encoder (audio_tower.*). moona3k's
    load_model() calls nn.quantize() on everything which breaks. We handle
    this by quantizing only the text decoder modules.

    Additionally, the quantized snapshot has tie_word_embeddings=True but
    no lm_head.weight in safetensors. After quantization, embed_tokens
    becomes QuantizedEmbedding (uint32) while lm_head retains random init.
    We fix this by dequantizing embed_tokens and assigning the result to
    lm_head.weight.
    """
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

    # Selective quantization: only quantize the text decoder (model.*),
    # not the audio encoder (audio_tower.*).
    nn.quantize(model.model, bits=bits, group_size=group_size)

    model.load_weights(list(weights.items()), strict=False)

    # Fix lm_head weight tying: dequantize embed_tokens -> assign to lm_head.
    # After nn.quantize + load_weights, embed_tokens has the correct quantized
    # weights from safetensors, but lm_head.weight is still random init (since
    # safetensors has no lm_head.weight key due to tie_word_embeddings=True).
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

    setattr(model, "_source_model_id", model_path)
    setattr(model, "_resolved_model_path", str(resolved))

    return model, config, str(resolved)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", help="Path to input WAV (16kHz mono)")
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen3-ASR-0.6B-8bit",
        help="Model path or HF repo",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "ref_data"),
        help="Output directory for reference data",
    )
    args = parser.parse_args()
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load or generate test audio.
    if args.audio:
        print(f"Loading audio from {args.audio}", file=sys.stderr)
        with open(args.audio, "rb") as f:
            data = f.read()
        if data[:4] != b"RIFF":
            raise ValueError("Not a WAV file")
        pos = 12
        while pos + 8 <= len(data):
            cid = data[pos : pos + 4]
            csz = struct.unpack_from("<I", data, pos + 4)[0]
            if cid == b"data":
                raw = data[pos + 8 : pos + 8 + csz]
                audio_np = (
                    np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                )
                break
            pos += 8 + csz + (csz % 2)
        else:
            raise ValueError("No data chunk")
    else:
        print("Generating synthetic test audio (3s sine sweep)", file=sys.stderr)
        audio_np = generate_test_audio(3.0)
        write_wav(os.path.join(out_dir, "test_audio.wav"), audio_np)

    save_float32_bin(os.path.join(out_dir, "test_audio.bin"), audio_np)
    print(
        f"  Audio: {len(audio_np)} samples ({len(audio_np) / 16000:.2f}s)",
        file=sys.stderr,
    )

    # 2. Load model with selective quantization.
    print("Loading model (selective quant)...", file=sys.stderr)
    model, config, model_resolved_path = load_model_selective_quant(args.model)
    print("  Model loaded", file=sys.stderr)

    # 3. Compute mel spectrogram.
    from mlx_qwen3_asr.audio import log_mel_spectrogram

    t0 = time.perf_counter()
    audio_mx = mx.array(audio_np)
    mel_mx = log_mel_spectrogram(audio_mx)
    mx.eval(mel_mx)
    mel_time = time.perf_counter() - t0
    mel_np = np.array(mel_mx)
    print(f"  Mel: shape={mel_np.shape}, time={mel_time * 1000:.1f}ms", file=sys.stderr)
    save_float32_bin(os.path.join(out_dir, "mel.bin"), mel_np)
    with open(os.path.join(out_dir, "mel_shape.json"), "w") as f:
        json.dump(list(mel_np.shape), f)

    # 4. Run encoder (audio_tower in moona3k).
    t0 = time.perf_counter()
    n_frames = mel_np.shape[1]
    mel_batched = mel_mx[None, :, :]  # (1, 128, frames)
    feature_lens = mx.array([n_frames])
    encoder_out, _ = model.audio_tower(mel_batched, feature_lens)
    mx.eval(encoder_out)
    enc_time = time.perf_counter() - t0
    enc_np = np.array(encoder_out)
    print(
        f"  Encoder: shape={enc_np.shape}, time={enc_time * 1000:.1f}ms",
        file=sys.stderr,
    )
    save_float32_bin(os.path.join(out_dir, "encoder.bin"), enc_np)
    with open(os.path.join(out_dir, "encoder_shape.json"), "w") as f:
        json.dump(list(enc_np.shape), f)

    # 5. Direct generation using our loaded model (not transcribe()).
    # This bypasses moona3k's transcribe() which may reload the model.
    from mlx_qwen3_asr.tokenizer import Tokenizer
    from mlx_qwen3_asr.generate import generate, GenerationConfig

    tokenizer = Tokenizer(model_resolved_path)
    n_audio_tokens = encoder_out.shape[1]
    print(f"  n_audio_tokens: {n_audio_tokens}", file=sys.stderr)

    # Build prompt tokens (no language forcing, empty context).
    prompt_tokens = tokenizer.build_prompt_tokens(
        n_audio_tokens=n_audio_tokens,
        language=None,
        context="",
    )
    print(f"  Prompt tokens ({len(prompt_tokens)}): {prompt_tokens}", file=sys.stderr)

    # Save prompt tokens.
    with open(os.path.join(out_dir, "prompt_tokens.json"), "w") as f:
        json.dump(
            {
                "tokens": prompt_tokens,
                "length": len(prompt_tokens),
                "n_audio_tokens": n_audio_tokens,
                "decoded": tokenizer.decode(prompt_tokens),
            },
            f,
            indent=2,
        )

    # Build position IDs (matching transcribe.py:704-705).
    input_ids = mx.array([prompt_tokens])
    seq_len = input_ids.shape[1]
    positions = mx.arange(seq_len)[None, :]
    position_ids = mx.stack([positions, positions, positions], axis=1)
    print(f"  input_ids shape: {input_ids.shape}", file=sys.stderr)
    print(f"  position_ids shape: {position_ids.shape}", file=sys.stderr)

    # Run prefill manually to capture logits.
    print("  Running prefill...", file=sys.stderr)
    max_seq_len = seq_len + 448
    cache = model.create_cache(max_seq_len=max_seq_len)

    prefill_logits = model.prefill(
        input_ids=input_ids,
        audio_features=encoder_out,
        position_ids=position_ids,
        cache=cache,
    )
    mx.eval(prefill_logits)

    # Get logits shape and top-10 values.
    logits_flat = prefill_logits.reshape(-1)
    top_k = 10
    # Use argsort to get top-k indices (descending).
    sorted_idx = mx.argsort(logits_flat)[::-1]
    top_indices = sorted_idx[:top_k]
    top_values = [float(logits_flat[int(i)]) for i in top_indices.tolist()]
    top_indices_list = [int(i) for i in top_indices.tolist()]

    first_token = int(mx.argmax(logits_flat).item())
    first_token_decoded = tokenizer.decode([first_token])
    print(f"  Prefill logits shape: {prefill_logits.shape}", file=sys.stderr)
    print(
        f"  First token (greedy): {first_token} = '{first_token_decoded}'",
        file=sys.stderr,
    )
    print(f"  Top-10 indices: {top_indices_list}", file=sys.stderr)
    print(f"  Top-10 values: {top_values}", file=sys.stderr)

    # Save prefill logits info.
    with open(os.path.join(out_dir, "prefill_logits.json"), "w") as f:
        json.dump(
            {
                "shape": list(prefill_logits.shape),
                "first_token": first_token,
                "first_token_decoded": first_token_decoded,
                "top_10_indices": top_indices_list,
                "top_10_values": top_values,
                "top_10_decoded": [tokenizer.decode([i]) for i in top_indices_list],
            },
            f,
            indent=2,
        )

    # Now run full generation.
    print("  Running full generation...", file=sys.stderr)
    t0 = time.perf_counter()
    gen_config = GenerationConfig(
        max_new_tokens=448,
        temperature=0.0,
    )
    gen_tokens = generate(
        model=model,
        input_ids=input_ids,
        audio_features=encoder_out,
        position_ids=position_ids,
        config=gen_config,
    )
    gen_time = time.perf_counter() - t0
    raw_text = tokenizer.decode(gen_tokens)
    print(
        f"  Generated {len(gen_tokens)} tokens in {gen_time * 1000:.1f}ms",
        file=sys.stderr,
    )
    print(f"  Raw text: '{raw_text}'", file=sys.stderr)

    # Parse ASR output.
    from mlx_qwen3_asr.tokenizer import parse_asr_output

    lang, parsed_text = parse_asr_output(raw_text, user_language=None)
    print(f"  Parsed: lang='{lang}', text='{parsed_text}'", file=sys.stderr)

    # Save generated tokens.
    with open(os.path.join(out_dir, "gen_tokens.json"), "w") as f:
        json.dump(
            {
                "tokens": gen_tokens,
                "length": len(gen_tokens),
                "raw_text": raw_text,
                "parsed_language": lang,
                "parsed_text": parsed_text,
                "gen_time_ms": gen_time * 1000,
            },
            f,
            indent=2,
        )

    with open(os.path.join(out_dir, "text.txt"), "w") as f:
        f.write(parsed_text)

    # 6. Save metadata.
    metadata = {
        "audio_samples": len(audio_np),
        "audio_duration_s": len(audio_np) / 16000,
        "mel_shape": list(mel_np.shape),
        "encoder_shape": list(enc_np.shape),
        "text": parsed_text,
        "raw_text": raw_text,
        "mel_time_ms": mel_time * 1000,
        "encoder_time_ms": enc_time * 1000,
        "gen_time_ms": gen_time * 1000,
        "model_path": args.model,
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nReference data saved to {out_dir}", file=sys.stderr)
    for fname in sorted(os.listdir(out_dir)):
        size = os.path.getsize(os.path.join(out_dir, fname))
        print(f"  {fname}: {size:,} bytes", file=sys.stderr)


if __name__ == "__main__":
    main()
