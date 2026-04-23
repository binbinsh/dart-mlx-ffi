from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

from huggingface_hub import snapshot_download
import mlx.core as mx
import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent

if str(ROOT / "benchmark") not in sys.path:
    sys.path.insert(0, str(ROOT / "benchmark"))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import benchmark_dart_export, compare_lists, cleanup_mlx, resolve_model_path
from decode import (
    ViterbiDecoder,
    decode_prediction,
    load_label_info,
    load_viterbi_biases,
)
from mlx_model import load_privacy_filter_model

MODEL_ID = "openai/privacy-filter"
BASE_PROMPT = (
    "Alice was born on 1990-01-02, lives at 123 Main Street, "
    "uses alice@example.com, and her phone number is 555-010-1234. "
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark OpenAI privacy-filter in Python MLX vs Dart MLX import.",
    )
    parser.add_argument(
        "--checkpoint",
        help="Local privacy-filter snapshot root or original/ directory.",
    )
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--moe-chunk-size", type=int, default=256)
    parser.add_argument(
        "--fast-attention",
        action="store_true",
        help="Use MLX fused SDPA. Faster, but currently has looser PyTorch parity.",
    )
    parser.add_argument(
        "--moe-matmul-dtype",
        choices=("float32", "bfloat16"),
        default="bfloat16",
        help="Accumulator/input dtype for gathered MoE matmuls.",
    )
    parser.add_argument(
        "--decode-mode",
        choices=("viterbi", "argmax", "none"),
        default="viterbi",
        help="Decode Python MLX logits into privacy spans for the report.",
    )
    parser.add_argument(
        "--viterbi-calibration",
        help="Optional viterbi_calibration.json override.",
    )
    parser.add_argument(
        "--out-dir",
        default="benchmark/out/privacy_filter",
        help="Directory for exported bundle and comparison report.",
    )
    args = parser.parse_args()
    validate_args(args)

    checkpoint = resolve_checkpoint(args.checkpoint)
    paths, _config, model = load_privacy_filter_model(
        checkpoint,
        moe_chunk_size=args.moe_chunk_size,
        use_fast_attention=args.fast_attention,
        moe_matmul_dtype=args.moe_matmul_dtype,
    )
    tokenizer = Tokenizer.from_file(str(paths.tokenizer_dir / "tokenizer.json"))
    prompt, token_ids, token_offsets = build_prompt(tokenizer, args.seq_len)
    tokens = mx.array([token_ids], dtype=mx.int32)

    python_values, python_ms, output_shape = benchmark_python(
        model,
        tokens,
        warmup=args.warmup,
        iters=args.iters,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    export_path = out_dir / "function.mlxfn"
    input_path = out_dir / "inputs.safetensors"
    if export_path.exists():
        export_path.unlink()
    if input_path.exists():
        input_path.unlink()

    def forward(input_ids):
        return model(input_ids)

    mx.export_function(str(export_path), forward, tokens)
    mx.save_safetensors(str(input_path), {"input_ids": tokens})

    dart_values, dart_ms = benchmark_dart_export(
        export_path=export_path,
        input_path=input_path,
        mx_module=mx,
        warmup=args.warmup,
        iters=args.iters,
    )
    max_diff, mean_diff = compare_lists(python_values, dart_values)
    decoded = decode_logits(
        values=python_values,
        shape=output_shape,
        token_offsets=token_offsets,
        text=prompt,
        paths=paths,
        mode=args.decode_mode,
        calibration_path=args.viterbi_calibration,
    )

    report = {
        "model_id": MODEL_ID,
        "snapshot_path": str(Path(checkpoint).resolve()),
        "model_dir": str(paths.model_dir),
        "tokenizer_dir": str(paths.tokenizer_dir),
        "seq_len": len(token_ids),
        "warmup": args.warmup,
        "iters": args.iters,
        "moe_chunk_size": args.moe_chunk_size,
        "moe_matmul_dtype": args.moe_matmul_dtype,
        "fast_attention": args.fast_attention,
        "prompt": prompt,
        "tokens": token_ids,
        "input_shape": list(tokens.shape),
        "output_shape": output_shape,
        "python_ms": python_ms,
        "dart_ms": dart_ms,
        "python_preview": preview(python_values),
        "dart_preview": preview(dart_values),
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "decode": decoded,
        "export_path": str(export_path),
        "input_path": str(input_path),
    }
    report_path = out_dir / "compare.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    cleanup_mlx(mx)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report_path={report_path}")


def resolve_checkpoint(raw: str | None) -> Path:
    if raw:
        return Path(raw).expanduser().resolve()
    return resolve_model_path(MODEL_ID, lambda model_id: Path(snapshot_download(model_id)))


def validate_args(args: argparse.Namespace) -> None:
    if args.seq_len <= 0:
        raise ValueError("--seq-len must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.iters <= 0:
        raise ValueError("--iters must be positive")
    if args.moe_chunk_size <= 0:
        raise ValueError("--moe-chunk-size must be positive")


def build_prompt(tokenizer, target_len: int) -> tuple[str, list[int], list[tuple[int, int]]]:
    if target_len <= 0:
        raise ValueError("--seq-len must be positive")
    text = BASE_PROMPT
    encoded = tokenizer.encode(text)
    while len(encoded.ids) < target_len:
        text += BASE_PROMPT
        encoded = tokenizer.encode(text)
    token_ids = [int(v) for v in encoded.ids[:target_len]]
    token_offsets = [(int(start), int(end)) for start, end in encoded.offsets[:target_len]]
    prompt_end = max((end for _start, end in token_offsets), default=0)
    prompt = text[:prompt_end]
    return prompt, token_ids, token_offsets


def decode_logits(
    *,
    values: list[float],
    shape: list[int],
    token_offsets: list[tuple[int, int]],
    text: str,
    paths,
    mode: str,
    calibration_path: str | None,
) -> dict[str, object] | None:
    if mode == "none":
        return None
    label_info = load_label_info(paths.root_dir, paths.model_dir)
    biases = load_viterbi_biases(
        paths.root_dir,
        paths.model_dir,
        calibration_path=calibration_path,
    )
    decoder = ViterbiDecoder(label_info, biases) if mode == "viterbi" else None
    prediction = decode_prediction(
        np.array(values, dtype=np.float32).reshape(shape),
        token_offsets=token_offsets,
        text=text,
        label_info=label_info,
        decoder=decoder,
    )
    return {
        "mode": mode,
        "viterbi_biases": biases if mode == "viterbi" else None,
        "token_labels": list(prediction.token_labels),
        "detected_spans": [
            {
                "label": span.label,
                "start": span.start,
                "end": span.end,
                "text": span.text,
                "placeholder": span.placeholder,
            }
            for span in prediction.detected_spans
        ],
        "redacted_text": prediction.redacted_text,
    }


def benchmark_python(
    model,
    tokens: mx.array,
    *,
    warmup: int,
    iters: int,
) -> tuple[list[float], float, list[int]]:
    for _ in range(warmup):
        logits = model(tokens)
        mx.eval(logits)
        mx.synchronize()

    started = time.perf_counter()
    last = None
    for _ in range(iters):
        last = model(tokens)
        mx.eval(last)
        mx.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0 / iters
    if last is None:
        raise RuntimeError("No benchmark iterations executed.")
    values = [float(v) for v in last.reshape([-1]).tolist()]
    return values, elapsed_ms, list(last.shape)


def preview(values: list[float], limit: int = 8) -> list[float]:
    return values[: min(limit, len(values))]


if __name__ == "__main__":
    main()
