from __future__ import annotations

import sys
from pathlib import Path
import time
from typing import Any

try:
    from ..common import add_vendor_to_path, cleanup_mlx, preview, resolve_model_path
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common import add_vendor_to_path, cleanup_mlx, preview, resolve_model_path

add_vendor_to_path("mlx-vlm")

import mlx.core as mx

TEXT_PROMPT = (
    "Summarize why MLX on Apple Silicon is useful for local inference, "
    "and mention memory efficiency and developer ergonomics."
)

MODEL_SPECS = [
    {
        "name": "qwen35_9b",
        "model_id": "mlx-community/Qwen3.5-9B-MLX-4bit",
        "backend": "mlx_vlm",
        "kind": "text",
    },
    {
        "name": "qwen35_35b_a3b",
        "model_id": "mlx-community/Qwen3.5-35B-A3B-4bit",
        "backend": "mlx_vlm",
        "kind": "text",
    },
]


def benchmark_model(
    spec: dict[str, Any],
    *,
    warmup: int,
    iters: int,
    seq_len: int,
) -> dict[str, Any]:
    from mlx_vlm.tokenizer_utils import load_tokenizer
    from mlx_vlm.utils import get_model_path, load_model

    model_path = resolve_model_path(spec["model_id"], get_model_path)
    model = load_model(model_path, lazy=False)
    tokenizer = load_tokenizer(model_path)
    token_ids = tokenizer.encode(TEXT_PROMPT)[:seq_len]
    tokens = mx.array(token_ids, dtype=mx.int32)[None, :]

    def forward() -> mx.array:
        logits = model(tokens).logits[:, -1, :16].astype(mx.float32)
        mx.eval(logits)
        mx.synchronize()
        return logits

    try:
        for _ in range(warmup):
            _ = forward()

        mx.reset_peak_memory()
        before_peak = int(mx.get_peak_memory())
        started = time.perf_counter()
        last = None
        for _ in range(iters):
            last = forward()
        total_ms = (time.perf_counter() - started) * 1000.0
        assert last is not None
        output_values = [float(v) for v in last.tolist()[0]]
        after_peak = int(mx.get_peak_memory())
        return {
            "name": spec["name"],
            "model_id": spec["model_id"],
            "backend": spec["backend"],
            "kind": spec["kind"],
            "snapshot_path": str(model_path.resolve()),
            "input_text": TEXT_PROMPT,
            "token_count": len(token_ids),
            "input_shape": [1, len(token_ids)],
            "output_kind": "logits",
            "output_shape": list(last.shape),
            "output_preview": preview(output_values),
            "output_values": output_values,
            "total_ms": total_ms,
            "per_iter_ms": total_ms / iters,
            "peak_bytes_delta": after_peak - before_peak,
        }
    finally:
        del tokens
        del tokenizer
        del model
        cleanup_mlx(mx)
