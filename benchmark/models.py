from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any

import mlx.core as mx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = {
        "runtime": "python_mlx",
        "mlx_version": getattr(mx, "__version__", "unknown"),
        "device": str(mx.default_device()),
        "metal": bool(mx.metal.is_available()),
        "warmup": args.warmup,
        "iters": args.iters,
        "models": [
            bench_mlp(args.warmup, args.iters),
            bench_conv(args.warmup, args.iters),
            bench_attention(args.warmup, args.iters),
        ],
    }

    if args.json:
        print(json.dumps(report))
        return

    print("python-mlx model benchmark")
    print(f"mlx_version: {report['mlx_version']}")
    print(f"device: {report['device']}")
    print(f"metal: {report['metal']}")
    print(f"warmup: {report['warmup']}")
    print(f"iters: {report['iters']}")
    for model in report["models"]:
        print()
        print(f"{model['name']}: {model['description']}")
        print(f"  output_shape: {model['output_shape']}")
        print(f"  output_preview: {model['output_preview']}")
        print(f"  per_iter_ms: {model['per_iter_ms']:.4f}")
        print(f"  peak_bytes_delta: {model['peak_bytes_delta']}")


def bench_mlp(warmup: int, iters: int) -> dict[str, Any]:
    input_shape = [32, 64]
    hidden_shape = [64, 96]
    output_shape = [96, 6]

    input_values = values(numel(input_shape), seed=3, divisor=64)
    w1_values = values(numel(hidden_shape), seed=5, divisor=128)
    b1_values = values(96, seed=7, divisor=256)
    w2_values = values(numel(output_shape), seed=11, divisor=128)
    b2_values = values(6, seed=13, divisor=256)

    x = array(input_values, input_shape)
    w1 = array(w1_values, hidden_shape)
    b1 = array(b1_values, [1, 96])
    w2 = array(w2_values, output_shape)
    b2 = array(b2_values, [1, 6])

    def forward() -> Any:
        hidden_linear = mx.add(mx.matmul(x, w1), b1)
        hidden = mx.sigmoid(hidden_linear)
        logits = mx.add(mx.matmul(hidden, w2), b2)
        output = mx.softmax(logits, axis=1)
        mx.eval(output)
        mx.synchronize()
        return output

    return measure(
        name="tiny_mlp",
        description="matmul + sigmoid + softmax",
        warmup=warmup,
        iters=iters,
        input_shapes={
            "input": input_shape,
            "w1": hidden_shape,
            "b1": [1, 96],
            "w2": output_shape,
            "b2": [1, 6],
        },
        input_preview=input_values,
        forward=forward,
    )


def bench_conv(warmup: int, iters: int) -> dict[str, Any]:
    input_shape = [8, 32, 32, 3]
    kernel_shape = [8, 3, 3, 3]
    bias_shape = [1, 1, 1, 8]
    head_shape = [8, 5]

    input_values = values(numel(input_shape), seed=17, divisor=128)
    kernel_values = values(numel(kernel_shape), seed=19, divisor=256)
    bias_values = values(numel(bias_shape), seed=23, divisor=512)
    head_values = values(numel(head_shape), seed=29, divisor=256)
    head_bias_values = values(5, seed=31, divisor=512)

    x = array(input_values, input_shape)
    kernel = array(kernel_values, kernel_shape)
    bias = array(bias_values, bias_shape)
    head = array(head_values, head_shape)
    head_bias = array(head_bias_values, [1, 5])

    def forward() -> Any:
        conv = mx.conv2d(x, kernel, padding=(1, 1))
        biased = mx.add(conv, bias)
        activated = mx.sigmoid(biased)
        pooled_h = mx.mean(activated, axis=1)
        pooled_w = mx.mean(pooled_h, axis=1)
        logits = mx.add(mx.matmul(pooled_w, head), head_bias)
        output = mx.softmax(logits, axis=1)
        mx.eval(output)
        mx.synchronize()
        return output

    return measure(
        name="tiny_conv",
        description="conv2d + sigmoid + global average pooling + softmax",
        warmup=warmup,
        iters=iters,
        input_shapes={
            "input": input_shape,
            "kernel": kernel_shape,
            "bias": bias_shape,
            "head": head_shape,
            "head_bias": [1, 5],
        },
        input_preview=input_values,
        forward=forward,
    )


def bench_attention(warmup: int, iters: int) -> dict[str, Any]:
    q_shape = [2, 4, 16, 64]
    norm_shape = [64]
    head_shape = [64, 4]

    q_values = values(numel(q_shape), seed=37, divisor=128)
    k_values = values(numel(q_shape), seed=41, divisor=128)
    v_values = values(numel(q_shape), seed=43, divisor=128)
    norm_weight_values = values(numel(norm_shape), seed=47, divisor=256)
    norm_bias_values = values(numel(norm_shape), seed=53, divisor=512)
    head_values = values(numel(head_shape), seed=59, divisor=256)
    head_bias_values = values(4, seed=61, divisor=512)

    q = array(q_values, q_shape)
    k = array(k_values, q_shape)
    v = array(v_values, q_shape)
    norm_weight = array(norm_weight_values, norm_shape)
    norm_bias = array(norm_bias_values, norm_shape)
    head = array(head_values, head_shape)
    head_bias = array(head_bias_values, [1, 4])

    def forward() -> Any:
        attention = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.25)
        normalized = mx.fast.layer_norm(attention, norm_weight, norm_bias, 1e-5)
        pooled_seq = mx.mean(normalized, axis=2)
        pooled_heads = mx.mean(pooled_seq, axis=1)
        logits = mx.add(mx.matmul(pooled_heads, head), head_bias)
        output = mx.softmax(logits, axis=1)
        mx.eval(output)
        mx.synchronize()
        return output

    return measure(
        name="tiny_attention",
        description="scaled dot product attention + layer norm + softmax",
        warmup=warmup,
        iters=iters,
        input_shapes={
            "q": q_shape,
            "k": q_shape,
            "v": q_shape,
            "norm_weight": norm_shape,
            "norm_bias": norm_shape,
            "head": head_shape,
            "head_bias": [1, 4],
        },
        input_preview=q_values,
        forward=forward,
    )


def measure(
    *,
    name: str,
    description: str,
    warmup: int,
    iters: int,
    input_shapes: dict[str, list[int]],
    input_preview: list[float],
    forward,
) -> dict[str, Any]:
    for _ in range(warmup):
        forward()

    mx.reset_peak_memory()
    before_peak = int(mx.get_peak_memory())
    start = time.perf_counter()
    last = None
    for _ in range(iters):
        last = forward()
    total_ms = (time.perf_counter() - start) * 1000.0
    assert last is not None
    output_flat = flatten(last.tolist())
    after_peak = int(mx.get_peak_memory())
    return {
        "name": name,
        "description": description,
        "input_shapes": input_shapes,
        "input_preview": preview(input_preview),
        "output_shape": list(last.shape),
        "output_preview": preview(output_flat),
        "output_flat": output_flat,
        "output_sum": sum(output_flat),
        "total_ms": total_ms,
        "per_iter_ms": total_ms / iters,
        "peak_bytes_delta": after_peak - before_peak,
    }


def array(data: list[float], shape: list[int]) -> Any:
    return mx.array(data, dtype=mx.float32).reshape(shape)


def values(count: int, *, seed: int, divisor: int) -> list[float]:
    result: list[float] = []
    for index in range(count):
        numerator = ((index * (seed * 2 + 1) + seed * 7 + 13) % 257) - 128
        result.append(numerator / divisor)
    return result


def preview(items: list[float], limit: int = 8) -> list[float]:
    return items[: min(limit, len(items))]


def numel(shape: list[int]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def flatten(value: Any) -> list[float]:
    if isinstance(value, list):
        result: list[float] = []
        for item in value:
            result.extend(flatten(item))
        return result
    return [float(value)]


if __name__ == "__main__":
    main()
