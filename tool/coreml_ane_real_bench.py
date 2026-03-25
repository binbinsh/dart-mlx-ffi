from __future__ import annotations

import argparse
import json
import time
from typing import Any

import coremltools as ct
import numpy as np
from coremltools.models import datatypes
from coremltools.models.neural_network import NeuralNetworkBuilder


IRIS3_WEIGHTS = [
    -2.4592809677124023,
    2.380864381790161,
    -4.534583568572998,
    -4.283433437347412,
    -0.3940414488315582,
    1.8275331258773804,
    -0.463300496339798,
    -1.8468471765518188,
    -1.7169692516326904,
    5.267071723937988,
    0.6317527890205383,
    -1.917563796043396,
    6.3814287185668945,
    6.00039005279541,
    -4.8730363845825195,
]

IRIS2_WEIGHTS = [
    -0.895805299282074,
    1.4704875946044922,
    -2.066481351852417,
    -1.8808748722076416,
    -2.147366523742676,
    0.8958030939102173,
    -1.470487117767334,
    2.0664796829223633,
    1.8808743953704834,
    2.1473684310913086,
]

IRIS3_SAMPLES = [
    {
        "label": 0,
        "input5": [-0.9006831645965576, 1.032057285308838, -1.3412725925445557, -1.3129769563674927, 1.0],
        "logits": [15.984342002859592, 7.8743573148955335, -23.858690481618893],
    },
    {
        "label": 1,
        "input5": [1.4015071392059326, 0.33784863352775574, 0.5352959036827087, 0.26469850540161133, 1.0],
        "logits": [-6.597531942408246, 6.228758084325478, 0.3687702827411634],
    },
]

IRIS2_SAMPLES = [
    {
        "label": 0,
        "input5": [-0.9006831645965576, 1.032057285308838, -1.3412725925445557, -1.3129769563674927, 1.0],
        "logits": [5.41835782830826, -5.41835057792872],
    },
    {
        "label": 1,
        "input5": [1.4015071392059326, 0.33784863352775574, 0.5352959036827087, 0.26469850540161133, 1.0],
        "logits": [-4.51008559177248, 4.510083549785989],
    },
]


def build_conv_model(output_channels: int, weights: list[float]) -> ct.models.MLModel:
    builder = NeuralNetworkBuilder(
        input_features=[("x", datatypes.Array(5, 1, 1))],
        output_features=[("y", datatypes.Array(output_channels, 1, 1))],
    )
    W = np.array(weights, dtype=np.float32).reshape(output_channels, 5, 1, 1)
    builder.add_convolution(
        name="conv",
        kernel_channels=5,
        output_channels=output_channels,
        height=1,
        width=1,
        stride_height=1,
        stride_width=1,
        border_mode="valid",
        groups=1,
        W=W,
        b=None,
        has_bias=False,
        input_name="x",
        output_name="y",
    )
    return ct.models.MLModel(builder.spec, compute_units=ct.ComputeUnit.CPU_AND_NE)


def argmax(values: list[float]) -> int:
    best_index = 0
    best_value = values[0]
    for index, value in enumerate(values[1:], start=1):
        if value > best_value:
            best_index = index
            best_value = value
    return best_index


def bench_model(
    *,
    name: str,
    output_channels: int,
    weights: list[float],
    samples: list[dict[str, Any]],
    warmup: int,
    iters: int,
    max_ms: float | None,
) -> dict[str, Any]:
    model = build_conv_model(output_channels, weights)
    sample_arrays = [
        np.array(sample["input5"], dtype=np.float32).reshape(5, 1, 1) for sample in samples
    ]

    for sample in samples:
        output = model.predict({"x": np.array(sample["input5"], dtype=np.float32).reshape(5, 1, 1)})["y"]
        values = output.flatten().tolist()
        assert len(values) == output_channels
        assert argmax(values) == sample["label"]

    for index in range(warmup):
        model.predict({"x": sample_arrays[index % len(sample_arrays)]})

    start = time.perf_counter()
    last_output = None
    for index in range(iters):
        last_output = model.predict({"x": sample_arrays[index % len(sample_arrays)]})["y"]
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    per_iter_ms = elapsed_ms / iters
    if max_ms is not None and per_iter_ms > max_ms:
        raise SystemExit(f"{name} exceeded max ms/iter: {per_iter_ms:.4f} > {max_ms:.4f}")

    return {
        "name": name,
        "output_channels": output_channels,
        "per_iter_ms": per_iter_ms,
        "last_output": last_output.flatten().tolist() if last_output is not None else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--max-ms", type=float, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = {
        "runtime": "coreml_cpu_and_ne",
        "coremltools_version": ct.__version__,
        "warmup": args.warmup,
        "iters": args.iters,
        "models": [
            bench_model(
                name="iris-3class",
                output_channels=3,
                weights=IRIS3_WEIGHTS,
                samples=IRIS3_SAMPLES,
                warmup=args.warmup,
                iters=args.iters,
                max_ms=args.max_ms,
            ),
            bench_model(
                name="iris-binary",
                output_channels=2,
                weights=IRIS2_WEIGHTS,
                samples=IRIS2_SAMPLES,
                warmup=args.warmup,
                iters=args.iters,
                max_ms=args.max_ms,
            ),
        ],
    }

    if args.json:
        print(json.dumps(report))
        return

    print("coreml ane real benchmark")
    print(f"coremltools_version: {report['coremltools_version']}")
    print(f"warmup: {report['warmup']}")
    print(f"iters: {report['iters']}")
    for model in report["models"]:
        print(f"{model['name']}: {model['per_iter_ms']:.4f} ms/iter")
        print(f"  last_output: {model['last_output']}")


if __name__ == "__main__":
    main()
