from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def build_model(output_channels: int, weights: list[float]) -> ct.models.MLModel:
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
    return ct.models.MLModel(builder.spec)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("iris-3class", 3, IRIS3_WEIGHTS, IRIS3_SAMPLES),
        ("iris-binary", 2, IRIS2_WEIGHTS, IRIS2_SAMPLES),
    ]

    metadata = {"models": []}
    for name, out_ch, weights, samples in specs:
        path = out_dir / f"{name}.mlmodel"
        build_model(out_ch, weights).save(str(path))
        metadata["models"].append(
            {
                "name": name,
                "path": str(path),
                "input_name": "x",
                "output_name": "y",
                "input_shape": [5, 1, 1],
                "output_count": out_ch,
                "samples": samples,
            }
        )

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
