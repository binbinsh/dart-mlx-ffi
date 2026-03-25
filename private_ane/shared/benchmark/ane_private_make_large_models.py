from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types


LARGE_MODEL_SPECS = (
    {
        "name": "64x-conv-512x64",
        "channels": 512,
        "spatial": 64,
        "depth": 64,
        "max_ms": 0.5,
    },
    {
        "name": "128x-conv-512x64",
        "channels": 512,
        "spatial": 64,
        "depth": 128,
        "max_ms": 1.2,
    },
    {
        "name": "64x-conv-1024x64",
        "channels": 1024,
        "spatial": 64,
        "depth": 64,
        "max_ms": 2.0,
    },
)


def make_deep_identity_model(
    out_dir: Path,
    *,
    channels: int,
    spatial: int,
    depth: int,
) -> Path:
    weights = np.eye(channels, dtype=np.float32).reshape(channels, channels, 1, 1)

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, channels, 1, spatial), dtype=types.fp32)]
    )
    def prog(x):
        y = x
        for index in range(depth):
            y = mb.conv(
                x=y,
                weight=weights,
                pad_type="valid",
                strides=[1, 1],
                dilations=[1, 1],
                groups=1,
                name=f"conv_{index}",
            )
        return y

    model = ct.convert(prog, convert_to="mlprogram")
    package = out_dir / f"deep_{depth}x_{channels}x{spatial}.mlpackage"
    model.save(str(package))
    return Path(ct.models.utils.compile_model(str(package)))


def copy_artifacts(compiled_dir: Path, dst_dir: Path) -> None:
    (dst_dir / "weights").mkdir(parents=True, exist_ok=True)
    shutil.copy2(compiled_dir / "model.mil", dst_dir / "model.mil")
    shutil.copy2(
        compiled_dir / "weights" / "weight.bin",
        dst_dir / "weights" / "weight.bin",
    )


def model_gflops(*, channels: int, spatial: int, depth: int) -> float:
    return (2.0 * channels * channels * spatial * depth) / 1e9


def weight_megabytes(*, channels: int, depth: int) -> float:
    return (channels * channels * 2 * depth) / (1024.0 * 1024.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {"runtime": "private_ane_large_mlprogram_artifacts", "models": []}
    for spec in LARGE_MODEL_SPECS:
        compiled_dir = make_deep_identity_model(
            out_dir,
            channels=spec["channels"],
            spatial=spec["spatial"],
            depth=spec["depth"],
        )
        model_dir = out_dir / spec["name"]
        copy_artifacts(compiled_dir, model_dir)
        report["models"].append(
            {
                **spec,
                "dir": str(model_dir),
                "model_mil": str(model_dir / "model.mil"),
                "weight_bin": str(model_dir / "weights" / "weight.bin"),
                "weight_offset": 64,
                "input_bytes": spec["channels"] * spec["spatial"] * 4,
                "output_bytes": spec["channels"] * spec["spatial"] * 4,
                "gflops": model_gflops(
                    channels=spec["channels"],
                    spatial=spec["spatial"],
                    depth=spec["depth"],
                ),
                "weight_mb": weight_megabytes(
                    channels=spec["channels"],
                    depth=spec["depth"],
                ),
            }
        )

    (out_dir / "metadata.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
