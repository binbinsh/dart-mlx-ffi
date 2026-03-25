from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types


FFN_MODEL_SPECS = (
    {
        "name": "ffn-512-1536-l64",
        "dim": 512,
        "hidden": 1536,
        "lane": 64,
        "seed": 7,
        "weight_scale": 0.02,
        "max_ms": 0.5,
    },
    {
        "name": "ffn-1024-3072-l32",
        "dim": 1024,
        "hidden": 3072,
        "lane": 32,
        "seed": 11,
        "weight_scale": 0.02,
        "max_ms": 0.8,
    },
)


def make_weights(spec: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(spec["seed"])
    dim = int(spec["dim"])
    hidden = int(spec["hidden"])
    scale = float(spec["weight_scale"])
    w1 = rng.normal(0.0, scale, size=(hidden, dim, 1, 1)).astype(np.float32)
    w3 = rng.normal(0.0, scale, size=(hidden, dim, 1, 1)).astype(np.float32)
    w2 = rng.normal(0.0, scale, size=(dim, hidden, 1, 1)).astype(np.float32)
    return w1, w3, w2


def make_ffn_model(
    out_dir: Path,
    *,
    dim: int,
    hidden: int,
    lane: int,
    w1: np.ndarray,
    w3: np.ndarray,
    w2: np.ndarray,
) -> Path:
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, dim, 1, lane), dtype=types.fp32)])
    def prog(x):
        h1 = mb.conv(
            x=x,
            weight=w1,
            pad_type="valid",
            strides=[1, 1],
            dilations=[1, 1],
            groups=1,
            name="w1",
        )
        h3 = mb.conv(
            x=x,
            weight=w3,
            pad_type="valid",
            strides=[1, 1],
            dilations=[1, 1],
            groups=1,
            name="w3",
        )
        sig = mb.sigmoid(x=h1, name="sig")
        silu = mb.mul(x=h1, y=sig, name="silu")
        gate = mb.mul(x=silu, y=h3, name="gate")
        y = mb.conv(
            x=gate,
            weight=w2,
            pad_type="valid",
            strides=[1, 1],
            dilations=[1, 1],
            groups=1,
            name="w2",
        )
        return mb.add(x=x, y=y, name="res")

    model = ct.convert(prog, convert_to="mlprogram")
    package = out_dir / f"ffn_{dim}_{hidden}_{lane}.mlpackage"
    model.save(str(package))
    return Path(ct.models.utils.compile_model(str(package)))


def copy_artifacts(compiled_dir: Path, dst_dir: Path) -> None:
    (dst_dir / "weights").mkdir(parents=True, exist_ok=True)
    shutil.copy2(compiled_dir / "model.mil", dst_dir / "model.mil")
    shutil.copy2(
        compiled_dir / "weights" / "weight.bin",
        dst_dir / "weights" / "weight.bin",
    )


def make_sample_input(spec: dict[str, object]) -> np.ndarray:
    dim = int(spec["dim"])
    lane = int(spec["lane"])
    values = np.array([(index % 97) / 97.0 for index in range(dim * lane)], dtype=np.float32)
    return values.reshape(dim, lane)


def compute_expected(
    x: np.ndarray,
    *,
    w1: np.ndarray,
    w3: np.ndarray,
    w2: np.ndarray,
) -> np.ndarray:
    w1m = w1[:, :, 0, 0]
    w3m = w3[:, :, 0, 0]
    w2m = w2[:, :, 0, 0]
    h1 = w1m @ x
    h3 = w3m @ x
    sig = 1.0 / (1.0 + np.exp(-h1))
    gate = (h1 * sig) * h3
    y = w2m @ gate
    return (x + y).astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {"runtime": "private_ane_ffn_artifacts", "models": []}
    for spec in FFN_MODEL_SPECS:
        w1, w3, w2 = make_weights(spec)
        compiled_dir = make_ffn_model(
            out_dir,
            dim=int(spec["dim"]),
            hidden=int(spec["hidden"]),
            lane=int(spec["lane"]),
            w1=w1,
            w3=w3,
            w2=w2,
        )
        model_dir = out_dir / str(spec["name"])
        copy_artifacts(compiled_dir, model_dir)

        sample = make_sample_input(spec)
        expected = compute_expected(sample, w1=w1, w3=w3, w2=w2)
        input_path = model_dir / "input_f32.bin"
        expected_path = model_dir / "expected_f32.bin"
        input_path.write_bytes(sample.astype(np.float32, copy=False).tobytes())
        expected_path.write_bytes(expected.astype(np.float32, copy=False).tobytes())

        report["models"].append(
            {
                **spec,
                "dir": str(model_dir),
                "model_mil": str(model_dir / "model.mil"),
                "weight_bin": str(model_dir / "weights" / "weight.bin"),
                "input_f32": str(input_path),
                "expected_f32": str(expected_path),
                "weight_offset": 64,
                "input_bytes": int(spec["dim"]) * int(spec["lane"]) * 4,
                "output_bytes": int(spec["dim"]) * int(spec["lane"]) * 4,
            }
        )

    (out_dir / "metadata.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
