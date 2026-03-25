from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types


def make_model(out_dir: Path, *, channels: int, spatial: int) -> Path:
    weights = np.eye(channels, dtype=np.float32).reshape(channels, channels, 1, 1)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, channels, 1, spatial), dtype=types.fp32)])
    def prog(x):
        return mb.conv(
            x=x,
            weight=weights,
            pad_type="valid",
            strides=[1, 1],
            dilations=[1, 1],
            groups=1,
        )

    model = ct.convert(prog, convert_to="mlprogram")
    package = out_dir / f"conv_{channels}x{spatial}.mlpackage"
    model.save(str(package))
    return Path(ct.models.utils.compile_model(str(package)))


def copy_artifacts(compiled_dir: Path, dst_dir: Path) -> None:
    (dst_dir / "weights").mkdir(parents=True, exist_ok=True)
    shutil.copy2(compiled_dir / "model.mil", dst_dir / "model.mil")
    shutil.copy2(
        compiled_dir / "weights" / "weight.bin",
        dst_dir / "weights" / "weight.bin",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {"runtime": "private_ane_mlprogram_artifacts", "models": []}
    for channels, spatial in [(256, 64), (512, 64)]:
        name = f"conv-{channels}x{spatial}"
        compiled_dir = make_model(out_dir, channels=channels, spatial=spatial)
        model_dir = out_dir / name
        copy_artifacts(compiled_dir, model_dir)
        report["models"].append(
            {
                "name": name,
                "dir": str(model_dir),
                "model_mil": str(model_dir / "model.mil"),
                "weight_bin": str(model_dir / "weights" / "weight.bin"),
                "weight_offset": 64,
                "channels": channels,
                "spatial": spatial,
                "input_bytes": channels * spatial * 4,
                "output_bytes": channels * spatial * 4,
            }
        )

    (out_dir / "metadata.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
