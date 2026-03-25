from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "private_ane" / "models" / "josie" / "python"))

from layer_ops_probe import build_helper, dequantize_linear_weight, load_josie, probe_one


def _reshape_linear_output(y, *, channels: int, spatial: int):
    y = mb.reshape(x=y, shape=[1, 1, spatial, channels])
    return mb.transpose(x=y, perm=[0, 3, 1, 2])


def compile_conv_probe(
    out_dir: Path,
    *,
    name: str,
    input_channels: int,
    output_channels: int,
    spatial: int,
    weight: np.ndarray,
    tile_size: int | None,
) -> Path:
    kernel = weight.reshape(output_channels, input_channels, 1, 1)
    use_tiles = tile_size is not None and tile_size > 0 and tile_size < output_channels

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, input_channels, 1, spatial), dtype=types.fp32),
        ]
    )
    def prog(x):
        if not use_tiles:
            return mb.conv(
                x=x,
                weight=kernel,
                pad_type="valid",
                strides=[1, 1],
                dilations=[1, 1],
                groups=1,
            )

        parts = []
        for start in range(0, output_channels, tile_size):
            stop = min(start + tile_size, output_channels)
            parts.append(
                mb.conv(
                    x=x,
                    weight=kernel[start:stop, :, :, :],
                    pad_type="valid",
                    strides=[1, 1],
                    dilations=[1, 1],
                    groups=1,
                )
            )
        return mb.concat(values=parts, axis=1)

    model = ct.convert(prog, convert_to="mlprogram")
    pkg = out_dir / f"{name}.mlpackage"
    model.save(str(pkg))
    return Path(ct.models.utils.compile_model(str(pkg)))


def compile_linear_probe(
    out_dir: Path,
    *,
    name: str,
    input_channels: int,
    output_channels: int,
    spatial: int,
    weight: np.ndarray,
    tile_size: int | None,
) -> Path:
    use_tiles = tile_size is not None and tile_size > 0 and tile_size < output_channels

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, input_channels, 1, spatial), dtype=types.fp32),
        ]
    )
    def prog(x):
        flat = mb.transpose(x=x, perm=[0, 2, 3, 1])
        flat = mb.reshape(x=flat, shape=[spatial, input_channels])

        if not use_tiles:
            y = mb.linear(x=flat, weight=weight)
            return _reshape_linear_output(
                y,
                channels=output_channels,
                spatial=spatial,
            )

        parts = []
        for start in range(0, output_channels, tile_size):
            stop = min(start + tile_size, output_channels)
            y = mb.linear(x=flat, weight=weight[start:stop, :])
            parts.append(
                _reshape_linear_output(
                    y,
                    channels=stop - start,
                    spatial=spatial,
                )
            )
        return mb.concat(values=parts, axis=1)

    model = ct.convert(prog, convert_to="mlprogram")
    pkg = out_dir / f"{name}.mlpackage"
    model.save(str(pkg))
    return Path(ct.models.utils.compile_model(str(pkg)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--spatial", type=int, default=1)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    model, _tokenizer = load_josie(lazy=True)
    attn = model.model.layers[args.layer].self_attn

    specs = [
        ("q_proj", 2560, 4096, dequantize_linear_weight(attn.q_proj)),
        ("o_proj", 4096, 2560, dequantize_linear_weight(attn.o_proj)),
    ]

    work_dir = Path(tempfile.mkdtemp(prefix="josie_tiled_export_probe_"))
    try:
        helper = build_helper(work_dir)
        reports = []
        for op_name, input_channels, output_channels, weight in specs:
            compiled = compile_conv_probe(
                work_dir,
                name=f"layer{args.layer}_{op_name}_conv",
                input_channels=input_channels,
                output_channels=output_channels,
                spatial=args.spatial,
                weight=weight,
                tile_size=None,
            )
            reports.append(
                probe_one(
                    helper,
                    compiled,
                    name=f"{op_name}_conv",
                    channels=input_channels,
                    spatial=args.spatial,
                    layer=args.layer,
                )
            )

            compiled = compile_linear_probe(
                work_dir,
                name=f"layer{args.layer}_{op_name}_linear",
                input_channels=input_channels,
                output_channels=output_channels,
                spatial=args.spatial,
                weight=weight,
                tile_size=None,
            )
            reports.append(
                probe_one(
                    helper,
                    compiled,
                    name=f"{op_name}_linear",
                    channels=input_channels,
                    spatial=args.spatial,
                    layer=args.layer,
                )
            )

            compiled = compile_conv_probe(
                work_dir,
                name=f"layer{args.layer}_{op_name}_tiled_conv",
                input_channels=input_channels,
                output_channels=output_channels,
                spatial=args.spatial,
                weight=weight,
                tile_size=args.tile_size,
            )
            reports.append(
                probe_one(
                    helper,
                    compiled,
                    name=f"{op_name}_tiled_conv_{args.tile_size}",
                    channels=input_channels,
                    spatial=args.spatial,
                    layer=args.layer,
                )
            )

            compiled = compile_linear_probe(
                work_dir,
                name=f"layer{args.layer}_{op_name}_tiled_linear",
                input_channels=input_channels,
                output_channels=output_channels,
                spatial=args.spatial,
                weight=weight,
                tile_size=args.tile_size,
            )
            reports.append(
                probe_one(
                    helper,
                    compiled,
                    name=f"{op_name}_tiled_linear_{args.tile_size}",
                    channels=input_channels,
                    spatial=args.spatial,
                    layer=args.layer,
                )
            )

        report = {
            "runtime": "josie_tiled_export_probe",
            "layer": args.layer,
            "spatial": args.spatial,
            "tile_size": args.tile_size,
            "models": reports,
        }
        if args.json:
            print(json.dumps(report))
            return
        print(json.dumps(report, indent=2))
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
