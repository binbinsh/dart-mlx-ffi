#!/usr/bin/env python3
"""Rewrite Ditto's warp_network.onnx to remove the custom GridSample3D op.

Ditto's `warp_network.onnx` ships with two `GridSample3D` nodes that use
the empty domain ('') but a custom op_type. ORT/CoreML/MLX all reject
this. The op is implemented by Ditto's bundled
`libgrid_sample_3d_plugin.so` (Linux/CUDA only).

Fix: replace each `GridSample3D` with the **standard** `GridSample`
(opset 20+, which natively supports 5D `(N, C, D, H, W)` inputs and 5D
grid `(N, D_out, H_out, W_out, 3)`). Then bump the model's
`opset_import` to 20. Functionally identical because:

  * Ditto's PyTorch source calls `F.grid_sample(..., align_corners=False)`
    (verified in `core/models/modules/{dense_motion,warping_network}.py`).
  * The HF community's `Live-Portrait-ONNX/generator_fix_grid.onnx`
    already does exactly this rewrite for vanilla LivePortrait, with
    attrs `align_corners=0, mode='linear', padding_mode='zeros'`.

Usage:
    python tool/rewrite_warp_gridsample3d.py \\
        --in  /path/to/warp_network.onnx \\
        --out /path/to/warp_network_v2.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx
from onnx import helper


GRIDSAMPLE_TARGET_OPSET = 20


def rewrite(input_path: Path, output_path: Path) -> int:
    model = onnx.load(str(input_path), load_external_data=False)
    graph = model.graph

    replaced = 0
    new_nodes = []
    for node in graph.node:
        if node.op_type == "GridSample3D" and node.domain == "":
            # Build a standard GridSample node with the same I/O.
            new_node = helper.make_node(
                "GridSample",
                inputs=list(node.input),
                outputs=list(node.output),
                name=node.name,
                domain="",
                align_corners=0,  # PyTorch align_corners=False
                mode="linear",
                padding_mode="zeros",
            )
            new_nodes.append(new_node)
            replaced += 1
            print(
                f"  replaced GridSample3D -> GridSample at {node.name!r} "
                f"(in={list(node.input)} out={list(node.output)})"
            )
        else:
            new_nodes.append(node)

    if replaced == 0:
        print("WARN: no GridSample3D nodes found; nothing to do.", file=sys.stderr)
        return 1

    del graph.node[:]
    graph.node.extend(new_nodes)

    # Bump opset to 20 (when GridSample gained 5D support). Preserve
    # other domain opsets (e.g. com.microsoft) untouched.
    bumped = False
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            if opset.version < GRIDSAMPLE_TARGET_OPSET:
                print(
                    f"  bumped opset {opset.domain or 'ai.onnx'} "
                    f"v{opset.version} -> v{GRIDSAMPLE_TARGET_OPSET}"
                )
                opset.version = GRIDSAMPLE_TARGET_OPSET
            bumped = True
    if not bumped:
        model.opset_import.append(
            helper.make_opsetid("", GRIDSAMPLE_TARGET_OPSET),
        )
        print(f"  added opset ai.onnx v{GRIDSAMPLE_TARGET_OPSET}")

    # Re-stamp ir_version to match opset 20 (ai.onnx 20 requires IR >= 9).
    if model.ir_version < 9:
        print(f"  bumped ir_version {model.ir_version} -> 9")
        model.ir_version = 9

    onnx.checker.check_model(model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(
        f"OK: wrote {output_path} ({size_mb:.1f} MB), "
        f"replaced {replaced} GridSample3D node(s).",
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="inp", required=True, type=Path)
    p.add_argument("--out", dest="out", required=True, type=Path)
    args = p.parse_args()
    return rewrite(args.inp, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
