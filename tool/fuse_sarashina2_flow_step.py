#!/usr/bin/env python3
"""Fuse Sarashina2/CosyVoice2 flow duplicate + guidance into an ONNX step."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import onnx
from onnx import TensorProto, helper, numpy_helper
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Directory containing flow.decoder.estimator.fp32.onnx.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Source estimator ONNX path. Defaults to <model-dir>/flow.decoder.estimator.fp32.onnx.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output fused step ONNX path. Defaults to <model-dir>/flow.decoder.step.fp32.onnx.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_fused_flow_step(src: Path, dst: Path, *, overwrite: bool = False) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    if dst.exists() and not overwrite:
        logging.info("skip existing %s", dst)
        return

    logging.info("loading %s", src)
    model = onnx.load_model(src, load_external_data=True)
    graph = model.graph

    _replace_node_input(graph, "x", "x_guidance_batch")
    graph.node.insert(
        0,
        helper.make_node(
            "Concat",
            ["x", "x"],
            ["x_guidance_batch"],
            name="dinf_guidance_duplicate_x",
            axis=0,
        ),
    )
    _set_input_shape(graph, "x", [1, 80, "seq_len"])
    graph.input.extend(
        [
            helper.make_tensor_value_info(
                "dt",
                TensorProto.FLOAT,
                [1],
            ),
        ],
    )

    graph.initializer.extend(
        [
            numpy_helper.from_array(np.array([0], dtype=np.int64), "dinf_slice_start0"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), "dinf_slice_end1"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), "dinf_slice_start1"),
            numpy_helper.from_array(np.array([2], dtype=np.int64), "dinf_slice_end2"),
            numpy_helper.from_array(np.array([0], dtype=np.int64), "dinf_slice_axis0"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), "dinf_slice_step1"),
            numpy_helper.from_array(np.array([1.7], dtype=np.float32), "dinf_guidance_cond_scale"),
            numpy_helper.from_array(np.array([-0.7], dtype=np.float32), "dinf_guidance_uncond_scale"),
        ],
    )
    graph.node.extend(
        [
            helper.make_node(
                "Slice",
                [
                    "estimator_out",
                    "dinf_slice_start0",
                    "dinf_slice_end1",
                    "dinf_slice_axis0",
                    "dinf_slice_step1",
                ],
                ["dinf_estimator_cond"],
                name="dinf_guidance_slice_cond",
            ),
            helper.make_node(
                "Slice",
                [
                    "estimator_out",
                    "dinf_slice_start1",
                    "dinf_slice_end2",
                    "dinf_slice_axis0",
                    "dinf_slice_step1",
                ],
                ["dinf_estimator_uncond"],
                name="dinf_guidance_slice_uncond",
            ),
            helper.make_node(
                "Mul",
                ["dinf_estimator_cond", "dinf_guidance_cond_scale"],
                ["dinf_guidance_cond_scaled"],
                name="dinf_guidance_scale_cond",
            ),
            helper.make_node(
                "Mul",
                ["dinf_estimator_uncond", "dinf_guidance_uncond_scale"],
                ["dinf_guidance_uncond_scaled"],
                name="dinf_guidance_scale_uncond",
            ),
            helper.make_node(
                "Add",
                ["dinf_guidance_cond_scaled", "dinf_guidance_uncond_scaled"],
                ["dinf_guidance"],
                name="dinf_guidance_add",
            ),
            helper.make_node(
                "Mul",
                ["dinf_guidance", "dt"],
                ["dinf_guidance_delta"],
                name="dinf_guidance_mul_dt",
            ),
            helper.make_node(
                "Add",
                ["x", "dinf_guidance_delta"],
                ["next_x"],
                name="dinf_guidance_next_x",
            ),
        ],
    )

    del graph.output[:]
    graph.output.extend(
        [
            helper.make_tensor_value_info(
                "next_x",
                TensorProto.FLOAT,
                [1, 80, "seq_len"],
            ),
        ],
    )
    onnx.checker.check_model(model, full_check=False)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, dst)
    logging.info("wrote %s", dst)


def _replace_node_input(graph: onnx.GraphProto, old: str, new: str) -> None:
    for node in graph.node:
        for index, value in enumerate(node.input):
            if value == old:
                node.input[index] = new


def _set_input_shape(graph: onnx.GraphProto, name: str, shape: list[int | str]) -> None:
    for value in graph.input:
        if value.name != name:
            continue
        dims = value.type.tensor_type.shape.dim
        del dims[:]
        for item in shape:
            dim = dims.add()
            if isinstance(item, int):
                dim.dim_value = item
            else:
                dim.dim_param = item
        return
    raise KeyError(f"input not found: {name}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve() if args.model_dir else None
    if args.input is None and model_dir is None:
        raise SystemExit("--model-dir or --input is required")
    src = (
        args.input.expanduser().resolve()
        if args.input
        else model_dir / "flow.decoder.estimator.fp32.onnx"
    )
    dst = (
        args.output.expanduser().resolve()
        if args.output
        else model_dir / "flow.decoder.step.fp32.onnx"
    )
    write_fused_flow_step(src, dst, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
