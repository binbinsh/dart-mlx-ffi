#!/usr/bin/env python3
"""Wrap the fused Sarashina2/CosyVoice2 flow step in a 10-step ONNX Loop."""

from __future__ import annotations

import argparse
import copy
import logging
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Directory containing flow.decoder.step.fp32.onnx.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Source fused step ONNX. Defaults to <model-dir>/flow.decoder.step.fp32.onnx.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output loop ONNX. Defaults to <model-dir>/flow.decoder.loop.fp32.onnx.",
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def write_fused_flow_loop(
    step_path: Path,
    output_path: Path,
    *,
    steps: int = 10,
    overwrite: bool = False,
    check: bool = False,
) -> None:
    if steps < 1:
        raise ValueError("--steps must be positive")
    if not step_path.is_file():
        raise FileNotFoundError(step_path)
    if output_path.exists() and not overwrite:
        logging.info("skip existing %s", output_path)
        return

    logging.info("loading %s", step_path)
    step = onnx.load_model(step_path, load_external_data=True)
    _validate_step_graph(step.graph)
    body = _make_loop_body(step.graph, steps)
    graph = helper.make_graph(
        [
            helper.make_node(
                "Loop",
                ["dinf_loop_trip_count", "dinf_loop_cond_init", "x"],
                ["next_x"],
                name="dinf_flow_loop",
                body=body,
            )
        ],
        "dinf_flow_10_step",
        [
            _value_info(step.graph, "x"),
            _value_info(step.graph, "mask"),
            _value_info(step.graph, "mu"),
            _value_info(step.graph, "spks"),
            _value_info(step.graph, "cond"),
        ],
        [
            helper.make_tensor_value_info(
                "next_x",
                TensorProto.FLOAT,
                [1, 80, "seq_len"],
            )
        ],
        [
            numpy_helper.from_array(np.array(steps, dtype=np.int64), "dinf_loop_trip_count"),
            numpy_helper.from_array(np.array(True, dtype=np.bool_), "dinf_loop_cond_init"),
        ],
    )
    model = helper.make_model(
        graph,
        ir_version=step.ir_version,
        producer_name="dart-inference-sarashina2-flow-loop",
        opset_imports=list(step.opset_import),
    )
    if check:
        onnx.checker.check_model(model, full_check=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, output_path)
    logging.info("wrote %s", output_path)


def _make_loop_body(step_graph: onnx.GraphProto, steps: int) -> onnx.GraphProto:
    t_steps, dt_steps = _time_steps(steps)
    body_nodes = [
        helper.make_node(
            "Gather",
            ["dinf_loop_t_steps", "iter_num"],
            ["dinf_loop_t"],
            name="dinf_loop_gather_t",
            axis=0,
        ),
        helper.make_node(
            "Gather",
            ["dinf_loop_dt_steps", "iter_num"],
            ["dinf_loop_dt"],
            name="dinf_loop_gather_dt",
            axis=0,
        ),
    ]
    for node in step_graph.node:
        copied = onnx.NodeProto()
        copied.CopyFrom(node)
        copied.name = _prefixed(copied.name)
        for index, value in enumerate(copied.input):
            copied.input[index] = {
                "x": "x_loop_in",
                "t": "dinf_loop_t",
                "dt": "dinf_loop_dt",
            }.get(value, value)
        for index, value in enumerate(copied.output):
            copied.output[index] = "x_loop_out" if value == "next_x" else value
        body_nodes.append(copied)
    body_nodes.append(
        helper.make_node(
            "Identity",
            ["cond_in"],
            ["cond_out"],
            name="dinf_loop_keep_going",
        )
    )
    return helper.make_graph(
        body_nodes,
        "dinf_flow_loop_body",
        [
            helper.make_tensor_value_info("iter_num", TensorProto.INT64, []),
            helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
            helper.make_tensor_value_info("x_loop_in", TensorProto.FLOAT, [1, 80, "seq_len"]),
        ],
        [
            helper.make_tensor_value_info("cond_out", TensorProto.BOOL, []),
            helper.make_tensor_value_info("x_loop_out", TensorProto.FLOAT, [1, 80, "seq_len"]),
        ],
        [
            *(copy.deepcopy(initializer) for initializer in step_graph.initializer),
            numpy_helper.from_array(t_steps, "dinf_loop_t_steps"),
            numpy_helper.from_array(dt_steps, "dinf_loop_dt_steps"),
        ],
    )


def _time_steps(steps: int) -> tuple[np.ndarray, np.ndarray]:
    t_values = []
    dt_values = []
    for step in range(1, steps + 1):
        prev = _cosine_schedule(step - 1, steps)
        nxt = _cosine_schedule(step, steps)
        t_values.append([prev, prev])
        dt_values.append([nxt - prev])
    return np.array(t_values, dtype=np.float32), np.array(dt_values, dtype=np.float32)


def _cosine_schedule(index: int, steps: int) -> float:
    return 1.0 - math.cos((index / steps) * 0.5 * math.pi)


def _validate_step_graph(graph: onnx.GraphProto) -> None:
    inputs = {value.name for value in graph.input}
    outputs = {value.name for value in graph.output}
    expected = {"x", "mask", "mu", "t", "spks", "cond", "dt"}
    if not expected.issubset(inputs):
        raise ValueError(f"step graph inputs missing {sorted(expected - inputs)}")
    if "next_x" not in outputs:
        raise ValueError("step graph must expose next_x")


def _value_info(graph: onnx.GraphProto, name: str) -> onnx.ValueInfoProto:
    for value in graph.input:
        if value.name == name:
            return copy.deepcopy(value)
    raise KeyError(name)


def _prefixed(name: str) -> str:
    return f"dinf_loop/{name}" if name else ""


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve() if args.model_dir else None
    if args.input is None and model_dir is None:
        raise SystemExit("--model-dir or --input is required")
    src = (
        args.input.expanduser().resolve()
        if args.input
        else model_dir / "flow.decoder.step.fp32.onnx"
    )
    dst = (
        args.output.expanduser().resolve()
        if args.output
        else model_dir / "flow.decoder.loop.fp32.onnx"
    )
    write_fused_flow_loop(
        src,
        dst,
        steps=args.steps,
        overwrite=args.overwrite,
        check=args.check,
    )


if __name__ == "__main__":
    main()
