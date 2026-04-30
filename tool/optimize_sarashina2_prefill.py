#!/usr/bin/env python3
"""Rewrite Sarashina2 LLM prefill ONNX to return only the last hidden state."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Source llm_prefill*.onnx model.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination llm_prefill_last*.onnx model.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--inline-weights",
        action="store_true",
        help="Write weights into the ONNX file instead of a .data sidecar.",
    )
    return parser.parse_args()


def write_prefill_last_hidden(
    src: Path,
    dst: Path,
    *,
    overwrite: bool = False,
    external_data: bool = True,
) -> None:
    src = src.expanduser().resolve()
    dst = dst.expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(src)
    if dst.exists() and not overwrite:
        logging.info("skip existing %s", dst)
        return
    if overwrite:
        _unlink_if_exists(dst)
        _unlink_if_exists(dst.with_name(dst.name + ".data"))

    logging.info("loading %s", src)
    model = onnx.load_model(src, load_external_data=True)
    graph = model.graph
    hidden_output = _graph_output(graph, "hidden")
    producer = _producer(graph, "hidden")
    if producer is None:
        raise ValueError(f"{src} has no node producing graph output 'hidden'")

    hidden_all = _unique_name(graph, "hidden_all")
    for index, name in enumerate(producer.output):
        if name == "hidden":
            producer.output[index] = hidden_all
            break
    for node in graph.node:
        for index, name in enumerate(node.input):
            if name == "hidden":
                node.input[index] = hidden_all

    index_name = _unique_name(graph, "last_hidden_index")
    graph.initializer.append(
        numpy_helper.from_array(np.asarray([-1], dtype=np.int64), index_name)
    )
    graph.node.append(
        helper.make_node(
            "Gather",
            [hidden_all, index_name],
            ["hidden"],
            name=_unique_name(graph, "GatherLastHidden"),
            axis=1,
        )
    )

    hidden_type = hidden_output.type.tensor_type
    hidden_type.elem_type = TensorProto.FLOAT
    shape = hidden_type.shape
    if len(shape.dim) >= 2:
        shape.dim[1].ClearField("dim_param")
        shape.dim[1].dim_value = 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    logging.info("writing %s", dst)
    if external_data:
        onnx.save_model(
            model,
            dst,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=dst.name + ".data",
            size_threshold=1024,
            convert_attribute=False,
        )
    else:
        onnx.save_model(model, dst)


def _graph_output(graph: onnx.GraphProto, name: str) -> onnx.ValueInfoProto:
    for output in graph.output:
        if output.name == name:
            return output
    raise ValueError(f"graph output {name!r} not found")


def _producer(graph: onnx.GraphProto, name: str) -> onnx.NodeProto | None:
    for node in graph.node:
        if name in node.output:
            return node
    return None


def _unique_name(graph: onnx.GraphProto, base: str) -> str:
    used = set()
    for node in graph.node:
        used.update(node.input)
        used.update(node.output)
        if node.name:
            used.add(node.name)
    used.update(value.name for value in graph.input)
    used.update(value.name for value in graph.output)
    used.update(value.name for value in graph.initializer)
    if base not in used:
        return base
    index = 1
    while f"{base}_{index}" in used:
        index += 1
    return f"{base}_{index}"


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    write_prefill_last_hidden(
        args.input,
        args.output,
        overwrite=args.overwrite,
        external_data=not args.inline_weights,
    )


if __name__ == "__main__":
    main()
