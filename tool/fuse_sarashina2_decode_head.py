#!/usr/bin/env python3
"""Fuse Sarashina2 LLM decode and decoder-head ONNX graphs."""

from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path

import onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Directory containing llm_decode*.onnx and llm_decoder_head*.onnx.",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default="fp32",
        help="Precision suffix to fuse when --input-decode/--input-head are omitted.",
    )
    parser.add_argument(
        "--input-decode",
        type=Path,
        help="Source decode ONNX. Defaults to <model-dir>/llm_decode[.<precision>].onnx.",
    )
    parser.add_argument(
        "--input-head",
        type=Path,
        help="Source decoder-head ONNX. Defaults to <model-dir>/llm_decoder_head[.<precision>].onnx.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Fused ONNX path. Defaults to <model-dir>/llm_decode_head[.<precision>].onnx.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def write_fused_decode_head(
    decode_path: Path,
    head_path: Path,
    output_path: Path,
    *,
    overwrite: bool = False,
    check: bool = False,
) -> None:
    if not decode_path.is_file():
        raise FileNotFoundError(decode_path)
    if not head_path.is_file():
        raise FileNotFoundError(head_path)
    if output_path.exists() and not overwrite:
        logging.info("skip existing %s", output_path)
        return

    logging.info("loading decode graph %s", decode_path)
    decode = onnx.load_model(decode_path, load_external_data=True)
    logging.info("loading decoder-head graph %s", head_path)
    head = onnx.load_model(head_path, load_external_data=True)

    _validate_decode_graph(decode)
    _validate_head_graph(head)
    _append_head_graph(decode.graph, head.graph)
    decode.producer_name = "dart-inference-sarashina2-fuser"

    if check:
        onnx.checker.check_model(decode, full_check=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        output_path.unlink(missing_ok=True)
        (output_path.parent / f"{output_path.name}.data").unlink(missing_ok=True)
    onnx.save_model(
        decode,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{output_path.name}.data",
        size_threshold=1024,
    )
    logging.info("wrote %s", output_path)


def _validate_decode_graph(model: onnx.ModelProto) -> None:
    output_names = {value.name for value in model.graph.output}
    if "hidden" not in output_names:
        raise ValueError("decode graph must expose a 'hidden' output")
    if not any(name.startswith("present_key_") for name in output_names):
        raise ValueError("decode graph must expose present_key_* outputs")


def _validate_head_graph(model: onnx.ModelProto) -> None:
    input_names = [value.name for value in model.graph.input]
    output_names = [value.name for value in model.graph.output]
    if input_names != ["hidden"]:
        raise ValueError(f"decoder-head inputs must be ['hidden'], got {input_names}")
    if output_names != ["logits"]:
        raise ValueError(f"decoder-head outputs must be ['logits'], got {output_names}")


def _append_head_graph(
    decode_graph: onnx.GraphProto,
    head_graph: onnx.GraphProto,
) -> None:
    rename = _head_name_map(head_graph)
    decode_outputs = _non_hidden_outputs(decode_graph)
    for initializer in head_graph.initializer:
        copied = onnx.TensorProto()
        copied.CopyFrom(initializer)
        copied.name = rename[initializer.name]
        decode_graph.initializer.append(copied)

    for value in head_graph.value_info:
        copied = onnx.ValueInfoProto()
        copied.CopyFrom(value)
        if copied.name in rename:
            copied.name = rename[copied.name]
        decode_graph.value_info.append(copied)

    for node in head_graph.node:
        copied = onnx.NodeProto()
        copied.CopyFrom(node)
        copied.name = _prefixed(node.name) if node.name else ""
        for index, value in enumerate(copied.input):
            copied.input[index] = rename.get(value, value)
        for index, value in enumerate(copied.output):
            copied.output[index] = rename.get(value, value)
        decode_graph.node.append(copied)

    logits = copy.deepcopy(head_graph.output[0])
    del decode_graph.output[:]
    decode_graph.output.append(logits)
    for value in decode_outputs:
        decode_graph.output.append(value)


def _head_name_map(head_graph: onnx.GraphProto) -> dict[str, str]:
    names = set()
    for initializer in head_graph.initializer:
        names.add(initializer.name)
    for node in head_graph.node:
        names.update(node.input)
        names.update(node.output)
    for value in head_graph.value_info:
        names.add(value.name)
    return {
        name: _prefixed(name)
        for name in names
        if name not in {"hidden", "logits", ""}
    }


def _non_hidden_outputs(graph: onnx.GraphProto) -> list[onnx.ValueInfoProto]:
    outputs = []
    for value in graph.output:
        if value.name != "hidden":
            outputs.append(copy.deepcopy(value))
    return outputs


def _prefixed(name: str) -> str:
    return f"dinf_decoder_head/{name}"


def _default_path(model_dir: Path, stem: str, precision: str) -> Path:
    suffix = "" if precision == "fp32" else f".{precision}"
    return model_dir / f"{stem}{suffix}.onnx"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve() if args.model_dir else None
    if model_dir is None and (
        args.input_decode is None or args.input_head is None or args.output is None
    ):
        raise SystemExit("--model-dir is required unless all explicit paths are set")
    decode = (
        args.input_decode.expanduser().resolve()
        if args.input_decode
        else _default_path(model_dir, "llm_decode", args.precision)
    )
    head = (
        args.input_head.expanduser().resolve()
        if args.input_head
        else _default_path(model_dir, "llm_decoder_head", args.precision)
    )
    output = (
        args.output.expanduser().resolve()
        if args.output
        else _default_path(model_dir, "llm_decode_head", args.precision)
    )
    write_fused_decode_head(
        decode,
        head,
        output,
        overwrite=args.overwrite,
        check=args.check,
    )


if __name__ == "__main__":
    main()
