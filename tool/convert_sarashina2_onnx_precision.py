#!/usr/bin/env python3
"""Convert Sarashina2 ONNX exports to lower precision variants."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import onnx
from onnxconverter_common import float16


LLM_MODELS = (
    ("llm_prefill.onnx", "llm_prefill.{precision}.onnx"),
    ("llm_decode.onnx", "llm_decode.{precision}.onnx"),
    ("llm_decoder_head.onnx", "llm_decoder_head.{precision}.onnx"),
)

FLOW_STEP_MODELS = (
    ("flow.decoder.step.fp32.onnx", "flow.decoder.step.{precision}.onnx"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        required=True,
        type=Path,
        help="Directory containing Sarashina2 ONNX files.",
    )
    parser.add_argument(
        "--precision",
        choices=("fp16",),
        default="fp16",
        help="Target precision. BF16 is supported by the PyTorch exporter path.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Only convert explicitly requested non-LLM graphs.",
    )
    parser.add_argument(
        "--include-flow-step",
        action="store_true",
        help="Also convert flow.decoder.step.fp32.onnx.",
    )
    parser.add_argument(
        "--disable-shape-infer",
        action="store_true",
        help="Skip converter shape inference to reduce peak RAM.",
    )
    return parser.parse_args()


def convert_one(src: Path, dst: Path, args: argparse.Namespace) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    if dst.exists() and not args.overwrite:
        logging.info("skip existing %s", dst)
        return
    logging.info("loading %s", src)
    model = onnx.load_model(src, load_external_data=True)
    logging.info("converting %s -> %s", src.name, dst.name)
    converted = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        disable_shape_infer=args.disable_shape_infer,
    )
    onnx.save_model(converted, dst)
    logging.info("wrote %s", dst)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    models = []
    if not args.skip_llm:
        models.extend(LLM_MODELS)
    if args.include_flow_step:
        models.extend(FLOW_STEP_MODELS)
    for src_name, dst_template in models:
        convert_one(
            model_dir / src_name,
            model_dir / dst_template.format(precision=args.precision),
            args,
        )


if __name__ == "__main__":
    main()
