from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from ..common import cleanup_mlx, find_cached_snapshot
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common import cleanup_mlx, find_cached_snapshot

import mlx.core as mx
from mlx_vlm import load as vlm_load
from mlx_vlm.models.paddleocr_vl.processing_paddleocr_vl import smart_resize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    return parser.parse_args()


def default_image_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "benchmark"
        / "assets"
        / "paddle_ocr_vl_test.jpg"
    ).resolve()


def build_prompt() -> str:
    return (
        "<|begin_of_sentence|>User: <|IMAGE_START|><|IMAGE_PLACEHOLDER|>"
        "<|IMAGE_END|>Extract all text from this image.\nAssistant:\n"
    )


def extract_logits(output):
    return output.logits if hasattr(output, "logits") else output


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    snapshot = find_cached_snapshot(args.model_id)
    if snapshot is None:
        raise RuntimeError(f"Model snapshot not found for {args.model_id}")

    model, processor = vlm_load(str(snapshot))
    image_path = args.image or default_image_path()
    if not image_path.exists():
        raise RuntimeError(f"Benchmark image not found: {image_path}")
    image = Image.open(image_path).convert("RGB")
    prompt = build_prompt()
    ip = processor.image_processor
    if args.min_pixels is not None:
        ip.min_pixels = args.min_pixels
    if args.max_pixels is not None:
        ip.max_pixels = args.max_pixels
    inputs = processor(images=[image], text=[prompt], return_tensors="np")

    input_ids = mx.array(inputs["input_ids"])
    pixel_values = mx.array(inputs["pixel_values"])
    image_grid_thw = mx.array(inputs["image_grid_thw"])
    pixel_values_for_vision = mx.array(
        pixel_values,
        dtype=model.visual.embeddings.patch_embedding.weight.dtype,
    )

    resized_h, resized_w = smart_resize(
        image.height,
        image.width,
        factor=ip.patch_size * ip.merge_size,
        min_pixels=ip.min_pixels,
        max_pixels=ip.max_pixels,
    )
    resized_img = image.resize((resized_w, resized_h), Image.BICUBIC)
    img_arr = np.array(resized_img, dtype=np.float32) / 255.0
    mean = np.array(ip.image_mean, dtype=np.float32)
    std = np.array(ip.image_std, dtype=np.float32)
    img_arr = (img_arr - mean) / std
    img_nhwc = img_arr[np.newaxis, :, :, :]

    np.save(args.out_dir / "input_ids.npy", np.array(input_ids.tolist(), dtype=np.int32))
    np.save(args.out_dir / "pixel_values.npy", np.array(pixel_values.tolist(), dtype=np.float32))
    np.save(
        args.out_dir / "image_grid_thw.npy",
        np.array(image_grid_thw.tolist(), dtype=np.int32),
    )
    np.save(args.out_dir / "image_nhwc.npy", img_nhwc)

    def forward():
        logits = extract_logits(
            model(
                input_ids,
                pixel_values=pixel_values_for_vision,
                image_grid_thw=image_grid_thw,
            )
        )[0, -1, :16].astype(mx.float32)
        mx.eval(logits)
        mx.synchronize()
        return logits

    for _ in range(args.warmup):
        out = forward()
        del out

    started = time.perf_counter()
    last = None
    for _ in range(args.iters):
        last = forward()
    py_ms = (time.perf_counter() - started) * 1000.0 / args.iters
    values = [float(v) for v in last.reshape([-1]).tolist()]

    print(
        json.dumps(
            {
                "model_id": args.model_id,
                "snapshot_path": str(snapshot),
                "input_ids_path": str(args.out_dir / "input_ids.npy"),
                "pixel_values_path": str(args.out_dir / "pixel_values.npy"),
                "image_grid_thw_path": str(args.out_dir / "image_grid_thw.npy"),
                "image_path": str(args.out_dir / "image_nhwc.npy"),
                "source_image_path": str(image_path),
                "min_pixels": int(ip.min_pixels),
                "max_pixels": int(ip.max_pixels),
                "resized_height": int(resized_h),
                "resized_width": int(resized_w),
                "comparison_source": "prefill_last_token_full_logits_slice",
                "python_ms": py_ms,
                "values": values,
            }
        )
    )
    cleanup_mlx(mx)


if __name__ == "__main__":
    main()
