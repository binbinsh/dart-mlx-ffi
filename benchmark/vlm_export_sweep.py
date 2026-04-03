from __future__ import annotations

import json
from pathlib import Path
import sys
import time

from PIL import Image, ImageDraw

try:
    from .common import (
        add_vendor_to_path,
        benchmark_dart_export,
        cleanup_mlx,
        resolve_model_path,
        slug,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import (
        add_vendor_to_path,
        benchmark_dart_export,
        cleanup_mlx,
        resolve_model_path,
        slug,
    )

add_vendor_to_path("mlx-vlm")

import mlx.core as mx
from mlx_vlm.utils import load, prepare_inputs

PROMPT = "Describe this image briefly."


def prompt_for_model(model_id: str) -> str:
    if "gemma" in model_id.lower():
        return "<start_of_image>\n" + PROMPT
    return PROMPT


def extract_logits(output):
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, tuple):
        return output[0]
    return output


def sample_image() -> Image.Image:
    image = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, 180, 180), fill="orange")
    draw.text((86, 100), "MLX", fill="black")
    return image


class _ProcessorWrapper:
    def __init__(self, tokenizer, image_processor):
        self.tokenizer = tokenizer
        self._hf_tokenizer = getattr(tokenizer, "_tokenizer", tokenizer)
        self.image_processor = image_processor
        self.pad_token = getattr(tokenizer, "pad_token", None)
        self.eos_token = getattr(tokenizer, "eos_token", None)
        self.pad_token_id = getattr(tokenizer, "pad_token_id", None)

    def __call__(self, *args, **kwargs):
        return self._hf_tokenizer(*args, **kwargs)


def prepare_model_inputs(model_id: str):
    from mlx_vlm.utils import get_model_path

    model_path = resolve_model_path(model_id, get_model_path)
    try:
        model, processor = load(model_path, lazy=False)
        inputs = prepare_inputs(
            processor,
            images=[sample_image()],
            prompts=[prompt_for_model(model_id)],
        )
        return model, processor, inputs
    except Exception as error:
        message = str(error)
        if "Torchvision" not in message and "PyTorch" not in message:
            raise
        from mlx_vlm.tokenizer_utils import load_tokenizer
        from mlx_vlm.utils import load_image_processor, load_model
        model = load_model(model_path, lazy=False)
        tokenizer = load_tokenizer(model_path)
        image_processor = load_image_processor(model_path, trust_remote_code=True)
        processor = _ProcessorWrapper(tokenizer, image_processor)
        inputs = prepare_inputs(
            processor,
            images=[sample_image()],
            prompts=[prompt_for_model(model_id)],
        )
        return model, processor, inputs


def python_forward(model_id: str):
    model, processor, inputs = prepare_model_inputs(model_id)
    input_names = [
        name
        for name, value in inputs.items()
        if isinstance(value, mx.array) and value.size > 0
    ]
    call_inputs = {name: inputs[name] for name in input_names}

    def forward():
        input_ids = call_inputs["input_ids"]
        attention_mask = call_inputs.get("attention_mask")
        pixel_values = call_inputs.get("pixel_values")
        kwargs = {
            key: value
            for key, value in call_inputs.items()
            if key not in {"input_ids", "attention_mask", "pixel_values"}
        }
        output = model(input_ids, pixel_values, attention_mask, **kwargs)
        logits = extract_logits(output)[:, -1, :16].astype(mx.float32)
        mx.eval(logits)
        mx.synchronize()
        return logits

    started = time.perf_counter()
    logits = forward()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    values = [float(v) for v in logits.reshape([-1]).tolist()]
    return model, processor, call_inputs, input_names, values, elapsed_ms


def export_model(model_id: str, export_dir: Path):
    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = export_dir / "function.mlxfn"
    input_path = export_dir / "inputs.safetensors"
    input_names_path = export_dir / "input_names.json"

    model, _processor, call_inputs = prepare_model_inputs(model_id)
    input_names = [
        name
        for name, value in call_inputs.items()
        if isinstance(value, mx.array) and value.size > 0
    ]
    if export_path.exists() and input_path.exists() and input_names_path.exists():
        cached_names = json.loads(input_names_path.read_text(encoding="utf-8"))
        if cached_names == input_names:
            del model, _processor, call_inputs
            cleanup_mlx(mx)
            return export_path, input_path, input_names

    def forward(*args):
        values_by_name = dict(zip(input_names, args, strict=True))
        input_ids = values_by_name["input_ids"]
        attention_mask = values_by_name.get("attention_mask")
        pixel_values = values_by_name.get("pixel_values")
        kwargs = {
            key: value
            for key, value in values_by_name.items()
            if key not in {"input_ids", "attention_mask", "pixel_values"}
        }
        output = model(input_ids, pixel_values, attention_mask, **kwargs)
        return extract_logits(output)[:, -1, :16].astype(mx.float32)

    if export_path.exists():
        export_path.unlink()
    mx.export_function(
        str(export_path),
        forward,
        *[call_inputs[name] for name in input_names],
    )
    mx.save_safetensors(str(input_path), {name: call_inputs[name] for name in input_names})
    input_names_path.write_text(json.dumps(input_names), encoding="utf-8")
    del model, _processor, call_inputs
    cleanup_mlx(mx)
    return export_path, input_path, input_names


def dart_forward(
    export_path: Path,
    input_path: Path,
    input_names: list[str],
    *,
    warmup: int = 0,
    iters: int = 1,
):
    return benchmark_dart_export(
        export_path=export_path,
        input_path=input_path,
        input_names=input_names,
        mx_module=mx,
        warmup=warmup,
        iters=iters,
    )
