from __future__ import annotations

import sys
from pathlib import Path

import argparse
import json

try:
    from .mlx_vlm_models import MODEL_SPECS
    from ..common import add_vendor_to_path, cleanup_mlx, resolve_model_path
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mlx_vlm_models import MODEL_SPECS
    from common import add_vendor_to_path, cleanup_mlx, resolve_model_path

add_vendor_to_path("mlx-vlm")

import mlx.core as mx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    spec = next(
        (item for item in MODEL_SPECS if item["name"] == args.model_name),
        None,
    )
    if spec is None:
        raise SystemExit(f"Unknown --model-name={args.model_name!r}")

    payload = generate_text(
        spec,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    if args.json:
        print(json.dumps(payload))
        return

    print(payload["text"])


def generate_text(
    spec: dict[str, object],
    *,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, object]:
    from mlx_vlm import generate
    from mlx_vlm.tokenizer_utils import load_tokenizer
    from mlx_vlm.utils import get_model_path, load_model

    model_path = resolve_model_path(str(spec["model_id"]), get_model_path)
    model = load_model(model_path, lazy=False)
    tokenizer = load_tokenizer(model_path)

    try:
        result = generate(
            model,
            tokenizer,
            prompt,
            verbose=False,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return {
            "name": spec["name"],
            "model_id": spec["model_id"],
            "snapshot_path": str(model_path.resolve()),
            "prompt": prompt,
            "text": result.text,
            "prompt_tokens": int(result.prompt_tokens),
            "generation_tokens": int(result.generation_tokens),
            "total_tokens": int(result.total_tokens),
            "prompt_tps": float(result.prompt_tps),
            "generation_tps": float(result.generation_tps),
            "peak_memory": float(result.peak_memory),
        }
    finally:
        del tokenizer
        del model
        cleanup_mlx(mx)


if __name__ == "__main__":
    main()
