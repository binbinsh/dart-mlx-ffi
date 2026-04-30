#!/usr/bin/env python3
"""Generate Sarashina2 llm_embeddings.npz from model.safetensors."""

from __future__ import annotations

import argparse
from pathlib import Path

EXPECTED_EMBEDDING_SHAPE = [108986, 1280]
EXPECTED_EMBEDDING_DTYPE = "F32"
EMBEDDING_KEY = "model.embed_tokens.weight"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--model-safetensors",
        type=Path,
        help="Path to sarashina2.2-tts/model.safetensors.",
    )
    input_group.add_argument(
        "--model-dir",
        type=Path,
        help="Directory containing model.safetensors; output defaults there.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output llm_embeddings.npz path. Defaults to <model-dir>/llm_embeddings.npz.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import numpy as np
    from safetensors import safe_open

    model_path = args.model_safetensors or args.model_dir / "model.safetensors"
    output = args.output or model_path.with_name("llm_embeddings.npz")
    with safe_open(str(model_path), framework="numpy") as tensors:
        if EMBEDDING_KEY not in tensors.keys():
            raise KeyError(f"{model_path} is missing {EMBEDDING_KEY}")
        tensor = tensors.get_slice(EMBEDDING_KEY)
        shape = tensor.get_shape()
        dtype = tensor.get_dtype()
        if shape != EXPECTED_EMBEDDING_SHAPE:
            raise ValueError(
                f"{EMBEDDING_KEY} shape must be {EXPECTED_EMBEDDING_SHAPE}, got {shape}"
            )
        if dtype != EXPECTED_EMBEDDING_DTYPE:
            raise ValueError(
                f"{EMBEDDING_KEY} dtype must be {EXPECTED_EMBEDDING_DTYPE}, got {dtype}"
            )
        value = tensors.get_tensor(EMBEDDING_KEY)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        token_embedding=np.asarray(value, dtype=np.float32),
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
