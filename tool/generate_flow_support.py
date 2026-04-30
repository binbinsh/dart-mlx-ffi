#!/usr/bin/env python3
"""Generate flow_support.npz for CosyVoice2-style flow runtimes.

The split ONNX flow encoder consumes token embeddings, while the PyTorch
checkpoint owns embedding/projection tables and the deterministic diffusion
noise seed. This extractor writes the small sidecar needed by Dart inference so
serving can stay on the Dart -> native FFI -> ONNX Runtime path without loading
PyTorch.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--flow-pt",
        type=Path,
        help="Path to a CosyVoice2-style flow.pt checkpoint.",
    )
    input_group.add_argument(
        "--model-dir",
        type=Path,
        help="Directory containing flow.pt; output defaults to flow_support.npz there.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output flow_support.npz path. Defaults to <model-dir>/flow_support.npz.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=50 * 300,
        help="Maximum flow mel frames covered by the deterministic noise table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import numpy as np
    import torch

    flow_pt = args.flow_pt or args.model_dir / "flow.pt"
    output = args.output or flow_pt.with_name("flow_support.npz")
    state = torch.load(flow_pt, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"expected flow checkpoint state dict, got {type(state)!r}")

    def tensor(name: str):
        value = state.get(name)
        if value is None:
            raise KeyError(f"{flow_pt} is missing {name}")
        return value.detach().cpu().numpy().astype(np.float32, copy=False)

    torch.manual_seed(0)
    rand_noise = torch.randn(1, 80, args.max_frames).cpu().numpy().astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        input_embedding=tensor("input_embedding.weight"),
        encoder_proj_weight=tensor("encoder_proj.weight"),
        encoder_proj_bias=tensor("encoder_proj.bias"),
        spk_embed_affine_weight=tensor("spk_embed_affine_layer.weight"),
        spk_embed_affine_bias=tensor("spk_embed_affine_layer.bias"),
        rand_noise=rand_noise,
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
