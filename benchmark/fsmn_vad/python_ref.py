from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--frames", type=int, default=30)
    return parser.parse_args()


def deterministic_features(frames: int, dims: int) -> torch.Tensor:
    values = []
    for t in range(frames):
        row = []
        for d in range(dims):
            row.append(
                (0.1 * math.sin(((t + 1) * 0.17) + (d * 0.013)))
                + (0.05 * math.cos(((t + 1) * 0.07) - (d * 0.019)))
            )
        values.append(row)
    return torch.tensor([values], dtype=torch.float32)


def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    y = torch.matmul(x, weight.t())
    if bias is not None:
        y = y + bias.view(1, 1, -1)
    return y


def run_layer(
    x: torch.Tensor,
    cache: torch.Tensor,
    linear_w: torch.Tensor,
    conv_left_w: torch.Tensor,
    affine_w: torch.Tensor,
    affine_b: torch.Tensor,
    cache_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    proj = linear(x, linear_w)
    proj4 = proj.unsqueeze(2)
    padded = torch.cat([cache, proj4], dim=1)
    conv_in = padded.permute(0, 3, 1, 2)
    conv_w = conv_left_w.permute(0, 3, 1, 2).contiguous()
    conv_out = F.conv2d(conv_in, conv_w, groups=proj.shape[-1])
    conv_out = conv_out.permute(0, 2, 3, 1)
    residual = proj4 + conv_out
    flat = residual.reshape(1, proj.shape[1], proj.shape[2])
    out = torch.relu(linear(flat, affine_w, affine_b))
    next_cache = padded[:, -cache_frames:, :, :].clone()
    return out, next_cache


def forward_once(state_dict: dict[str, torch.Tensor], frames: int) -> torch.Tensor:
    x = deterministic_features(frames, 400)
    x = torch.relu(
        linear(
            linear(
                x,
                state_dict["encoder.in_linear1.linear.weight"],
                state_dict["encoder.in_linear1.linear.bias"],
            ),
            state_dict["encoder.in_linear2.linear.weight"],
            state_dict["encoder.in_linear2.linear.bias"],
        )
    )
    cache_frames = 19
    for index in range(4):
        cache = torch.zeros((1, cache_frames, 1, 128), dtype=torch.float32)
        x, _ = run_layer(
            x,
            cache,
            state_dict[f"encoder.fsmn.{index}.linear.linear.weight"],
            state_dict[f"encoder.fsmn.{index}.fsmn_block.conv_left.weight"],
            state_dict[f"encoder.fsmn.{index}.affine.linear.weight"],
            state_dict[f"encoder.fsmn.{index}.affine.linear.bias"],
            cache_frames,
        )
    x = linear(
        linear(
            x,
            state_dict["encoder.out_linear1.linear.weight"],
            state_dict["encoder.out_linear1.linear.bias"],
        ),
        state_dict["encoder.out_linear2.linear.weight"],
        state_dict["encoder.out_linear2.linear.bias"],
    )
    probs = torch.softmax(x, dim=-1)
    return 1.0 - probs[0, :, 0]


def main() -> None:
    args = parse_args()
    bundle = Path(args.bundle)
    state_dict = load_file(bundle / "model.safetensors")

    for _ in range(args.warmup):
        forward_once(state_dict, args.frames)

    started = time.perf_counter()
    result = None
    for _ in range(args.iters):
        result = forward_once(state_dict, args.frames)
    python_ms = (time.perf_counter() - started) * 1000.0 / args.iters

    assert result is not None
    print(
        json.dumps(
            {
                "python_ms": python_ms,
                "speech_preview": [float(v) for v in result[:16].tolist()],
                "frames": args.frames,
            }
        )
    )


if __name__ == "__main__":
    main()
