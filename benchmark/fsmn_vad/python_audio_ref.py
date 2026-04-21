from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--pcm", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=160000)
    return parser.parse_args()


def parse_cmvn(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    lines = path.read_text(encoding="utf-8").splitlines()

    def section(header: str) -> list[float]:
        for index, line in enumerate(lines):
            parts = line.strip().split()
            if not parts or parts[0] != header:
                continue
            next_parts = lines[index + 1].strip().split()
            if not next_parts or next_parts[0] != "<LearnRateCoef>":
                break
            return [float(v) for v in next_parts[3:-1]]
        raise RuntimeError(f"Missing {header} section")

    return (
        torch.tensor(section("<AddShift>"), dtype=torch.float32),
        torch.tensor(section("<Rescale>"), dtype=torch.float32),
    )


def hamming_window(length: int) -> torch.Tensor:
    if length <= 1:
        return torch.ones(length, dtype=torch.float32)
    positions = torch.arange(length, dtype=torch.float32)
    return 0.54 - 0.46 * torch.cos((2.0 * math.pi * positions) / (length - 1))


def hz_to_mel(hz: float) -> float:
    return 1127.0 * math.log(1.0 + (hz / 700.0))


def mel_to_hz(mel: float) -> float:
    return 700.0 * (math.exp(mel / 1127.0) - 1.0)


def kaldi_mel_filterbank(sample_rate: int, fft_size: int, n_mels: int) -> torch.Tensor:
    bins = (fft_size // 2) + 1
    mel_min = hz_to_mel(0.0)
    mel_max = hz_to_mel(sample_rate / 2.0)
    mel_points = [mel_min + (mel_max - mel_min) * i / (n_mels + 1) for i in range(n_mels + 2)]
    hz_points = [mel_to_hz(m) for m in mel_points]
    fft_freqs = [sample_rate * i / fft_size for i in range(bins)]
    out = torch.zeros((bins, n_mels), dtype=torch.float32)
    for mel in range(n_mels):
        lower, center, upper = hz_points[mel], hz_points[mel + 1], hz_points[mel + 2]
        for bin_index, freq in enumerate(fft_freqs):
            left = (freq - lower) / (center - lower) if center > lower else 0.0
            right = (upper - freq) / (upper - center) if upper > center else 0.0
            out[bin_index, mel] = max(0.0, min(left, right))
    return out


def compute_fbank(audio: torch.Tensor) -> torch.Tensor:
    frame_length = 400
    frame_shift = 160
    fft_size = 512
    if audio.numel() < frame_length:
        return torch.zeros((0, 80), dtype=torch.float32)
    frame_count = ((audio.numel() - frame_length) // frame_shift) + 1
    indices = (
        torch.arange(frame_count, dtype=torch.long).unsqueeze(1) * frame_shift
        + torch.arange(frame_length, dtype=torch.long).unsqueeze(0)
    )
    frames = audio.index_select(0, indices.reshape(-1)).reshape(frame_count, frame_length)
    frames = frames - frames.mean(dim=1, keepdim=True)
    previous = torch.cat([frames[:, :1], frames[:, :-1]], dim=1)
    frames = frames - (0.97 * previous)
    frames = frames * hamming_window(frame_length).unsqueeze(0)
    frames = F.pad(frames, (0, fft_size - frame_length))
    spectrum = torch.fft.rfft(frames, n=fft_size, dim=1)
    power = spectrum.abs().pow(2)
    mel = power @ kaldi_mel_filterbank(16000, fft_size, 80)
    return torch.log(torch.clamp(mel, min=1e-10))


def apply_lfr(features: torch.Tensor) -> torch.Tensor:
    lfr_m = 5
    lfr_n = 1
    left_padding = features[0:1].repeat((lfr_m - 1) // 2, 1)
    features = torch.cat([left_padding, features], dim=0)
    total = features.shape[0]
    left_context = (lfr_m - 1) // 2
    t_lfr = math.ceil((total - left_context) / lfr_n)
    last_idx = ((total - lfr_m) // lfr_n) + 1
    num_padding = lfr_m - (total - last_idx * lfr_n)
    if num_padding > 0:
        extra = ((2 * lfr_m) - (2 * total) + ((t_lfr - 1 + last_idx) * lfr_n)) // 2
        copies = extra * (t_lfr - last_idx)
        if copies > 0:
            features = torch.cat([features] + [features[-1:]] * copies, dim=0)
    rows = []
    for row in range(t_lfr):
        start = row * lfr_n
        rows.append(features[start : start + lfr_m].reshape(-1))
    return torch.stack(rows)


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


def forward_once(state_dict: dict[str, torch.Tensor], offsets: torch.Tensor, scales: torch.Tensor, pcm_path: Path, max_samples: int) -> tuple[torch.Tensor, int, int]:
    audio_np = np.fromfile(pcm_path, dtype=np.float32)
    audio = torch.from_numpy(audio_np[: min(max_samples, len(audio_np))].copy())
    fbank = compute_fbank(audio)
    lfr = apply_lfr(fbank)
    lfr = (lfr + offsets) * scales
    x = torch.relu(
        linear(
            linear(
                lfr.unsqueeze(0),
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
    return 1.0 - probs[0, :, 0], lfr.shape[0], audio.numel()


def main() -> None:
    args = parse_args()
    bundle = Path(args.bundle)
    state_dict = load_file(bundle / "model.safetensors")
    offsets, scales = parse_cmvn(bundle / "am.mvn")

    for _ in range(args.warmup):
        forward_once(state_dict, offsets, scales, Path(args.pcm), args.max_samples)

    started = time.perf_counter()
    result = None
    frames = 0
    used_samples = 0
    for _ in range(args.iters):
        result, frames, used_samples = forward_once(
            state_dict,
            offsets,
            scales,
            Path(args.pcm),
            args.max_samples,
        )
    python_ms = (time.perf_counter() - started) * 1000.0 / args.iters

    assert result is not None
    print(
        json.dumps(
            {
                "python_ms": python_ms,
                "speech_preview": [float(v) for v in result[:16].tolist()],
                "frames": frames,
                "samples": used_samples,
            }
        )
    )


if __name__ == "__main__":
    main()
