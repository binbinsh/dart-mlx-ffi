"""Export pyannote/segmentation-3.0 weights + reference fixtures for MLX parity.

Usage:

    uv run --with torch --with pyannote.audio --with numpy --with soundfile \
        --with safetensors python tool/pyannote_seg_export.py \
            --out models/pyannote_seg \
            --fixtures test/data/pyannote_seg

Outputs
-------
models/pyannote_seg/
    cmdspace_mlx_pyannote_seg.json  -- manifest (model shape + class layout)
    weights.safetensors              -- all PyanNet parameters (fp32)
test/data/pyannote_seg/
    synthetic_10s.wav                -- deterministic 10s test audio
    reference_waveform.npy           -- (160000,) float32
    reference_sincnet.npy            -- SincNet output (C, T_out)
    reference_lstm.npy               -- after BiLSTM + linear stack
    reference_logits.npy             -- (frames, 7) pre-softmax
    reference_powerset.npy           -- (frames, 7) softmax probabilities
    reference_meta.json              -- summary

The saved fixtures give the MLX implementation staged parity targets:
    stage 1: SincNet frontend output shape / values
    stage 2: per-layer BiLSTM states
    stage 3: final powerset logits (what we consume at runtime)

All input audio is seeded-synthetic so no third-party sample ships.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np


def _set_determinism(seed: int = 20251009) -> None:
    import torch

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)


def _synth_waveform(sample_rate: int = 16000, seconds: float = 10.0,
                    seed: int = 20251009) -> np.ndarray:
    """Two 'speakers' + silence gap deterministic waveform.

    0.0-3.0s : speaker A-ish tone stack
    3.0-3.5s : silence
    3.5-7.0s : speaker B-ish tone stack (different fundamental)
    7.0-8.0s : overlap zone (A + B simultaneously)
    8.0-10.0s : low-amp noise
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    signal = np.zeros_like(t, dtype=np.float32)

    def _speaker(freq_base: float, t0: float, t1: float) -> np.ndarray:
        mask = (t >= t0) & (t < t1)
        env = 0.5 * (1.0 + np.sin(2.0 * math.pi * 0.9 * (t - t0)))
        voice = (
            0.6 * np.sin(2.0 * math.pi * freq_base * t)
            + 0.3 * np.sin(2.0 * math.pi * (freq_base * 3) * t)
            + 0.15 * np.sin(2.0 * math.pi * (freq_base * 5) * t)
        )
        return (voice * env * mask).astype(np.float32)

    signal += _speaker(170.0, 0.0, 3.0)
    signal += _speaker(240.0, 3.5, 7.0)
    signal += _speaker(170.0, 7.0, 8.0) + _speaker(240.0, 7.0, 8.0)
    signal += rng.normal(scale=0.02, size=signal.shape).astype(np.float32)

    peak = float(np.max(np.abs(signal)))
    if peak > 0:
        signal = (0.9 * signal / peak).astype(np.float32)
    return signal


def _dump_block_outputs(model, waveform: "torch.Tensor") -> dict:
    """Forward through PyanNet capturing intermediate tensors."""
    import torch

    outs: dict = {}
    with torch.no_grad():
        # PyanNet forward expects (batch, 1, samples)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        outs["input"] = waveform.cpu().numpy()

        # SincNet frontend
        sincnet_out = model.sincnet(waveform)
        outs["sincnet"] = sincnet_out.cpu().numpy()

        # LSTM expects (batch, T, C) so transpose
        lstm_in = sincnet_out.transpose(1, 2)
        lstm_out, _ = model.lstm(lstm_in)
        outs["lstm"] = lstm_out.cpu().numpy()

        # Linear stack
        x = lstm_out
        for i, layer in enumerate(model.linear):
            x = torch.nn.functional.leaky_relu(layer(x))
            outs[f"linear_{i}"] = x.cpu().numpy()

        # Classifier (raw logits), then log-softmax activation as in PyanNet
        logits = model.classifier(x)
        outs["logits"] = logits.cpu().numpy()
        # The public output of PyanNet is log-softmax; exp to get probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        outs["log_probs"] = log_probs.cpu().numpy()
        outs["powerset"] = torch.exp(log_probs).cpu().numpy()
    return outs


def _export_weights(model, dest: Path) -> dict:
    import torch

    state_dict = model.state_dict()
    tensors: dict = {}
    shapes: dict = {}
    for key, value in state_dict.items():
        array = value.detach().cpu().numpy().astype(np.float32)
        tensors[key] = array
        shapes[key] = list(array.shape)
    try:
        from safetensors.numpy import save_file
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "safetensors is required. Install with `uv add safetensors`."
        ) from exc
    dest.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(dest))
    return shapes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="models/pyannote_seg")
    parser.add_argument("--fixtures", default="test/data/pyannote_seg")
    parser.add_argument("--seed", type=int, default=20251009)
    args = parser.parse_args()

    _set_determinism(args.seed)

    import soundfile as sf
    import torch
    from pyannote.audio import Model

    out_dir = Path(args.out)
    fixture_dir = Path(args.fixtures)
    out_dir.mkdir(parents=True, exist_ok=True)
    fixture_dir.mkdir(parents=True, exist_ok=True)

    print("Loading pyannote/segmentation-3.0...")
    model = Model.from_pretrained("pyannote/segmentation-3.0")
    model.eval()

    # Capture specifications (duration, classes, powerset max classes)
    spec = model.specifications
    classes = list(spec.classes)
    powerset_max = int(spec.powerset_max_classes)
    duration = float(spec.duration)

    print("Synthesizing 10s reference waveform...")
    waveform = _synth_waveform(seconds=duration, seed=args.seed)
    wav_path = fixture_dir / "synthetic_10s.wav"
    sf.write(wav_path, waveform, 16000, subtype="PCM_16")
    np.save(fixture_dir / "reference_waveform.npy", waveform)

    print("Running forward pass, capturing activations...")
    wav_tensor = torch.from_numpy(waveform).unsqueeze(0).unsqueeze(0).float()
    activations = _dump_block_outputs(model, wav_tensor)

    np.save(fixture_dir / "reference_sincnet.npy", activations["sincnet"])
    np.save(fixture_dir / "reference_lstm.npy", activations["lstm"])
    np.save(fixture_dir / "reference_logits.npy", activations["logits"])
    np.save(fixture_dir / "reference_log_probs.npy", activations["log_probs"])
    np.save(fixture_dir / "reference_powerset.npy", activations["powerset"])

    print("Exporting weights...")
    shapes = _export_weights(model, out_dir / "weights.safetensors")

    manifest = {
        "format": "cmdspace-mlx-pyannote-seg/v1",
        "model_id": "pyannote/segmentation-3.0",
        "weights": "weights.safetensors",
        "sample_rate": 16000,
        "window_duration_seconds": duration,
        "window_samples": int(duration * 16000),
        "sincnet": {
            "stride": 10,
            "kernel_size": 251,
            "sample_rate": 16000,
            "n_filters": [80, 60, 60],
        },
        "lstm": {
            "input_size": 60,
            "hidden_size": 128,
            "num_layers": 4,
            "bidirectional": True,
            "dropout": 0.0,
        },
        "linear_hidden_sizes": [128, 128],
        "num_classes": int(activations["logits"].shape[-1]),
        "powerset_max_classes": powerset_max,
        "num_speakers": len(classes),
        "classes": classes,
        # Powerset class layout for pyannote 3.0 with max_classes=2, n=3:
        #   index 0 = silence
        #   index 1 = spk0
        #   index 2 = spk1
        #   index 3 = spk2
        #   index 4 = spk0+spk1 overlap
        #   index 5 = spk0+spk2 overlap
        #   index 6 = spk1+spk2 overlap
        "powerset_index_layout": [
            [],
            [0],
            [1],
            [2],
            [0, 1],
            [0, 2],
            [1, 2],
        ],
        "tensor_shapes": shapes,
    }
    (out_dir / "cmdspace_mlx_pyannote_seg.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    meta = {
        "format": "cmdspace-mlx-pyannote-seg-fixtures/v1",
        "seed": args.seed,
        "sample_rate": 16000,
        "duration_seconds": duration,
        "waveform_shape": list(waveform.shape),
        "sincnet_shape": list(activations["sincnet"].shape),
        "lstm_shape": list(activations["lstm"].shape),
        "logits_shape": list(activations["logits"].shape),
        "powerset_shape": list(activations["powerset"].shape),
        "classes": classes,
        "powerset_max_classes": powerset_max,
    }
    (fixture_dir / "reference_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n"
    )

    print("Done.")
    print(f"  weights:  {out_dir / 'weights.safetensors'}")
    print(f"  manifest: {out_dir / 'cmdspace_mlx_pyannote_seg.json'}")
    print(f"  fixtures: {fixture_dir}")
    print(f"  logits:   {activations['logits'].shape}")


if __name__ == "__main__":
    main()
