"""Export SpeechBrain ECAPA-TDNN into MLX-ready fused weights + parity fixtures.

Usage:

    uv sync
    uv run --with torch --with speechbrain --with numpy --with soundfile \
        python tool/ecapa_export.py \
            --out models/ecapa_tdnn \
            --fixtures test/data/speaker_embedding

What this script produces
-------------------------

models/ecapa_tdnn/
    cmdspace_mlx_ecapa_tdnn.json  -- manifest (architecture + fused key list)
    weights.safetensors           -- fused tensors in MLX layout:
        blocks.0.conv.w                (C_out, kW, C_in)   Conv1d kernel
        blocks.0.conv.b                (C_out,)            Conv1d bias
        blocks.0.bn.scale              (C_out,)  fused BN  w / sqrt(var+eps)
        blocks.0.bn.bias               (C_out,)  fused BN  b - mean*scale
        blocks.1.tdnn1.{w,b,bn.scale,bn.bias}               (TDNN 1x1)
        blocks.1.res2net.blocks.{0..6}.{w,b,bn.scale,bn.bias}
        blocks.1.tdnn2.{w,b,bn.scale,bn.bias}
        blocks.1.se.conv1.{w,b}        (no BN)
        blocks.1.se.conv2.{w,b}
        ... same for blocks.2, blocks.3 ...
        mfa.{w,b,bn.scale,bn.bias}
        asp.tdnn.{w,b,bn.scale,bn.bias}
        asp.conv.{w,b}                 (no BN)
        asp_bn.{scale,bias}
        fc.{w,b}

test/data/speaker_embedding/
    synthetic_tone_3s.wav
    reference_waveform.npy
    reference_fbank_raw.npy        (T, 80) SpeechBrain pre-mean-norm log mel
    reference_fbank.npy            (T, 80) post mean-norm
    reference_block_outputs.npz    per-stage activations captured channel-first
    reference_embedding.npy        (192,) raw embedding (no L2 norm)
    reference_meta.json            shapes + norms (debug aids)

Key layout notes
----------------
* Conv1d weight PyTorch layout is (C_out, C_in, kW). MLX `mx.conv1d` accepts
  (C_out, kW, C_in). We transpose at export so runtime never has to reshape.
* BatchNorm1d eval mode is `y = (x - mean) / sqrt(var + eps) * weight + bias`.
  We pre-compute `scale = weight / sqrt(var + eps)` and
  `bias' = bias - mean * scale`; runtime does one broadcast multiply + add.
* The exporter is deterministic; synthetic waveform uses a seeded RNG.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _set_determinism(seed: int) -> None:
    import torch

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)


def _synth_reference_waveform(
    sample_rate: int = 16000, seconds: float = 3.0, seed: int = 20251008
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    env = 0.5 * (1.0 + np.sin(2.0 * math.pi * 0.8 * t))
    signal = (
        0.6 * np.sin(2.0 * math.pi * 180.0 * t)
        + 0.3 * np.sin(2.0 * math.pi * 540.0 * t)
        + 0.15 * np.sin(2.0 * math.pi * 1100.0 * t)
    )
    noise = rng.normal(scale=0.02, size=signal.shape)
    waveform = (signal * env + noise).astype(np.float32)
    peak = float(np.max(np.abs(waveform)))
    if peak > 0:
        waveform = 0.95 * waveform / peak
    return waveform


def _load_speechbrain(cache_dir: Path):
    from speechbrain.inference.speaker import EncoderClassifier

    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(cache_dir),
    )


def _fuse_bn(
    sd: Dict[str, "torch.Tensor"], prefix: str, eps: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Fuse (weight, bias, running_mean, running_var) into (scale, bias')."""
    w = sd[f"{prefix}.weight"].detach().cpu().numpy().astype(np.float64)
    b = sd[f"{prefix}.bias"].detach().cpu().numpy().astype(np.float64)
    mean = sd[f"{prefix}.running_mean"].detach().cpu().numpy().astype(np.float64)
    var = sd[f"{prefix}.running_var"].detach().cpu().numpy().astype(np.float64)
    scale = w / np.sqrt(var + eps)
    bias = b - mean * scale
    return scale.astype(np.float32), bias.astype(np.float32)


def _conv_layout(sd: Dict[str, "torch.Tensor"], prefix: str) -> Tuple[np.ndarray, np.ndarray]:
    """Transpose Conv1d weight (out, in, k) -> MLX (out, k, in). Copy bias."""
    w = sd[f"{prefix}.weight"].detach().cpu().numpy()  # (out, in, k)
    b = sd[f"{prefix}.bias"].detach().cpu().numpy()
    w_mlx = np.transpose(w, (0, 2, 1)).astype(np.float32).copy()
    return w_mlx, b.astype(np.float32).copy()


def _build_fused_tensors(model) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Produce the fused tensor dict in deterministic key order."""
    import torch.nn as nn

    # Discover BN eps from any BatchNorm1d in the model.
    bn_eps = 1e-5
    for _, m in model.named_modules():
        if isinstance(m, nn.BatchNorm1d):
            bn_eps = float(m.eps)
            break

    sd = model.state_dict()
    t: Dict[str, np.ndarray] = {}
    keys: List[str] = []

    def add(name: str, arr: np.ndarray) -> None:
        t[name] = arr
        keys.append(name)

    def fuse_block(cprefix: str, bprefix: str, out_prefix: str) -> None:
        w, b = _conv_layout(sd, cprefix)
        add(f"{out_prefix}.w", w)
        add(f"{out_prefix}.b", b)
        scale, bias = _fuse_bn(sd, bprefix, bn_eps)
        add(f"{out_prefix}.bn.scale", scale)
        add(f"{out_prefix}.bn.bias", bias)

    # block 0: TDNNBlock (conv + norm)
    fuse_block("blocks.0.conv.conv", "blocks.0.norm.norm", "blocks.0")

    # blocks 1..3: SERes2NetBlock (tdnn1 + res2net(blocks.0..scale-2) + tdnn2 + se)
    res2net_scale = 8  # from manifest
    for i in (1, 2, 3):
        fuse_block(
            f"blocks.{i}.tdnn1.conv.conv",
            f"blocks.{i}.tdnn1.norm.norm",
            f"blocks.{i}.tdnn1",
        )
        for j in range(res2net_scale - 1):
            fuse_block(
                f"blocks.{i}.res2net_block.blocks.{j}.conv.conv",
                f"blocks.{i}.res2net_block.blocks.{j}.norm.norm",
                f"blocks.{i}.res2net.{j}",
            )
        fuse_block(
            f"blocks.{i}.tdnn2.conv.conv",
            f"blocks.{i}.tdnn2.norm.norm",
            f"blocks.{i}.tdnn2",
        )
        # SE conv1/conv2: no BN, no activation before final sigmoid in conv2.
        w, b = _conv_layout(sd, f"blocks.{i}.se_block.conv1.conv")
        add(f"blocks.{i}.se.conv1.w", w)
        add(f"blocks.{i}.se.conv1.b", b)
        w, b = _conv_layout(sd, f"blocks.{i}.se_block.conv2.conv")
        add(f"blocks.{i}.se.conv2.w", w)
        add(f"blocks.{i}.se.conv2.b", b)

    # MFA: TDNNBlock
    fuse_block("mfa.conv.conv", "mfa.norm.norm", "mfa")

    # ASP: internal tdnn (conv+BN) + final 1x1 conv (no BN)
    fuse_block("asp.tdnn.conv.conv", "asp.tdnn.norm.norm", "asp.tdnn")
    w, b = _conv_layout(sd, "asp.conv.conv")
    add("asp.conv.w", w)
    add("asp.conv.b", b)

    # Final BN on pooled stats (no preceding conv): 6144-dim
    # Fuse weight/bias/running_mean/running_var.
    scale, bias = _fuse_bn(sd, "asp_bn.norm", bn_eps)
    add("asp_bn.scale", scale)
    add("asp_bn.bias", bias)

    # FC: final 1x1 Conv1d (no BN)
    w, b = _conv_layout(sd, "fc.conv")
    add("fc.w", w)
    add("fc.b", b)

    return t, keys


def _build_frontend_tensors(classifier) -> Dict[str, np.ndarray]:
    """Extract the Hamming window and mel filterbank matrix baked into the
    SpeechBrain feature extractor so the MLX runtime can use the exact same
    numerical frontend without re-implementing mel scale details.

    Rather than replicate SpeechBrain's `Filterbank._create_fbank_matrix`
    from scratch (which has subtle mel-band width conventions), we invoke
    the real module on a synthetic power spectrum and solve for the matrix
    algebraically — the frontend is strictly linear between the power
    spectrogram and its log-mel output (log is applied *after* the matmul).
    """
    import torch

    cf = classifier.mods.compute_features
    stft = cf.compute_STFT
    fb = cf.compute_fbanks
    window = stft.window.detach().cpu().numpy().astype(np.float32).copy()

    n_fft = fb.n_fft
    n_stft = n_fft // 2 + 1
    n_mels = fb.n_mels
    # fb.forward signature: fb(spectrogram) where spectrogram has shape
    # (batch, time, n_stft, 2-or-1) or (batch, time, n_stft) for power.
    # The caller passes power spectrogram (spectrogram ** 2) with shape
    # (batch, time, n_stft). We need the matmul weight only, so poke it.
    # Build identity: for each of n_stft bins, a one-hot power spectrum of
    # shape (1, 1, n_stft). After fb.forward we'd get the log-mel row; but
    # log is the user-facing op in the recipe. In this module specifically,
    # fb(spectrogram) returns torch.matmul(spectrogram, fbank_matrix) where
    # fbank_matrix has shape (n_stft, n_mels). So the resulting rows ARE
    # the fbank matrix (no log is applied inside Filterbank).
    eye = torch.eye(n_stft).unsqueeze(0)  # (1, n_stft, n_stft)
    saved_log_mel = fb.log_mel
    fb.log_mel = False
    try:
        with torch.no_grad():
            out = fb(eye)  # (1, n_stft, n_mels)
    finally:
        fb.log_mel = saved_log_mel
    fm = out[0].detach().cpu().numpy().astype(np.float32).copy()
    assert fm.shape == (n_stft, n_mels), fm.shape
    return {"frontend.window": window, "frontend.mel_fb": fm}


def _dump_block_outputs(model, mel: "torch.Tensor") -> Dict[str, np.ndarray]:
    import torch

    out: Dict[str, np.ndarray] = {}
    with torch.no_grad():
        x = mel.transpose(1, 2)  # (B, n_mels, T)
        out["input_channel_first"] = x.detach().cpu().numpy()
        xl = []
        for i, layer in enumerate(model.blocks):
            x = layer(x) if i == 0 else layer(x, lengths=None)
            out[f"block_{i}"] = x.detach().cpu().numpy()
            xl.append(x)
        x = torch.cat(xl[1:], dim=1)
        out["pre_mfa_concat"] = x.detach().cpu().numpy()
        x = model.mfa(x)
        out["mfa"] = x.detach().cpu().numpy()
        x = model.asp(x, lengths=None)
        out["asp"] = x.detach().cpu().numpy()
        x = model.asp_bn(x)
        out["asp_bn"] = x.detach().cpu().numpy()
        x = model.fc(x)
        out["fc"] = x.detach().cpu().numpy()
        x = x.transpose(1, 2)
        out["embedding"] = x.squeeze().detach().cpu().numpy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="models/ecapa_tdnn")
    parser.add_argument("--fixtures", default="test/data/speaker_embedding")
    parser.add_argument("--cache", default="/tmp/ecapa_sb")
    parser.add_argument("--seed", type=int, default=20251008)
    args = parser.parse_args()

    _set_determinism(args.seed)

    import soundfile as sf
    import torch

    out_dir = Path(args.out)
    fx_dir = Path(args.fixtures)
    out_dir.mkdir(parents=True, exist_ok=True)
    fx_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SpeechBrain ECAPA-TDNN...")
    classifier = _load_speechbrain(Path(args.cache))
    model = classifier.mods.embedding_model
    model.eval()

    print("Synthesizing reference waveform...")
    waveform = _synth_reference_waveform(seed=args.seed)
    sf.write(fx_dir / "synthetic_tone_3s.wav", waveform, 16000, subtype="PCM_16")
    np.save(fx_dir / "reference_waveform.npy", waveform.astype(np.float32))

    print("Running SpeechBrain feature pipeline...")
    wav = torch.from_numpy(waveform).unsqueeze(0).float()
    with torch.no_grad():
        fbank_raw = classifier.mods.compute_features(wav)  # (1, T, 80)
        fbank_norm = classifier.mods.mean_var_norm(
            fbank_raw, torch.ones(wav.shape[0])
        )
    np.save(fx_dir / "reference_fbank_raw.npy", fbank_raw.squeeze(0).cpu().numpy())
    np.save(fx_dir / "reference_fbank.npy", fbank_norm.squeeze(0).cpu().numpy())

    print("Capturing per-block activations...")
    block_outs = _dump_block_outputs(model, fbank_norm)
    np.savez(fx_dir / "reference_block_outputs.npz", **block_outs)
    # Also dump each block output as its own .npy so the Dart parity test
    # can load them directly with `mx.io.load` (no npz parser needed).
    for name, arr in block_outs.items():
        np.save(fx_dir / f"reference_{name}.npy", arr.astype(np.float32))

    print("Exporting fused weights...")
    tensors, ordered_keys = _build_fused_tensors(model)

    print("Extracting frontend (window + mel filterbank) tensors...")
    frontend = _build_frontend_tensors(classifier)
    for k, v in frontend.items():
        tensors[k] = v
        ordered_keys.append(k)

    weights_path = out_dir / "weights.safetensors"
    from safetensors.numpy import save_file

    save_file(tensors, str(weights_path))

    print("Computing final embedding via classifier (for fixture only)...")
    with torch.no_grad():
        emb = (
            classifier.encode_batch(torch.from_numpy(waveform).unsqueeze(0).float())
            .squeeze()
            .cpu()
            .numpy()
        )
    np.save(fx_dir / "reference_embedding.npy", emb.astype(np.float32))

    # Manifest
    # Mel filterbank parameters must match SpeechBrain's `Filterbank` defaults
    # used by the `spkrec-ecapa-voxceleb` recipe: sample_rate=16000, n_fft=400,
    # win_length=25ms, hop_length=10ms, n_mels=80, f_min=0, f_max=8000,
    # log10*10 and sentence-level mean subtraction without std normalization.
    manifest: Dict[str, Any] = {
        "format": "cmdspace-mlx-ecapa-tdnn/v2",
        "model_id": "speechbrain/spkrec-ecapa-voxceleb",
        "weights": weights_path.name,
        "sample_rate": 16000,
        "n_fft": 400,
        "win_length": 400,
        "hop_length": 160,
        "window": "hamming",
        "n_mels": 80,
        "f_min": 0,
        "f_max": 8000,
        "log_scale": 10.0,
        "log_floor": 1e-10,
        "mean_norm": "sentence",
        "std_norm": False,
        "embedding_dim": 192,
        "channels": [1024, 1024, 1024, 1024, 3072],
        "kernel_sizes": [5, 3, 3, 3, 1],
        "dilations": [1, 2, 3, 4, 1],
        "attention_channels": 128,
        "res2net_scale": 8,
        "se_channels": 128,
        "bn_eps": 1e-5,
        "asp_eps": 1e-12,
        "conv_padding_mode": "reflect",
        "fused_keys": ordered_keys,
    }
    (out_dir / "cmdspace_mlx_ecapa_tdnn.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    meta = {
        "format": "cmdspace-mlx-ecapa-tdnn-fixtures/v2",
        "seed": args.seed,
        "sample_rate": 16000,
        "duration_seconds": 3.0,
        "fbank_raw_shape": list(fbank_raw.squeeze(0).shape),
        "fbank_norm_shape": list(fbank_norm.squeeze(0).shape),
        "embedding_shape": list(emb.shape),
        "embedding_norm": float(np.linalg.norm(emb)),
        "block_shapes": {k: list(v.shape) for k, v in block_outs.items()},
    }
    (fx_dir / "reference_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n"
    )

    print("Done.")
    print(f"  weights:  {weights_path}  ({len(tensors)} tensors)")
    print(f"  manifest: {out_dir / 'cmdspace_mlx_ecapa_tdnn.json'}")
    print(f"  fixtures: {fx_dir}")
    print(f"  embedding: shape={emb.shape} norm={np.linalg.norm(emb):.4f}")


if __name__ == "__main__":
    main()
