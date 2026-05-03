from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from safetensors.torch import load_file


@dataclass(frozen=True)
class MingTtsShapes:
    hidden_size: int
    latent_dim: int
    patch_size: int
    history_patch_size: int
    audio_output_dim: int
    audio_decoder_config: dict[str, Any]
    ditar_config: dict[str, Any]
    aggregator_config: dict[str, Any]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export executable Ming-omni TTS native ONNX components."
    )
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--components-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--component",
        action="append",
        choices=[
            "flowloss_dit_step",
            "linear_proj_audio",
            "stop_head",
            "audio_decode_chunk",
        ],
        help="Component to export. Defaults to all supported components.",
    )
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--dtype", choices=["float32"], default="float32")
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Export a dynamic batch axis. Mobile LiteRT conversion uses static batch.",
    )
    args = parser.parse_args()

    report = export_native_components(
        source_dir=args.source_dir,
        components_dir=args.components_dir,
        output_dir=args.output_dir,
        components=args.component
        or [
            "flowloss_dit_step",
            "linear_proj_audio",
            "stop_head",
            "audio_decode_chunk",
        ],
        opset=args.opset,
        dtype=args.dtype,
        dynamic_batch=args.dynamic_batch,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def export_native_components(
    *,
    source_dir: Path,
    components_dir: Path,
    output_dir: Path,
    components: Iterable[str],
    opset: int = 18,
    dtype: str = "float32",
    dynamic_batch: bool = False,
) -> dict[str, Any]:
    source_dir = source_dir.expanduser().resolve()
    components_dir = components_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shapes = read_ming_tts_shapes(source_dir / "config.json")
    report_path = output_dir / "native_component_export_report.json"
    exported: dict[str, Any] = _existing_native_components(report_path)
    for component in components:
        if component == "flowloss_dit_step":
            exported[component] = export_flowloss_dit_step_onnx(
                source_dir=source_dir,
                component_path=components_dir / "flowloss.safetensors",
                output_path=output_dir / "flowloss_dit_step.onnx",
                shapes=shapes,
                opset=opset,
                dynamic_batch=dynamic_batch,
            )
        elif component == "stop_head":
            exported[component] = export_stop_head_onnx(
                component_path=components_dir / "stop_head.safetensors",
                output_path=output_dir / "stop_head.onnx",
                hidden_size=shapes.hidden_size,
                opset=opset,
                dynamic_batch=dynamic_batch,
            )
        elif component == "linear_proj_audio":
            exported[component] = export_linear_proj_audio_onnx(
                source_dir=source_dir,
                component_path=components_dir / "linear_proj_audio.safetensors",
                output_path=output_dir / "linear_proj_audio.onnx",
                shapes=shapes,
                opset=opset,
                dynamic_batch=dynamic_batch,
            )
        elif component == "audio_decode_chunk":
            exported[component] = export_audio_decode_chunk_onnx(
                source_dir=source_dir,
                component_path=components_dir / "audio.safetensors",
                output_path=output_dir / "audio_decode_chunk.onnx",
                shapes=shapes,
                opset=opset,
                dynamic_batch=dynamic_batch,
            )
        else:
            raise ValueError(f"Unsupported Ming component: {component}")
    report = {
        "format": "dart_mlx_ffi.ming_omni_tts_native_components.v1",
        "source_dir": str(source_dir),
        "components_dir": str(components_dir),
        "output_dir": str(output_dir),
        "dtype": dtype,
        "opset": opset,
        "shape_mode": "dynamic_batch" if dynamic_batch else "static_batch1",
        "components": exported,
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def _existing_native_components(report_path: Path) -> dict[str, Any]:
    if not report_path.exists():
        return {}
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if report.get("format") != "dart_mlx_ffi.ming_omni_tts_native_components.v1":
        return {}
    components = report.get("components")
    return dict(components) if isinstance(components, dict) else {}


def read_ming_tts_shapes(config_path: Path) -> MingTtsShapes:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    llm_config = _required_dict(config, "llm_config")
    audio_config = _required_dict(config, "audio_tokenizer_config")
    ditar_config = _required_dict(config, "ditar_config")
    ditar_config = dict(ditar_config)
    aggregator_config = dict(_required_dict(config, "aggregator_config"))
    enc_kwargs = _required_dict(audio_config, "enc_kwargs")
    dec_kwargs = _required_dict(audio_config, "dec_kwargs")
    hidden_size = _required_int(llm_config, "hidden_size")
    latent_dim = _required_int(enc_kwargs, "latent_dim")
    audio_output_dim = _required_int(dec_kwargs, "output_dim")
    audio_decoder_config = dict(_required_dict(dec_kwargs, "backbone"))
    patch_size = _required_int(ditar_config, "patch_size")
    history_patch_size = _required_int(ditar_config, "history_patch_size")
    return MingTtsShapes(
        hidden_size=hidden_size,
        latent_dim=latent_dim,
        patch_size=patch_size,
        history_patch_size=history_patch_size,
        audio_output_dim=audio_output_dim,
        audio_decoder_config=audio_decoder_config,
        ditar_config=ditar_config,
        aggregator_config=aggregator_config,
    )


def export_flowloss_dit_step_onnx(
    *,
    source_dir: Path,
    component_path: Path,
    output_path: Path,
    shapes: MingTtsShapes,
    opset: int,
    dynamic_batch: bool = False,
) -> dict[str, Any]:
    torch = _torch()
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    from fm.flowloss import FlowLoss  # type: ignore

    state = _float_state_dict(load_file(component_path))
    flowloss = FlowLoss(
        z_channels=shapes.latent_dim,
        llm_cond_dim=shapes.hidden_size,
        **shapes.ditar_config,
    ).float()
    flowloss.load_state_dict(state, strict=True)
    model = _flowloss_denoiser_step(torch, flowloss.cfm.model, shapes.patch_size)
    model.eval()
    noised = torch.zeros((1, shapes.patch_size, shapes.latent_dim), dtype=torch.float32)
    timestep = torch.zeros((1,), dtype=torch.float32)
    cond = torch.zeros((1, 1, shapes.hidden_size), dtype=torch.float32)
    history = torch.zeros(
        (1, shapes.history_patch_size, shapes.latent_dim),
        dtype=torch.float32,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    expected = _export_onnx(
        model,
        (noised, timestep, cond, history),
        output_path=output_path,
        input_names=[
            "noised_latent_patch",
            "timestep",
            "llm_cond",
            "latent_history",
        ],
        output_names=["flow_velocity"],
        dynamic_axes={
            "noised_latent_patch": {0: "batch"},
            "timestep": {0: "batch"},
            "llm_cond": {0: "batch"},
            "latent_history": {0: "batch"},
            "flow_velocity": {0: "batch"},
        }
        if dynamic_batch
        else None,
        opset=opset,
    )
    return {
        "artifact": str(output_path),
        "input_names": [
            "noised_latent_patch",
            "timestep",
            "llm_cond",
            "latent_history",
        ],
        "output_names": ["flow_velocity"],
        "input_shapes": {
            "noised_latent_patch": list(noised.shape),
            "timestep": list(timestep.shape),
            "llm_cond": list(cond.shape),
            "latent_history": list(history.shape),
        },
        "output_shape": list(expected.shape),
        "tensor_count": len(state),
        "max_abs_error": _max_abs_error(
            model,
            (noised, timestep, cond, history),
            expected,
        ),
    }


def export_stop_head_onnx(
    *,
    component_path: Path,
    output_path: Path,
    hidden_size: int,
    opset: int,
    dynamic_batch: bool = False,
) -> dict[str, Any]:
    torch = _torch()
    state = _float_state_dict(load_file(component_path))
    model = torch.nn.Linear(hidden_size, 2)
    model.load_state_dict(state, strict=True)
    model.eval()
    dummy = torch.zeros((1, 1, hidden_size), dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    expected = _export_onnx(
        model,
        (dummy,),
        output_path=output_path,
        input_names=["z_diff"],
        output_names=["stop_logits"],
        dynamic_axes={
            "z_diff": {0: "batch", 1: "sequence"},
            "stop_logits": {0: "batch", 1: "sequence"},
        }
        if dynamic_batch
        else None,
        opset=opset,
    )
    return {
        "artifact": str(output_path),
        "input_names": ["z_diff"],
        "output_names": ["stop_logits"],
        "input_shape": list(dummy.shape),
        "output_shape": list(expected.shape),
        "tensor_count": len(state),
        "max_abs_error": _max_abs_error(model, (dummy,), expected),
    }


def export_linear_proj_audio_onnx(
    *,
    source_dir: Path,
    component_path: Path,
    output_path: Path,
    shapes: MingTtsShapes,
    opset: int,
    dynamic_batch: bool = False,
) -> dict[str, Any]:
    torch = _torch()
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    from fm.dit import Aggregator  # type: ignore

    state = _float_state_dict(load_file(component_path))
    model = Aggregator(
        in_channels=shapes.latent_dim,
        llm_input_dim=shapes.hidden_size,
        **shapes.aggregator_config,
    ).float()
    model.load_state_dict(state, strict=True)
    model.eval()
    dummy = torch.zeros((1, shapes.patch_size, shapes.latent_dim), dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    expected = _export_onnx(
        model,
        (dummy,),
        output_path=output_path,
        input_names=["sampled_latent_patch"],
        output_names=["audio_inputs_embeds"],
        dynamic_axes={
            "sampled_latent_patch": {0: "batch"},
            "audio_inputs_embeds": {0: "batch"},
        }
        if dynamic_batch
        else None,
        opset=opset,
    )
    return {
        "artifact": str(output_path),
        "input_names": ["sampled_latent_patch"],
        "output_names": ["audio_inputs_embeds"],
        "input_shape": list(dummy.shape),
        "output_shape": list(expected.shape),
        "tensor_count": len(state),
        "max_abs_error": _max_abs_error(model, (dummy,), expected),
    }


def export_audio_decode_chunk_onnx(
    *,
    source_dir: Path,
    component_path: Path,
    output_path: Path,
    shapes: MingTtsShapes,
    opset: int,
    dynamic_batch: bool = False,
) -> dict[str, Any]:
    torch = _torch()
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    from audio_tokenizer.vae_modules import Decoder  # type: ignore

    state = _audio_decoder_state_dict(load_file(component_path))
    decoder_config = _onnx_decoder_config(shapes.audio_decoder_config)
    reference = Decoder(
        decoder_config,
        output_dim=shapes.audio_output_dim,
        latent_dim=shapes.latent_dim,
        semantic_model=None,
        patch_size=shapes.patch_size,
    ).float()
    reference.load_state_dict(state, strict=True)
    reference.eval()
    model = _audio_decode_chunk_module(torch, reference, shapes)
    model.eval()
    dummy = torch.zeros((1, shapes.patch_size, shapes.latent_dim), dtype=torch.float32)
    with torch.no_grad():
        official = reference.low_level_reconstruct(
            dummy,
            use_cache=False,
            stream_state=(None, None, None),
            last_chunk=True,
        )[0].detach().cpu()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    expected = _export_onnx(
        model,
        (dummy,),
        output_path=output_path,
        input_names=["sampled_latent_patch"],
        output_names=["waveform"],
        dynamic_axes={
            "sampled_latent_patch": {0: "batch"},
            "waveform": {0: "batch"},
        }
        if dynamic_batch
        else None,
        opset=opset,
    )
    return {
        "artifact": str(output_path),
        "input_names": ["sampled_latent_patch"],
        "output_names": ["waveform"],
        "input_shape": list(dummy.shape),
        "output_shape": list(expected.shape),
        "tensor_count": len(state),
        "decoder_layers": decoder_config.get("num_hidden_layers"),
        "audio_output_dim": shapes.audio_output_dim,
        "istft": {
            "n_fft": shapes.audio_output_dim * 4,
            "hop_length": shapes.audio_output_dim,
            "frame_count": shapes.patch_size * shapes.patch_size,
        },
        "official_max_abs_error": _tensor_max_abs_error(expected, official),
        "max_abs_error": _max_abs_error(model, (dummy,), expected),
    }


def _export_onnx(
    model: Any,
    args: tuple[Any, ...],
    *,
    output_path: Path,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]] | None,
    opset: int,
) -> Any:
    torch = _torch()
    with torch.no_grad():
        expected = model(*args)
    torch.onnx.export(
        model,
        args,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        export_params=True,
        do_constant_folding=True,
        external_data=True,
        dynamo=False,
    )
    return expected.detach().cpu()


def _max_abs_error(model: Any, args: tuple[Any, ...], expected: Any) -> float:
    torch = _torch()
    with torch.no_grad():
        actual = model(*args).detach().cpu()
    return float(torch.max(torch.abs(actual - expected)).item())


def _tensor_max_abs_error(left: Any, right: Any) -> float:
    torch = _torch()
    return float(torch.max(torch.abs(left - right)).item())


def _flowloss_denoiser_step(torch: Any, model: Any, patch_size: int) -> Any:
    class FlowLossDenoiserStep(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model
            self.patch_size = patch_size

        def forward(self, x: Any, t: Any, c: Any, latent_history: Any) -> Any:
            output = self.model(x=x, t=t, c=c, latent_history=latent_history)
            return output[:, -self.patch_size :, :]

    return FlowLossDenoiserStep()


def _audio_decode_chunk_module(torch: Any, decoder: Any, shapes: MingTtsShapes) -> Any:
    class AudioDecodeChunk(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = decoder.fc1
            self.upsampler = decoder.upsampling.upsampler
            self.decoder = decoder.decoder
            frame_count = shapes.patch_size * shapes.patch_size
            mask = torch.full(
                (1, 1, frame_count, frame_count),
                -1.0e9,
                dtype=torch.float32,
            )
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("attention_mask", mask)
            position_ids = torch.arange(frame_count, dtype=torch.long)
            self.register_buffer("position_ids", position_ids.view(1, -1))
            self.register_buffer("cache_position", position_ids)
            self.head = _onnx_friendly_istft_head(
                torch,
                decoder.head.out,
                n_fft=shapes.audio_output_dim * 4,
                hop_length=shapes.audio_output_dim,
                frame_count=frame_count,
            )

        def forward(self, sampled_latent_patch: Any) -> Any:
            x = self.fc1(sampled_latent_patch)
            x = self.upsampler(x.transpose(1, 2)).transpose(1, 2)
            attention_mask = {
                "full_attention": self.attention_mask,
                "sliding_attention": self.attention_mask,
            }
            x = self.decoder(
                inputs_embeds=x,
                attention_mask=attention_mask,
                position_ids=self.position_ids,
                cache_position=self.cache_position,
                use_cache=False,
            ).last_hidden_state
            return self.head(x)

    return AudioDecodeChunk()


def _onnx_friendly_istft_head(
    torch: Any,
    linear: Any,
    *,
    n_fft: int,
    hop_length: int,
    frame_count: int,
) -> Any:
    class RealIstftHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.out = linear
            self.n_fft = n_fft
            self.hop_length = hop_length
            self.frame_count = frame_count
            self.output_size = (frame_count - 1) * hop_length + n_fft
            self.pad = (n_fft - hop_length) // 2
            freq_bins = n_fft // 2 + 1
            bins = torch.arange(freq_bins, dtype=torch.float32).view(-1, 1)
            samples = torch.arange(n_fft, dtype=torch.float32).view(1, -1)
            angle = 2 * math.pi * bins * samples / n_fft
            scale = torch.full((freq_bins, 1), 2.0 / n_fft, dtype=torch.float32)
            scale[0] = 1.0 / n_fft
            scale[-1] = 1.0 / n_fft
            self.register_buffer("cos_basis", torch.cos(angle) * scale)
            self.register_buffer("sin_basis", torch.sin(angle) * scale)
            window = torch.hann_window(n_fft, dtype=torch.float32)
            self.register_buffer("window", window.view(1, n_fft, 1))
            envelope = torch.zeros(self.output_size, dtype=torch.float32)
            window_sq = window.square()
            for index in range(frame_count):
                start = index * hop_length
                envelope[start : start + n_fft] += window_sq
            envelope = envelope[self.pad : self.output_size - self.pad]
            self.register_buffer("envelope", envelope.clamp_min(1e-11).view(1, -1))

        def forward(self, x: Any) -> Any:
            predicted = self.out(x).transpose(1, 2)
            magnitude, phase = predicted.chunk(2, dim=1)
            magnitude = torch.exp(magnitude).clamp(max=1e2)
            real = magnitude * torch.cos(phase)
            imaginary = magnitude * torch.sin(phase)
            frames = torch.matmul(real.transpose(1, 2), self.cos_basis)
            frames = frames - torch.matmul(imaginary.transpose(1, 2), self.sin_basis)
            frames = frames.transpose(1, 2) * self.window
            chunks = []
            for index in range(self.frame_count):
                start = index * self.hop_length
                end_pad = self.output_size - start - self.n_fft
                chunks.append(torch.nn.functional.pad(frames[:, :, index], (start, end_pad)))
            waveform = chunks[0]
            for chunk in chunks[1:]:
                waveform = waveform + chunk
            waveform = waveform[:, self.pad : self.output_size - self.pad]
            waveform = waveform / self.envelope
            return waveform.unsqueeze(1)

    return RealIstftHead()


def _audio_decoder_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    torch = _torch()
    converted = {}
    for key, value in state.items():
        if not key.startswith("decoder."):
            continue
        converted_key = key[len("decoder.") :]
        converted[converted_key] = (
            value.float() if torch.is_floating_point(value) else value
        )
    return converted


def _onnx_decoder_config(config: dict[str, Any]) -> dict[str, Any]:
    converted = dict(config)
    converted["_attn_implementation"] = "eager"
    converted["attn_implementation"] = "eager"
    converted["torch_dtype"] = "float32"
    converted["use_cache"] = False
    return converted


def _float_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    torch = _torch()
    converted = {}
    for key, value in state.items():
        converted[key] = value.float() if torch.is_floating_point(value) else value
    return converted


def _required_dict(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Ming config field {key!r} must be an object.")
    return value


def _required_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Ming config field {key!r} must be an integer.")
    return value


def _torch() -> Any:
    import torch

    return torch


if __name__ == "__main__":
    main()
