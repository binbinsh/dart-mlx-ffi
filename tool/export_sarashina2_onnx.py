#!/usr/bin/env python3
"""Export sarashina2.2-tts source weights for the Dart/FFI ONNX runtime."""

from __future__ import annotations

import argparse
import logging
import math
import shutil
import sys
import types
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from fuse_sarashina2_flow_step import write_fused_flow_step
from fuse_sarashina2_decode_head import write_fused_decode_head
from fuse_sarashina2_flow_loop import write_fused_flow_loop
from optimize_sarashina2_prefill import write_prefill_last_hidden


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory containing sbintuitions/sarashina2.2-tts files.",
    )
    parser.add_argument(
        "--sarashina-src",
        type=Path,
        help="Path to the cloned sbintuitions/sarashina2.2-tts source tree.",
    )
    parser.add_argument(
        "--cosyvoice2-model-dir",
        type=Path,
        help="Optional CosyVoice2 model dir used as speech_tokenizer_v2.onnx source.",
    )
    parser.add_argument("--device", default="cpu", help="Export device for flow/vocoder modules.")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--skip-flow", action="store_true")
    parser.add_argument("--skip-hift", action="store_true")
    parser.add_argument("--skip-campplus", action="store_true")
    parser.add_argument("--skip-speech-tokenizer", action="store_true")
    parser.add_argument(
        "--llm-dtype",
        choices=("fp32", "fp16", "bf16"),
        default="fp32",
        help="LLM ONNX compute precision. Non-fp32 exports keep fp32 I/O and use a precision suffix.",
    )
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def add_sarashina_src(src: Path | None) -> None:
    if src is None:
        return
    root = src.expanduser().resolve()
    if not (root / "sarashina_tts").is_dir():
        raise FileNotFoundError(f"{root} does not contain sarashina_tts/")
    sys.path.insert(0, str(root))


class PrefillWrapper(torch.nn.Module):
    def __init__(self, causal_lm: torch.nn.Module, compute_dtype: torch.dtype):
        super().__init__()
        self.model = causal_lm.model
        self.compute_dtype = compute_dtype

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
        outs = self.model(
            inputs_embeds=inputs_embeds.to(dtype=self.compute_dtype),
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=True,
            return_dict=True,
        )
        pkv = outs.past_key_values
        if hasattr(pkv, "to_legacy_cache"):
            pkv = pkv.to_legacy_cache()
        flat_kv: List[torch.Tensor] = []
        for layer_kv in pkv:
            flat_kv.append(layer_kv[0].float())
            flat_kv.append(layer_kv[1].float())
        return (outs.last_hidden_state.float(), *flat_kv)


class DecodeWrapper(torch.nn.Module):
    def __init__(self, causal_lm: torch.nn.Module, num_layers: int, compute_dtype: torch.dtype):
        super().__init__()
        self.model = causal_lm.model
        self.num_layers = num_layers
        self.compute_dtype = compute_dtype

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, *past_kv_flat: torch.Tensor):
        from transformers.cache_utils import DynamicCache

        if len(past_kv_flat) != 2 * self.num_layers:
            raise RuntimeError(f"expected {2 * self.num_layers} KV tensors")
        legacy_kv: Tuple[Tuple[torch.Tensor, torch.Tensor], ...] = tuple(
            (
                past_kv_flat[2 * i].to(dtype=self.compute_dtype),
                past_kv_flat[2 * i + 1].to(dtype=self.compute_dtype),
            )
            for i in range(self.num_layers)
        )
        if hasattr(DynamicCache, "from_legacy_cache"):
            past_key_values = DynamicCache.from_legacy_cache(legacy_kv)
        else:
            past_key_values = DynamicCache(legacy_kv)
        outs = self.model(
            inputs_embeds=inputs_embeds.to(dtype=self.compute_dtype),
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )
        pkv = outs.past_key_values
        if hasattr(pkv, "to_legacy_cache"):
            pkv = pkv.to_legacy_cache()
        flat_kv: List[torch.Tensor] = []
        for layer_kv in pkv:
            flat_kv.append(layer_kv[0].float())
            flat_kv.append(layer_kv[1].float())
        return (outs.last_hidden_state.float(), *flat_kv)


class DecoderHead(torch.nn.Module):
    def __init__(self, lm_head: torch.nn.Module, compute_dtype: torch.dtype):
        super().__init__()
        self.proj = lm_head
        self.compute_dtype = compute_dtype

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden.to(dtype=self.compute_dtype)).float()


def torch_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def llm_onnx_path(model_dir: Path, stem: str, dtype_name: str) -> Path:
    suffix = "" if dtype_name == "fp32" else f".{dtype_name}"
    return model_dir / f"{stem}{suffix}.onnx"


def load_causal_lm(model_dir: Path, dtype: torch.dtype, device: torch.device) -> torch.nn.Module:
    from transformers import AutoModelForCausalLM

    kwargs = {
        "torch_dtype": dtype,
        "local_files_only": True,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            attn_implementation="eager",
            **kwargs,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(str(model_dir), **kwargs)
    return model.to(device=device, dtype=dtype).eval()


@torch.no_grad()
def export_llm(model_dir: Path, dtype_name: str, device: torch.device) -> None:
    require_file(model_dir / "model.safetensors")
    dtype = torch_dtype(dtype_name)
    logging.info("loading LLM from %s dtype=%s device=%s", model_dir, dtype_name, device)
    model = load_causal_lm(model_dir, dtype, device)
    cfg = model.config
    hidden_size = int(cfg.hidden_size)
    num_layers = int(cfg.num_hidden_layers)
    num_heads = int(cfg.num_attention_heads)
    num_kv_heads = int(getattr(cfg, "num_key_value_heads", num_heads))
    head_dim = hidden_size // num_heads

    if dtype_name == "fp32" or not (model_dir / "llm_embeddings.npz").is_file():
        token_embedding = model.model.embed_tokens.weight.detach().float().cpu().numpy().astype(np.float32)
        np.savez(model_dir / "llm_embeddings.npz", token_embedding=token_embedding)
        logging.info("wrote llm_embeddings.npz token_embedding=%s", token_embedding.shape)

    hidden = torch.randn(1, 1, hidden_size, dtype=torch.float32, device=device)
    torch.onnx.export(
        DecoderHead(model.lm_head, dtype).eval(),
        (hidden,),
        llm_onnx_path(model_dir, "llm_decoder_head", dtype_name),
        input_names=["hidden"],
        output_names=["logits"],
        dynamic_axes={"hidden": {0: "batch", 1: "seq"}, "logits": {0: "batch", 1: "seq"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    logging.info("wrote %s", llm_onnx_path(model_dir, "llm_decoder_head", dtype_name))

    seq = 8
    inputs_embeds = torch.randn(1, seq, hidden_size, dtype=torch.float32, device=device)
    attention_mask = torch.ones(1, seq, dtype=torch.int64, device=device)
    output_names = ["hidden"]
    dynamic_axes = {
        "inputs_embeds": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "hidden": {0: "batch", 1: "seq"},
    }
    for i in range(num_layers):
        output_names.extend([f"present_key_{i}", f"present_value_{i}"])
        dynamic_axes[f"present_key_{i}"] = {0: "batch", 2: "seq"}
        dynamic_axes[f"present_value_{i}"] = {0: "batch", 2: "seq"}
    torch.onnx.export(
        PrefillWrapper(model, dtype).eval(),
        (inputs_embeds, attention_mask),
        llm_onnx_path(model_dir, "llm_prefill", dtype_name),
        input_names=["inputs_embeds", "attention_mask"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        external_data=True,
        dynamo=False,
    )
    logging.info("wrote %s", llm_onnx_path(model_dir, "llm_prefill", dtype_name))
    write_prefill_last_hidden(
        llm_onnx_path(model_dir, "llm_prefill", dtype_name),
        llm_onnx_path(model_dir, "llm_prefill_last", dtype_name),
        overwrite=True,
    )

    past_seq = 8
    step_embed = torch.randn(1, 1, hidden_size, dtype=torch.float32, device=device)
    step_mask = torch.ones(1, past_seq + 1, dtype=torch.int64, device=device)
    past_kv_flat: List[torch.Tensor] = []
    for _ in range(num_layers):
        past_kv_flat.append(torch.zeros(1, num_kv_heads, past_seq, head_dim, dtype=torch.float32, device=device))
        past_kv_flat.append(torch.zeros(1, num_kv_heads, past_seq, head_dim, dtype=torch.float32, device=device))
    input_names = ["inputs_embeds", "attention_mask"]
    output_names = ["hidden"]
    dynamic_axes = {
        "inputs_embeds": {0: "batch"},
        "attention_mask": {0: "batch", 1: "total_seq"},
        "hidden": {0: "batch"},
    }
    for i in range(num_layers):
        input_names.extend([f"past_key_{i}", f"past_value_{i}"])
        output_names.extend([f"present_key_{i}", f"present_value_{i}"])
        dynamic_axes[f"past_key_{i}"] = {0: "batch", 2: "past_seq"}
        dynamic_axes[f"past_value_{i}"] = {0: "batch", 2: "past_seq"}
        dynamic_axes[f"present_key_{i}"] = {0: "batch", 2: "total_seq"}
        dynamic_axes[f"present_value_{i}"] = {0: "batch", 2: "total_seq"}
    torch.onnx.export(
        DecodeWrapper(model, num_layers, dtype).eval(),
        (step_embed, step_mask, *past_kv_flat),
        llm_onnx_path(model_dir, "llm_decode", dtype_name),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        external_data=True,
        dynamo=False,
    )
    logging.info("wrote %s", llm_onnx_path(model_dir, "llm_decode", dtype_name))
    write_fused_decode_head(
        llm_onnx_path(model_dir, "llm_decode", dtype_name),
        llm_onnx_path(model_dir, "llm_decoder_head", dtype_name),
        llm_onnx_path(model_dir, "llm_decode_head", dtype_name),
        overwrite=True,
    )


def load_flow(model_dir: Path, device: torch.device) -> torch.nn.Module:
    from sarashina_tts.flow_matching.flow import CausalMaskedDiffWithXvec

    flow = CausalMaskedDiffWithXvec()
    flow.load_state_dict(torch.load(model_dir / "flow.pt", map_location="cpu", weights_only=True), strict=True)
    return flow.to(device).eval()


@torch.no_grad()
def export_flow(model_dir: Path, device: torch.device) -> None:
    require_file(model_dir / "flow.pt")
    flow = load_flow(model_dir, device)

    xs = torch.rand(1, 64, 512, dtype=torch.float32, device=device)
    xs_lens = torch.full((1,), 64, dtype=torch.int32, device=device)
    torch.onnx.export(
        flow.encoder,
        (xs, xs_lens),
        model_dir / "flow.encoder.fp32.onnx",
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
        input_names=["xs", "xs_lens"],
        output_names=["encoder_out", "encoder_mask"],
        dynamic_axes={
            "xs": {0: "batch", 1: "seq_len"},
            "xs_lens": {0: "batch"},
            "encoder_out": {0: "batch", 1: "out_seq_len"},
            "encoder_mask": {0: "batch", 2: "out_seq_len"},
        },
    )
    logging.info("wrote flow.encoder.fp32.onnx")

    seq_len = 256
    x = torch.rand(2, 80, seq_len, dtype=torch.float32, device=device)
    mask = torch.ones(2, 1, seq_len, dtype=torch.float32, device=device)
    mu = torch.rand(2, 80, seq_len, dtype=torch.float32, device=device)
    t = torch.rand(2, dtype=torch.float32, device=device)
    spks = torch.rand(2, 80, dtype=torch.float32, device=device)
    cond = torch.rand(2, 80, seq_len, dtype=torch.float32, device=device)
    torch.onnx.export(
        flow.decoder.estimator,
        (x, mask, mu, t, spks, cond),
        model_dir / "flow.decoder.estimator.fp32.onnx",
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
        input_names=["x", "mask", "mu", "t", "spks", "cond"],
        output_names=["estimator_out"],
        dynamic_axes={
            "x": {2: "seq_len"},
            "mask": {2: "seq_len"},
            "mu": {2: "seq_len"},
            "cond": {2: "seq_len"},
            "estimator_out": {2: "seq_len"},
        },
    )
    logging.info("wrote flow.decoder.estimator.fp32.onnx")
    write_fused_flow_step(
        model_dir / "flow.decoder.estimator.fp32.onnx",
        model_dir / "flow.decoder.step.fp32.onnx",
        overwrite=True,
    )
    write_fused_flow_loop(
        model_dir / "flow.decoder.step.fp32.onnx",
        model_dir / "flow.decoder.loop.fp32.onnx",
        overwrite=True,
    )


def patch_hift_stft(hift: torch.nn.Module, device: torch.device) -> None:
    n_fft = int(hift.istft_params["n_fft"])
    hop_len = int(hift.istft_params["hop_len"])
    stft_window = hift.stft_window.detach().to(device)
    win_sq = stft_window.pow(2).detach().to(device)
    n_freq = n_fft // 2 + 1
    k_idx = torch.arange(n_fft, dtype=torch.float32, device=device)
    f_idx = torch.arange(n_freq, dtype=torch.float32, device=device)
    angle = 2.0 * math.pi * k_idx.unsqueeze(1) * f_idx.unsqueeze(0) / float(n_fft)
    cos_basis = torch.cos(angle)
    sin_basis = torch.sin(angle)
    cos_inv = torch.cos(angle).t().contiguous()
    sin_inv = torch.sin(angle).t().contiguous()
    weights = torch.full((n_freq,), 2.0, dtype=torch.float32, device=device)
    weights[0] = 1.0
    if n_fft % 2 == 0:
        weights[-1] = 1.0
    w_cos_inv = (cos_inv * weights.unsqueeze(1)) / float(n_fft)
    w_sin_inv = (sin_inv * weights.unsqueeze(1)) / float(n_fft)
    overlap_weight = torch.eye(n_fft, dtype=torch.float32, device=device).view(n_fft, 1, n_fft)

    def exportable_stft(self, x):
        pad_amount = n_fft // 2
        x_padded = torch.nn.functional.pad(x.unsqueeze(1), (pad_amount, pad_amount), mode="reflect")
        frames = torch.nn.functional.unfold(x_padded.unsqueeze(2), kernel_size=(1, n_fft), stride=(1, hop_len))
        frames = frames.transpose(1, 2) * stft_window.view(1, 1, -1)
        real = torch.matmul(frames, cos_basis).transpose(1, 2)
        imag = -torch.matmul(frames, sin_basis).transpose(1, 2)
        return real, imag

    def exportable_istft(self, magnitude, phase):
        magnitude = torch.clip(magnitude, max=1e2)
        real = (magnitude * torch.cos(phase)).transpose(1, 2).contiguous()
        imag = (magnitude * torch.sin(phase)).transpose(1, 2).contiguous()
        time_frames = torch.matmul(real, w_cos_inv) - torch.matmul(imag, w_sin_inv)
        time_frames = time_frames * stft_window.view(1, 1, -1)
        time_frames_t = time_frames.transpose(1, 2).contiguous()
        audio = torch.nn.functional.conv_transpose1d(time_frames_t, overlap_weight, bias=None, stride=hop_len).squeeze(1)
        t_frames = time_frames.shape[1]
        win_sq_frames = win_sq.view(1, n_fft, 1).expand(1, n_fft, t_frames).contiguous()
        win_sum = torch.nn.functional.conv_transpose1d(
            win_sq_frames, overlap_weight, bias=None, stride=hop_len
        ).squeeze(1).squeeze(0).clamp(min=1e-11)
        return audio / win_sum.unsqueeze(0)

    hift._stft = types.MethodType(exportable_stft, hift)
    hift._istft = types.MethodType(exportable_istft, hift)


@torch.no_grad()
def export_hift(model_dir: Path, device: torch.device) -> None:
    from sarashina_tts.flow_matching.hifigan import HiFTGenerator

    require_file(model_dir / "hift.pt")
    hift = HiFTGenerator()
    state = {
        key.replace("generator.", ""): value
        for key, value in torch.load(model_dir / "hift.pt", map_location="cpu", weights_only=True).items()
    }
    hift.load_state_dict(state, strict=True)
    hift.remove_weight_norm()
    hift.to(device).eval()
    patch_hift_stft(hift, device)

    class HiftWrapper(torch.nn.Module):
        def __init__(self, module: torch.nn.Module):
            super().__init__()
            self.hift = module

        def forward(self, speech_feat: torch.Tensor):
            audio, source = self.hift(speech_feat=speech_feat)
            return audio, source

    speech_feat = torch.rand(1, 80, 64, dtype=torch.float32, device=device)
    torch.onnx.export(
        HiftWrapper(hift).eval(),
        (speech_feat,),
        model_dir / "hift.onnx",
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
        input_names=["speech_feat"],
        output_names=["audio", "source_cache"],
        dynamic_axes={
            "speech_feat": {0: "batch", 2: "t_mel"},
            "audio": {0: "batch", 1: "t_audio"},
            "source_cache": {0: "batch", 2: "t_source"},
        },
    )
    logging.info("wrote hift.onnx")


@torch.no_grad()
def export_campplus(model_dir: Path, device: torch.device) -> None:
    from sarashina_tts.speech_encoder.campplus import CAMPPlus

    require_file(model_dir / "campplus_cn_common.bin")
    model = CAMPPlus()
    model.load_state_dict(torch.load(model_dir / "campplus_cn_common.bin", map_location="cpu", weights_only=False))
    model.to(device).eval()
    inputs = torch.rand(1, 200, 80, dtype=torch.float32, device=device)
    torch.onnx.export(
        model,
        (inputs,),
        model_dir / "campplus.onnx",
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch", 1: "frames"}, "output": {0: "batch"}},
    )
    logging.info("wrote campplus.onnx")


def copy_speech_tokenizer(model_dir: Path, cosyvoice2_model_dir: Path | None) -> None:
    if (model_dir / "speech_tokenizer_v2.onnx").is_file():
        logging.info("speech_tokenizer_v2.onnx already exists")
        return
    candidates = []
    if cosyvoice2_model_dir is not None:
        candidates.append(cosyvoice2_model_dir / "speech_tokenizer_v2.onnx")
    for candidate in candidates:
        if candidate.is_file():
            shutil.copy2(candidate, model_dir / "speech_tokenizer_v2.onnx")
            logging.info("copied speech_tokenizer_v2.onnx from %s", candidate)
            return
    raise FileNotFoundError(
        "speech_tokenizer_v2.onnx is not in the Sarashina model dir. "
        "Pass --cosyvoice2-model-dir pointing at a CosyVoice2 snapshot with speech_tokenizer_v2.onnx."
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    model_dir = args.model_dir.expanduser().resolve()
    add_sarashina_src(args.sarashina_src)
    device = torch.device(args.device)

    if not args.skip_speech_tokenizer:
        copy_speech_tokenizer(
            model_dir,
            args.cosyvoice2_model_dir.expanduser().resolve() if args.cosyvoice2_model_dir else None,
        )
    if not args.skip_campplus:
        export_campplus(model_dir, device)
    if not args.skip_flow:
        export_flow(model_dir, device)
    if not args.skip_hift:
        export_hift(model_dir, device)
    if not args.skip_llm:
        export_llm(model_dir, args.llm_dtype, device)


if __name__ == "__main__":
    main()
