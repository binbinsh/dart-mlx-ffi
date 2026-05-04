"""End-to-end conversion orchestrator.

Run as ``python -m benchmark.runtime.converters.paddleocr_vl_coreml.pipeline``
or via the legacy ``paddleocr_vl_coreml.py`` shim.

Outputs in ``--output-dir``:
  vision_embed.mlpackage/      Model A
  prefill_decoder.mlpackage/   Model B (stateful)
  decode_decoder.mlpackage/    Model C (stateful)
  pipeline.json                runtime wiring + bucket metadata
  parity_report.json           PT vs CoreML logit comparison
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .decode_decoder_model import (
    DecodeDecoderWrapper,
    make_trace_example as make_decode_example,
)
from .enumerated_shapes import (
    GRID_BUCKETS,
    MAX_KV_LEN,
    PATCH_SIZE,
    PROMPT_BUCKETS,
    ImageBucket,
    all_image_buckets,
    default_image_bucket,
    default_prompt_len,
    merged_token_buckets,
    patch_count_buckets,
)
from .palettization import palettize_embed, quantize_decoder
from .parity import ParityReport, compare_logits, write_report
from .prefill_decoder_model import (
    PrefillDecoderWrapper,
    make_trace_example as make_prefill_example,
)
from .vision_embed_model import (
    VisionEmbedWrapper,
    make_trace_example as make_vision_example,
)

IMAGE_PLACEHOLDER = "<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>"


@dataclass
class PipelineConfig:
    hf_snapshot: Path
    output_dir: Path
    image: Path | None
    prompt: str
    skip_quantization: bool
    skip_parity: bool
    parity_tolerance: float
    deployment_target: str  # "iOS18"
    keep_packages: bool


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-snapshot", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--image", type=Path, default=None)
    p.add_argument("--prompt", type=str, default="Convert this image to markdown.")
    p.add_argument("--skip-quantization", action="store_true")
    p.add_argument("--skip-parity", action="store_true")
    p.add_argument("--parity-tolerance", type=float, default=5e-3)
    p.add_argument("--deployment-target", default="iOS18")
    p.add_argument("--keep-packages", action="store_true")
    args = p.parse_args()
    cfg = PipelineConfig(
        hf_snapshot=args.hf_snapshot,
        output_dir=args.output_dir,
        image=args.image,
        prompt=args.prompt,
        skip_quantization=args.skip_quantization,
        skip_parity=args.skip_parity,
        parity_tolerance=args.parity_tolerance,
        deployment_target=args.deployment_target,
        keep_packages=args.keep_packages,
    )
    build_pipeline(cfg)


def build_pipeline(cfg: PipelineConfig) -> dict[str, Any]:
    import coremltools as ct

    _patch_transformers_mask_alias()

    if cfg.output_dir.exists():
        shutil.rmtree(cfg.output_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    target = _resolve_target(ct, cfg.deployment_target)
    precision = _mixed_precision(ct)

    print("[1/4] loading HF model + processor")
    model, processor, image = _load_model_and_image(cfg)

    # Bug V: text-decoder hyperparameters live on `config.text_config` for
    # the LIB PaddleOCRVLConfig (multimodal).
    text_cfg = model.config.text_config
    head_dim = int(text_cfg.head_dim)
    hidden_size = int(text_cfg.hidden_size)
    num_layers = int(text_cfg.num_hidden_layers)

    # --------------------------------------------------------------- #
    print("[2/4] converting Model A (vision + projector → image_embeds)")
    vision_pkg = cfg.output_dir / "vision_embed.mlpackage"
    _convert_vision_embed(
        ct=ct,
        model=model,
        package_path=vision_pkg,
        target=target,
        precision=precision,
        skip_quantization=cfg.skip_quantization,
    )
    _free_torch_mem()

    # --------------------------------------------------------------- #
    print("[3/4] converting Model B (prefill, stateful)")
    prefill_pkg = cfg.output_dir / "prefill_decoder.mlpackage"
    _convert_prefill(
        ct=ct,
        model=model,
        package_path=prefill_pkg,
        target=target,
        precision=precision,
        head_dim=head_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        skip_quantization=cfg.skip_quantization,
    )
    _free_torch_mem()

    # --------------------------------------------------------------- #
    print("[4/4] converting Model C (decode, stateful)")
    decode_pkg = cfg.output_dir / "decode_decoder.mlpackage"
    _convert_decode(
        ct=ct,
        model=model,
        package_path=decode_pkg,
        target=target,
        precision=precision,
        head_dim=head_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        skip_quantization=cfg.skip_quantization,
    )
    _free_torch_mem()

    # --------------------------------------------------------------- #
    pipeline_json = _pipeline_spec(
        vision_pkg=vision_pkg,
        prefill_pkg=prefill_pkg,
        decode_pkg=decode_pkg,
        head_dim=head_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        deployment_target=cfg.deployment_target,
    )
    (cfg.output_dir / "pipeline.json").write_text(
        json.dumps(pipeline_json, indent=2) + "\n"
    )

    parity: ParityReport | None = None
    if not cfg.skip_parity:
        print("[parity] (deferred — see parity.py; reference path TBD)")
        # Parity check requires loading the saved mlpackages and replaying the
        # full prompt. Implemented as a follow-up — the converter exits cleanly
        # here so Phase 1 review can proceed.
    return {
        "vision_embed": str(vision_pkg),
        "prefill_decoder": str(prefill_pkg),
        "decode_decoder": str(decode_pkg),
        "pipeline_json": str(cfg.output_dir / "pipeline.json"),
        "parity": parity.to_dict() if parity else None,
    }


# -------------------------------------------------------------------- #
# Per-model converters
# -------------------------------------------------------------------- #
def _convert_vision_embed(
    *,
    ct: Any,
    model: torch.nn.Module,
    package_path: Path,
    target: Any,
    precision: Any,
    skip_quantization: bool,
) -> None:
    """Trace one VisionEmbedWrapper per bucket, then assemble EnumeratedShapes.

    Hybrid-OCR contract (issue #1): vision_embed.mlpackage emits the
    projector output ``image_embeds`` directly. Scatter into prompt embeds
    happens host-side. For Phase 1 we ship the smallest bucket only and
    add enumeration in Phase 2 — keeps trace memory bounded.
    """
    bucket = default_image_bucket()
    wrapper = VisionEmbedWrapper(model, bucket).eval()
    ex = make_vision_example(bucket, dtype=torch.float32)

    inputs = [
        ct.TensorType(
            name="pixel_values",
            shape=tuple(ex.pixel_values.shape),
            dtype=np.float16,
        ),
        ct.TensorType(
            name="image_grid_thw",
            shape=tuple(ex.image_grid_thw.shape),
            dtype=np.int32,
        ),
    ]
    outputs = [ct.TensorType(name="image_embeds", dtype=np.float16)]

    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper,
            (ex.pixel_values, ex.image_grid_thw),
            strict=False,
        )
        mlmodel = ct.convert(
            traced,
            source="pytorch",
            inputs=inputs,
            outputs=outputs,
            minimum_deployment_target=target,
            compute_precision=precision,
        )
    if not skip_quantization:
        mlmodel = palettize_embed(mlmodel)
    _save(mlmodel, package_path)


def _convert_prefill(
    *,
    ct: Any,
    model: torch.nn.Module,
    package_path: Path,
    target: Any,
    precision: Any,
    head_dim: int,
    hidden_size: int,
    num_layers: int,
    skip_quantization: bool,
) -> None:
    prompt_len = default_prompt_len()
    wrapper = PrefillDecoderWrapper(model, prompt_len=prompt_len).eval()
    ex = make_prefill_example(
        prompt_len, head_dim=head_dim, hidden_size=hidden_size, dtype=torch.float32,
    )

    inputs = [
        ct.TensorType(name="inputs_embeds", shape=tuple(ex.inputs_embeds.shape), dtype=np.float16),
        ct.TensorType(name="attention_mask", shape=tuple(ex.attention_mask.shape), dtype=np.int32),
        ct.TensorType(name="rope_cos", shape=tuple(ex.rope_cos.shape), dtype=np.float16),
        ct.TensorType(name="rope_sin", shape=tuple(ex.rope_sin.shape), dtype=np.float16),
        ct.TensorType(name="prompt_len_used", shape=tuple(ex.prompt_len_used.shape), dtype=np.int32),
    ]
    outputs = [ct.TensorType(name="logits", dtype=np.float16)]
    states = _kv_state_specs(ct, num_layers, head_dim, model)

    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper,
            (
                ex.inputs_embeds,
                ex.attention_mask,
                ex.rope_cos,
                ex.rope_sin,
                ex.prompt_len_used,
            ),
            strict=False,
        )
        mlmodel = ct.convert(
            traced,
            source="pytorch",
            inputs=inputs,
            outputs=outputs,
            states=states,
            minimum_deployment_target=target,
            compute_precision=precision,
        )
    if not skip_quantization:
        mlmodel = quantize_decoder(mlmodel)
    _save(mlmodel, package_path)


def _convert_decode(
    *,
    ct: Any,
    model: torch.nn.Module,
    package_path: Path,
    target: Any,
    precision: Any,
    head_dim: int,
    hidden_size: int,
    num_layers: int,
    skip_quantization: bool,
) -> None:
    wrapper = DecodeDecoderWrapper(model).eval()
    ex = make_decode_example(head_dim=head_dim, hidden_size=hidden_size, dtype=torch.float32)

    inputs = [
        ct.TensorType(name="inputs_embeds", shape=tuple(ex.inputs_embeds.shape), dtype=np.float16),
        ct.TensorType(name="rope_cos", shape=tuple(ex.rope_cos.shape), dtype=np.float16),
        ct.TensorType(name="rope_sin", shape=tuple(ex.rope_sin.shape), dtype=np.float16),
        ct.TensorType(name="cur_len", shape=tuple(ex.cur_len.shape), dtype=np.int32),
        ct.TensorType(name="kv_len", shape=tuple(ex.kv_len.shape), dtype=np.int32),
    ]
    outputs = [ct.TensorType(name="logits", dtype=np.float16)]
    states = _kv_state_specs(ct, num_layers, head_dim, model)

    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper,
            (ex.inputs_embeds, ex.rope_cos, ex.rope_sin, ex.cur_len, ex.kv_len),
            strict=False,
        )
        mlmodel = ct.convert(
            traced,
            source="pytorch",
            inputs=inputs,
            outputs=outputs,
            states=states,
            minimum_deployment_target=target,
            compute_precision=precision,
        )
    if not skip_quantization:
        mlmodel = quantize_decoder(mlmodel)
    _save(mlmodel, package_path)


def _kv_state_specs(ct: Any, num_layers: int, head_dim: int, model: torch.nn.Module) -> list:
    num_kv = int(model.config.text_config.num_key_value_heads)
    states = []
    for i in range(num_layers):
        for tag in ("k_cache", "v_cache"):
            states.append(
                ct.StateType(
                    wrapped_type=ct.TensorType(
                        shape=(1, num_kv, MAX_KV_LEN, head_dim),
                        dtype=np.float16,
                    ),
                    name=f"{tag}_{i}",
                )
            )
    return states


# -------------------------------------------------------------------- #
# Pipeline metadata
# -------------------------------------------------------------------- #
def _pipeline_spec(
    *,
    vision_pkg: Path,
    prefill_pkg: Path,
    decode_pkg: Path,
    head_dim: int,
    hidden_size: int,
    num_layers: int,
    deployment_target: str,
) -> dict[str, Any]:
    return {
        "format": "dart_inference.coreml_pipeline.v2",
        "deployment_target": deployment_target,
        "config": {
            "head_dim": head_dim,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "max_kv_len": MAX_KV_LEN,
            "patch_size": PATCH_SIZE,
        },
        "buckets": {
            "image_grids": [list(g) for g in GRID_BUCKETS],
            "patch_counts": list(patch_count_buckets()),
            "merged_token_counts": list(merged_token_buckets()),
            "prompt_lens": list(PROMPT_BUCKETS),
        },
        "models": {
            "vision_embed": vision_pkg.name,
            "prefill_decoder": prefill_pkg.name,
            "decode_decoder": decode_pkg.name,
        },
        "io": {
            "vision_embed": {
                "inputs": ["pixel_values", "image_grid_thw"],
                "outputs": ["image_embeds"],
            },
            "prefill_decoder": {
                "inputs": [
                    "inputs_embeds",
                    "attention_mask",
                    "rope_cos",
                    "rope_sin",
                    "prompt_len_used",
                ],
                "outputs": ["logits"],
                "states_per_layer": ["k_cache_{i}", "v_cache_{i}"],
            },
            "decode_decoder": {
                "inputs": ["inputs_embeds", "rope_cos", "rope_sin", "cur_len", "kv_len"],
                "outputs": ["logits"],
                "states_per_layer": ["k_cache_{i}", "v_cache_{i}"],
            },
        },
        "phase_1_status": {
            "vision_embed_buckets_enumerated": False,
            "prefill_buckets_enumerated": False,
            "decode_kv_rangedim_enabled": False,
            "stateful_kv_enabled": True,
            "image_tokens_must_be_at_prompt_prefix": True,
            "notes": (
                "Phase 1 ships the default bucket only; bucket enumeration "
                "lands in Phase 2. vision_embed.mlpackage emits image_embeds "
                "(projector output, [num_image_tokens, hidden]); the host "
                "scatters into the prompt embedding buffer at IMAGE_PLACEHOLDER "
                "positions (see paddleOcrVlScatterImageEmbeddings, embed.dart)."
            ),
        },
    }


# -------------------------------------------------------------------- #
# Helpers
# -------------------------------------------------------------------- #
def _load_model_and_image(cfg: PipelineConfig):
    from PIL import Image
    # Bug V fix: load via the in-tree (LIB) PaddleOCRVLForConditionalGeneration
    # rather than AutoModelForCausalLM(trust_remote_code=True). The remote-code
    # ("SHIPPED") variant exports an mlpackage whose vision tower disagrees
    # with the LIB reference at cos≈0.46 even though weights are nominally
    # the same; every downstream consumer (parity.py, e2e_token_golden.py,
    # diag_*) is already on LIB so the converter must match.
    from transformers import AutoProcessor, PaddleOCRVLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(str(cfg.hf_snapshot))
    model = PaddleOCRVLForConditionalGeneration.from_pretrained(
        str(cfg.hf_snapshot),
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="cpu",
    ).eval()
    model.requires_grad_(False)

    if cfg.image is not None and cfg.image.exists():
        image = Image.open(cfg.image).convert("RGB")
    else:
        image = Image.new("RGB", (336, 336), "white")
    return model, processor, image


def _save(mlmodel: Any, path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    mlmodel.save(str(path))
    print(f"   wrote {path}")


def _free_torch_mem() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


# -------------------------------------------------------------------- #
# Mixed precision policy
# -------------------------------------------------------------------- #
# These MIL ops are numerically unsafe in fp16 for transformer decoders:
#   * RMSNorm pipeline: ``pow``/``square`` blows up for hidden_dim≥1024
#     (max abs activation^2 easily exceeds fp16 max 65504), and the
#     subsequent ``reduce_sum``/``reduce_mean`` accumulates the error.
#     ``rsqrt`` of a tiny variance also loses precision in fp16.
#   * Softmax: the max-subtraction trick still leaves large positive
#     exponents that overflow / underflow in fp16, particularly for the
#     decode step where we attend over MAX_KV=4096 columns. Upstream HF
#     models explicitly do ``softmax(..., dtype=torch.float32)``.
#   * ``layer_norm`` is included for safety (vision tower uses it).
#
# Returning False from the op_selector tells coremltools' FP16 pass to
# leave that op (and its surrounding cast pair) in fp32. The model
# weights and the rest of the graph stay in fp16 — package size is
# unchanged from the all-fp16 baseline (verified empirically: only a
# handful of activation tensors get a 2x-size cast inserted).
_FP32_SAFE_OPS = frozenset({
    "rsqrt",
    "softmax",
    "reduce_sum",
    "reduce_mean",
    "reduce_l2_norm",
    "pow",
    "square",
    "layer_norm",
    "rms_norm",
})


def _mixed_precision(ct: Any) -> Any:
    def _op_selector(op: Any) -> bool:
        # True  → cast this op to fp16
        # False → keep this op (and its inputs/outputs) in fp32
        return op.op_type not in _FP32_SAFE_OPS

    return ct.transform.FP16ComputePrecision(op_selector=_op_selector)


def _resolve_target(ct: Any, value: str) -> Any:
    v = value.lower()
    if v in {"ios18", "macos15"}:
        return ct.target.iOS18
    if v in {"ios17", "macos14"}:
        # Stateful conversion REQUIRES iOS18; fall through with a warning.
        print("[warn] stateful KV cache requires iOS18 — using iOS18 instead of", value)
        return ct.target.iOS18
    raise ValueError(f"unknown deployment target: {value}")


def _patch_transformers_mask_alias() -> None:
    """Compat shims for transformers 5.x against PaddleOCR-VL's bundled modeling.

    1. ``masking_utils.create_causal_mask`` argument rename
       ``inputs_embeds`` → ``input_embeds``.
    2. ``transformers.utils.generic.check_model_inputs`` was removed; the
       PaddleOCR-VL modeling file uses it as a no-op decorator on
       ``forward`` — inject an identity decorator so the import succeeds.
    """
    # (1) mask alias
    try:
        from transformers import masking_utils  # type: ignore

        fn = getattr(masking_utils, "create_causal_mask", None)
        if fn is not None and not getattr(fn, "_dinf_alias", False):
            def alias(*args, **kwargs):
                if "inputs_embeds" in kwargs and "input_embeds" not in kwargs:
                    kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
                return fn(*args, **kwargs)

            alias._dinf_alias = True  # type: ignore[attr-defined]
            masking_utils.create_causal_mask = alias
    except ImportError:
        pass

    # (2) check_model_inputs stub
    try:
        from transformers.utils import generic as _generic  # type: ignore

        if not hasattr(_generic, "check_model_inputs"):
            def check_model_inputs(fn=None, **_kw):
                # Supports both ``@check_model_inputs`` and
                # ``@check_model_inputs(...)`` usages.
                if callable(fn):
                    return fn
                def deco(inner):
                    return inner
                return deco

            _generic.check_model_inputs = check_model_inputs  # type: ignore[attr-defined]
    except ImportError:
        pass

    # (3) ROPE_INIT_FUNCTIONS["default"] was removed in transformers 5.x.
    #     Re-implement the historical no-scaling default: standard
    #     base^(2i/d) inv-freq computation. The bundled PaddleOCR-VL
    #     modeling file declares ``rope_type="default"``.
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS  # type: ignore

        def _default_rope_init(config, device=None, seq_len=None, layer_type=None):
            base = getattr(config, "rope_theta", None)
            if base is None:
                rp = getattr(config, "rope_parameters", {}) or {}
                if isinstance(rp, dict):
                    if layer_type is not None and layer_type in rp:
                        base = rp[layer_type].get("rope_theta")
                    else:
                        base = rp.get("rope_theta")
            base = float(base or 10000.0)
            head_dim = getattr(config, "head_dim", None) or (
                config.hidden_size // config.num_attention_heads
            )
            partial = 1.0
            if hasattr(config, "rope_parameters"):
                rp = config.rope_parameters or {}
                if isinstance(rp, dict):
                    rp_eff = (
                        rp[layer_type]
                        if layer_type is not None and layer_type in rp
                        else rp
                    )
                    partial = (
                        rp_eff.get("partial_rotary_factor", 1.0)
                        if isinstance(rp_eff, dict)
                        else 1.0
                    )
            dim = int(head_dim * partial)
            inv_freq = 1.0 / (
                base
                ** (
                    torch.arange(0, dim, 2, dtype=torch.int64).to(
                        device=device, dtype=torch.float
                    )
                    / dim
                )
            )
            return inv_freq, 1.0

        if "default" not in ROPE_INIT_FUNCTIONS:
            ROPE_INIT_FUNCTIONS["default"] = _default_rope_init

        # The 5.x weight initialiser checks
        # ``module.compute_default_rope_parameters`` for "default" rope_type
        # — short-circuit by patching ``_init_weights``.
        from transformers import modeling_utils as _mu  # type: ignore
        from torch import nn as _nn

        if not getattr(_mu.PreTrainedModel._init_weights, "_dinf_rope_patched", False):
            orig_init = _mu.PreTrainedModel._init_weights

            def _patched_init_weights(self, module):
                cls_name = module.__class__.__name__
                if "RotaryEmbedding" in cls_name and hasattr(module, "original_inv_freq"):
                    rope_type = getattr(module, "rope_type", "default")
                    fn = ROPE_INIT_FUNCTIONS.get(rope_type) or ROPE_INIT_FUNCTIONS.get("default")
                    inv_freq, _ = fn(module.config)
                    with torch.no_grad():
                        module.inv_freq.copy_(inv_freq)
                        module.original_inv_freq.copy_(inv_freq)
                    return
                return orig_init(self, module)

            _patched_init_weights._dinf_rope_patched = True  # type: ignore[attr-defined]
            _mu.PreTrainedModel._init_weights = _patched_init_weights
    except ImportError:
        pass


if __name__ == "__main__":
    main()
