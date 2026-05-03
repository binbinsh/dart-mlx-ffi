"""Quantization configs centralised so all 3 sub-converters agree.

Decoder linears: INT4 grouped (block_size=64) per ADR §6.1.
Embed table:    INT8 LUT palettize per ADR §6.2.
Vision tower:   FP16 only (no quant) per ADR §6.3.

These functions return ``coremltools`` op-config objects ready to feed
``ct.optimize.coreml.linear_quantize_weights`` /
``palettize_weights``.  Keeping the values here means a single edit changes
prefill and decode together.
"""

from __future__ import annotations

import coremltools as ct
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig,
    OpPalettizerConfig,
    OptimizationConfig,
)


# Layers we never quantise even when the global decoder pass runs. The lm_head
# is the most error-amplifying linear in the network; norms are tiny anyway.
DECODER_QUANT_SKIP_PATTERNS: tuple[str, ...] = (
    "lm_head",
    "norm",
    "embed_tokens",
)


def decoder_int4_config() -> OptimizationConfig:
    """INT4 grouped weight quant for ALL decoder ``Linear`` ops.

    block_size=64 chosen to align with head_dim (128) and MLP intermediate
    (3072) — both divisible — giving zero padding and even group counts.

    NOTE: coremltools-9 renamed the grouped-channel API. The old
    ``granularity="per_grouped_channel"`` + ``group_size=...`` pair was
    replaced by ``granularity="per_block"`` + ``block_size=...`` with
    semantically equivalent behaviour (block_size=64 here matches what was
    previously group_size=64).
    """
    op_cfg = OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype="int4",
        granularity="per_block",
        block_size=64,
        weight_threshold=2048,
    )
    return OptimizationConfig(
        global_config=op_cfg,
        op_name_configs={pat: None for pat in DECODER_QUANT_SKIP_PATTERNS},
    )


def embed_int8_lut_config() -> OptimizationConfig:
    """8-bit LUT palettize for the token-embedding table ONLY.

    Vocab × hidden ≈ 103k × 1024 ≈ 105.9M weights — biggest single tensor in
    the network.  Per-tensor (granularity=per_tensor) keeps lookup cheap.

    ADR §6.3 compliance — vision tower MUST stay FP16 (no quant). The
    previous version used ``global_config=op_cfg`` with
    ``weight_threshold=2048``, which is a *catch-all*: it palettized every
    weight ≥ 2048 elems, i.e. all 162 vision linears + patch_embedding conv
    + projector + embed_tokens. That violated the ADR and dropped vision
    cosine to 0.622 worst row.

    The fix flips polarity: ``global_config=None`` palettizes nothing by
    default; ``op_name_configs`` opts IN exactly the embed table by its
    MIL output name (``embed_tokens_weight``, confirmed via spec inspect
    on 2026-05-03).
    """
    op_cfg = OpPalettizerConfig(
        mode="kmeans",
        nbits=8,
        granularity="per_tensor",
        # No weight_threshold here: when a config is bound to a specific op
        # name we want it to apply unconditionally to that op.
    )
    return OptimizationConfig(
        global_config=None,
        op_name_configs={"embed_tokens_weight": op_cfg},
    )


def quantize_decoder(mlmodel: ct.models.MLModel) -> ct.models.MLModel:
    from coremltools.optimize.coreml import linear_quantize_weights

    return linear_quantize_weights(mlmodel, config=decoder_int4_config())


def palettize_embed(mlmodel: ct.models.MLModel) -> ct.models.MLModel:
    from coremltools.optimize.coreml import palettize_weights

    return palettize_weights(mlmodel, config=embed_int8_lut_config())
