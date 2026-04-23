from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

_RMS_NORM_EPS = 1e-5
_SWIGLU_ALPHA = 1.702
_SWIGLU_LIMIT = 7.0


@dataclass(frozen=True)
class PrivacyFilterConfig:
    num_hidden_layers: int
    num_experts: int
    experts_per_token: int
    vocab_size: int
    num_labels: int
    hidden_size: int
    intermediate_size: int
    head_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    bidirectional_left_context: int
    bidirectional_right_context: int
    initial_context_length: int
    max_position_embeddings: int
    rope_theta: float
    rope_scaling_factor: float
    rope_ntk_alpha: float
    rope_ntk_beta: float

    @property
    def q_mult(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @staticmethod
    def load(config_path: Path) -> "PrivacyFilterConfig":
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        if raw.get("model_type") != "privacy_filter":
            raise ValueError(
                f"Expected original privacy_filter config, got {raw.get('model_type')!r}",
            )
        return PrivacyFilterConfig(
            num_hidden_layers=int(raw["num_hidden_layers"]),
            num_experts=int(raw["num_experts"]),
            experts_per_token=int(raw["experts_per_token"]),
            vocab_size=int(raw["vocab_size"]),
            num_labels=int(raw["num_labels"]),
            hidden_size=int(raw["hidden_size"]),
            intermediate_size=int(raw["intermediate_size"]),
            head_dim=int(raw["head_dim"]),
            num_attention_heads=int(raw["num_attention_heads"]),
            num_key_value_heads=int(raw["num_key_value_heads"]),
            bidirectional_left_context=int(raw["bidirectional_left_context"]),
            bidirectional_right_context=int(raw["bidirectional_right_context"]),
            initial_context_length=int(raw["initial_context_length"]),
            max_position_embeddings=int(raw["max_position_embeddings"]),
            rope_theta=float(raw["rope_theta"]),
            rope_scaling_factor=float(raw["rope_scaling_factor"]),
            rope_ntk_alpha=float(raw["rope_ntk_alpha"]),
            rope_ntk_beta=float(raw["rope_ntk_beta"]),
        )


@dataclass(frozen=True)
class PrivacyFilterPaths:
    root_dir: Path
    model_dir: Path
    tokenizer_dir: Path


@dataclass(frozen=True)
class BlockWeights:
    attn_norm: mx.array
    qkv_weight: mx.array
    qkv_bias: mx.array
    attn_out_weight: mx.array
    attn_out_bias: mx.array
    attn_sinks: mx.array
    mlp_norm: mx.array
    gate_weight: mx.array
    gate_bias: mx.array
    swiglu_weight: mx.array
    swiglu_bias: mx.array
    mlp_out_weight: mx.array
    mlp_out_bias: mx.array


def resolve_privacy_filter_paths(checkpoint: str | Path) -> PrivacyFilterPaths:
    root = Path(checkpoint).expanduser().resolve()
    if (root / "original" / "config.json").exists() and (
        root / "original" / "model.safetensors"
    ).exists():
        return PrivacyFilterPaths(
            root_dir=root,
            model_dir=root / "original",
            tokenizer_dir=root,
        )
    if (root / "config.json").exists() and (root / "model.safetensors").exists():
        tokenizer_dir = root if (root / "tokenizer.json").exists() else root.parent
        return PrivacyFilterPaths(
            root_dir=root if tokenizer_dir == root else root.parent,
            model_dir=root,
            tokenizer_dir=tokenizer_dir,
        )
    raise FileNotFoundError(
        "Could not find privacy-filter checkpoint. Expected either a snapshot root "
        "with original/config.json or the original/ directory itself.",
    )


def rms_norm(x: mx.array, scale: mx.array, eps: float = _RMS_NORM_EPS) -> mx.array:
    return mx.fast.rms_norm(x, scale.astype(mx.float32), eps).astype(x.dtype)


def linear(x: mx.array, weight: mx.array, bias: mx.array | None = None) -> mx.array:
    weight_t = weight.astype(x.dtype).T
    out = mx.matmul(x, weight_t)
    if bias is not None:
        out = out + bias.astype(out.dtype)
    return out


def swiglu(x: mx.array) -> mx.array:
    gate, value = mx.split(x, 2, axis=-1)
    gate = mx.minimum(gate, _SWIGLU_LIMIT)
    value = mx.clip(value, -_SWIGLU_LIMIT, _SWIGLU_LIMIT)
    return (gate * mx.sigmoid(_SWIGLU_ALPHA * gate)) * (value + 1.0)


def apply_rotary(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return mx.reshape(mx.stack([o1, o2], axis=-1), x.shape)


def build_rope_cache(config: PrivacyFilterConfig) -> tuple[mx.array, mx.array]:
    d_half = config.head_dim // 2
    freq = config.rope_theta ** (
        np.arange(0, config.head_dim, 2, dtype=np.float32) / config.head_dim
    )
    if config.rope_scaling_factor > 1.0:
        concentration = 0.1 * math.log(config.rope_scaling_factor) + 1.0
        low = (
            d_half
            * math.log(
                config.initial_context_length
                / (config.rope_ntk_beta * 2.0 * math.pi),
            )
            / math.log(config.rope_theta)
        )
        high = (
            d_half
            * math.log(
                config.initial_context_length
                / (config.rope_ntk_alpha * 2.0 * math.pi),
            )
            / math.log(config.rope_theta)
        )
        ramp = (np.arange(d_half, dtype=np.float32) - low) / (high - low)
        mask = 1.0 - np.clip(ramp, 0.0, 1.0)
        interpolation = 1.0 / (config.rope_scaling_factor * freq)
        extrapolation = 1.0 / freq
        inv_freq = interpolation * (1.0 - mask) + extrapolation * mask
    else:
        concentration = 1.0
        inv_freq = 1.0 / freq
    positions = np.arange(config.max_position_embeddings, dtype=np.float32)
    freqs = np.einsum("i,j->ij", positions, inv_freq)
    cos = (np.cos(freqs) * concentration).astype(np.float32)
    sin = (np.sin(freqs) * concentration).astype(np.float32)
    return mx.array(cos), mx.array(sin)


class PrivacyFilterMlxModel:
    def __init__(
        self,
        *,
        config: PrivacyFilterConfig,
        embedding: mx.array,
        final_norm: mx.array,
        classifier: mx.array,
        blocks: list[BlockWeights],
        moe_chunk_size: int = 256,
        use_fast_attention: bool = False,
        moe_matmul_dtype: str = "bfloat16",
    ) -> None:
        self.config = config
        self.embedding = embedding
        self.final_norm = final_norm
        self.classifier = classifier
        self.blocks = blocks
        self.moe_chunk_size = moe_chunk_size
        self.use_fast_attention = use_fast_attention
        self.moe_matmul_dtype = moe_matmul_dtype
        self.rope_cos, self.rope_sin = build_rope_cache(config)
        self._attn_scale = 1.0 / math.sqrt(math.sqrt(config.head_dim))
        self._sink_log2_scale = math.log(2.0)

    def __call__(self, input_ids: mx.array) -> mx.array:
        if input_ids.ndim != 2:
            raise ValueError("Expected input_ids with shape [batch, tokens].")
        seq_len = int(input_ids.shape[1])
        if seq_len > self.rope_cos.shape[0]:
            raise ValueError(
                f"Sequence length {seq_len} exceeds RoPE cache {self.rope_cos.shape[0]}",
            )
        hidden = mx.take(self.embedding, input_ids, axis=0)
        band_mask = self._band_mask(seq_len)
        for block in self.blocks:
            hidden = self._attention(block, hidden, band_mask)
            hidden = self._mlp(block, hidden)
        hidden = rms_norm(hidden, self.final_norm)
        return linear(hidden, self.classifier).astype(mx.float32)

    def _attention(
        self,
        block: BlockWeights,
        x: mx.array,
        band_mask: mx.array,
    ) -> mx.array:
        batch_size, seq_len, _ = x.shape
        q_dim = self.config.num_attention_heads * self.config.head_dim
        kv_dim = self.config.num_key_value_heads * self.config.head_dim

        hidden = rms_norm(x, block.attn_norm)
        qkv = linear(hidden, block.qkv_weight, block.qkv_bias)
        q = qkv[:, :, :q_dim].reshape(
            (batch_size, seq_len, self.config.num_attention_heads, self.config.head_dim),
        )
        k = qkv[:, :, q_dim : q_dim + kv_dim].reshape(
            (batch_size, seq_len, self.config.num_key_value_heads, self.config.head_dim),
        )
        v = qkv[:, :, q_dim + kv_dim :].reshape(
            (batch_size, seq_len, self.config.num_key_value_heads, self.config.head_dim),
        )

        cos = self.rope_cos[:seq_len].reshape((1, seq_len, 1, self.config.head_dim // 2))
        sin = self.rope_sin[:seq_len].reshape((1, seq_len, 1, self.config.head_dim // 2))
        q = apply_rotary(q, cos, sin) * self._attn_scale
        k = apply_rotary(k, cos, sin) * self._attn_scale

        if self.use_fast_attention:
            attn_out = mx.fast.scaled_dot_product_attention(
                q.transpose(0, 2, 1, 3),
                k.transpose(0, 2, 1, 3),
                v.transpose(0, 2, 1, 3),
                scale=1.0,
                mask=band_mask,
                sinks=block.attn_sinks.astype(mx.float32) * self._sink_log2_scale,
            )
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(
                (
                    batch_size,
                    seq_len,
                    self.config.num_attention_heads * self.config.head_dim,
                ),
            )
        else:
            window = (
                self.config.bidirectional_left_context
                + self.config.bidirectional_right_context
                + 1
            )
            if seq_len > window:
                attn_out = self._attention_windowed(block, q, k, v, seq_len)
            else:
                attn_out = self._attention_reference(block, q, k, v, seq_len, band_mask)
        proj = linear(attn_out, block.attn_out_weight, block.attn_out_bias)
        return x + proj.astype(x.dtype)

    def _attention_reference(
        self,
        block: BlockWeights,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        seq_len: int,
        band_mask: mx.array,
    ) -> mx.array:
        batch_size = int(q.shape[0])
        q = q.reshape(
            (
                batch_size,
                seq_len,
                self.config.num_key_value_heads,
                self.config.q_mult,
                self.config.head_dim,
            ),
        )
        k = mx.broadcast_to(
            k[:, :, :, None, :],
            (
                batch_size,
                seq_len,
                self.config.num_key_value_heads,
                self.config.q_mult,
                self.config.head_dim,
            ),
        )
        v = mx.broadcast_to(
            v[:, :, :, None, :],
            (
                batch_size,
                seq_len,
                self.config.num_key_value_heads,
                self.config.q_mult,
                self.config.head_dim,
            ),
        )

        scores = mx.einsum("bthqd,bshqd->bthqs", q, k).astype(mx.float32)
        mask_bias = mx.where(band_mask, 0.0, -1e9).astype(mx.float32)
        scores = scores + mask_bias.reshape((1, seq_len, 1, 1, seq_len))

        sinks = (
            block.attn_sinks.reshape(
                (self.config.num_key_value_heads, self.config.q_mult),
            ).astype(mx.float32)
            * self._sink_log2_scale
        )
        sink_scores = mx.broadcast_to(
            sinks.reshape(
                (1, 1, self.config.num_key_value_heads, self.config.q_mult, 1),
            ),
            (
                batch_size,
                seq_len,
                self.config.num_key_value_heads,
                self.config.q_mult,
                1,
            ),
        )
        scores = mx.concatenate([scores, sink_scores], axis=-1)
        attn = mx.softmax(scores, axis=-1)[..., :-1].astype(v.dtype)
        attn_out = mx.einsum("bthqs,bshqd->bthqd", attn, v)
        return attn_out.reshape(
            (batch_size, seq_len, self.config.num_attention_heads * self.config.head_dim),
        )

    def _attention_windowed(
        self,
        block: BlockWeights,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        seq_len: int,
    ) -> mx.array:
        batch_size = int(q.shape[0])
        left_context = self.config.bidirectional_left_context
        right_context = self.config.bidirectional_right_context
        window = left_context + right_context + 1
        q = q.reshape(
            (
                batch_size,
                seq_len,
                self.config.num_key_value_heads,
                self.config.q_mult,
                self.config.head_dim,
            ),
        )
        positions = mx.arange(seq_len, dtype=mx.int32)
        offsets = mx.arange(window, dtype=mx.int32) - left_context
        raw_indices = positions[:, None] + offsets[None, :]
        valid = (raw_indices >= 0) & (raw_indices < seq_len)
        indices = mx.clip(raw_indices, 0, seq_len - 1)
        k_window = mx.take(k, indices, axis=1)
        v_window = mx.take(v, indices, axis=1)

        scores = mx.einsum("bthqd,btwhd->bthqw", q, k_window).astype(mx.float32)
        mask_bias = mx.where(valid, 0.0, -1e9).astype(mx.float32)
        scores = scores + mask_bias.reshape((1, seq_len, 1, 1, window))

        sinks = (
            block.attn_sinks.reshape(
                (self.config.num_key_value_heads, self.config.q_mult),
            ).astype(mx.float32)
            * self._sink_log2_scale
        )
        sink_scores = mx.broadcast_to(
            sinks.reshape(
                (1, 1, self.config.num_key_value_heads, self.config.q_mult, 1),
            ),
            (
                batch_size,
                seq_len,
                self.config.num_key_value_heads,
                self.config.q_mult,
                1,
            ),
        )
        scores = mx.concatenate([scores, sink_scores], axis=-1)
        attn = mx.softmax(scores, axis=-1)[..., :-1].astype(v_window.dtype)
        attn_out = mx.einsum("bthqw,btwhd->bthqd", attn, v_window)
        return attn_out.reshape(
            (batch_size, seq_len, self.config.num_attention_heads * self.config.head_dim),
        )

    def _mlp(self, block: BlockWeights, x: mx.array) -> mx.array:
        batch_size, seq_len, hidden_size = x.shape
        hidden = rms_norm(x, block.mlp_norm)
        flat = hidden.reshape((batch_size * seq_len, hidden_size))

        router_logits = linear(
            flat.astype(mx.float32),
            block.gate_weight.astype(mx.float32),
            block.gate_bias.astype(mx.float32),
        )
        top_idx = mx.argpartition(
            router_logits,
            -self.config.experts_per_token,
            axis=-1,
        )[:, -self.config.experts_per_token :]
        top_scores = mx.take_along_axis(router_logits, top_idx, axis=-1)
        top_order = mx.argsort(top_scores, axis=-1)
        top_idx = mx.take_along_axis(top_idx, top_order, axis=-1)
        top_scores = mx.take_along_axis(top_scores, top_order, axis=-1)
        expert_weights = mx.softmax(top_scores, axis=-1).astype(flat.dtype)
        expert_weights = expert_weights / float(self.config.experts_per_token)
        matmul_dtype = mx.bfloat16 if self.moe_matmul_dtype == "bfloat16" else mx.float32

        chunks: list[mx.array] = []
        total = int(flat.shape[0])
        for start in range(0, total, self.moe_chunk_size):
            end = min(start + self.moe_chunk_size, total)
            token_chunk = flat[start:end]
            idx_chunk = top_idx[start:end]
            weights_chunk = expert_weights[start:end]
            chunk_tokens = end - start
            token_indices = mx.broadcast_to(
                mx.arange(chunk_tokens, dtype=mx.int32)[:, None],
                (chunk_tokens, self.config.experts_per_token),
            ).reshape(-1)
            expert_indices = idx_chunk.reshape(-1)

            up_bias = mx.take(block.swiglu_bias, idx_chunk, axis=0).astype(matmul_dtype)
            up = mx.gather_mm(
                token_chunk.astype(matmul_dtype).reshape((chunk_tokens, 1, hidden_size)),
                block.swiglu_weight,
                token_indices,
                expert_indices,
            ).reshape((chunk_tokens, self.config.experts_per_token, -1))
            up = up + up_bias
            up = swiglu(up)

            down_bias = mx.take(block.mlp_out_bias, idx_chunk, axis=0).astype(matmul_dtype)
            expert_count = chunk_tokens * self.config.experts_per_token
            down = mx.gather_mm(
                up.astype(matmul_dtype).reshape(
                    (expert_count, 1, self.config.intermediate_size),
                ),
                block.mlp_out_weight,
                mx.arange(expert_count, dtype=mx.int32),
                expert_indices,
            ).reshape((chunk_tokens, self.config.experts_per_token, hidden_size))
            down = down + down_bias
            mixed = mx.sum(down * weights_chunk[:, :, None], axis=1)
            chunks.append(mixed * float(self.config.experts_per_token))

        out = mx.concatenate(chunks, axis=0).reshape((batch_size, seq_len, hidden_size))
        return x + out.astype(x.dtype)

    def _band_mask(self, seq_len: int) -> mx.array:
        positions = mx.arange(seq_len, dtype=mx.int32)
        row = positions.reshape((seq_len, 1))
        col = positions.reshape((1, seq_len))
        return (col >= (row - self.config.bidirectional_left_context)) & (
            col <= (row + self.config.bidirectional_right_context)
        )


def load_privacy_filter_model(
    checkpoint: str | Path,
    *,
    moe_chunk_size: int = 256,
    use_fast_attention: bool = False,
    moe_matmul_dtype: str = "bfloat16",
) -> tuple[PrivacyFilterPaths, PrivacyFilterConfig, PrivacyFilterMlxModel]:
    paths = resolve_privacy_filter_paths(checkpoint)
    config = PrivacyFilterConfig.load(paths.model_dir / "config.json")
    weights = mx.load(str(paths.model_dir / "model.safetensors"))
    blocks = [
        BlockWeights(
            attn_norm=weights[f"block.{i}.attn.norm.scale"],
            qkv_weight=weights[f"block.{i}.attn.qkv.weight"],
            qkv_bias=weights[f"block.{i}.attn.qkv.bias"],
            attn_out_weight=weights[f"block.{i}.attn.out.weight"],
            attn_out_bias=weights[f"block.{i}.attn.out.bias"],
            attn_sinks=weights[f"block.{i}.attn.sinks"],
            mlp_norm=weights[f"block.{i}.mlp.norm.scale"],
            gate_weight=weights[f"block.{i}.mlp.gate.weight"],
            gate_bias=weights[f"block.{i}.mlp.gate.bias"],
            swiglu_weight=weights[f"block.{i}.mlp.swiglu.weight"],
            swiglu_bias=weights[f"block.{i}.mlp.swiglu.bias"],
            mlp_out_weight=weights[f"block.{i}.mlp.out.weight"],
            mlp_out_bias=weights[f"block.{i}.mlp.out.bias"],
        )
        for i in range(config.num_hidden_layers)
    ]
    model = PrivacyFilterMlxModel(
        config=config,
        embedding=weights["embedding.weight"],
        final_norm=weights["norm.scale"],
        classifier=weights["unembedding.weight"],
        blocks=blocks,
        moe_chunk_size=moe_chunk_size,
        use_fast_attention=use_fast_attention,
        moe_matmul_dtype=moe_matmul_dtype,
    )
    return paths, config, model
