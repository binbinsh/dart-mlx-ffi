from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    from .private_attn_post_runtime import build_attn_post_runtimes, close_attn_post_runtimes
    from .private_sdpa_runtime import PrivateSdpaRuntime
    from ..common import add_vendor_to_path, cleanup_mlx, resolve_model_path
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from private_attn_post_runtime import build_attn_post_runtimes, close_attn_post_runtimes
    from private_sdpa_runtime import PrivateSdpaRuntime
    from common import add_vendor_to_path, cleanup_mlx, resolve_model_path

add_vendor_to_path("mlx-vlm")

import mlx.core as mx

from mlx_vlm.models.qwen3_5.language import apply_multimodal_rotary_pos_emb
from mlx_vlm.models.base import create_attention_mask, create_ssm_mask
from mlx_vlm.tokenizer_utils import load_tokenizer
from mlx_vlm.utils import get_model_path, load_model


DEFAULT_PROMPT = (
    "Explain why MLX on Apple Silicon is useful for local inference, "
    "and mention latency, memory efficiency, and developer ergonomics."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="qwen35_0p8b", choices=["qwen35_0p8b"])
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--token-limit", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--attn-layers")
    parser.add_argument("--q-scale", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def repeat_kv(tensor, num_heads: int, num_kv_heads: int, seq_len: int, head_dim: int):
    repeat = num_heads // num_kv_heads
    return mx.broadcast_to(
        mx.expand_dims(tensor, axis=2),
        (1, num_kv_heads, repeat, seq_len, head_dim),
    ).reshape(1, num_heads, seq_len, head_dim)


def to_numpy(array, *, dtype=None) -> np.ndarray:
    cast = array if dtype is None else array.astype(dtype)
    mx.eval(cast)
    mx.synchronize()
    return np.asarray(cast)


def to_numpy_f16(array) -> np.ndarray:
    return to_numpy(array, dtype=mx.float16).astype(np.float16, copy=False)


def to_numpy_f32(array) -> np.ndarray:
    return to_numpy(array, dtype=mx.float32).astype(np.float32, copy=False)


def parse_attn_layers(raw: str | None, model) -> set[int]:
    available = {
        index
        for index, layer in enumerate(model.language_model.model.layers)
        if not layer.is_linear
    }
    if raw is None or not raw.strip():
        return available
    if raw.strip().lower() == "all":
        return available
    layers = {int(item.strip()) for item in raw.split(",") if item.strip()}
    invalid = sorted(layer for layer in layers if layer not in available)
    if invalid:
        raise SystemExit(f"Invalid --attn-layers for Qwen3.5 full-attn layers: {invalid}")
    return layers


def baseline_next_logits(model, token_ids: list[int]) -> np.ndarray:
    tokens = mx.array([token_ids], dtype=mx.int32)
    out = model(tokens).logits[:, -1, :].astype(mx.float32)
    mx.eval(out)
    mx.synchronize()
    return np.asarray(out).astype(np.float32, copy=False)[0]


def _array_to_numpy_f32(array) -> np.ndarray:
    return to_numpy_f32(array)


def run_sdpa_prefixes(
    sdpa_runtime,
    *,
    q_np: np.ndarray,
    k_np: np.ndarray,
    v_np: np.ndarray,
    scale: float,
    q_scale_extra: float,
) -> np.ndarray:
    seq_len = q_np.shape[2]
    scaled_q = (
        q_np.astype(np.float32, copy=False)
        * np.float32(scale)
        * np.float32(q_scale_extra)
    ).astype(np.float16)
    if seq_len > 1 and k_np.shape[2] == seq_len and v_np.shape[2] == seq_len:
        try:
            blobs = sdpa_runtime.run_prefill(
                seq_len,
                scaled_q,
                k_np,
                v_np,
            )
            ctx = [
                np.frombuffer(blob, dtype=np.float16).reshape(
                    1,
                    q_np.shape[1],
                    1,
                    q_np.shape[3],
                )
                for blob in blobs
            ]
            return np.concatenate(ctx, axis=2).astype(np.float32, copy=False)
        except Exception:
            pass

    ctx_slices = []
    for prefix in range(1, seq_len + 1):
        qq = scaled_q[:, :, prefix - 1 : prefix, :]
        kk = k_np[:, :, :prefix, :]
        vv = v_np[:, :, :prefix, :]
        ctx_bytes = sdpa_runtime.run(prefix, qq, kk, vv)
        ctx = np.frombuffer(ctx_bytes, dtype=np.float16).astype(np.float32)
        ctx_slices.append(ctx.reshape(1, q_np.shape[1], 1, q_np.shape[3]))
    return np.concatenate(ctx_slices, axis=2)


def baseline_forward_trace(model, token_ids: list[int]) -> dict[str, object]:
    lm = model.language_model
    base = lm.model
    tokens = mx.array([token_ids], dtype=mx.int32)
    h = base.embed_tokens(tokens)
    cache = [None] * len(base.layers)
    position_ids, _rope_deltas = lm.get_rope_index(tokens)
    fa_mask = create_attention_mask(h, cache[base.fa_idx])
    ssm_mask = create_ssm_mask(h, cache[base.ssm_idx])
    layers = []

    for layer_index, layer in enumerate(base.layers):
        mask = ssm_mask if layer.is_linear else fa_mask
        h = layer(h, mask, None, position_ids)
        layers.append(
            {
                "layer": layer_index,
                "is_linear": bool(layer.is_linear),
                "hidden": _array_to_numpy_f32(h),
            }
        )

    out = base.norm(h)
    out = base.embed_tokens.as_linear(out)
    out = out[:, -1, :].astype(mx.float32)
    mx.eval(out)
    mx.synchronize()
    return {
        "layers": layers,
        "logits": np.asarray(out).astype(np.float32, copy=False)[0],
    }


def hybrid_next_logits(
    model,
    token_ids: list[int],
    sdpa_runtime,
    *,
    attn_layers: set[int],
    attn_post_runtimes: dict[int, object] | None,
    q_scale_extra: float,
) -> np.ndarray:
    lm = model.language_model
    base = lm.model
    tokens = mx.array([token_ids], dtype=mx.int32)
    h = base.embed_tokens(tokens)
    cache = [None] * len(base.layers)
    position_ids, _rope_deltas = lm.get_rope_index(tokens)
    fa_mask = create_attention_mask(h, cache[base.fa_idx])
    ssm_mask = create_ssm_mask(h, cache[base.ssm_idx])

    for layer_index, layer in enumerate(base.layers):
        if layer.is_linear:
            h = layer(h, ssm_mask, None, position_ids)
            continue

        if layer_index not in attn_layers:
            h = layer(h, fa_mask, None, position_ids)
            continue

        x_norm = layer.input_layernorm(h)
        sa = layer.self_attn
        bsz, seq_len, _dim = x_norm.shape

        q_proj_output = sa.q_proj(x_norm)
        queries, gate = mx.split(
            q_proj_output.reshape(bsz, seq_len, sa.num_attention_heads, -1),
            2,
            axis=-1,
        )
        gate = gate.reshape(bsz, seq_len, -1)
        keys = sa.k_proj(x_norm)
        values = sa.v_proj(x_norm)

        queries = sa.q_norm(queries).transpose(0, 2, 1, 3)
        keys = sa.k_norm(
            keys.reshape(bsz, seq_len, sa.num_key_value_heads, -1)
        ).transpose(0, 2, 1, 3)
        values = values.reshape(bsz, seq_len, sa.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        cos, sin = sa.rotary_emb(values, position_ids)
        queries, keys = apply_multimodal_rotary_pos_emb(queries, keys, cos, sin)

        keys = repeat_kv(
            keys,
            sa.num_attention_heads,
            sa.num_key_value_heads,
            seq_len,
            sa.head_dim,
        )
        values = repeat_kv(
            values,
            sa.num_attention_heads,
            sa.num_key_value_heads,
            seq_len,
            sa.head_dim,
        )

        q_np = to_numpy_f16(queries)
        k_np = to_numpy_f16(keys)
        v_np = to_numpy_f16(values)

        ctx_all = run_sdpa_prefixes(
            sdpa_runtime,
            q_np=q_np,
            k_np=k_np,
            v_np=v_np,
            scale=sa.scale,
            q_scale_extra=q_scale_extra,
        )
        if attn_post_runtimes is not None and layer_index in attn_post_runtimes:
            attn_out = attn_post_runtimes[layer_index].run(
                ctx_np=ctx_all.astype(np.float32, copy=False),
                gate_np=to_numpy_f32(gate),
            )
        else:
            ctx_mx = mx.array(ctx_all.astype(np.float16), dtype=mx.float16)
            attn_out = sa.o_proj(
                ctx_mx.transpose(0, 2, 1, 3).reshape(bsz, seq_len, -1)
                * mx.sigmoid(gate)
            )

        h = h + attn_out
        h = h + layer.mlp(layer.post_attention_layernorm(h))

    out = base.norm(h)
    out = base.embed_tokens.as_linear(out)
    out = out[:, -1, :].astype(mx.float32)
    mx.eval(out)
    mx.synchronize()
    return np.asarray(out).astype(np.float32, copy=False)[0]


def hybrid_forward_trace(
    model,
    token_ids: list[int],
    sdpa_runtime,
    *,
    attn_layers: set[int],
    attn_post_runtimes: dict[int, object] | None,
    q_scale_extra: float,
) -> dict[str, object]:
    lm = model.language_model
    base = lm.model
    tokens = mx.array([token_ids], dtype=mx.int32)
    h = base.embed_tokens(tokens)
    cache = [None] * len(base.layers)
    position_ids, _rope_deltas = lm.get_rope_index(tokens)
    fa_mask = create_attention_mask(h, cache[base.fa_idx])
    ssm_mask = create_ssm_mask(h, cache[base.ssm_idx])
    layers = []

    for layer_index, layer in enumerate(base.layers):
        if layer.is_linear:
            h = layer(h, ssm_mask, None, position_ids)
            layers.append(
                {
                    "layer": layer_index,
                    "is_linear": True,
                    "attn_replaced": False,
                    "hidden": _array_to_numpy_f32(h),
                }
            )
            continue

        if layer_index not in attn_layers:
            h = layer(h, fa_mask, None, position_ids)
            layers.append(
                {
                    "layer": layer_index,
                    "is_linear": False,
                    "attn_replaced": False,
                    "hidden": _array_to_numpy_f32(h),
                }
            )
            continue

        x_norm = layer.input_layernorm(h)
        sa = layer.self_attn
        bsz, seq_len, _dim = x_norm.shape

        q_proj_output = sa.q_proj(x_norm)
        queries, gate = mx.split(
            q_proj_output.reshape(bsz, seq_len, sa.num_attention_heads, -1),
            2,
            axis=-1,
        )
        gate = gate.reshape(bsz, seq_len, -1)
        keys = sa.k_proj(x_norm)
        values = sa.v_proj(x_norm)

        queries = sa.q_norm(queries).transpose(0, 2, 1, 3)
        keys = sa.k_norm(
            keys.reshape(bsz, seq_len, sa.num_key_value_heads, -1)
        ).transpose(0, 2, 1, 3)
        values = values.reshape(bsz, seq_len, sa.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        cos, sin = sa.rotary_emb(values, position_ids)
        queries, keys = apply_multimodal_rotary_pos_emb(queries, keys, cos, sin)

        keys = repeat_kv(
            keys,
            sa.num_attention_heads,
            sa.num_key_value_heads,
            seq_len,
            sa.head_dim,
        )
        values = repeat_kv(
            values,
            sa.num_attention_heads,
            sa.num_key_value_heads,
            seq_len,
            sa.head_dim,
        )

        q_np = to_numpy_f16(queries)
        k_np = to_numpy_f16(keys)
        v_np = to_numpy_f16(values)

        ctx_all = run_sdpa_prefixes(
            sdpa_runtime,
            q_np=q_np,
            k_np=k_np,
            v_np=v_np,
            scale=sa.scale,
            q_scale_extra=q_scale_extra,
        )
        if attn_post_runtimes is not None and layer_index in attn_post_runtimes:
            attn_out = attn_post_runtimes[layer_index].run(
                ctx_np=ctx_all.astype(np.float32, copy=False),
                gate_np=to_numpy_f32(gate),
            )
        else:
            ctx_mx = mx.array(ctx_all.astype(np.float16), dtype=mx.float16)
            attn_out = sa.o_proj(
                ctx_mx.transpose(0, 2, 1, 3).reshape(bsz, seq_len, -1)
                * mx.sigmoid(gate)
            )

        h = h + attn_out
        h = h + layer.mlp(layer.post_attention_layernorm(h))
        layers.append(
            {
                "layer": layer_index,
                "is_linear": False,
                "attn_replaced": True,
                "hidden": _array_to_numpy_f32(h),
            }
        )

    out = base.norm(h)
    out = base.embed_tokens.as_linear(out)
    out = out[:, -1, :].astype(mx.float32)
    mx.eval(out)
    mx.synchronize()
    return {
        "layers": layers,
        "logits": np.asarray(out).astype(np.float32, copy=False)[0],
    }


def generate_greedy(fn, token_ids: list[int], max_new_tokens: int, eos_token_id: int | None):
    tokens = list(token_ids)
    start = time.perf_counter()
    for _ in range(max_new_tokens):
        logits = fn(tokens)
        next_token = int(np.argmax(logits))
        tokens.append(next_token)
        if eos_token_id is not None and next_token == eos_token_id:
            break
    total_ms = (time.perf_counter() - start) * 1000.0
    return tokens, total_ms


def generate_baseline_tokens(model, token_ids: list[int], max_new_tokens: int) -> list[int]:
    tokens = list(token_ids)
    for _ in range(max_new_tokens):
        logits = baseline_next_logits(model, tokens)
        tokens.append(int(np.argmax(logits)))
    return tokens


def main() -> None:
    args = parse_args()
    spec = {
        "name": "qwen35_0p8b",
        "model_id": "mlx-community/Qwen3.5-0.8B-4bit",
    }
    model_path = resolve_model_path(str(spec["model_id"]), get_model_path)
    model = load_model(model_path, lazy=True)
    tokenizer = load_tokenizer(model_path)
    token_ids = tokenizer.encode(args.prompt)[: args.token_limit]
    max_seq_len = len(token_ids) + args.max_new_tokens
    attn_layers = parse_attn_layers(args.attn_layers, model)
    sdpa_runtime = PrivateSdpaRuntime.build(
        max_seq_len=max_seq_len,
        num_heads=model.language_model.model.layers[3].self_attn.num_attention_heads,
        head_dim=model.language_model.model.layers[3].self_attn.head_dim,
    )
    sdpa_runtime.prepare_prefill(len(token_ids))
    attn_post_runtimes = build_attn_post_runtimes(
        model,
        lane=max_seq_len,
        attn_layers=attn_layers,
    )

    try:
        baseline_tokens, baseline_ms = generate_greedy(
            lambda ids: baseline_next_logits(model, ids),
            token_ids,
            args.max_new_tokens,
            tokenizer.eos_token_id,
        )
        hybrid_tokens, hybrid_ms = generate_greedy(
            lambda ids: hybrid_next_logits(
                model,
                ids,
                sdpa_runtime,
                attn_layers=attn_layers,
                attn_post_runtimes=attn_post_runtimes,
                q_scale_extra=args.q_scale,
            ),
            token_ids,
            args.max_new_tokens,
            tokenizer.eos_token_id,
        )
    finally:
        close_attn_post_runtimes(attn_post_runtimes)
        sdpa_runtime.close()
        cleanup_mlx(mx)

    report = {
        "runtime": "qwen35_private_full_attn_hybrid",
        "model_id": spec["model_id"],
        "snapshot_path": str(model_path.resolve()),
        "prompt": args.prompt,
        "prompt_token_ids": token_ids,
        "baseline_token_ids": baseline_tokens,
        "hybrid_token_ids": hybrid_tokens,
        "baseline_text": tokenizer.decode(baseline_tokens, skip_special_tokens=False),
        "hybrid_text": tokenizer.decode(hybrid_tokens, skip_special_tokens=False),
        "baseline_new_text": tokenizer.decode(
            baseline_tokens[len(token_ids) :], skip_special_tokens=False
        ),
        "hybrid_new_text": tokenizer.decode(
            hybrid_tokens[len(token_ids) :], skip_special_tokens=False
        ),
        "token_match": baseline_tokens == hybrid_tokens,
        "attn_layers": sorted(attn_layers),
        "q_scale": args.q_scale,
        "baseline_generate_ms": baseline_ms,
        "hybrid_generate_ms": hybrid_ms,
        "hybrid_speedup_vs_baseline": baseline_ms / hybrid_ms if hybrid_ms else None,
    }

    if args.json:
        print(json.dumps(report))
        return
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
