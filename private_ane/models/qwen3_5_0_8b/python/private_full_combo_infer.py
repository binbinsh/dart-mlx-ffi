from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

try:
    from .private_attn_post_runtime import build_attn_post_runtimes, close_attn_post_runtimes
    from .private_full_attn_infer import (
        DEFAULT_PROMPT,
        parse_attn_layers,
        repeat_kv,
        run_sdpa_prefixes,
        to_numpy_f16,
        to_numpy_f32,
    )
    from .private_ane_ctypes import is_enabled as ane_ffi_enabled
    from .private_ffn_runtime import build_ffn_runtimes, close_ffn_runtimes
    from .private_sdpa_runtime import PrivateSdpaRuntime
    from ..common import add_vendor_to_path, cleanup_mlx, resolve_model_path
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from private_attn_post_runtime import build_attn_post_runtimes, close_attn_post_runtimes
    from private_full_attn_infer import (
        DEFAULT_PROMPT,
        parse_attn_layers,
        repeat_kv,
        run_sdpa_prefixes,
        to_numpy_f16,
        to_numpy_f32,
    )
    from private_ane_ctypes import is_enabled as ane_ffi_enabled
    from private_ffn_runtime import build_ffn_runtimes, close_ffn_runtimes
    from private_sdpa_runtime import PrivateSdpaRuntime
    from common import add_vendor_to_path, cleanup_mlx, resolve_model_path

add_vendor_to_path("mlx-vlm")

import mlx.core as mx

from mlx_vlm.models.qwen3_5.language import apply_multimodal_rotary_pos_emb
from mlx_vlm.models.base import create_attention_mask, create_ssm_mask
from mlx_vlm.tokenizer_utils import load_tokenizer
from mlx_vlm.utils import get_model_path, load_model

ROOT = Path(__file__).resolve().parents[4]
MODEL_ID = "mlx-community/Qwen3.5-0.8B-4bit"
_PACK_F32_CACHE: dict[tuple[int, int], np.ndarray] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--token-limit", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--attn-layers")
    parser.add_argument("--q-scale", type=float, default=1.0)
    parser.add_argument("--lane", type=int, default=32)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def pack_f32(values: np.ndarray, *, dim: int, lane: int, seq_len: int) -> np.ndarray:
    key = (dim, lane)
    packed = _PACK_F32_CACHE.get(key)
    if packed is None:
        packed = np.zeros((dim, lane), dtype=np.float32)
        _PACK_F32_CACHE[key] = packed
    else:
        packed.fill(0.0)
    src = values.reshape(seq_len, dim)
    packed[:, :seq_len] = src.T
    return packed


def unpack_f32(blob: bytes, *, dim: int, lane: int, seq_len: int) -> np.ndarray:
    packed = np.frombuffer(blob, dtype=np.float32).reshape(dim, lane)
    return packed[:, :seq_len].T.reshape(1, seq_len, dim)


def baseline_next_logits(model, token_ids: list[int]) -> np.ndarray:
    tokens = mx.array([token_ids], dtype=mx.int32)
    out = model(tokens).logits[:, -1, :].astype(mx.float32)
    mx.eval(out)
    mx.synchronize()
    return to_numpy_f32(out)[0]


def generate_greedy(fn, token_ids: list[int], max_new_tokens: int, eos_token_id: int | None):
    tokens = list(token_ids)
    started = time.perf_counter()
    for _ in range(max_new_tokens):
        logits = fn(tokens)
        next_token = int(np.argmax(logits))
        tokens.append(next_token)
        if eos_token_id is not None and next_token == eos_token_id:
            break
    total_ms = (time.perf_counter() - started) * 1000.0
    return tokens, total_ms


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(MODEL_ID, get_model_path)
    model = load_model(model_path, lazy=True)
    tokenizer = load_tokenizer(model_path)
    token_ids = tokenizer.encode(args.prompt)[: args.token_limit]
    attn_layers = parse_attn_layers(args.attn_layers, model)

    work_dir = Path(tempfile.mkdtemp(prefix="qwen35_private_combo_"))
    try:
        artifacts_dir = work_dir / "artifacts"
        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "private_ane" / "models" / "qwen3_5_0_8b" / "export" / "make_private_ffn.py"),
                "--snapshot-dir",
                str(model_path),
                "--out-dir",
                str(artifacts_dir),
                "--lane",
                str(args.lane),
            ]
        )

        max_seq_len = len(token_ids) + args.max_new_tokens
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
        ffn_models = {}
        try:
            if not ane_ffi_enabled():
                raise RuntimeError("Private ANE FFI bridge is unavailable.")
            ffn_models = build_ffn_runtimes(artifacts_dir)

            def combo_next_logits(ids: list[int]) -> np.ndarray:
                lm = model.language_model
                base = lm.model
                tokens = mx.array([ids], dtype=mx.int32)
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
                        x_norm = layer.input_layernorm(h)
                        attn_out = layer.self_attn(x_norm, fa_mask, None, position_ids)
                    else:
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
                        values = values.reshape(
                            bsz, seq_len, sa.num_key_value_heads, -1
                        ).transpose(0, 2, 1, 3)
                        cos, sin = sa.rotary_emb(values, position_ids)
                        queries, keys = apply_multimodal_rotary_pos_emb(
                            queries, keys, cos, sin
                        )
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
                            q_scale_extra=args.q_scale,
                        )
                        if layer_index in attn_post_runtimes:
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

                    norm2 = layer.post_attention_layernorm(h)
                    mlp_out = ffn_models[layer_index].run(norm2)
                    h = h + mlp_out.astype(h.dtype)

                out = base.norm(h)
                out = base.embed_tokens.as_linear(out)
                out = out[:, -1, :].astype(mx.float32)
                mx.eval(out)
                mx.synchronize()
                return to_numpy_f32(out)[0]

            baseline_tokens, baseline_ms = generate_greedy(
                lambda ids: baseline_next_logits(model, ids),
                token_ids,
                args.max_new_tokens,
                tokenizer.eos_token_id,
            )
            combo_tokens, combo_ms = generate_greedy(
                combo_next_logits,
                token_ids,
                args.max_new_tokens,
                tokenizer.eos_token_id,
            )
        finally:
            close_attn_post_runtimes(attn_post_runtimes)
            sdpa_runtime.close()
            close_ffn_runtimes(ffn_models)

        report = {
            "runtime": "qwen35_private_full_combo_hybrid",
            "model_id": MODEL_ID,
            "snapshot_path": str(model_path.resolve()),
            "prompt": args.prompt,
            "prompt_token_ids": token_ids,
            "baseline_token_ids": baseline_tokens,
            "combo_token_ids": combo_tokens,
            "baseline_text": tokenizer.decode(baseline_tokens, skip_special_tokens=False),
            "combo_text": tokenizer.decode(combo_tokens, skip_special_tokens=False),
            "baseline_new_text": tokenizer.decode(
                baseline_tokens[len(token_ids) :], skip_special_tokens=False
            ),
            "combo_new_text": tokenizer.decode(
                combo_tokens[len(token_ids) :], skip_special_tokens=False
            ),
            "token_match": baseline_tokens == combo_tokens,
            "attn_layers": sorted(attn_layers),
            "lane": args.lane,
            "baseline_generate_ms": baseline_ms,
            "combo_generate_ms": combo_ms,
            "combo_speedup_vs_baseline": baseline_ms / combo_ms if combo_ms else None,
        }
        if args.json:
            print(json.dumps(report))
            return
        print(json.dumps(report, indent=2, ensure_ascii=False))
    finally:
        cleanup_mlx(mx)
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
