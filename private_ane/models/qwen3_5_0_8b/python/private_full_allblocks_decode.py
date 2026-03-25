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
    from .private_linear_runtime import (
        PrivateLinearLayerRuntime,
    )
    from .private_sdpa_runtime import PrivateSdpaRuntime, pack_sdpa_input
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
    from private_linear_runtime import (
        PrivateLinearLayerRuntime,
    )
    from private_sdpa_runtime import PrivateSdpaRuntime, pack_sdpa_input
    from common import add_vendor_to_path, cleanup_mlx, resolve_model_path

add_vendor_to_path("mlx-vlm")

import mlx.core as mx
from mlx_vlm.models.qwen3_5.language import apply_multimodal_rotary_pos_emb
from mlx_vlm.models.base import create_attention_mask, create_ssm_mask
from mlx_vlm.tokenizer_utils import load_tokenizer
from mlx_vlm.utils import get_model_path, load_model


ROOT = Path(__file__).resolve().parents[4]
MODEL_ID = "mlx-community/Qwen3.5-0.8B-4bit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--token-limit", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--attn-layers")
    parser.add_argument("--q-scale", type=float, default=1.0)
    parser.add_argument("--lane", type=int, default=32)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def baseline_decode(model, token_ids: list[int], max_new_tokens: int) -> tuple[list[int], float]:
    cache = model.language_model.make_cache()
    started = time.perf_counter()
    prompt = mx.array([token_ids], dtype=mx.int32)
    out = model(prompt, cache=cache).logits[:, -1, :].astype(mx.float32)
    mx.eval(out)
    mx.synchronize()
    tokens = list(token_ids)
    next_token = int(np.argmax(to_numpy_f32(out)[0]))
    tokens.append(next_token)
    for _ in range(max_new_tokens - 1):
        step = mx.array([[next_token]], dtype=mx.int32)
        out = model(step, cache=cache).logits[:, -1, :].astype(mx.float32)
        mx.eval(out)
        mx.synchronize()
        next_token = int(np.argmax(to_numpy_f32(out)[0]))
        tokens.append(next_token)
    return tokens, (time.perf_counter() - started) * 1000.0


def build_linear_runtimes(model, lane: int):
    runtimes = {}
    for index, layer in enumerate(model.language_model.model.layers):
        if layer.is_linear:
            runtimes[index] = PrivateLinearLayerRuntime.build(layer.linear_attn, lane=lane)
    return runtimes


def close_linear_runtimes(runtimes) -> None:
    for runtime in runtimes.values():
        runtime.close()

def _run_private_full_attn(layer_index, x_norm, sa, position_ids, cache, servers, attn_post_runtimes, q_scale: float):
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
    values = values.reshape(bsz, seq_len, sa.num_key_value_heads, -1).transpose(0, 2, 1, 3)
    cos, sin = sa.rotary_emb(values, position_ids)
    queries, keys = apply_multimodal_rotary_pos_emb(queries, keys, cos, sin)
    keys, values = cache.update_and_fetch(keys, values)
    key_len = int(keys.shape[2])
    keys = repeat_kv(keys, sa.num_attention_heads, sa.num_key_value_heads, key_len, sa.head_dim)
    values = repeat_kv(
        values,
        sa.num_attention_heads,
        sa.num_key_value_heads,
        key_len,
        sa.head_dim,
    )
    q_np = to_numpy_f16(queries)
    k_np = to_numpy_f16(keys)
    v_np = to_numpy_f16(values)
    if key_len == seq_len:
        ctx_all = run_sdpa_prefixes(
            servers,
            q_np=q_np,
            k_np=k_np,
            v_np=v_np,
            scale=sa.scale,
            q_scale_extra=q_scale,
        )
    else:
        ctx_slices = []
        for prefix in range(1, seq_len + 1):
            q_pos = prefix - 1
            key_prefix = key_len - seq_len + prefix
            qq = (
                q_np[:, :, q_pos : q_pos + 1, :].astype(np.float32, copy=False)
                * np.float32(sa.scale)
                * np.float32(q_scale)
            ).astype(np.float16)
            kk = k_np[:, :, :key_prefix, :]
            vv = v_np[:, :, :key_prefix, :]
            if seq_len == 1 and q_pos == 0:
                try:
                    ctx_bytes = servers.run_packed(key_prefix, pack_sdpa_input(qq, kk, vv))
                except Exception:
                    ctx_bytes = servers.run(key_prefix, qq, kk, vv)
            else:
                ctx_bytes = servers.run(key_prefix, qq, kk, vv)
            ctx = np.frombuffer(ctx_bytes, dtype=np.float16).astype(np.float32)
            ctx_slices.append(ctx.reshape(1, sa.num_attention_heads, 1, sa.head_dim))
        ctx_all = np.concatenate(ctx_slices, axis=2)
    if layer_index in attn_post_runtimes:
        return attn_post_runtimes[layer_index].run(
            ctx_np=ctx_all.astype(np.float32, copy=False),
            gate_np=to_numpy_f32(gate),
        )
    ctx_mx = mx.array(ctx_all.astype(np.float16), dtype=mx.float16)
    return sa.o_proj(
        ctx_mx.transpose(0, 2, 1, 3).reshape(bsz, seq_len, -1) * mx.sigmoid(gate)
    )


def allblocks_decode(model, token_ids: list[int], max_new_tokens: int, *, attn_layers, q_scale, lane, servers, attn_post_runtimes, ffn_models, linear_runtimes):
    lm = model.language_model
    base = lm.model
    caches = lm.make_cache()
    started = time.perf_counter()

    def run_step(step_ids: list[int]) -> np.ndarray:
        tokens = mx.array([step_ids], dtype=mx.int32)
        h = base.embed_tokens(tokens)
        if len(step_ids) > 1:
            position_ids, _ = lm.get_rope_index(tokens)
        else:
            offset = caches[base.fa_idx].offset
            pos = mx.arange(offset, offset + 1).reshape(1, 1)
            position_ids = mx.broadcast_to(pos, (3, 1, 1))
        fa_mask = create_attention_mask(h, caches[base.fa_idx])
        ssm_mask = create_ssm_mask(h, caches[base.ssm_idx])

        for layer_index, layer in enumerate(base.layers):
            x_norm = layer.input_layernorm(h)
            if layer.is_linear:
                r = linear_runtimes[layer_index].run(
                    x_norm,
                    caches[layer_index],
                    ssm_mask,
                )
            else:
                if layer_index in attn_layers:
                    r = _run_private_full_attn(
                        layer_index,
                        x_norm,
                        layer.self_attn,
                        position_ids,
                        caches[layer_index],
                        servers,
                        attn_post_runtimes,
                        q_scale,
                    )
                else:
                    r = layer.self_attn(x_norm, fa_mask, caches[layer_index], position_ids)
            h = h + r
            norm2 = layer.post_attention_layernorm(h)
            mlp_out = ffn_models[layer_index].run(norm2)
            h = h + mlp_out.astype(h.dtype)

        out = base.norm(h)
        out = base.embed_tokens.as_linear(out)
        out = out[:, -1, :].astype(mx.float32)
        mx.eval(out)
        mx.synchronize()
        return to_numpy_f32(out)[0]

    tokens = list(token_ids)
    logits = run_step(tokens)
    next_token = int(np.argmax(logits))
    tokens.append(next_token)
    for _ in range(max_new_tokens - 1):
        logits = run_step([next_token])
        next_token = int(np.argmax(logits))
        tokens.append(next_token)
    return tokens, (time.perf_counter() - started) * 1000.0


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(MODEL_ID, get_model_path)
    model = load_model(model_path, lazy=True)
    tokenizer = load_tokenizer(model_path)
    token_ids = tokenizer.encode(args.prompt)[: args.token_limit]
    attn_layers = parse_attn_layers(args.attn_layers, model)

    if not ane_ffi_enabled():
        raise RuntimeError("Private ANE FFI bridge is unavailable.")

    work_dir = Path(tempfile.mkdtemp(prefix="qwen35_private_allblocks_decode_"))
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
        for key_len in range(2, max_seq_len + 1):
            sdpa_runtime.prepare_packed(key_len)
        attn_post_runtimes = build_attn_post_runtimes(
            model,
            lane=max_seq_len,
            attn_layers=attn_layers,
        )
        ffn_models = build_ffn_runtimes(artifacts_dir)
        linear_runtimes = build_linear_runtimes(model, args.lane)
        try:
            baseline_tokens, baseline_ms = baseline_decode(model, token_ids, args.max_new_tokens)
            all_tokens, all_ms = allblocks_decode(
                model,
                token_ids,
                args.max_new_tokens,
                attn_layers=attn_layers,
                q_scale=args.q_scale,
                lane=args.lane,
                servers=sdpa_runtime,
                attn_post_runtimes=attn_post_runtimes,
                ffn_models=ffn_models,
                linear_runtimes=linear_runtimes,
            )
        finally:
            close_attn_post_runtimes(attn_post_runtimes)
            sdpa_runtime.close()
            close_ffn_runtimes(ffn_models)
            close_linear_runtimes(linear_runtimes)

        report = {
            "runtime": "qwen35_private_full_allblocks_decode",
            "model_id": MODEL_ID,
            "snapshot_path": str(model_path.resolve()),
            "prompt": args.prompt,
            "prompt_token_ids": token_ids,
            "baseline_token_ids": baseline_tokens,
            "allblocks_token_ids": all_tokens,
            "baseline_text": tokenizer.decode(baseline_tokens, skip_special_tokens=False),
            "allblocks_text": tokenizer.decode(all_tokens, skip_special_tokens=False),
            "baseline_new_text": tokenizer.decode(
                baseline_tokens[len(token_ids) :], skip_special_tokens=False
            ),
            "allblocks_new_text": tokenizer.decode(
                all_tokens[len(token_ids) :], skip_special_tokens=False
            ),
            "token_match": baseline_tokens == all_tokens,
            "attn_layers": sorted(attn_layers),
            "lane": args.lane,
            "baseline_generate_ms": baseline_ms,
            "allblocks_generate_ms": all_ms,
            "allblocks_speedup_vs_baseline": baseline_ms / all_ms if all_ms else None,
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
