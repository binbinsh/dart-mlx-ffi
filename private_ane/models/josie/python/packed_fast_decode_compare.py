from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "private_ane" / "models" / "josie" / "python"))

from ane_ops import build_server, close_server, run_server
from fast_decode_compare import (
    fast_incremental_forward,
    load_josie,
    mlx_incremental_forward,
)
from full_model_compare import PROMPT, repeat_kv


def make_packed_sdpa_mil(
    *,
    key_len: int,
    num_heads: int,
    head_dim: int,
) -> str:
    channels = num_heads * head_dim
    total_steps = 1 + (2 * key_len)
    q_shape = [1, num_heads, head_dim, 1]
    kv_shape = [1, num_heads, head_dim, key_len]
    q_begin = [0, 0, 0, 0]
    q_end = [1, channels, 1, 1]
    k_begin = [0, 0, 0, 1]
    k_end = [1, channels, 1, 1 + key_len]
    v_begin = [0, 0, 0, 1 + key_len]
    v_end = [1, channels, 1, total_steps]
    perm = [0, 1, 3, 2]
    return f"""program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}, {{"coremlc-version", "3505.4.1"}}, {{"coremltools-component-milinternal", ""}}, {{"coremltools-version", "9.0"}}}})]
{{
    func main<ios18>(
        tensor<fp16, [1, {channels}, 1, {total_steps}]> packed
    ) {{
        tensor<int32, [4]> q_begin = const()[name = string("q_begin"), val = tensor<int32, [4]>([{q_begin[0]},{q_begin[1]},{q_begin[2]},{q_begin[3]}])];
        tensor<int32, [4]> q_end = const()[name = string("q_end"), val = tensor<int32, [4]>([{q_end[0]},{q_end[1]},{q_end[2]},{q_end[3]}])];
        tensor<int32, [4]> k_begin = const()[name = string("k_begin"), val = tensor<int32, [4]>([{k_begin[0]},{k_begin[1]},{k_begin[2]},{k_begin[3]}])];
        tensor<int32, [4]> k_end = const()[name = string("k_end"), val = tensor<int32, [4]>([{k_end[0]},{k_end[1]},{k_end[2]},{k_end[3]}])];
        tensor<int32, [4]> v_begin = const()[name = string("v_begin"), val = tensor<int32, [4]>([{v_begin[0]},{v_begin[1]},{v_begin[2]},{v_begin[3]}])];
        tensor<int32, [4]> v_end = const()[name = string("v_end"), val = tensor<int32, [4]>([{v_end[0]},{v_end[1]},{v_end[2]},{v_end[3]}])];
        tensor<fp16, [1, {channels}, 1, 1]> q_flat = slice_by_index(begin = q_begin, end = q_end, x = packed)[name = string("q_flat")];
        tensor<fp16, [1, {channels}, 1, {key_len}]> k_flat = slice_by_index(begin = k_begin, end = k_end, x = packed)[name = string("k_flat")];
        tensor<fp16, [1, {channels}, 1, {key_len}]> v_flat = slice_by_index(begin = v_begin, end = v_end, x = packed)[name = string("v_flat")];
        tensor<int32, [4]> q_shape = const()[name = string("q_shape"), val = tensor<int32, [4]>([{q_shape[0]},{q_shape[1]},{q_shape[2]},{q_shape[3]}])];
        tensor<int32, [4]> kv_shape = const()[name = string("kv_shape"), val = tensor<int32, [4]>([{kv_shape[0]},{kv_shape[1]},{kv_shape[2]},{kv_shape[3]}])];
        tensor<int32, [4]> perm = const()[name = string("perm"), val = tensor<int32, [4]>([{perm[0]},{perm[1]},{perm[2]},{perm[3]}])];
        tensor<fp16, [1, {num_heads}, {head_dim}, 1]> q_reshaped = reshape(shape = q_shape, x = q_flat)[name = string("q_reshaped")];
        tensor<fp16, [1, {num_heads}, {head_dim}, {key_len}]> k_reshaped = reshape(shape = kv_shape, x = k_flat)[name = string("k_reshaped")];
        tensor<fp16, [1, {num_heads}, {head_dim}, {key_len}]> v_reshaped = reshape(shape = kv_shape, x = v_flat)[name = string("v_reshaped")];
        tensor<fp16, [1, {num_heads}, 1, {head_dim}]> q =
            transpose(perm = perm, x = q_reshaped)[name = string("q")];
        tensor<fp16, [1, {num_heads}, {key_len}, {head_dim}]> k =
            transpose(perm = perm, x = k_reshaped)[name = string("k")];
        tensor<fp16, [1, {num_heads}, {key_len}, {head_dim}]> v =
            transpose(perm = perm, x = v_reshaped)[name = string("v")];
        tensor<fp16, [1, {num_heads}, 1, {head_dim}]> out =
            scaled_dot_product_attention(query = q, key = k, value = v)[name = string("sdpa")];
    }} -> (out);
}}
"""


def build_packed_sdpa_servers(
    token_count: int,
    *,
    num_heads: int,
    head_dim: int,
) -> dict[int, tuple[Path, subprocess.Popen[bytes]]]:
    servers: dict[int, tuple[Path, subprocess.Popen[bytes]]] = {}
    for key_len in range(2, token_count + 1):
        channels = num_heads * head_dim
        total_steps = 1 + (2 * key_len)
        servers[key_len] = build_server(
            make_packed_sdpa_mil(
                key_len=key_len,
                num_heads=num_heads,
                head_dim=head_dim,
            ),
            [channels * total_steps * 2],
            channels * 2,
            prefix=f"josie_packed_sdpa_{key_len}_",
        )
    return servers


def close_packed_sdpa_servers(
    servers: dict[int, tuple[Path, subprocess.Popen[bytes]]],
) -> None:
    for work_dir, proc in servers.values():
        close_server(work_dir, proc)


def pack_sdpa_input(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> bytes:
    q_flat = np.transpose(q, (0, 1, 3, 2)).reshape(q.shape[0], -1, 1, 1)
    key_len = k.shape[2]
    k_flat = np.transpose(k, (0, 1, 3, 2)).reshape(k.shape[0], -1, 1, key_len)
    v_flat = np.transpose(v, (0, 1, 3, 2)).reshape(v.shape[0], -1, 1, key_len)
    packed = np.concatenate([q_flat, k_flat, v_flat], axis=3)
    return packed.astype(np.float16, copy=False).tobytes()


def packed_fast_incremental_forward(
    model,
    token_ids: list[int],
    servers: dict[int, tuple[Path, subprocess.Popen[bytes]]],
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> np.ndarray:
    from mlx_lm.models.cache import make_prompt_cache

    caches = make_prompt_cache(model.model)
    logits = []
    for token_id in token_ids:
        h = model.model.embed_tokens(mx.array([[token_id]], dtype=mx.int32)).astype(
            mx.float16
        )
        mx.eval(h)
        mx.synchronize()

        for layer, cache in zip(model.model.layers, caches):
            x_norm = layer.input_layernorm(h).astype(mx.float16)
            q = layer.self_attn.q_proj(x_norm).astype(mx.float16)
            k = layer.self_attn.k_proj(x_norm).astype(mx.float16)
            v = layer.self_attn.v_proj(x_norm).astype(mx.float16)

            q = layer.self_attn.q_norm(
                q.reshape(1, 1, num_heads, -1)
            ).transpose(0, 2, 1, 3).astype(mx.float16)
            k = layer.self_attn.k_norm(
                k.reshape(1, 1, num_kv_heads, -1)
            ).transpose(0, 2, 1, 3).astype(mx.float16)
            v = v.reshape(1, 1, num_kv_heads, -1).transpose(0, 2, 1, 3).astype(
                mx.float16
            )

            q = layer.self_attn.rope(q, offset=cache.offset).astype(mx.float16)
            k = layer.self_attn.rope(k, offset=cache.offset).astype(mx.float16)
            keys, values = cache.update_and_fetch(k, v)
            key_len = int(keys.shape[2])

            if key_len == 1:
                ctx = repeat_kv(values, num_heads, num_kv_heads, 1, head_dim).astype(
                    mx.float16
                )
            else:
                keys = repeat_kv(
                    keys,
                    num_heads,
                    num_kv_heads,
                    key_len,
                    head_dim,
                ).astype(mx.float16)
                values = repeat_kv(
                    values,
                    num_heads,
                    num_kv_heads,
                    key_len,
                    head_dim,
                ).astype(mx.float16)
                q_np = np.array(q, copy=False).astype(np.float16, copy=False)
                k_np = np.array(keys, copy=False).astype(np.float16, copy=False)
                v_np = np.array(values, copy=False).astype(np.float16, copy=False)
                packed = pack_sdpa_input(q_np, k_np, v_np)
                ctx_bytes = run_server(
                    servers[key_len][1],
                    [packed],
                    num_heads * head_dim * 2,
                )
                ctx = mx.array(
                    np.frombuffer(ctx_bytes, dtype=np.float16).reshape(
                        1,
                        num_heads,
                        1,
                        head_dim,
                    ),
                    dtype=mx.float16,
                )

            attn_out = layer.self_attn.o_proj(
                ctx.transpose(0, 2, 1, 3).reshape(1, 1, -1)
            ).astype(mx.float16)
            h = (h + attn_out).astype(mx.float16)
            h = (
                h
                + layer.mlp(layer.post_attention_layernorm(h)).astype(mx.float16)
            ).astype(mx.float16)

        h = model.model.norm(h).astype(mx.float16)
        out = model.model.embed_tokens.as_linear(h)
        mx.eval(out)
        mx.synchronize()
        logits.append(
            np.array(out[:, -1, :].astype(mx.float32).tolist(), dtype=np.float32)[0]
        )

    return np.stack(logits, axis=0)


def run_benchmark(fn, *, warmup: int, iters: int):
    for _ in range(warmup):
        fn()
    started = time.perf_counter()
    last = None
    for _ in range(iters):
        last = fn()
    total_ms = (time.perf_counter() - started) * 1000.0 / iters
    return last, total_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--token-limit", type=int, default=4)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load_josie(lazy=True)
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    token_ids = tokenizer.encode(args.prompt)[: args.token_limit]

    mlx_out, mlx_ms = run_benchmark(
        lambda: mlx_incremental_forward(model, token_ids),
        warmup=args.warmup,
        iters=args.iters,
    )

    from fast_decode_compare import build_sdpa_servers, close_sdpa_servers

    multi_servers = build_sdpa_servers(
        len(token_ids),
        num_heads=num_heads,
        head_dim=head_dim,
    )
    multi_error = None
    try:
        try:
            multi_out, multi_ms = run_benchmark(
                lambda: fast_incremental_forward(
                    model,
                    token_ids,
                    multi_servers,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                ),
                warmup=args.warmup,
                iters=args.iters,
            )
        except Exception as error:
            multi_out = None
            multi_ms = None
            multi_error = str(error)
    finally:
        close_sdpa_servers(multi_servers)

    packed_servers = build_packed_sdpa_servers(
        len(token_ids),
        num_heads=num_heads,
        head_dim=head_dim,
    )
    packed_error = None
    try:
        try:
            packed_out, packed_ms = run_benchmark(
                lambda: packed_fast_incremental_forward(
                    model,
                    token_ids,
                    packed_servers,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                ),
                warmup=args.warmup,
                iters=args.iters,
            )
        except Exception as error:
            packed_out = None
            packed_ms = None
            packed_error = str(error)
    finally:
        close_packed_sdpa_servers(packed_servers)

    report = {
        "runtime": "josie_packed_fast_decode_compare",
        "prompt": args.prompt,
        "token_count": len(token_ids),
        "mlx_total_ms": mlx_ms,
        "ane_multi_total_ms": multi_ms,
        "ane_packed_total_ms": packed_ms,
        "ane_multi_error": multi_error,
        "ane_packed_error": packed_error,
        "packed_speedup_vs_multi": (
            multi_ms / packed_ms if multi_ms is not None and packed_ms else None
        ),
        "multi_argmax_matches_mlx": (
            [int(np.argmax(row)) for row in multi_out]
            == [int(np.argmax(row)) for row in mlx_out]
            if multi_out is not None
            else None
        ),
        "packed_argmax_matches_mlx": (
            [int(np.argmax(row)) for row in packed_out]
            == [int(np.argmax(row)) for row in mlx_out]
            if packed_out is not None
            else None
        ),
        "multi_mean_abs_diff": (
            float(np.mean(np.abs(multi_out - mlx_out))) if multi_out is not None else None
        ),
        "packed_mean_abs_diff": (
            float(np.mean(np.abs(packed_out - mlx_out)))
            if packed_out is not None
            else None
        ),
        "multi_max_abs_diff": (
            float(np.max(np.abs(multi_out - mlx_out))) if multi_out is not None else None
        ),
        "packed_max_abs_diff": (
            float(np.max(np.abs(packed_out - mlx_out)))
            if packed_out is not None
            else None
        ),
    }
    if args.json:
        print(json.dumps(report))
        return
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
