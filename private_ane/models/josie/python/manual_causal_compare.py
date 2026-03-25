from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "private_ane" / "models" / "josie" / "python"))

from ane_ops import build_server, close_server, run_server
from full_model_compare import PROMPT, repeat_kv


def make_scores_mil(num_heads: int, key_len: int, head_dim: int) -> str:
    return f'''program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}, {{"coremlc-version", "3505.4.1"}}, {{"coremltools-component-milinternal", ""}}, {{"coremltools-version", "9.0"}}}})]
{{
    func main<ios18>(
        tensor<fp16, [1, {num_heads}, 1, {head_dim}]> a,
        tensor<fp16, [1, {num_heads}, {key_len}, {head_dim}]> b
    ) {{
        bool mm_tx = const()[name = string("mm_tx"), val = bool(false)];
        bool mm_ty = const()[name = string("mm_ty"), val = bool(true)];
        tensor<fp16, [1, {num_heads}, 1, {key_len}]> out =
            matmul(transpose_x = mm_tx, transpose_y = mm_ty, x = a, y = b)[name = string("mm")];
    }} -> (out);
}}
'''


def make_softmax_mil(num_heads: int, key_len: int) -> str:
    return f'''program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}, {{"coremlc-version", "3505.4.1"}}, {{"coremltools-component-milinternal", ""}}, {{"coremltools-version", "9.0"}}}})]
{{
    func main<ios18>(
        tensor<fp16, [1, {num_heads}, 1, {key_len}]> x
    ) {{
        int32 sm_ax = const()[name = string("sm_ax"), val = int32(-1)];
        tensor<fp16, [1, {num_heads}, 1, {key_len}]> out =
            softmax(axis = sm_ax, x = x)[name = string("sm")];
    }} -> (out);
}}
'''


def make_ctx_mil(num_heads: int, key_len: int, head_dim: int) -> str:
    return f'''program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}, {{"coremlc-version", "3505.4.1"}}, {{"coremltools-component-milinternal", ""}}, {{"coremltools-version", "9.0"}}}})]
{{
    func main<ios18>(
        tensor<fp16, [1, {num_heads}, 1, {key_len}]> a,
        tensor<fp16, [1, {num_heads}, {key_len}, {head_dim}]> b
    ) {{
        bool mm_tx = const()[name = string("mm_tx"), val = bool(false)];
        bool mm_ty = const()[name = string("mm_ty"), val = bool(false)];
        tensor<fp16, [1, {num_heads}, 1, {head_dim}]> out =
            matmul(transpose_x = mm_tx, transpose_y = mm_ty, x = a, y = b)[name = string("mm")];
    }} -> (out);
}}
'''


def standard_forward(model, tokens):
    out = model(tokens)
    mx.eval(out)
    mx.synchronize()
    return out[:, -1, :].astype(mx.float32)


def build_prefix_servers(
    token_count: int,
    *,
    num_heads: int,
    head_dim: int,
) -> dict[int, dict[str, tuple[Path, object]]]:
    servers: dict[int, dict[str, tuple[Path, object]]] = {}
    score_in_a = num_heads * head_dim * 2
    ctx_out = num_heads * head_dim * 2
    for prefix in range(1, token_count + 1):
        score_in_b = num_heads * prefix * head_dim * 2
        score_out = num_heads * prefix * 2
        servers[prefix] = {
            "scores": build_server(
                make_scores_mil(num_heads, prefix, head_dim),
                [score_in_a, score_in_b],
                score_out,
                prefix=f"josie_scores_{prefix}_",
            ),
            "softmax": build_server(
                make_softmax_mil(num_heads, prefix),
                [score_out],
                score_out,
                prefix=f"josie_softmax_{prefix}_",
            ),
            "ctx": build_server(
                make_ctx_mil(num_heads, prefix, head_dim),
                [score_out, score_in_b],
                ctx_out,
                prefix=f"josie_ctx_{prefix}_",
            ),
        }
    return servers


def close_prefix_servers(servers: dict[int, dict[str, tuple[Path, object]]]) -> None:
    for server_group in servers.values():
        for work_dir, proc in server_group.values():
            close_server(work_dir, proc)


def manual_decode_forward(
    model,
    tokens,
    servers,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
):
    seq_len = tokens.shape[1]
    h = model.model.embed_tokens(tokens).astype(mx.float16)
    mx.eval(h)
    mx.synchronize()
    scale = np.float32(head_dim ** -0.5)

    for layer in model.model.layers:
        x_norm = layer.input_layernorm(h)
        q = layer.self_attn.q_proj(x_norm)
        k = layer.self_attn.k_proj(x_norm)
        v = layer.self_attn.v_proj(x_norm)
        q = layer.self_attn.q_norm(q.reshape(1, seq_len, num_heads, -1)).transpose(0, 2, 1, 3)
        k = layer.self_attn.k_norm(k.reshape(1, seq_len, num_kv_heads, -1)).transpose(0, 2, 1, 3)
        v = v.reshape(1, seq_len, num_kv_heads, -1).transpose(0, 2, 1, 3)
        q = layer.self_attn.rope(q)
        k = layer.self_attn.rope(k)
        k = repeat_kv(k, num_heads, num_kv_heads, seq_len, head_dim)
        v = repeat_kv(v, num_heads, num_kv_heads, seq_len, head_dim)

        q_np = np.array(q, copy=False).astype(np.float16, copy=False)
        k_np = np.array(k, copy=False).astype(np.float16, copy=False)
        v_np = np.array(v, copy=False).astype(np.float16, copy=False)

        ctx_slices = []
        for prefix in range(1, seq_len + 1):
            qq = np.array(q_np[:, :, prefix - 1 : prefix, :], copy=True)
            qq = (qq.astype(np.float32) * scale).astype(np.float16)
            kk = k_np[:, :, :prefix, :]
            vv = v_np[:, :, :prefix, :]

            score_bytes = run_server(
                servers[prefix]["scores"][1],
                [qq.tobytes(), kk.tobytes()],
                num_heads * prefix * 2,
            )
            prob_bytes = run_server(
                servers[prefix]["softmax"][1],
                [score_bytes],
                num_heads * prefix * 2,
            )
            ctx_bytes = run_server(
                servers[prefix]["ctx"][1],
                [prob_bytes, vv.tobytes()],
                num_heads * head_dim * 2,
            )
            ctx = np.frombuffer(ctx_bytes, dtype=np.float16).astype(np.float32)
            ctx_slices.append(ctx.reshape(1, num_heads, 1, head_dim))

        ctx_all = np.concatenate(ctx_slices, axis=2)
        ctx_mx = mx.array(ctx_all.astype(np.float16), dtype=mx.float16)
        attn_out = layer.self_attn.o_proj(
            ctx_mx.transpose(0, 2, 1, 3).reshape(1, seq_len, -1)
        )
        h = h + attn_out
        h = h + layer.mlp(layer.post_attention_layernorm(h))

    h = model.model.norm(h)
    out = model.model.embed_tokens.as_linear(h)
    mx.eval(out)
    mx.synchronize()
    return out[:, -1, :].astype(mx.float32)


def run_prefixes(fn, token_ids, *, warmup: int, iters: int):
    prefixes = [mx.array([token_ids[:index]], dtype=mx.int32) for index in range(1, len(token_ids) + 1)]
    for _ in range(warmup):
        for prefix in prefixes:
            fn(prefix)
    started = time.perf_counter()
    last = []
    for _ in range(iters):
        last = [fn(prefix) for prefix in prefixes]
    total_ms = (time.perf_counter() - started) * 1000.0 / iters
    stacked = np.stack([np.array(item.tolist(), dtype=np.float32)[0] for item in last], axis=0)
    return stacked, total_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--token-limit", type=int, default=4)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load("mlx-community/JOSIE-1.1-4B-Instruct-4bit", lazy=True)
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    token_ids = tokenizer.encode(args.prompt)[: args.token_limit]

    mlx_out, mlx_ms = run_prefixes(
        lambda prefix: standard_forward(model, prefix),
        token_ids,
        warmup=args.warmup,
        iters=args.iters,
    )

    servers = build_prefix_servers(
        len(token_ids),
        num_heads=num_heads,
        head_dim=head_dim,
    )
    try:
        ane_out, ane_ms = run_prefixes(
            lambda prefix: manual_decode_forward(
                model,
                prefix,
                servers,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
            ),
            token_ids,
            warmup=args.warmup,
            iters=args.iters,
        )
    finally:
        close_prefix_servers(servers)

    diffs = np.abs(ane_out - mlx_out)
    ane_argmax = [int(np.argmax(row)) for row in ane_out]
    mlx_argmax = [int(np.argmax(row)) for row in mlx_out]
    report = {
        "runtime": "josie_manual_causal_private_vs_mlx",
        "prompt": args.prompt,
        "token_count": len(token_ids),
        "ane_total_ms": ane_ms,
        "mlx_total_ms": mlx_ms,
        "ane_speedup_vs_mlx": mlx_ms / ane_ms if ane_ms else None,
        "max_abs_diff": float(np.max(diffs)),
        "mean_abs_diff": float(np.mean(diffs)),
        "argmax_matches_mlx": ane_argmax == mlx_argmax,
        "ane_argmax": ane_argmax,
        "mlx_argmax": mlx_argmax,
    }
    if args.json:
        print(json.dumps(report))
        return
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
