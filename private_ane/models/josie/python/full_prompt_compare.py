from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time

import mlx.core as mx
import numpy as np
from mlx_lm import load

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "private_ane" / "models" / "josie" / "python"))

from full_model_compare import PROMPT, make_sdpa_server, repeat_kv

PREFIX_Q_SCALE = {
    1: 1.0,
    2: 0.35,
    3: 0.4,
    4: 0.4,
}


def standard_forward(model, tokens):
    out = model(tokens)
    mx.eval(out)
    mx.synchronize()
    return out[:, -1, :].astype(mx.float32)


def fp16_reference_forward(model, tokens, *, num_heads: int, num_kv_heads: int, head_dim: int):
    seq_len = tokens.shape[1]
    h = model.model.embed_tokens(tokens).astype(mx.float16)
    mx.eval(h)
    mx.synchronize()
    for layer in model.model.layers:
        x_norm = layer.input_layernorm(h).astype(mx.float16)
        q = layer.self_attn.q_proj(x_norm).astype(mx.float16)
        k = layer.self_attn.k_proj(x_norm).astype(mx.float16)
        v = layer.self_attn.v_proj(x_norm).astype(mx.float16)
        q = layer.self_attn.q_norm(q.reshape(1, seq_len, num_heads, -1)).transpose(0, 2, 1, 3).astype(mx.float16)
        k = layer.self_attn.k_norm(k.reshape(1, seq_len, num_kv_heads, -1)).transpose(0, 2, 1, 3).astype(mx.float16)
        v = v.reshape(1, seq_len, num_kv_heads, -1).transpose(0, 2, 1, 3).astype(mx.float16)
        q = layer.self_attn.rope(q).astype(mx.float16)
        k = layer.self_attn.rope(k).astype(mx.float16)
        k = repeat_kv(k, num_heads, num_kv_heads, seq_len, head_dim).astype(mx.float16)
        v = repeat_kv(v, num_heads, num_kv_heads, seq_len, head_dim).astype(mx.float16)
        ctx = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=head_dim ** -0.5,
        ).astype(mx.float16)
        attn_out = layer.self_attn.o_proj(
            ctx.transpose(0, 2, 1, 3).reshape(1, seq_len, -1)
        ).astype(mx.float16)
        h = (h + attn_out).astype(mx.float16)
        h = (h + layer.mlp(layer.post_attention_layernorm(h))).astype(mx.float16)
    h = model.model.norm(h).astype(mx.float16)
    out = model.model.embed_tokens.as_linear(h)
    mx.eval(out)
    mx.synchronize()
    return out[:, -1, :].astype(mx.float32)


def hybrid_forward(
    model,
    tokens,
    proc,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
):
    seq_len = tokens.shape[1]
    h = model.model.embed_tokens(tokens).astype(mx.float16)
    mx.eval(h)
    mx.synchronize()
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

        q_scale = PREFIX_Q_SCALE.get(seq_len, 0.4)
        q_np = np.array((q * q_scale), copy=False).astype(np.float16, copy=False)
        k_np = np.array(k, copy=False).astype(np.float16, copy=False)
        v_np = np.array(v, copy=False).astype(np.float16, copy=False)

        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(q_np.tobytes())
        proc.stdin.write(k_np.tobytes())
        proc.stdin.write(v_np.tobytes())
        proc.stdin.flush()

        out_bytes = q_np.nbytes
        chunk = proc.stdout.read(out_bytes)
        if chunk is None or len(chunk) != out_bytes:
            stderr = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
            raise RuntimeError(f"ANE SDPA server failed. stderr={stderr}")

        ctx = np.frombuffer(chunk, dtype=np.float16).astype(np.float32)
        ctx = ctx.reshape(1, num_heads, seq_len, head_dim)
        ctx = mx.array(ctx.astype(np.float16), dtype=mx.float16)
        attn_out = layer.self_attn.o_proj(
            ctx.transpose(0, 2, 1, 3).reshape(1, seq_len, -1)
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

    std_out, std_ms = run_prefixes(
        lambda prefix: standard_forward(model, prefix),
        token_ids,
        warmup=args.warmup,
        iters=args.iters,
    )

    fp16_out, fp16_ms = run_prefixes(
        lambda prefix: fp16_reference_forward(
            model,
            prefix,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        ),
        token_ids,
        warmup=args.warmup,
        iters=args.iters,
    )

    servers = {}
    try:
        for seq_len in range(1, len(token_ids) + 1):
            server_dir, server_exe, mil_path = make_sdpa_server(seq_len, num_heads, head_dim)
            proc = subprocess.Popen(
                [
                    str(server_exe),
                    str(mil_path),
                    str(num_heads * seq_len * head_dim * 2),
                    str(num_heads * seq_len * head_dim * 2),
                    str(num_heads * seq_len * head_dim * 2),
                    str(num_heads * seq_len * head_dim * 2),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            servers[seq_len] = (server_dir, proc)

        def hybrid(prefix):
            seq_len = prefix.shape[1]
            _server_dir, proc = servers[seq_len]
            return hybrid_forward(
                model,
                prefix,
                proc,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
            )

        ane_out, ane_ms = run_prefixes(
            hybrid,
            token_ids,
            warmup=args.warmup,
            iters=args.iters,
        )
    finally:
        for server_dir, proc in servers.values():
            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
            shutil.rmtree(server_dir, ignore_errors=True)

    diffs_std = np.abs(ane_out - std_out)
    diffs_fp16 = np.abs(ane_out - fp16_out)
    fp16_std = np.abs(fp16_out - std_out)

    ane_argmax = [int(np.argmax(row)) for row in ane_out]
    std_argmax = [int(np.argmax(row)) for row in std_out]
    fp16_argmax = [int(np.argmax(row)) for row in fp16_out]

    report = {
        "runtime": "josie_full_prompt_hybrid_private_vs_mlx",
        "prompt": args.prompt,
        "token_count": len(token_ids),
        "ane_total_ms": ane_ms,
        "mlx_total_ms": std_ms,
        "ane_speedup_vs_mlx": std_ms / ane_ms if ane_ms else None,
        "fp16_ref_total_ms": fp16_ms,
        "max_abs_diff": float(np.max(diffs_std)),
        "mean_abs_diff": float(np.mean(diffs_std)),
        "max_abs_diff_vs_fp16_ref": float(np.max(diffs_fp16)),
        "mean_abs_diff_vs_fp16_ref": float(np.mean(diffs_fp16)),
        "fp16_ref_max_abs_diff_vs_mlx": float(np.max(fp16_std)),
        "fp16_ref_mean_abs_diff_vs_mlx": float(np.mean(fp16_std)),
        "argmax_matches_mlx": ane_argmax == std_argmax,
        "argmax_matches_fp16_ref": ane_argmax == fp16_argmax,
        "ane_argmax": ane_argmax,
        "mlx_argmax": std_argmax,
        "fp16_ref_argmax": fp16_argmax,
    }

    if args.json:
        print(json.dumps(report))
        return

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
