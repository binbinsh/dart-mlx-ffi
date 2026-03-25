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
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "private_ane" / "models" / "josie" / "python"))

from full_model_compare import PROMPT, make_sdpa_server, repeat_kv

JOSIE_REPO = "mlx-community/JOSIE-1.1-4B-Instruct-4bit"
JOSIE_CACHE = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--mlx-community--JOSIE-1.1-4B-Instruct-4bit"
    / "snapshots"
)


def load_josie(*, lazy: bool):
    if JOSIE_CACHE.exists():
        snapshots = sorted(
            (path for path in JOSIE_CACHE.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if snapshots:
            return load(str(snapshots[0]), lazy=lazy)
    return load(JOSIE_REPO, lazy=lazy)


def build_sdpa_servers(
    token_count: int,
    *,
    num_heads: int,
    head_dim: int,
):
    servers: dict[int, tuple[Path, subprocess.Popen[bytes]]] = {}
    for key_len in range(1, token_count + 1):
        server_dir, server_exe, mil_path = make_sdpa_server(
            key_len,
            num_heads,
            head_dim,
            query_len=1,
            key_len=key_len,
        )
        proc = subprocess.Popen(
            [
                str(server_exe),
                str(mil_path),
                str(num_heads * head_dim * 2),
                str(num_heads * key_len * head_dim * 2),
                str(num_heads * key_len * head_dim * 2),
                str(num_heads * head_dim * 2),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        servers[key_len] = (server_dir, proc)
    return servers


def close_sdpa_servers(servers: dict[int, tuple[Path, subprocess.Popen[bytes]]]) -> None:
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


def mlx_incremental_forward(model, token_ids: list[int]) -> np.ndarray:
    cache = make_prompt_cache(model.model)
    logits = []
    for token_id in token_ids:
        out = model(mx.array([[token_id]], dtype=mx.int32), cache=cache)
        mx.eval(out)
        mx.synchronize()
        logits.append(
            np.array(out[:, -1, :].astype(mx.float32).tolist(), dtype=np.float32)[0]
        )
    return np.stack(logits, axis=0)


def fast_incremental_forward(
    model,
    token_ids: list[int],
    servers: dict[int, tuple[Path, subprocess.Popen[bytes]]],
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> np.ndarray:
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
                q_np = np.array(q, copy=False).astype(np.float16, copy=False)
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
                k_np = np.array(keys, copy=False).astype(np.float16, copy=False)
                v_np = np.array(values, copy=False).astype(np.float16, copy=False)
                _server_dir, proc = servers[key_len]
                assert proc.stdin is not None and proc.stdout is not None
                proc.stdin.write(q_np.tobytes())
                proc.stdin.write(k_np.tobytes())
                proc.stdin.write(v_np.tobytes())
                proc.stdin.flush()
                out_bytes = q_np.nbytes
                chunk = proc.stdout.read(out_bytes)
                if chunk is None or len(chunk) != out_bytes:
                    stderr = (
                        proc.stderr.read().decode("utf-8", errors="ignore")
                        if proc.stderr
                        else ""
                    )
                    raise RuntimeError(f"ANE fast decode server failed. stderr={stderr}")
                ctx = mx.array(
                    np.frombuffer(chunk, dtype=np.float16).reshape(
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

    servers = build_sdpa_servers(
        len(token_ids),
        num_heads=num_heads,
        head_dim=head_dim,
    )
    try:
        ane_out, ane_ms = run_benchmark(
            lambda: fast_incremental_forward(
                model,
                token_ids,
                servers,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
    finally:
        close_sdpa_servers(servers)

    diffs = np.abs(ane_out - mlx_out)
    ane_argmax = [int(np.argmax(row)) for row in ane_out]
    mlx_argmax = [int(np.argmax(row)) for row in mlx_out]
    report = {
        "runtime": "josie_fast_decode_fused_private_vs_mlx",
        "prompt": args.prompt,
        "token_count": len(token_ids),
        "ane_mode": "incremental_fused_full_head",
        "ane_backend": "coreml_builtin_sdpa",
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
