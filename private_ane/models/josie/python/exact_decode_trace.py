from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys

import mlx.core as mx
import numpy as np
from mlx_lm import load

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "private_ane" / "models" / "josie" / "python"))

from full_model_compare import PROMPT, make_sdpa_server, repeat_kv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--token-limit", type=int, default=4)
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load("mlx-community/JOSIE-1.1-4B-Instruct-4bit", lazy=True)
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128

    token_ids = tokenizer.encode(args.prompt)[: args.token_limit]
    seq_len = len(token_ids)
    tokens = mx.array([token_ids], dtype=mx.int32)

    exact_h = model.model.embed_tokens(tokens).astype(mx.float16)
    hybrid_h = model.model.embed_tokens(tokens).astype(mx.float16)
    mx.eval(exact_h, hybrid_h)
    mx.synchronize()

    servers = {}
    try:
        for prefix in range(1, seq_len + 1):
            server_dir, server_exe, mil_path = make_sdpa_server(
                prefix,
                num_heads,
                head_dim,
                query_len=1,
                key_len=prefix,
            )
            proc = subprocess.Popen(
                [
                    str(server_exe),
                    str(mil_path),
                    str(num_heads * 1 * head_dim * 2),
                    str(num_heads * prefix * head_dim * 2),
                    str(num_heads * prefix * head_dim * 2),
                    str(num_heads * 1 * head_dim * 2),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            servers[prefix] = (server_dir, proc)

        layers = []
        for layer_index, layer in enumerate(model.model.layers[: args.layers]):
            exact_h = layer(exact_h, "causal", None)
            mx.eval(exact_h)
            mx.synchronize()

            x_norm = layer.input_layernorm(hybrid_h)
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
                _server_dir, proc = servers[prefix]
                qq = q_np[:, :, prefix - 1 : prefix, :]
                kk = k_np[:, :, :prefix, :]
                vv = v_np[:, :, :prefix, :]
                assert proc.stdin is not None and proc.stdout is not None
                proc.stdin.write(qq.tobytes())
                proc.stdin.write(kk.tobytes())
                proc.stdin.write(vv.tobytes())
                proc.stdin.flush()
                out_bytes = qq.nbytes
                chunk = proc.stdout.read(out_bytes)
                if chunk is None or len(chunk) != out_bytes:
                    stderr = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
                    raise RuntimeError(f"ANE exact decode trace failed. stderr={stderr}")
                ctx = np.frombuffer(chunk, dtype=np.float16).astype(np.float32)
                ctx = ctx.reshape(1, num_heads, 1, head_dim)
                ctx_slices.append(ctx)

            ctx_all = np.concatenate(ctx_slices, axis=2)
            ctx_mx = mx.array(ctx_all.astype(np.float16), dtype=mx.float16)
            attn_out = layer.self_attn.o_proj(
                ctx_mx.transpose(0, 2, 1, 3).reshape(1, seq_len, -1)
            )
            hybrid_h = hybrid_h + attn_out
            hybrid_h = hybrid_h + layer.mlp(layer.post_attention_layernorm(hybrid_h))
            mx.eval(hybrid_h)
            mx.synchronize()

            exact_np = np.array(exact_h.astype(mx.float32).tolist(), dtype=np.float32)
            hybrid_np = np.array(hybrid_h.astype(mx.float32).tolist(), dtype=np.float32)
            diffs = np.abs(hybrid_np - exact_np)
            layers.append(
                {
                    "layer": layer_index,
                    "hidden_max_abs_diff": float(np.max(diffs)),
                    "hidden_mean_abs_diff": float(np.mean(diffs)),
                }
            )

        report = {
            "runtime": "josie_exact_decode_trace",
            "prompt": args.prompt,
            "token_count": seq_len,
            "layers": layers,
        }
        if args.json:
            print(json.dumps(report))
            return
        print(json.dumps(report, indent=2))
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


if __name__ == "__main__":
    main()
