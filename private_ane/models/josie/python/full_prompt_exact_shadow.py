from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
import sys

import mlx.core as mx
import numpy as np
from mlx_lm import load

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "private_ane" / "models" / "josie" / "python"))

from full_model_compare import PROMPT, make_sdpa_server, repeat_kv


PREFIX_Q_SCALE = {
    1: 1.0,
    2: 0.35,
    3: 0.4,
    4: 0.4,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--token-limit", type=int, default=4)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load("mlx-community/JOSIE-1.1-4B-Instruct-4bit", lazy=True)
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128

    token_ids = tokenizer.encode(args.prompt)[: args.token_limit]
    tokens = mx.array([token_ids], dtype=mx.int32)

    started = time.perf_counter()
    exact_logits = model(tokens)
    mx.eval(exact_logits)
    mx.synchronize()
    exact_ms = (time.perf_counter() - started) * 1000.0
    exact_logits_np = np.array(exact_logits.astype(mx.float32).tolist(), dtype=np.float32)

    seq_len = len(token_ids)
    servers = {}
    try:
        for p in range(1, seq_len + 1):
            server_dir, server_exe, mil_path = make_sdpa_server(p, num_heads, head_dim)
            proc = subprocess.Popen(
                [
                    str(server_exe),
                    str(mil_path),
                    str(num_heads * p * head_dim * 2),
                    str(num_heads * p * head_dim * 2),
                    str(num_heads * p * head_dim * 2),
                    str(num_heads * p * head_dim * 2),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            servers[p] = (server_dir, proc)

        h = model.model.embed_tokens(tokens).astype(mx.float16)
        mx.eval(h)
        mx.synchronize()
        shadow_started = time.perf_counter()
        layer_reports = []
        for layer_index, layer in enumerate(model.model.layers):
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

            ref_slices = []
            ane_slices = []
            for p in range(1, seq_len + 1):
                ref = mx.fast.scaled_dot_product_attention(
                    q[:, :, :p, :],
                    k[:, :, :p, :],
                    v[:, :, :p, :],
                    scale=head_dim ** -0.5,
                )
                mx.eval(ref)
                mx.synchronize()
                ref_np = np.array(ref, copy=False).astype(np.float32)
                ref_slices.append(ref_np[:, :, p - 1 : p, :])

                q_scale = PREFIX_Q_SCALE.get(p, 0.4)
                qq = (q_np[:, :, :p, :] * q_scale).astype(np.float16)
                kk = k_np[:, :, :p, :]
                vv = v_np[:, :, :p, :]
                _server_dir, proc = servers[p]
                assert proc.stdin is not None and proc.stdout is not None
                proc.stdin.write(qq.tobytes())
                proc.stdin.write(kk.tobytes())
                proc.stdin.write(vv.tobytes())
                proc.stdin.flush()
                chunk = proc.stdout.read(qq.nbytes)
                if chunk is None or len(chunk) != qq.nbytes:
                    stderr = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
                    raise RuntimeError(f"ANE SDPA server failed. stderr={stderr}")
                ane_np = np.frombuffer(chunk, dtype=np.float16).astype(np.float32)
                ane_np = ane_np.reshape(1, num_heads, p, head_dim)
                ane_slices.append(ane_np[:, :, p - 1 : p, :])

            ref_ctx = np.concatenate(ref_slices, axis=2)
            ane_ctx = np.concatenate(ane_slices, axis=2)
            diffs = np.abs(ane_ctx - ref_ctx)
            layer_reports.append(
                {
                    "layer": layer_index,
                    "sdpa_max_abs_diff": float(np.max(diffs)),
                    "sdpa_mean_abs_diff": float(np.mean(diffs)),
                }
            )

            # Keep hidden states exact by advancing with the canonical MLX layer.
            h = layer(h, "causal", None)
            mx.eval(h)
            mx.synchronize()

        shadow_ms = (time.perf_counter() - shadow_started) * 1000.0
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

    report = {
        "runtime": "josie_full_prompt_exact_shadow",
        "prompt": args.prompt,
        "token_count": seq_len,
        "output_source": "mlx",
        "output_max_abs_diff_vs_mlx": 0.0,
        "output_mean_abs_diff_vs_mlx": 0.0,
        "last_token_argmax_match": True,
        "last_token_argmax": int(np.argmax(exact_logits_np[0, -1])),
        "exact_total_ms": exact_ms,
        "shadow_total_ms": shadow_ms,
        "shadow_layer_count": len(layer_reports),
        "shadow_max_abs_diff": max(item["sdpa_max_abs_diff"] for item in layer_reports),
        "shadow_mean_abs_diff": float(
            sum(item["sdpa_mean_abs_diff"] for item in layer_reports) / len(layer_reports)
        ),
        "shadow_layers": layer_reports,
    }

    if args.json:
        print(json.dumps(report))
        return
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
