from __future__ import annotations

import argparse
import json
import shutil
import sys

import mlx.core as mx
import numpy as np
from mlx_lm import load

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "private_ane" / "models" / "josie" / "python"))

from full_model_compare import PROMPT, make_sdpa_server, repeat_kv


ALPHA_GRID = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--token-limit", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load("mlx-community/JOSIE-1.1-4B-Instruct-4bit", lazy=True)
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128

    token_ids = tokenizer.encode(args.prompt)[: args.token_limit]
    tokens = mx.array([token_ids], dtype=mx.int32)
    h = model.model.embed_tokens(tokens).astype(mx.float16)
    mx.eval(h)
    mx.synchronize()

    report = {"prompt": args.prompt, "token_count": len(token_ids), "layers": {}}
    for layer_index, layer in enumerate(model.model.layers[: args.layers]):
        seq_len = h.shape[1]
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

        layer_report = {}
        servers = {}
        try:
            for p in range(2, seq_len + 1):
                server_dir, _exe, _mil = make_sdpa_server(p, num_heads, head_dim)
                # make_sdpa_server already spawned only helper build pieces; full server startup is inside compare script
                # here we use its returned paths to spawn a server per prefix
                proc = __import__("subprocess").Popen(
                    [
                        str(_exe),
                        str(_mil),
                        str(num_heads * p * head_dim * 2),
                        str(num_heads * p * head_dim * 2),
                        str(num_heads * p * head_dim * 2),
                        str(num_heads * p * head_dim * 2),
                    ],
                    stdin=__import__("subprocess").PIPE,
                    stdout=__import__("subprocess").PIPE,
                    stderr=__import__("subprocess").PIPE,
                )
                servers[p] = (server_dir, proc)

            for p in range(2, seq_len + 1):
                ref = mx.fast.scaled_dot_product_attention(
                    q[:, :, :p, :],
                    k[:, :, :p, :],
                    v[:, :, :p, :],
                    scale=head_dim ** -0.5,
                )
                mx.eval(ref)
                mx.synchronize()
                ref_np = np.array(ref, copy=False).astype(np.float32, copy=False)

                best_alpha = None
                best_mean = None
                best_max = None
                server_dir, proc = servers[p]
                for alpha in ALPHA_GRID:
                    qq = (q_np[:, :, :p, :] * alpha).astype(np.float16)
                    kk = k_np[:, :, :p, :]
                    vv = v_np[:, :, :p, :]
                    assert proc.stdin is not None and proc.stdout is not None
                    proc.stdin.write(qq.tobytes())
                    proc.stdin.write(kk.tobytes())
                    proc.stdin.write(vv.tobytes())
                    proc.stdin.flush()
                    chunk = proc.stdout.read(qq.nbytes)
                    ane = np.frombuffer(chunk, dtype=np.float16).astype(np.float32)
                    ane = ane.reshape(1, num_heads, p, head_dim)
                    diff = np.abs(ane[:, :, p - 1 : p, :] - ref_np[:, :, p - 1 : p, :])
                    mean = float(np.mean(diff))
                    if best_mean is None or mean < best_mean:
                        best_alpha = alpha
                        best_mean = mean
                        best_max = float(np.max(diff))
                layer_report[str(p)] = {
                    "alpha": best_alpha,
                    "mean": best_mean,
                    "max": best_max,
                }
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

        report["layers"][str(layer_index)] = layer_report
        h = layer(h, "causal", None)
        mx.eval(h)
        mx.synchronize()

    if args.json:
        print(json.dumps(report))
        return
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
