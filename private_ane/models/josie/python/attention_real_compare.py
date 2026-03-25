from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "private_ane" / "models" / "josie" / "python"))

from attention_hybrid_compare import HELPER_SRC, make_mil


PROMPT = "Explain why MLX is useful for local inference."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load("mlx-community/JOSIE-1.1-4B-Instruct-4bit", lazy=True)
    attn = model.model.layers[0].self_attn
    dim = 2560
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    repeat = num_heads // num_kv_heads

    token_ids = tokenizer.encode(PROMPT)[:16]
    seq_len = len(token_ids)
    tokens = mx.array([token_ids], dtype=mx.int32)
    x = model.model.embed_tokens(tokens).astype(mx.float16)
    mx.eval(x)
    mx.synchronize()

    real_attention = attn(x)
    mx.eval(real_attention)
    mx.synchronize()
    real_out = np.array(real_attention, copy=False).astype(np.float32, copy=False)

    def repeat_kv(tensor):
        return mx.broadcast_to(
            mx.expand_dims(tensor, axis=2),
            (1, num_kv_heads, repeat, seq_len, head_dim),
        ).reshape(1, num_heads, seq_len, head_dim)

    q = attn.q_proj(x)
    k = attn.k_proj(x)
    v = attn.v_proj(x)
    q = attn.q_norm(q.reshape(1, seq_len, num_heads, -1)).transpose(0, 2, 1, 3)
    k = attn.k_norm(k.reshape(1, seq_len, num_kv_heads, -1)).transpose(0, 2, 1, 3)
    v = v.reshape(1, seq_len, num_kv_heads, -1).transpose(0, 2, 1, 3)
    q = attn.rope(q)
    k = attn.rope(k)
    k = repeat_kv(k)
    v = repeat_kv(v)

    def mlx_attention():
        ctx = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=head_dim ** -0.5,
        )
        out = attn.o_proj(ctx.transpose(0, 2, 1, 3).reshape(1, seq_len, -1))
        mx.eval(out)
        mx.synchronize()
        return out

    for _ in range(args.warmup):
        mlx_attention()
    started = time.perf_counter()
    mlx_last = None
    for _ in range(args.iters):
        mlx_last = mlx_attention()
    mlx_ms = (time.perf_counter() - started) * 1000.0 / args.iters
    mlx_out = np.array(mlx_last, copy=False).astype(np.float32, copy=False)
    manual_diffs = np.abs(mlx_out - real_out)

    q_np = np.array(q, copy=False).astype(np.float16, copy=False)
    k_np = np.array(k, copy=False).astype(np.float16, copy=False)
    v_np = np.array(v, copy=False).astype(np.float16, copy=False)

    work_dir = Path(tempfile.mkdtemp(prefix="josie_real_attention_"))
    try:
        root = work_dir / "artifact"
        root.mkdir(parents=True, exist_ok=True)
        (root / "q.bin").write_bytes(q_np.tobytes())
        (root / "k.bin").write_bytes(k_np.tobytes())
        (root / "v.bin").write_bytes(v_np.tobytes())
        (root / "model.mil").write_text(
            make_mil(num_heads=num_heads, seq_len=seq_len, head_dim=head_dim),
            encoding="utf-8",
        )
        src = work_dir / "ane_sdpa_helper.m"
        exe = work_dir / "ane_sdpa_helper"
        src.write_text(HELPER_SRC, encoding="utf-8")
        subprocess.check_call(
            [
                "clang",
                "-fobjc-arc",
                "-framework",
                "Foundation",
                "-framework",
                "CoreML",
                "-framework",
                "IOSurface",
                "-ldl",
                "-o",
                str(exe),
                str(src),
            ]
        )
        out_path = work_dir / "out.bin"
        proc = subprocess.run(
            [
                str(exe),
                str(root),
                str(q_np.nbytes),
                str(k_np.nbytes),
                str(v_np.nbytes),
                str(q_np.nbytes),
                str(args.iters),
                str(out_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(proc.stdout.strip())
        ane_ctx = np.frombuffer(out_path.read_bytes(), dtype=np.float16).astype(np.float32)
        ane_ctx = ane_ctx.reshape(1, num_heads, seq_len, head_dim)
        ane_ctx = mx.array(ane_ctx.astype(np.float16), dtype=mx.float16)
        ane_out = attn.o_proj(ane_ctx.transpose(0, 2, 1, 3).reshape(1, seq_len, -1))
        mx.eval(ane_out)
        mx.synchronize()
        ane_out_np = np.array(ane_out, copy=False).astype(np.float32, copy=False)
        diffs = np.abs(ane_out_np - real_out)
        report = {
            "runtime": "josie_attention_real_private_vs_mlx",
            "prompt": PROMPT,
            "token_count": seq_len,
            "ane_sdpa_per_iter_ms": float(payload["per_iter_ms"]),
            "mlx_attention_per_iter_ms": mlx_ms,
            "ane_sdpa_speedup_vs_mlx_attention": mlx_ms / float(payload["per_iter_ms"]),
            "max_abs_diff": float(np.max(diffs)),
            "mean_abs_diff": float(np.mean(diffs)),
            "manual_mlx_max_abs_diff_vs_real": float(np.max(manual_diffs)),
            "manual_mlx_mean_abs_diff_vs_real": float(np.mean(manual_diffs)),
        }
        if args.json:
            print(json.dumps(report))
            return
        print(json.dumps(report, indent=2))
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
