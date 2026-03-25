from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "private_ane" / "models" / "josie" / "python"))

from ane_ops import build_multi_server, close_server, run_multi_server
from full_model_compare import PROMPT, repeat_kv


def make_ctx_prefix_group_mil(seq_len: int, head_dim: int) -> str:
    bodies = []
    outputs = []
    scale = head_dim ** -0.5
    for prefix in range(2, seq_len + 1):
        q_begin = [0, 0, prefix - 1, 0]
        q_end = [1, 1, prefix, head_dim]
        kv_begin = [0, 0, 0, 0]
        kv_end = [1, 1, prefix, head_dim]
        out_name = f"z{prefix}"
        outputs.append(out_name)
        bodies.append(
            f'''        tensor<int32, [4]> q{prefix}_b = const()[name = string("q{prefix}_b"), val = tensor<int32, [4]>([{q_begin[0]},{q_begin[1]},{q_begin[2]},{q_begin[3]}])];
        tensor<int32, [4]> q{prefix}_e = const()[name = string("q{prefix}_e"), val = tensor<int32, [4]>([{q_end[0]},{q_end[1]},{q_end[2]},{q_end[3]}])];
        tensor<int32, [4]> k{prefix}_b = const()[name = string("k{prefix}_b"), val = tensor<int32, [4]>([{kv_begin[0]},{kv_begin[1]},{kv_begin[2]},{kv_begin[3]}])];
        tensor<int32, [4]> k{prefix}_e = const()[name = string("k{prefix}_e"), val = tensor<int32, [4]>([{kv_end[0]},{kv_end[1]},{kv_end[2]},{kv_end[3]}])];
        tensor<fp16, [1, 1, 1, {head_dim}]> q{prefix} =
            slice_by_index(begin = q{prefix}_b, end = q{prefix}_e, x = a)[name = string("q{prefix}")];
        tensor<fp16, [1, 1, {prefix}, {head_dim}]> k{prefix} =
            slice_by_index(begin = k{prefix}_b, end = k{prefix}_e, x = b)[name = string("k{prefix}")];
        tensor<fp16, [1, 1, {prefix}, {head_dim}]> v{prefix} =
            slice_by_index(begin = k{prefix}_b, end = k{prefix}_e, x = c)[name = string("v{prefix}")];
        bool s{prefix}_tx = const()[name = string("s{prefix}_tx"), val = bool(false)];
        bool s{prefix}_ty = const()[name = string("s{prefix}_ty"), val = bool(true)];
        int32 p{prefix}_ax = const()[name = string("p{prefix}_ax"), val = int32(-1)];
        fp16 p{prefix}_sc = const()[name = string("p{prefix}_sc"), val = fp16({scale:.8f})];
        tensor<fp16, [1, 1, 1, {prefix}]> s{prefix} =
            matmul(transpose_x = s{prefix}_tx, transpose_y = s{prefix}_ty, x = q{prefix}, y = k{prefix})[name = string("s{prefix}")];
        tensor<fp16, [1, 1, 1, {prefix}]> u{prefix} =
            mul(x = s{prefix}, y = p{prefix}_sc)[name = string("u{prefix}")];
        tensor<fp16, [1, 1, 1, {prefix}]> p{prefix} =
            softmax(axis = p{prefix}_ax, x = u{prefix})[name = string("p{prefix}")];
        tensor<fp16, [1, 1, 1, {head_dim}]> {out_name} =
            matmul(transpose_x = s{prefix}_tx, transpose_y = s{prefix}_tx, x = p{prefix}, y = v{prefix})[name = string("{out_name}")];'''
        )

    return f'''program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}, {{"coremlc-version", "3505.4.1"}}, {{"coremltools-component-milinternal", ""}}, {{"coremltools-version", "9.0"}}}})]
{{
    func main<ios18>(
        tensor<fp16, [1, 1, {seq_len}, {head_dim}]> a,
        tensor<fp16, [1, 1, {seq_len}, {head_dim}]> b,
        tensor<fp16, [1, 1, {seq_len}, {head_dim}]> c
    ) {{
{chr(10).join(bodies)}
    }} -> ({", ".join(outputs)});
}}
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--token-limit", type=int, default=4)
    args = parser.parse_args()

    model, tokenizer = load("mlx-community/JOSIE-1.1-4B-Instruct-4bit", lazy=True)
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    token_ids = tokenizer.encode(PROMPT)[: args.token_limit]
    tokens = mx.array([token_ids], dtype=mx.int32)
    layer = model.model.layers[0]

    h = model.model.embed_tokens(tokens).astype(mx.float16)
    mx.eval(h)
    mx.synchronize()
    x = layer.input_layernorm(h)
    q = layer.self_attn.q_proj(x)
    k = layer.self_attn.k_proj(x)
    v = layer.self_attn.v_proj(x)
    q = layer.self_attn.q_norm(q.reshape(1, args.token_limit, num_heads, -1)).transpose(0, 2, 1, 3)
    k = layer.self_attn.k_norm(k.reshape(1, args.token_limit, num_kv_heads, -1)).transpose(0, 2, 1, 3)
    v = v.reshape(1, args.token_limit, num_kv_heads, -1).transpose(0, 2, 1, 3)
    q = layer.self_attn.rope(q)
    k = layer.self_attn.rope(k)
    k = repeat_kv(k, num_heads, num_kv_heads, args.token_limit, head_dim)
    v = repeat_kv(v, num_heads, num_kv_heads, args.token_limit, head_dim)
    q_np = np.array(q, copy=False).astype(np.float16, copy=False)
    k_np = np.array(k, copy=False).astype(np.float16, copy=False)
    v_np = np.array(v, copy=False).astype(np.float16, copy=False)

    work_dir = None
    proc = None
    try:
        output_bytes = [head_dim * 2] * max(0, args.token_limit - 1)
        work_dir, proc = build_multi_server(
            make_ctx_prefix_group_mil(args.token_limit, head_dim),
            [args.token_limit * head_dim * 2] * 3,
            output_bytes,
            prefix="josie_prefix_group_",
        )
        reports = []
        for head_index in range(num_heads):
            qh = q_np[:, head_index : head_index + 1, :, :]
            kh = k_np[:, head_index : head_index + 1, :, :]
            vh = v_np[:, head_index : head_index + 1, :, :]
            blobs = run_multi_server(
                proc,
                [qh.tobytes(), kh.tobytes(), vh.tobytes()],
                output_bytes,
            )
            head_reports = []
            for prefix, blob in enumerate(blobs, start=2):
                ref = mx.fast.scaled_dot_product_attention(
                    q[:, head_index : head_index + 1, prefix - 1 : prefix, :],
                    k[:, head_index : head_index + 1, :prefix, :],
                    v[:, head_index : head_index + 1, :prefix, :],
                    scale=head_dim ** -0.5,
                )
                mx.eval(ref)
                mx.synchronize()
                ref_np = np.array(ref, copy=False).astype(np.float32)
                got = np.frombuffer(blob, dtype=np.float16).astype(np.float32).reshape(1, 1, 1, head_dim)
                diffs = np.abs(got - ref_np)
                head_reports.append(
                    {
                        "prefix": prefix,
                        "max_abs_diff": float(np.max(diffs)),
                        "mean_abs_diff": float(np.mean(diffs)),
                    }
                )
            reports.append({"head": head_index, "prefixes": head_reports})
    finally:
        if work_dir is not None and proc is not None:
            close_server(work_dir, proc)

    payload = {"runtime": "josie_private_ane_prefix_group_probe", "reports": reports}
    if args.json:
        print(json.dumps(payload))
        return
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
