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


def make_score_softmax_group_mil(prefix: int, head_dim: int, heads: int) -> str:
    inputs = []
    body = []
    outputs = []
    for index in range(heads):
        q = f"a{index}"
        k = f"b{index}"
        scores = f"s{index}"
        scaled = f"u{index}"
        out = f"z{index}"
        inputs.append(f"tensor<fp16, [1, 1, 1, {head_dim}]> {q}")
        inputs.append(f"tensor<fp16, [1, 1, {prefix}, {head_dim}]> {k}")
        body.append(
            f'''        bool {scores}_tx = const()[name = string("{scores}_tx"), val = bool(false)];
        bool {scores}_ty = const()[name = string("{scores}_ty"), val = bool(true)];
        int32 {out}_ax = const()[name = string("{out}_ax"), val = int32(-1)];
        fp16 {scaled}_sc = const()[name = string("{scaled}_sc"), val = fp16({(head_dim ** -0.5):.8f})];
        tensor<fp16, [1, 1, 1, {prefix}]> {scores} =
            matmul(transpose_x = {scores}_tx, transpose_y = {scores}_ty, x = {q}, y = {k})[name = string("{scores}")];
        tensor<fp16, [1, 1, 1, {prefix}]> {scaled} =
            mul(x = {scores}, y = {scaled}_sc)[name = string("{scaled}")];
        tensor<fp16, [1, 1, 1, {prefix}]> {out} =
            softmax(axis = {out}_ax, x = {scaled})[name = string("{out}")];'''
        )
        outputs.append(out)

    return f'''program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}, {{"coremlc-version", "3505.4.1"}}, {{"coremltools-component-milinternal", ""}}, {{"coremltools-version", "9.0"}}}})]
{{
    func main<ios18>(
        {",\n        ".join(inputs)}
    ) {{
{chr(10).join(body)}
    }} -> ({", ".join(outputs)});
}}
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--group-heads", type=int, default=2)
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
    q = layer.self_attn.q_norm(q.reshape(1, args.token_limit, num_heads, -1)).transpose(0, 2, 1, 3)
    k = layer.self_attn.k_norm(k.reshape(1, args.token_limit, num_kv_heads, -1)).transpose(0, 2, 1, 3)
    q = layer.self_attn.rope(q)
    k = layer.self_attn.rope(k)
    k = repeat_kv(k, num_heads, num_kv_heads, args.token_limit, head_dim)
    q_np = np.array(q, copy=False).astype(np.float16, copy=False)
    k_np = np.array(k, copy=False).astype(np.float16, copy=False)

    reports = []
    for prefix in range(2, args.token_limit + 1):
        work_dir = None
        proc = None
        try:
            work_dir, proc = build_multi_server(
                make_score_softmax_group_mil(prefix, head_dim, args.group_heads),
                [head_dim * 2, prefix * head_dim * 2] * args.group_heads,
                [prefix * 2] * args.group_heads,
                prefix=f"josie_group_probe_{prefix}_",
            )
            groups = []
            for start in range(0, num_heads, args.group_heads):
                inputs = []
                refs = []
                for offset in range(args.group_heads):
                    head = start + offset
                    qh = q_np[:, head : head + 1, prefix - 1 : prefix, :]
                    kh = k_np[:, head : head + 1, :prefix, :]
                    inputs.extend([qh.tobytes(), kh.tobytes()])
                    score = np.matmul(qh.astype(np.float32), np.swapaxes(kh.astype(np.float32), -1, -2)).astype(np.float32)
                    score *= head_dim ** -0.5
                    ref = mx.softmax(mx.array(score.astype(np.float16), dtype=mx.float16), axis=-1)
                    mx.eval(ref)
                    mx.synchronize()
                    refs.append(np.array(ref, copy=False).astype(np.float32))
                blobs = run_multi_server(proc, inputs, [prefix * 2] * args.group_heads)
                diffs = []
                for offset, blob in enumerate(blobs):
                    got = np.frombuffer(blob, dtype=np.float16).astype(np.float32).reshape(1, 1, 1, prefix)
                    diffs.append(np.abs(got - refs[offset]))
                merged = np.concatenate(diffs, axis=1)
                groups.append(
                    {
                        "start_head": start,
                        "max_abs_diff": float(np.max(merged)),
                        "mean_abs_diff": float(np.mean(merged)),
                    }
                )
            reports.append({"prefix": prefix, "groups": groups})
        finally:
            if work_dir is not None and proc is not None:
                close_server(work_dir, proc)

    payload = {
        "runtime": "josie_private_ane_head_group_probe",
        "group_heads": args.group_heads,
        "reports": reports,
    }
    if args.json:
        print(json.dumps(payload))
        return
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
