from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from pathlib import Path

try:
    from .private_attn_post_runtime import build_attn_post_runtimes, close_attn_post_runtimes
    from .private_full_attn_infer import (
        DEFAULT_PROMPT,
        baseline_forward_trace,
        generate_baseline_tokens,
        hybrid_forward_trace,
        parse_attn_layers,
    )
    from .private_sdpa_runtime import PrivateSdpaRuntime
    from ..common import add_vendor_to_path, cleanup_mlx, resolve_model_path
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from private_attn_post_runtime import build_attn_post_runtimes, close_attn_post_runtimes
    from private_full_attn_infer import (
        DEFAULT_PROMPT,
        baseline_forward_trace,
        generate_baseline_tokens,
        hybrid_forward_trace,
        parse_attn_layers,
    )
    from private_sdpa_runtime import PrivateSdpaRuntime
    from common import add_vendor_to_path, cleanup_mlx, resolve_model_path

add_vendor_to_path("mlx-vlm")

import mlx.core as mx

from mlx_vlm.tokenizer_utils import load_tokenizer
from mlx_vlm.utils import get_model_path, load_model


MODEL_ID = "mlx-community/Qwen3.5-0.8B-4bit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--token-limit", type=int, default=8)
    parser.add_argument("--decode-step", type=int, default=0)
    parser.add_argument("--attn-layers")
    parser.add_argument("--q-scale", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(MODEL_ID, get_model_path)
    model = load_model(model_path, lazy=True)
    tokenizer = load_tokenizer(model_path)
    token_ids = tokenizer.encode(args.prompt)[: args.token_limit]
    if args.decode_step > 0:
        token_ids = generate_baseline_tokens(model, token_ids, args.decode_step)
    attn_layers = parse_attn_layers(args.attn_layers, model)
    max_seq_len = len(token_ids)
    sdpa_runtime = PrivateSdpaRuntime.build(
        max_seq_len=max_seq_len,
        num_heads=model.language_model.model.layers[3].self_attn.num_attention_heads,
        head_dim=model.language_model.model.layers[3].self_attn.head_dim,
    )
    attn_post_runtimes = build_attn_post_runtimes(
        model,
        lane=max_seq_len,
        attn_layers=attn_layers,
    )

    try:
        baseline = baseline_forward_trace(model, token_ids)
        hybrid = hybrid_forward_trace(
            model,
            token_ids,
            sdpa_runtime,
            attn_layers=attn_layers,
            attn_post_runtimes=attn_post_runtimes,
            q_scale_extra=args.q_scale,
        )
    finally:
        close_attn_post_runtimes(attn_post_runtimes)
        sdpa_runtime.close()
        cleanup_mlx(mx)

    layer_reports = []
    for base_layer, hybrid_layer in zip(baseline["layers"], hybrid["layers"]):
        base_hidden = base_layer["hidden"]
        hybrid_hidden = hybrid_layer["hidden"]
        diffs = np.abs(hybrid_hidden - base_hidden)
        layer_reports.append(
            {
                "layer": int(base_layer["layer"]),
                "is_linear": bool(base_layer["is_linear"]),
                "attn_replaced": bool(hybrid_layer.get("attn_replaced", False)),
                "max_abs_diff": float(np.max(diffs)),
                "mean_abs_diff": float(np.mean(diffs)),
            }
        )

    logits_diff = np.abs(hybrid["logits"] - baseline["logits"])
    report = {
        "runtime": "qwen35_private_full_attn_trace",
        "model_id": MODEL_ID,
        "prompt": args.prompt,
        "trace_token_ids": token_ids,
        "decode_step": args.decode_step,
        "attn_layers": sorted(attn_layers),
        "q_scale": args.q_scale,
        "baseline_argmax": int(np.argmax(baseline["logits"])),
        "hybrid_argmax": int(np.argmax(hybrid["logits"])),
        "argmax_match": int(np.argmax(baseline["logits"]))
        == int(np.argmax(hybrid["logits"])),
        "logits_max_abs_diff": float(np.max(logits_diff)),
        "logits_mean_abs_diff": float(np.mean(logits_diff)),
        "layers": layer_reports,
    }

    if args.json:
        print(json.dumps(report))
        return
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
