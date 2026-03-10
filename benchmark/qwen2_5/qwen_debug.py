from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask


def main() -> None:
    manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    model_name = sys.argv[2]
    spec = next(model for model in manifest["models"] if model["name"] == model_name)
    model, _ = load(spec["snapshot_path"])
    tokens = spec["tokens"]
    x = mx.array(tokens, dtype=mx.int32)[None, :]

    h = model.model.embed_tokens(x)
    dump: dict[str, list[float]] = {"embed": slice_last(h)}
    mask = create_attention_mask(h, None)

    for index, layer in enumerate(model.model.layers):
        r = layer.self_attn(layer.input_layernorm(h), mask, None)
        h1 = h + r
        dump[f"block_{index}_resid1"] = slice_last(h1)
        r2 = layer.mlp(layer.post_attention_layernorm(h1))
        h = h1 + r2
        dump[f"block_{index}_out"] = slice_last(h)

    norm = model.model.norm(h)
    dump["final_norm"] = slice_last(norm)
    logits = model(x)[:, -1, :16].astype(mx.float32)
    mx.eval(logits)
    mx.synchronize()
    dump["logits16"] = [float(v) for v in logits.tolist()[0]]
    print(json.dumps(dump))


def slice_last(h):
    y = h[:, -1, :16].astype(mx.float32)
    mx.eval(y)
    return [float(v) for v in y.tolist()[0]]


if __name__ == "__main__":
    main()
