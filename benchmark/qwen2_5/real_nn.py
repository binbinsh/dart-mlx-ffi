from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download
import mlx.core as mx
from mlx_lm import load


MODELS = [
    {
        "name": "qwen25_05b",
        "model_id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    },
    {
        "name": "qwen25_15b",
        "model_id": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    },
    {
        "name": "qwen25_3b",
        "model_id": "mlx-community/Qwen2.5-3B-Instruct-4bit",
    },
]

PROMPT = (
    "Summarize why MLX on Apple Silicon is useful for local inference, "
    "and mention memory efficiency and developer ergonomics."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--model-name", default=None)
    parser.add_argument(
        "--out-dir",
        default="benchmark/out/real_nn",
        help="Directory for manifest and Python benchmark results.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_models: list[dict[str, Any]] = []
    python_models: list[dict[str, Any]] = []

    selected = [
        item for item in MODELS if args.model_name is None or item["name"] == args.model_name
    ]
    for item in selected:
        manifest_entry, python_entry = benchmark_model(
            item["name"],
            item["model_id"],
            seq_len=args.seq_len,
            warmup=args.warmup,
            iters=args.iters,
        )
        manifest_models.append(manifest_entry)
        python_models.append(python_entry)

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompt": PROMPT,
        "seq_len": args.seq_len,
        "models": manifest_models,
    }
    python_results = {
        "runtime": "python_mlx_lm",
        "mlx_version": getattr(mx, "__version__", "unknown"),
        "device": str(mx.default_device()),
        "metal": bool(mx.metal.is_available()),
        "warmup": args.warmup,
        "iters": args.iters,
        "prompt": PROMPT,
        "seq_len": args.seq_len,
        "models": python_models,
    }

    manifest_path = out_dir / "real_manifest.json"
    python_path = out_dir / "real_python.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    python_path.write_text(json.dumps(python_results, indent=2), encoding="utf-8")

    payload = {
        "manifest_path": str(manifest_path),
        "python_results_path": str(python_path),
        "manifest": manifest,
        "python": python_results,
    }
    if args.json:
        print(json.dumps(payload))
        return

    print(f"manifest_path: {manifest_path}")
    print(f"python_results_path: {python_path}")
    for model in python_models:
        print()
        print(f"{model['name']}: {model['model_id']}")
        print(f"  token_count: {model['token_count']}")
        print(f"  output_shape: {model['output_shape']}")
        print(f"  output_preview: {model['output_preview']}")
        print(f"  per_iter_ms: {model['per_iter_ms']:.4f}")
        print(f"  peak_bytes_delta: {model['peak_bytes_delta']}")


def benchmark_model(
    name: str,
    model_id: str,
    *,
    seq_len: int,
    warmup: int,
    iters: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    snapshot_path = Path(snapshot_download(model_id))
    model, tokenizer, config = load(str(snapshot_path), return_config=True)
    token_ids = tokenizer.encode(PROMPT)[:seq_len]
    x = mx.array(token_ids, dtype=mx.int32)[None, :]

    def forward(tokens):
        logits = model(tokens)
        return logits[:, -1, :16].astype(mx.float32)

    for _ in range(warmup):
        y = forward(x)
        mx.eval(y)
        mx.synchronize()

    mx.reset_peak_memory()
    before_peak = int(mx.get_peak_memory())
    start = time.perf_counter()
    y = None
    for _ in range(iters):
        y = forward(x)
        mx.eval(y)
        mx.synchronize()
    total_ms = (time.perf_counter() - start) * 1000.0
    assert y is not None
    output_list = [float(v) for v in y.tolist()[0]]
    after_peak = int(mx.get_peak_memory())

    param_count = extract_param_count(config)
    repo_size = repo_size_hint(model_id)
    safetensors_files = sorted(path.name for path in snapshot_path.glob("*.safetensors"))
    config_basics = config_subset(config)

    manifest_entry = {
        "name": name,
        "model_id": model_id,
        "snapshot_path": str(snapshot_path.resolve()),
        "config": config_basics,
        "safetensors_files": safetensors_files,
        "tokens": token_ids,
        "input_shape": [1, len(token_ids)],
        "expected_output_shape": list(y.shape),
        "python_output": output_list,
    }
    python_entry = {
        "name": name,
        "model_id": model_id,
        "token_count": len(token_ids),
        "input_shape": [1, len(token_ids)],
        "output_shape": list(y.shape),
        "output_preview": output_list[:8],
        "output_values": output_list,
        "total_ms": total_ms,
        "per_iter_ms": total_ms / iters,
        "peak_bytes_delta": after_peak - before_peak,
        "config_param_count": param_count,
        "repo_size_bytes": repo_size,
        "snapshot_path": str(snapshot_path.resolve()),
        "safetensors_files": safetensors_files,
        "config": config_basics,
    }

    del y
    del x
    del config
    del tokenizer
    del model
    gc.collect()
    return manifest_entry, python_entry


def extract_param_count(config: dict[str, Any]) -> int | None:
    for key in ("n_parameters", "num_parameters", "parameter_count"):
        value = config.get(key)
        if isinstance(value, int):
            return value
    return None


def repo_size_hint(model_id: str) -> int | None:
    try:
        info = HfApi().model_info(model_id, files_metadata=True)
        return int(sum((s.size or 0) for s in info.siblings))
    except Exception:
        return None


def config_subset(config: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "rms_norm_eps",
        "vocab_size",
        "max_position_embeddings",
        "rope_theta",
        "rope_scaling",
        "tie_word_embeddings",
        "quantization",
    )
    return {key: config.get(key) for key in keys}


if __name__ == "__main__":
    main()
