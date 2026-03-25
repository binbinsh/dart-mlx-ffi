from __future__ import annotations

import json
from pathlib import Path
import sys
import time

from huggingface_hub import HfApi

try:
    from .common import benchmark_dart_export, cleanup_mlx, slug
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import benchmark_dart_export, cleanup_mlx, slug

import mlx.core as mx
from mlx_lm import load

PROMPT = "Explain why MLX is useful for local inference on Apple Silicon."


def extract_logits(output):
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, tuple):
        return output[0]
    return output


def latest_text_models(limit: int = 10) -> list[dict[str, object]]:
    api = HfApi()
    models = list(api.list_models(author="mlx-community", sort="lastModified", limit=30, full=True))
    selected = []
    for model in models:
        if getattr(model, "pipeline_tag", None) != "text-generation":
            continue
        selected.append(
            {
                "model_id": model.id,
                "last_modified": str(model.last_modified),
                "pipeline_tag": model.pipeline_tag,
            }
        )
        if len(selected) == limit:
            break
    return selected


def benchmark_python(model_id: str) -> tuple[list[int], list[float], float]:
    model, tokenizer = load(model_id, lazy=False)
    token_ids = tokenizer.encode(PROMPT)[:24]
    tokens = mx.array([token_ids], dtype=mx.int32)

    def forward():
        output = extract_logits(model(tokens))
        logits = output[:, -1, :16].astype(mx.float32)
        mx.eval(logits)
        mx.synchronize()
        return logits

    started = time.perf_counter()
    logits = forward()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    values = [float(v) for v in logits.reshape([-1]).tolist()]
    del model, tokenizer
    cleanup_mlx(mx)
    return token_ids, values, elapsed_ms


def export_model(model_id: str, token_ids: list[int], export_dir: Path) -> Path:
    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = export_dir / "function.mlxfn"
    input_path = export_dir / "inputs.safetensors"
    if export_path.exists() and input_path.exists():
        return input_path

    model, _tokenizer = load(model_id, lazy=False)
    tokens = mx.array([token_ids], dtype=mx.int32)

    def forward(input_ids):
        output = extract_logits(model(input_ids))
        return output[:, -1, :16].astype(mx.float32)

    if export_path.exists():
        export_path.unlink()
    mx.export_function(str(export_path), forward, tokens)
    mx.save_safetensors(str(input_path), {"input_ids": tokens})
    del model, _tokenizer
    cleanup_mlx(mx)
    return input_path


def main() -> None:
    models = latest_text_models(limit=10)
    root = Path("benchmark/out/text_export")
    results = []
    for item in models:
        model_id = item["model_id"]
        export_dir = root / slug(model_id)
        token_ids, py_values, py_ms = benchmark_python(model_id)
        input_path = export_model(model_id, token_ids, export_dir)
        dart_values, dart_ms = benchmark_dart_export(
            export_path=export_dir / "function.mlxfn",
            input_path=input_path,
            mx_module=mx,
        )
        diffs = [abs(a - b) for a, b in zip(py_values, dart_values)]
        results.append(
            {
                **item,
                "token_count": len(token_ids),
                "python_ms": py_ms,
                "dart_ms": dart_ms,
                "max_abs_diff": max(diffs) if diffs else 0.0,
                "mean_abs_diff": (sum(diffs) / len(diffs)) if diffs else 0.0,
                "python_preview": py_values[:8],
                "dart_preview": dart_values[:8],
            }
        )
        print(json.dumps(results[-1], ensure_ascii=False))

    report_path = root / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"report_path={report_path}")


if __name__ == "__main__":
    main()
