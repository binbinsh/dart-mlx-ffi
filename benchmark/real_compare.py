from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="benchmark/out/real_nn/real_python.json")
    parser.add_argument("--dart", default="benchmark/out/real_nn/real_dart.json")
    parser.add_argument("--output", default="benchmark/out/real_nn/real_compare.json")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    python = load_json(args.python)
    dart = load_json(args.dart)
    report = compare(python, dart)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report))
        return

    print(f"compare_path: {output_path}")
    for model in report["models"]:
        print()
        print(f"{model['name']}: {model['model_id']}")
        print(f"  max_abs_diff: {model['max_abs_diff']:.9f}")
        print(f"  mean_abs_diff: {model['mean_abs_diff']:.9f}")
        print(f"  dart_per_iter_ms: {model['dart_per_iter_ms']:.4f}")
        print(f"  python_per_iter_ms: {model['python_per_iter_ms']:.4f}")
        print(f"  dart_speedup_vs_python: {model['dart_speedup_vs_python']:.3f}x")


def load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare(python: dict[str, Any], dart: dict[str, Any]) -> dict[str, Any]:
    py_models = {model["name"]: model for model in python["models"]}
    dart_models = {model["name"]: model for model in dart["models"]}

    models: list[dict[str, Any]] = []
    for name in py_models.keys():
        py = py_models[name]
        da = dart_models[name]
        py_values = [float(v) for v in py["output_values"]]
        da_values = [float(v) for v in da["output_values"]]
        diffs = [abs(a - b) for a, b in zip(py_values, da_values)]
        max_abs = max(diffs) if diffs else 0.0
        mean_abs = sum(diffs) / len(diffs) if diffs else 0.0
        speedup = py["per_iter_ms"] / da["per_iter_ms"] if da["per_iter_ms"] else None
        models.append(
            {
                "name": name,
                "model_id": py["model_id"],
                "token_count": py["token_count"],
                "snapshot_path": py.get("snapshot_path"),
                "safetensors_files": py.get("safetensors_files"),
                "config": py.get("config"),
                "repo_size_bytes": py.get("repo_size_bytes"),
                "config_param_count": py.get("config_param_count"),
                "output_shape": py["output_shape"],
                "python_output_preview": py["output_preview"],
                "dart_output_preview": da["output_preview"],
                "python_per_iter_ms": py["per_iter_ms"],
                "dart_per_iter_ms": da["per_iter_ms"],
                "dart_speedup_vs_python": speedup,
                "python_peak_bytes_delta": py["peak_bytes_delta"],
                "dart_peak_bytes_delta": da["peak_bytes_delta"],
                "max_abs_diff": max_abs,
                "mean_abs_diff": mean_abs,
            }
        )

    return {
        "python": {
            "runtime": python["runtime"],
            "mlx_version": python["mlx_version"],
            "device": python["device"],
            "metal": python["metal"],
        },
        "dart": {
            "runtime": dart["runtime"],
            "mlx_version": dart["mlx_version"],
            "device": dart["device"],
            "metal": dart["metal"],
        },
        "warmup": python["warmup"],
        "iters": python["iters"],
        "prompt": python["prompt"],
        "seq_len": python["seq_len"],
        "models": models,
    }


if __name__ == "__main__":
    main()
