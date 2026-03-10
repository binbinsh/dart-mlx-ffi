from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dart-json", required=True)
    parser.add_argument("--python-json", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    dart = load_json(args.dart_json)
    python = load_json(args.python_json)
    report = compare_reports(dart, python)

    if args.json:
        print(json.dumps(report))
        return

    print("dart_mlx_ffi vs python-mlx")
    print(
        f"Dart MLX: {report['dart']['mlx_version']} | "
        f"Python MLX: {report['python']['mlx_version']}"
    )
    print(
        f"Dart device: {report['dart']['device']} | "
        f"Python device: {report['python']['device']}"
    )
    if report["version_note"]:
        print(f"Version note: {report['version_note']}")
    print()
    print(
        "model            max_abs_diff   mean_abs_diff   "
        "dart_ms   python_ms   dart_vs_py"
    )
    for model in report["models"]:
        print(
            f"{model['name']:<16}"
            f"{model['max_abs_diff']:<15.9f}"
            f"{model['mean_abs_diff']:<16.9f}"
            f"{model['dart_per_iter_ms']:<10.4f}"
            f"{model['python_per_iter_ms']:<12.4f}"
            f"{model['dart_speedup_vs_python']:.3f}x"
        )
    print()
    for model in report["models"]:
        print(f"{model['name']}: {model['description']}")
        print(f"  input_shapes: {model['input_shapes']}")
        print(f"  output_shape: {model['output_shape']}")
        print(f"  dart_output_preview: {model['dart_output_preview']}")
        print(f"  python_output_preview: {model['python_output_preview']}")
        print(
            f"  peak_bytes_delta: dart={model['dart_peak_bytes_delta']}, "
            f"python={model['python_peak_bytes_delta']}"
        )
def load_json(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8")) if not Path(path).is_absolute() else json.loads(Path(path).read_text(encoding="utf-8"))


def compare_reports(dart: dict[str, Any], python: dict[str, Any]) -> dict[str, Any]:
    dart_models = {model["name"]: model for model in dart["models"]}
    python_models = {model["name"]: model for model in python["models"]}
    shared_names = sorted(dart_models.keys() & python_models.keys())

    models: list[dict[str, Any]] = []
    for name in shared_names:
        d = dart_models[name]
        p = python_models[name]
        if d["output_shape"] != p["output_shape"]:
            raise RuntimeError(
                f"Output shape mismatch for {name}: "
                f"{d['output_shape']} != {p['output_shape']}"
            )

        d_flat = d["output_flat"]
        p_flat = p["output_flat"]
        if len(d_flat) != len(p_flat):
            raise RuntimeError(
                f"Output length mismatch for {name}: {len(d_flat)} != {len(p_flat)}"
            )

        diffs = [abs(float(a) - float(b)) for a, b in zip(d_flat, p_flat)]
        max_abs_diff = max(diffs) if diffs else 0.0
        mean_abs_diff = sum(diffs) / len(diffs) if diffs else 0.0
        dart_ms = float(d["per_iter_ms"])
        python_ms = float(p["per_iter_ms"])
        speedup = python_ms / dart_ms if dart_ms else math.inf

        models.append(
            {
                "name": name,
                "description": d["description"],
                "input_shapes": d["input_shapes"],
                "output_shape": d["output_shape"],
                "dart_output_preview": d["output_preview"],
                "python_output_preview": p["output_preview"],
                "dart_peak_bytes_delta": d["peak_bytes_delta"],
                "python_peak_bytes_delta": p["peak_bytes_delta"],
                "dart_per_iter_ms": dart_ms,
                "python_per_iter_ms": python_ms,
                "dart_speedup_vs_python": speedup,
                "max_abs_diff": max_abs_diff,
                "mean_abs_diff": mean_abs_diff,
                "dart_output_sum": d["output_sum"],
                "python_output_sum": p["output_sum"],
            }
        )

    version_note = None
    if dart["mlx_version"] != python["mlx_version"]:
        version_note = (
            "Bundled Dart MLX runtime and python-mlx package are on different patch "
            f"versions ({dart['mlx_version']} vs {python['mlx_version']})."
        )

    return {
        "dart": {
            "runtime": dart["runtime"],
            "mlx_version": dart["mlx_version"],
            "device": dart["device"],
            "metal": dart["metal"],
        },
        "python": {
            "runtime": python["runtime"],
            "mlx_version": python["mlx_version"],
            "device": python["device"],
            "metal": python["metal"],
        },
        "warmup": dart["warmup"],
        "iters": dart["iters"],
        "version_note": version_note,
        "models": models,
    }


if __name__ == "__main__":
    main()
