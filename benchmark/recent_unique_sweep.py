from __future__ import annotations

import json
from pathlib import Path
import sys

try:
    from .recent_text_sweep import benchmark_python as benchmark_text_python
    from .recent_text_sweep import benchmark_dart as benchmark_text_dart
    from .recent_text_sweep import export_model as export_text_model
    from .recent_text_sweep import slug
    from .recent_vlm_sweep import dart_forward as benchmark_vlm_dart
    from .recent_vlm_sweep import export_model as export_vlm_model
    from .recent_vlm_sweep import python_forward as benchmark_vlm_python
    from .recent_tts_sweep import dart_forward as benchmark_tts_dart
    from .recent_tts_sweep import export_model as export_tts_model
    from .recent_tts_sweep import python_forward as benchmark_tts_python
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from recent_text_sweep import benchmark_python as benchmark_text_python
    from recent_text_sweep import benchmark_dart as benchmark_text_dart
    from recent_text_sweep import export_model as export_text_model
    from recent_text_sweep import slug
    from recent_vlm_sweep import dart_forward as benchmark_vlm_dart
    from recent_vlm_sweep import export_model as export_vlm_model
    from recent_vlm_sweep import python_forward as benchmark_vlm_python
    from recent_tts_sweep import dart_forward as benchmark_tts_dart
    from recent_tts_sweep import export_model as export_tts_model
    from recent_tts_sweep import python_forward as benchmark_tts_python


def compare_lists(a: list[float], b: list[float]) -> dict[str, float]:
    diffs = [abs(x - y) for x, y in zip(a, b)]
    return {
        "max_abs_diff": max(diffs) if diffs else 0.0,
        "mean_abs_diff": (sum(diffs) / len(diffs)) if diffs else 0.0,
    }


def run_text(model_id: str, root: Path) -> dict[str, object]:
    token_ids, py_values, py_ms = benchmark_text_python(model_id)
    input_path = export_text_model(model_id, token_ids, root)
    dart_values, dart_ms = benchmark_text_dart(root / "function.mlxfn", input_path)
    return {
        "token_count": len(token_ids),
        "python_ms": py_ms,
        "dart_ms": dart_ms,
        **compare_lists(py_values, dart_values),
        "python_preview": py_values[:8],
        "dart_preview": dart_values[:8],
    }


def run_vlm(model_id: str, root: Path) -> dict[str, object]:
    (
        _model,
        _processor,
        _inputs,
        input_names,
        py_values,
        py_ms,
    ) = benchmark_vlm_python(model_id)
    export_path, input_path, input_names = export_vlm_model(model_id, root)
    dart_values, dart_ms = benchmark_vlm_dart(export_path, input_path, input_names)
    return {
        "input_names": input_names,
        "python_ms": py_ms,
        "dart_ms": dart_ms,
        **compare_lists(py_values, dart_values),
        "python_preview": py_values[:8],
        "dart_preview": dart_values[:8],
    }


def run_tts(model_id: str, root: Path) -> dict[str, object]:
    _model, _inputs, py_values, py_ms = benchmark_tts_python(model_id)
    export_path, input_path, input_names = export_tts_model(model_id, root)
    dart_values, dart_ms = benchmark_tts_dart(export_path, input_path, input_names)
    return {
        "input_names": input_names,
        "python_ms": py_ms,
        "dart_ms": dart_ms,
        **compare_lists(py_values, dart_values),
        "python_preview": py_values[:8],
        "dart_preview": dart_values[:8],
    }


def main() -> None:
    models = json.loads(Path("benchmark/recent_unique_models.json").read_text(encoding="utf-8"))
    results = []
    for item in models[:10]:
        model_id = item["model_id"]
        kind = item["kind"]
        root = Path("benchmark/out/recent_unique") / slug(model_id)
        if kind == "text":
            extra = run_text(model_id, root)
        elif kind == "vlm":
            extra = run_vlm(model_id, root)
        elif kind == "tts":
            extra = run_tts(model_id, root)
        else:
            extra = {"status": "pending", "note": f"Unsupported kind: {kind}"}
        record = {**item, **extra}
        results.append(record)
        print(json.dumps(record, ensure_ascii=False))

    report_path = Path("benchmark/out/recent_unique/report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"report_path={report_path}")


if __name__ == "__main__":
    main()
