from __future__ import annotations

import json
from pathlib import Path
import sys

import mlx.core as mx

try:
    from .common import benchmark_dart_export, compare_lists, slug
    from .parakeet_tdt_sweep import asr_bench
    from .text_export_sweep import benchmark_python as benchmark_text_python
    from .text_export_sweep import export_model as export_text_model
    from .vlm_export_sweep import export_model as export_vlm_model
    from .vlm_export_sweep import python_forward as benchmark_vlm_python
    from .tts_export_sweep import export_model as export_tts_model
    from .tts_export_sweep import python_forward as benchmark_tts_python
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import benchmark_dart_export, compare_lists, slug
    from parakeet_tdt_sweep import asr_bench
    from text_export_sweep import benchmark_python as benchmark_text_python
    from text_export_sweep import export_model as export_text_model
    from vlm_export_sweep import export_model as export_vlm_model
    from vlm_export_sweep import python_forward as benchmark_vlm_python
    from tts_export_sweep import export_model as export_tts_model
    from tts_export_sweep import python_forward as benchmark_tts_python

def run_text(model_id: str, root: Path) -> dict[str, object]:
    token_ids, py_values, py_ms = benchmark_text_python(model_id)
    input_path = export_text_model(model_id, token_ids, root)
    dart_values, dart_ms = benchmark_dart_export(
        export_path=root / "function.mlxfn",
        input_path=input_path,
        mx_module=mx,
    )
    return {
        "token_count": len(token_ids),
        "python_ms": py_ms,
        "dart_ms": dart_ms,
        **{
            "max_abs_diff": compare_lists(py_values, dart_values)[0],
            "mean_abs_diff": compare_lists(py_values, dart_values)[1],
        },
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
    dart_values, dart_ms = benchmark_dart_export(
        export_path=export_path,
        input_path=input_path,
        input_names=input_names,
        mx_module=mx,
    )
    return {
        "input_names": input_names,
        "python_ms": py_ms,
        "dart_ms": dart_ms,
        **{
            "max_abs_diff": compare_lists(py_values, dart_values)[0],
            "mean_abs_diff": compare_lists(py_values, dart_values)[1],
        },
        "python_preview": py_values[:8],
        "dart_preview": dart_values[:8],
    }


def run_tts(model_id: str, root: Path) -> dict[str, object]:
    _model, _inputs, py_values, py_ms = benchmark_tts_python(model_id)
    export_path, input_path, input_names = export_tts_model(model_id, root)
    dart_values, dart_ms = benchmark_dart_export(
        export_path=export_path,
        input_path=input_path,
        input_names=input_names,
        mx_module=mx,
    )
    return {
        "input_names": input_names,
        "python_ms": py_ms,
        "dart_ms": dart_ms,
        **{
            "max_abs_diff": compare_lists(py_values, dart_values)[0],
            "mean_abs_diff": compare_lists(py_values, dart_values)[1],
        },
        "python_preview": py_values[:8],
        "dart_preview": dart_values[:8],
    }


def main() -> None:
    models = json.loads(Path("benchmark/publish_model_list.json").read_text(encoding="utf-8"))
    results = []
    for item in models[:10]:
        model_id = item["model_id"]
        kind = item["kind"]
        root = Path("benchmark/out/model_list") / slug(model_id)
        if kind == "text":
            extra = run_text(model_id, root)
        elif kind == "vlm":
            extra = run_vlm(model_id, root)
        elif kind == "tts":
            extra = run_tts(model_id, root)
        elif kind == "asr":
            extra = asr_bench(model_id)
        else:
            extra = {"status": "pending", "note": f"Unsupported kind: {kind}"}
        record = {**item, **extra}
        results.append(record)
        print(json.dumps(record, ensure_ascii=False))

    report_path = Path("benchmark/out/model_list/report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"report_path={report_path}")


if __name__ == "__main__":
    main()
