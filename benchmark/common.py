from __future__ import annotations

import gc
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
VENDORS = ROOT / "vendors"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def add_vendor_to_path(name: str) -> None:
    vendor_path = str(VENDORS / name)
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)


def resolve_model_path(model_id: str, fallback: Callable[[str], Path]) -> Path:
    cached = find_cached_snapshot(model_id)
    if cached is not None:
        return cached
    return fallback(model_id)


def find_cached_snapshot(model_id: str) -> Path | None:
    cache_dir = HF_CACHE / ("models--" + model_id.replace("/", "--"))
    snapshots_dir = cache_dir / "snapshots"
    if not snapshots_dir.exists():
        return None
    candidates = [
        path
        for path in snapshots_dir.iterdir()
        if path.is_dir() and _is_complete_snapshot(path)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def preview(values: list[float], limit: int = 8) -> list[float]:
    return values[: min(limit, len(values))]


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def run_script_capture(cmd: list[str], *, env: dict[str, str]) -> str:
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    if completed.stderr:
        return completed.stdout + completed.stderr
    return completed.stdout


def parse_last_json(raw: str) -> dict[str, object]:
    decoder = json.JSONDecoder()
    best: tuple[int, dict[str, object]] | None = None
    for match in re.finditer(r"\{", raw):
        try:
            value, end = decoder.raw_decode(raw[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            if best is None or end > best[0]:
                best = (end, value)
    if best is None:
        raise RuntimeError(f"No JSON payload found in output:\n{raw}")
    return best[1]


def compare_lists(a: list[float], b: list[float]) -> tuple[float, float]:
    diffs = [abs(x - y) for x, y in zip(a, b)]
    return (
        max(diffs) if diffs else 0.0,
        (sum(diffs) / len(diffs)) if diffs else 0.0,
    )


def benchmark_dart_export(
    *,
    export_path: Path,
    input_path: Path,
    input_names: list[str] | None = None,
    mx_module,
    warmup: int = 0,
    iters: int = 1,
    values_env: str = "GENERIC_VALUES_PATH",
) -> tuple[list[float], float]:
    temp_dir = Path(tempfile.mkdtemp())
    input_json_path = temp_dir / "inputs.json"
    _write_runtime_input_json(
        mx_module=mx_module,
        input_path=input_path,
        input_names=input_names,
        output_path=input_json_path,
    )
    env = dict(os.environ)
    env.pop(values_env, None)
    cmd = [
        "dart",
        "run",
        "benchmark/runtime/dart_runtime_runner.dart",
        "--model-id",
        "mlx_export",
        "--engine",
        "mlx",
        "--artifact",
        str(export_path),
        "--input-json",
        str(input_json_path),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--include-output-values",
    ]
    raw = run_script_capture(cmd, env=env)
    payload = parse_last_json(raw)
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError(f"Missing runtime metrics in output:\n{raw}")
    return _runtime_output_values(payload), float(metrics["end_to_end_ms"])


def _write_runtime_input_json(
    *,
    mx_module,
    input_path: Path,
    input_names: list[str] | None,
    output_path: Path,
) -> None:
    tensors = mx_module.load(str(input_path))
    names = input_names or sorted(tensors.keys())
    inputs: dict[str, dict[str, object]] = {}
    for name in names:
        array = tensors[name]
        inputs[name] = {
            "dtype": _runtime_dtype_name(array.dtype),
            "shape": [int(dim) for dim in array.shape],
            "values": array.reshape([-1]).tolist(),
        }
    output_path.write_text(
        json.dumps({"input_order": names, "inputs": inputs}),
        encoding="utf-8",
    )


def _runtime_dtype_name(dtype) -> str:
    text = str(dtype).split(".")[-1]
    aliases = {
        "bool_": "bool",
        "bool": "bool",
        "uint8": "uint8",
        "int32": "int32",
        "int64": "int64",
        "float16": "float16",
        "float32": "float32",
        "float64": "float64",
    }
    if text not in aliases:
        raise ValueError(f"Unsupported runtime input dtype: {dtype}")
    return aliases[text]


def _runtime_output_values(payload: dict[str, object]) -> list[float]:
    correctness = payload.get("correctness")
    if not isinstance(correctness, dict):
        raise RuntimeError(f"Missing runtime correctness payload: {payload}")
    output_values = correctness.get("output_values")
    if not isinstance(output_values, dict):
        raise RuntimeError(f"Runtime output values were omitted: {payload}")
    output = output_values.get("output")
    if not isinstance(output, dict):
        raise RuntimeError(f"Runtime output tensor 'output' was not produced: {payload}")
    values = output.get("values")
    if not isinstance(values, list):
        raise RuntimeError(f"Runtime output tensor has no values: {payload}")
    return [float(value) for value in values]


def cleanup_mlx(mx_module) -> None:
    gc.collect()
    mx_module.clear_cache()


def _is_complete_snapshot(path: Path) -> bool:
    config = path / "config.json"
    if not config.exists():
        return False
    return any(path.glob("*.safetensors"))
