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
    temp_dir = Path(tempfile.mkdtemp())
    stdout_path = temp_dir / "runner.stdout"
    subprocess.run(
        ["script", "-q", str(stdout_path), *cmd],
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
    )
    return stdout_path.read_text(encoding="utf-8")


def parse_last_json(raw: str) -> dict[str, object]:
    matches = re.findall(r"\{.*?\}", raw, flags=re.DOTALL)
    if not matches:
        raise RuntimeError(f"No JSON payload found in output:\n{raw}")
    return json.loads(matches[-1])


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
    values_path = temp_dir / "dart_output.safetensors"
    env = dict(os.environ)
    env[values_env] = str(values_path)
    env["GENERIC_WARMUP"] = str(warmup)
    env["GENERIC_ITERS"] = str(iters)
    cmd = [
        "dart",
        "run",
        "models/common/import_run.dart",
        str(export_path),
        str(input_path),
    ]
    if input_names is not None:
        cmd.append(json.dumps(input_names))
    raw = run_script_capture(cmd, env=env)
    payload = parse_last_json(raw)
    values = [
        float(v)
        for v in mx_module.load(str(values_path))["output"]
        .reshape([-1])
        .astype(mx_module.float32)
        .tolist()
    ]
    return values, float(payload["per_iter_ms"])


def cleanup_mlx(mx_module) -> None:
    gc.collect()
    mx_module.clear_cache()


def _is_complete_snapshot(path: Path) -> bool:
    config = path / "config.json"
    if not config.exists():
        return False
    return any(path.glob("*.safetensors"))
