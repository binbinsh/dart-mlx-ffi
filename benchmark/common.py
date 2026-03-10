from __future__ import annotations

import gc
import sys
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


def cleanup_mlx(mx_module) -> None:
    gc.collect()
    mx_module.clear_cache()


def _is_complete_snapshot(path: Path) -> bool:
    config = path / "config.json"
    if not config.exists():
        return False
    return any(path.glob("*.safetensors"))
