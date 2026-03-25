from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
CACHE_DIR = ROOT / ".dart_tool" / "qwen3_5_private_ane"
CACHE_FILE = CACHE_DIR / "probe_cache.json"


def _load() -> dict[str, bool]:
    if not CACHE_FILE.exists():
        return {}
    try:
        data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): bool(v) for k, v in data.items()}


_CACHE = _load()


def _persist() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(_CACHE, sort_keys=True), encoding="utf-8")
    tmp.replace(CACHE_FILE)


def make_key(kind: str, **kwargs) -> str:
    payload = {"kind": kind, **kwargs}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def get(key: str) -> bool | None:
    value = _CACHE.get(key)
    if value is None:
        return None
    return bool(value)


def set(key: str, value: bool) -> None:
    current = _CACHE.get(key)
    if current is value:
        return
    _CACHE[key] = bool(value)
    _persist()
