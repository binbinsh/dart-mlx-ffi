from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import hf_hub_download, snapshot_download


DEFAULT_FALLBACK_ENDPOINT = "https://hf-mirror.com"
ROOT = Path(__file__).resolve().parents[3]


def snapshot_download_with_fallback(
    *,
    repo_id: str,
    endpoint: str | None = None,
    fallback_endpoint: str | None = None,
    attempts_per_endpoint: int | None = None,
    retry_backoff_seconds: float | None = None,
    **kwargs: Any,
) -> str:
    return str(
        _download_with_fallback(
            snapshot_download,
            repo_id=repo_id,
            endpoint=endpoint,
            fallback_endpoint=fallback_endpoint,
            attempts_per_endpoint=attempts_per_endpoint,
            retry_backoff_seconds=retry_backoff_seconds,
            **kwargs,
        )
    )


def hf_hub_download_with_fallback(
    *,
    repo_id: str,
    filename: str,
    endpoint: str | None = None,
    fallback_endpoint: str | None = None,
    attempts_per_endpoint: int | None = None,
    retry_backoff_seconds: float | None = None,
    **kwargs: Any,
) -> str:
    return str(
        _download_with_fallback(
            hf_hub_download,
            repo_id=repo_id,
            filename=filename,
            endpoint=endpoint,
            fallback_endpoint=fallback_endpoint,
            attempts_per_endpoint=attempts_per_endpoint,
            retry_backoff_seconds=retry_backoff_seconds,
            **kwargs,
        )
    )


def _download_with_fallback(
    fn: Callable[..., Any],
    *,
    endpoint: str | None,
    fallback_endpoint: str | None,
    attempts_per_endpoint: int | None,
    retry_backoff_seconds: float | None,
    **kwargs: Any,
) -> Any:
    _ensure_default_cache(kwargs)
    attempts = []
    first_error: Exception | None = None
    max_attempts = _retry_count(attempts_per_endpoint)
    backoff = _retry_backoff(retry_backoff_seconds)
    for candidate in _download_endpoints(endpoint, fallback_endpoint):
        for attempt in range(1, max_attempts + 1):
            try:
                return fn(endpoint=candidate, **kwargs)
            except Exception as error:  # noqa: BLE001 - preserve all attempts.
                if first_error is None:
                    first_error = error
                attempts.append(
                    f"{candidate or 'default'}[{attempt}/{max_attempts}]: "
                    f"{type(error).__name__}: {error}"
                )
                if attempt < max_attempts and backoff > 0:
                    time.sleep(backoff * attempt)
    message = "HF download failed across endpoints: " + " | ".join(attempts)
    raise RuntimeError(message) from first_error


def _ensure_default_cache(kwargs: dict[str, Any]) -> None:
    if kwargs.get("cache_dir"):
        Path(str(kwargs["cache_dir"])).expanduser().mkdir(parents=True, exist_ok=True)
        return
    if os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return
    cache_dir = ROOT / "benchmark" / ".hf_home" / "hub"
    cache_dir.mkdir(parents=True, exist_ok=True)
    kwargs["cache_dir"] = str(cache_dir)


def _download_endpoints(
    endpoint: str | None,
    fallback_endpoint: str | None,
) -> list[str | None]:
    primary = _clean_endpoint(endpoint or os.environ.get("HF_ENDPOINT"))
    result: list[str | None] = [primary]
    fallback_raw = (
        fallback_endpoint
        if fallback_endpoint is not None
        else os.environ.get("HF_FALLBACK_ENDPOINT") or DEFAULT_FALLBACK_ENDPOINT
    )
    for candidate in _split_endpoints(fallback_raw):
        cleaned = _clean_endpoint(candidate)
        if cleaned == primary or cleaned in result:
            continue
        result.append(cleaned)
    return result


def _split_endpoints(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _clean_endpoint(endpoint: str | None) -> str | None:
    text = (endpoint or "").strip()
    return text.rstrip("/") if text else None


def _retry_count(explicit: int | None) -> int:
    if explicit is not None:
        return max(1, explicit)
    raw = os.environ.get("HF_DOWNLOAD_RETRIES")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return 3


def _retry_backoff(explicit: float | None) -> float:
    if explicit is not None:
        return max(0.0, explicit)
    raw = os.environ.get("HF_DOWNLOAD_RETRY_BACKOFF_SECONDS")
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    return 0.5
