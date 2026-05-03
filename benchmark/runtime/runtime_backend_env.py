from __future__ import annotations

import os
from typing import Any, Mapping

from litert_env import resolve_litert_environment
from ort_env import resolve_ort_environment


RUNTIME_BUILD_ENV_KEYS = (
    "DART_MLX_ENABLE_ORT",
    "DART_MLX_ORT_INCLUDE_DIR",
    "DART_MLX_ORT_LIBRARY",
    "DART_MLX_ORT_RUNTIME_LIBRARY",
    "DART_MLX_LITERT_LIBRARY",
    "DART_MLX_TFLITE_LIBRARY",
    "DART_MLX_LITERT_EXTRA_LIBRARIES",
)


def prepare_runtime_environment(
    *,
    engine: str,
    platform: str | None = None,
    target_os: str = "host",
    target_arch: str | None = None,
    base_env: Mapping[str, str] | None = None,
    fetch_dependencies: bool = True,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Resolve native runtime env vars for a backend and merge into process env."""
    env = dict(base_env or os.environ)
    metadata: dict[str, Any] = {}
    normalized_target = target_os.strip().lower()
    if normalized_target not in {"host", "android"}:
        normalized_target = "host"
    if normalized_target == "host":
        # Runtime probes and matrix runners execute on the host process even
        # when they validate Android-target artifacts.
        normalized_target = "host"

    if engine == "onnx":
        ort = resolve_ort_environment(
            fetch_headers=fetch_dependencies,
            target_os=normalized_target,
            target_arch=target_arch,
        )
        metadata["ort_env"] = ort.to_json()
        if ort.ready:
            env.update(ort.to_env())
        return env, metadata

    if engine == "litert":
        litert = resolve_litert_environment(
            fetch_library=fetch_dependencies,
            target_os=normalized_target,
            target_arch=target_arch,
        )
        metadata["litert_env"] = litert.to_json()
        if litert.ready:
            env.update(litert.to_env())
        return env, metadata

    return env, metadata


def runtime_build_env_values(env: Mapping[str, str]) -> dict[str, str]:
    return {
        key: value
        for key in RUNTIME_BUILD_ENV_KEYS
        if (value := env.get(key)) is not None and value != ""
    }
