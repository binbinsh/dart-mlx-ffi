from __future__ import annotations

from typing import Any

DEFAULT_PRODUCTION_PLATFORMS = ["ios", "macos", "android"]
DEFAULT_OPTIONAL_PLATFORMS = ["linux"]
DEFAULT_ABANDONED_PLATFORMS = ["windows"]
DEFAULT_ENGINE_ORDER = {
    "ios": ["coreml", "mlx", "onnx"],
    "macos": ["coreml", "mlx", "onnx"],
    "linux": ["onnx"],
    "android": ["litert", "onnx"],
}


def production_platforms(config: dict[str, Any]) -> list[str]:
    policy = config.get("support_policy") or {}
    production = policy.get("production_requires") or {}
    platforms = production.get("platforms")
    if isinstance(platforms, list) and platforms:
        return [str(platform) for platform in platforms]
    return list(DEFAULT_PRODUCTION_PLATFORMS)


def optional_platforms(config: dict[str, Any]) -> list[str]:
    policy = config.get("support_policy") or {}
    platforms = policy.get("optional_platforms") or policy.get("optionalPlatforms")
    if isinstance(platforms, list):
        return [str(platform) for platform in platforms]
    return list(DEFAULT_OPTIONAL_PLATFORMS)


def abandoned_platforms(config: dict[str, Any]) -> list[str]:
    policy = config.get("support_policy") or {}
    platforms = policy.get("abandoned_platforms") or policy.get("abandonedPlatforms")
    if isinstance(platforms, list):
        return [str(platform) for platform in platforms]
    return list(DEFAULT_ABANDONED_PLATFORMS)


def blocked_platforms(model: dict[str, Any]) -> dict[str, str]:
    raw = model.get("blocked_platforms") or model.get("blockedPlatforms") or {}
    if isinstance(raw, dict):
        return {
            str(platform): str(reason)
            for platform, reason in raw.items()
            if platform is not None
        }
    if isinstance(raw, list):
        return {
            str(platform): _generic_blocker(str(platform))
            for platform in raw
            if platform is not None
        }
    return {}


def blocked_platform_reason(model: dict[str, Any], platform: str) -> str | None:
    return blocked_platforms(model).get(platform)


def blocked_engines(model: dict[str, Any], platform: str) -> dict[str, str]:
    raw = model.get("blocked_engines") or model.get("blockedEngines") or {}
    if not isinstance(raw, dict):
        return {}
    platform_value = raw.get(platform) or {}
    if isinstance(platform_value, dict):
        return {
            str(engine): str(reason)
            for engine, reason in platform_value.items()
            if engine is not None
        }
    return {}


def blocked_engine_reason(
    model: dict[str, Any],
    platform: str,
    engine: str,
) -> str | None:
    return blocked_engines(model, platform).get(engine)


def artifact_unblocks_platform(model: dict[str, Any], platform: str) -> bool:
    cell = platform_artifact(model, platform)
    if not cell:
        return False
    source = str(cell.get("artifact_source") or cell.get("artifactSource") or "")
    return bool(cell.get("unblock_platform") or source == "converted")


def platform_artifact(model: dict[str, Any], platform: str) -> dict[str, Any]:
    platforms = model.get("platforms") or {}
    if not isinstance(platforms, dict):
        return {}
    cell = platforms.get(platform) or {}
    return cell if isinstance(cell, dict) else {}


def fallback_engines(model: dict[str, Any], platform: str) -> list[str]:
    cell = platform_artifact(model, platform)
    raw = cell.get("fallback_from") or cell.get("fallbackFrom") or []
    if isinstance(raw, list):
        return [str(engine) for engine in raw if engine is not None]
    if raw:
        return [str(raw)]
    return []


def fallback_reason(
    model: dict[str, Any],
    platform: str,
    *,
    preferred_engine: str | None = None,
) -> str | None:
    selected = platform_artifact(model, platform).get("engine")
    preferred = preferred_engine or preferred_engine_for_platform(model, platform)
    if preferred and selected == preferred:
        return None
    engines = fallback_engines(model, platform)
    if not engines:
        return None
    return (
        "Selected artifact uses a fallback engine; missing preferred "
        f"engine(s): {', '.join(engines)}."
    )


def artifact_coverage(model: dict[str, Any]) -> str:
    return str(model.get("artifact_coverage") or "full")


def engine_order_for_platform(
    config: dict[str, Any],
    model: dict[str, Any],
    platform: str,
) -> list[str]:
    for source in (model, config):
        raw = source.get("engine_order") or source.get("engineOrder") or {}
        if not isinstance(raw, dict):
            continue
        value = raw.get(platform)
        if isinstance(value, list) and value:
            return [str(engine) for engine in value if engine is not None]
    return list(DEFAULT_ENGINE_ORDER.get(platform) or ["coreml", "mlx", "onnx", "litert"])


def preferred_engine_for_platform(
    model: dict[str, Any],
    platform: str,
    config: dict[str, Any] | None = None,
) -> str | None:
    order = engine_order_for_platform(config or {}, model, platform)
    return order[0] if order else None


def _generic_blocker(platform: str) -> str:
    return (
        "No directly loadable runtime artifact is currently available for "
        f"{platform}."
    )
