from __future__ import annotations

from typing import Any


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


def fallback_reason(model: dict[str, Any], platform: str) -> str | None:
    engines = fallback_engines(model, platform)
    if not engines:
        return None
    return (
        "Selected artifact uses a fallback engine; missing preferred "
        f"engine(s): {', '.join(engines)}."
    )


def artifact_coverage(model: dict[str, Any]) -> str:
    return str(model.get("artifact_coverage") or "full")


def _generic_blocker(platform: str) -> str:
    return (
        "No directly loadable runtime artifact is currently available for "
        f"{platform}."
    )
