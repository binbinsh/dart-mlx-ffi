from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from matrix_config import (
    artifact_unblocks_platform,
    blocked_engine_reason,
    blocked_platform_reason,
)


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = Path(__file__).resolve().parent
DEFAULT_CATALOG = RUNTIME_DIR / "hf_artifacts.yaml"
DEFAULT_RECIPES = RUNTIME_DIR / "conversion_recipes.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Report runtime engine gaps that are hidden by platform fallbacks. "
            "For example, Apple may be ready through MLX while Core ML is still "
            "missing for the same model."
        )
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument(
        "--artifacts",
        type=Path,
        help=(
            "Optional resolved/converted artifact map. Converted platform "
            "entries override catalog gaps when computing remaining work."
        ),
    )
    parser.add_argument("--recipes", type=Path, default=DEFAULT_RECIPES)
    parser.add_argument("--model-id", action="append")
    parser.add_argument("--platform", action="append")
    parser.add_argument("--engine", action="append")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    catalog = _read_yaml(args.catalog)
    artifacts = _read_yaml(args.artifacts) if args.artifacts else {}
    recipes = _read_yaml(args.recipes) if args.recipes.exists() else {}
    report = build_report(
        catalog,
        artifacts=artifacts,
        recipes=recipes,
        catalog_path=args.catalog,
        artifacts_path=args.artifacts,
        recipes_path=args.recipes,
        model_filter=set(args.model_id or []),
        platform_filter=set(args.platform or []),
        engine_filter=set(args.engine or []),
    )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


def build_report(
    catalog: dict[str, Any],
    *,
    artifacts: dict[str, Any] | None = None,
    recipes: dict[str, Any],
    catalog_path: Path,
    artifacts_path: Path | None = None,
    recipes_path: Path,
    model_filter: set[str],
    platform_filter: set[str],
    engine_filter: set[str],
) -> dict[str, Any]:
    cells = []
    recipe_index = _recipe_index(recipes)
    artifact_models = (artifacts or {}).get("models") or {}
    for model_id, model in _models(catalog):
        if model_filter and model_id not in model_filter:
            continue
        artifact_model = artifact_models.get(model_id) or {}
        if not isinstance(artifact_model, dict):
            artifact_model = {}
        for platform in _platforms(catalog, model, platform_filter):
            order = _selected_order(catalog, platform, engine_filter)
            if not order:
                continue
            cell = _cell(
                catalog=catalog,
                model_id=model_id,
                model=model,
                artifact_model=artifact_model,
                platform=platform,
                order=order,
                recipe_index=recipe_index,
            )
            cells.append(cell)
    return {
        "version": 1,
        "catalog": str(catalog_path),
        **({"artifacts": str(artifacts_path)} if artifacts_path else {}),
        "recipes": str(recipes_path),
        "cell_count": len(cells),
        "preferred_ready_count": sum(
            1 for cell in cells if cell["state"] == "preferred_ready"
        ),
        "fallback_ready_count": sum(
            1 for cell in cells if cell["state"] == "fallback_ready"
        ),
        "missing_count": sum(1 for cell in cells if cell["state"] == "missing"),
        "blocked_count": sum(1 for cell in cells if cell["state"] == "blocked"),
        "cells": cells,
    }


def _cell(
    *,
    catalog: dict[str, Any],
    model_id: str,
    model: dict[str, Any],
    artifact_model: dict[str, Any],
    platform: str,
    order: list[str],
    recipe_index: dict[str, set[str]],
) -> dict[str, Any]:
    preferred = order[0]
    blocker = blocked_platform_reason(model, platform)
    if blocker and artifact_unblocks_platform(artifact_model, platform):
        blocker = None
    if blocker:
        cell = {
            "model_id": model_id,
            "family": model.get("family"),
            "platform": platform,
            "preferred_engine": preferred,
            "state": "blocked",
            "reason": blocker,
            "conversion": _conversion_commands(model_id, order, recipe_index),
        }
        cell.update(_blocked_metadata(model, artifact_model, platform))
        return cell

    artifacts = _effective_artifacts(model, artifact_model, platform)

    missing: list[str] = []
    blocked: dict[str, str] = {}
    for engine in order:
        engine_blocker = _blocked_engine_reason(
            model=model,
            artifact_model=artifact_model,
            platform=platform,
            engine=engine,
        )
        if engine_blocker:
            missing.append(engine)
            blocked[engine] = engine_blocker
            continue
        artifact = artifacts.get(engine)
        if isinstance(artifact, dict) and _artifact_supports(
            catalog,
            engine,
            artifact,
            platform,
        ):
            return {
                "model_id": model_id,
                "family": model.get("family"),
                "platform": platform,
                "preferred_engine": preferred,
                "selected_engine": engine,
                "state": "fallback_ready" if missing else "preferred_ready",
                "missing_preferred_engines": missing,
                "source_uri": _source_uri(artifact),
                "conversion": _conversion_commands(model_id, missing, recipe_index),
                **({"blocked_engines": blocked} if blocked else {}),
                **_blocked_engine_metadata(
                    model=model,
                    artifact_model=artifact_model,
                    platform=platform,
                    engines=blocked.keys(),
                ),
            }
        missing.append(engine)

    if blocked:
        return {
            "model_id": model_id,
            "family": model.get("family"),
            "platform": platform,
            "preferred_engine": preferred,
            "state": "blocked",
            "reason": _blocked_engine_summary(blocked),
            "missing_engines": missing,
            "blocked_engines": blocked,
            "conversion": _conversion_commands(model_id, missing, recipe_index),
            **_blocked_metadata(model, artifact_model, platform),
            **_blocked_engine_metadata(
                model=model,
                artifact_model=artifact_model,
                platform=platform,
                engines=blocked.keys(),
            ),
        }

    return {
        "model_id": model_id,
        "family": model.get("family"),
        "platform": platform,
        "preferred_engine": preferred,
        "state": "missing",
        "missing_engines": missing,
        "conversion": _conversion_commands(model_id, missing, recipe_index),
    }


def _models(catalog: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    raw = catalog.get("models") or {}
    return [
        (str(model_id), model)
        for model_id, model in raw.items()
        if isinstance(model, dict)
    ]


def _platforms(
    catalog: dict[str, Any],
    model: dict[str, Any],
    platform_filter: set[str],
) -> list[str]:
    if platform_filter:
        return sorted(platform_filter)
    policy = (catalog.get("support_policy") or {}).get("production_requires") or {}
    platforms = policy.get("platforms")
    if isinstance(platforms, list) and platforms:
        return [str(platform) for platform in platforms]
    model_platforms = model.get("platforms")
    if isinstance(model_platforms, list) and model_platforms:
        return [str(platform) for platform in model_platforms]
    return ["ios", "macos", "windows", "linux", "android"]


def _selected_order(
    catalog: dict[str, Any],
    platform: str,
    engine_filter: set[str],
) -> list[str]:
    raw = catalog.get("engine_order") or {}
    order = raw.get(platform)
    if not isinstance(order, list) or not order:
        order = ["coreml", "mlx", "onnx", "litert"]
    selected = [
        str(engine)
        for engine in order
        if _engine_supports(catalog, str(engine), platform)
    ]
    if engine_filter:
        selected = [engine for engine in selected if engine in engine_filter]
    return selected


def _engine_supports(catalog: dict[str, Any], engine: str, platform: str) -> bool:
    platforms = (catalog.get("engine_platforms") or {}).get(engine)
    if isinstance(platforms, list):
        return platform in {str(item) for item in platforms}
    return True


def _artifact_supports(
    catalog: dict[str, Any],
    engine: str,
    artifact: dict[str, Any],
    platform: str,
) -> bool:
    platforms = artifact.get("platforms")
    if isinstance(platforms, list):
        return platform in {str(item) for item in platforms}
    return _engine_supports(catalog, engine, platform)


def _source_uri(artifact: dict[str, Any]) -> str:
    source_uri = artifact.get("source_uri") or artifact.get("sourceUri")
    if isinstance(source_uri, str) and source_uri:
        return source_uri
    if "repo" not in artifact:
        return str(artifact.get("artifact") or "")
    return f"hf://{artifact['repo']}/{artifact.get('artifact') or '.'}"


def _effective_artifacts(
    model: dict[str, Any],
    artifact_model: dict[str, Any],
    platform: str,
) -> dict[str, Any]:
    raw = model.get("artifacts") or {}
    artifacts = dict(raw) if isinstance(raw, dict) else {}
    platform_cell = ((artifact_model.get("platforms") or {}).get(platform) or {})
    if not isinstance(platform_cell, dict):
        return artifacts
    engine = platform_cell.get("engine")
    artifact = platform_cell.get("artifact")
    if not isinstance(engine, str) or not engine:
        return artifacts
    if not isinstance(artifact, str) or not artifact:
        return artifacts
    artifacts[engine] = {
        "artifact": artifact,
        "source_uri": platform_cell.get("source_uri")
        or platform_cell.get("sourceUri")
        or artifact,
        "platforms": [platform],
        "artifact_source": platform_cell.get("artifact_source")
        or platform_cell.get("artifactSource"),
    }
    return artifacts


def _blocked_engine_reason(
    *,
    model: dict[str, Any],
    artifact_model: dict[str, Any],
    platform: str,
    engine: str,
) -> str | None:
    return blocked_engine_reason(
        artifact_model,
        platform,
        engine,
    ) or blocked_engine_reason(model, platform, engine)


def _blocked_engine_summary(blocked: dict[str, str]) -> str:
    return "; ".join(f"{engine}: {reason}" for engine, reason in blocked.items())


def _recipe_index(recipes: dict[str, Any]) -> dict[str, set[str]]:
    raw = recipes.get("models") or {}
    result: dict[str, set[str]] = {}
    for model_id, model in raw.items():
        if not isinstance(model, dict):
            continue
        model_recipes = model.get("recipes") or {}
        if isinstance(model_recipes, dict):
            result[str(model_id)] = {str(engine) for engine in model_recipes}
    return result


def _conversion_commands(
    model_id: str,
    engines: list[str],
    recipe_index: dict[str, set[str]],
) -> dict[str, str]:
    available = recipe_index.get(model_id) or set()
    commands = {}
    for engine in engines:
        if engine in available:
            commands[engine] = (
                "uv run python benchmark/runtime/convert_artifacts.py "
                f"--model-id {model_id} --engine {engine}"
            )
    return commands


def _blocked_metadata(
    model: dict[str, Any],
    artifact_model: dict[str, Any],
    platform: str,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for output_key, source_key in [
        ("failure_class", "blocked_platform_failure_classes"),
        ("failure_reason", "blocked_platform_failure_reasons"),
        ("report", "blocked_platform_reports"),
        ("log", "blocked_platform_logs"),
    ]:
        value = _blocked_platform_value(
            artifact_model=artifact_model,
            model=model,
            key=source_key,
            platform=platform,
        )
        if value:
            result[output_key] = value
    return result


def _blocked_engine_metadata(
    *,
    model: dict[str, Any],
    artifact_model: dict[str, Any],
    platform: str,
    engines: Any,
) -> dict[str, dict[str, str]]:
    engine_set = {str(engine) for engine in engines}
    if not engine_set:
        return {}
    result: dict[str, dict[str, str]] = {}
    for output_key, source_key in [
        ("blocked_engine_failure_classes", "blocked_engine_failure_classes"),
        ("blocked_engine_failure_reasons", "blocked_engine_failure_reasons"),
        ("blocked_engine_reports", "blocked_engine_reports"),
        ("blocked_engine_logs", "blocked_engine_logs"),
    ]:
        values = _blocked_engine_values(
            artifact_model=artifact_model,
            model=model,
            key=source_key,
            platform=platform,
            engines=engine_set,
        )
        if values:
            result[output_key] = values
    return result


def _blocked_engine_values(
    *,
    artifact_model: dict[str, Any],
    model: dict[str, Any],
    key: str,
    platform: str,
    engines: set[str],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for source in [artifact_model, model]:
        raw = source.get(key)
        if not isinstance(raw, dict):
            continue
        platform_value = raw.get(platform)
        if not isinstance(platform_value, dict):
            continue
        for engine in engines:
            if engine in result:
                continue
            value = platform_value.get(engine)
            if isinstance(value, str) and value:
                result[engine] = value
    return result


def _blocked_platform_value(
    *,
    artifact_model: dict[str, Any],
    model: dict[str, Any],
    key: str,
    platform: str,
) -> str:
    for source in [artifact_model, model]:
        raw = source.get(key)
        if not isinstance(raw, dict):
            continue
        value = raw.get(platform)
        if isinstance(value, str) and value:
            return value
    return ""


def _read_yaml(path: Path) -> dict[str, Any]:
    decoded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return decoded if isinstance(decoded, dict) else {}


if __name__ == "__main__":
    main()
