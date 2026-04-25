from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from matrix_config import blocked_platform_reason


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
    parser.add_argument("--recipes", type=Path, default=DEFAULT_RECIPES)
    parser.add_argument("--model-id", action="append")
    parser.add_argument("--platform", action="append")
    parser.add_argument("--engine", action="append")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    catalog = _read_yaml(args.catalog)
    recipes = _read_yaml(args.recipes) if args.recipes.exists() else {}
    report = build_report(
        catalog,
        recipes=recipes,
        catalog_path=args.catalog,
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
    recipes: dict[str, Any],
    catalog_path: Path,
    recipes_path: Path,
    model_filter: set[str],
    platform_filter: set[str],
    engine_filter: set[str],
) -> dict[str, Any]:
    cells = []
    recipe_index = _recipe_index(recipes)
    for model_id, model in _models(catalog):
        if model_filter and model_id not in model_filter:
            continue
        for platform in _platforms(catalog, model, platform_filter):
            order = _selected_order(catalog, platform, engine_filter)
            if not order:
                continue
            cell = _cell(
                catalog=catalog,
                model_id=model_id,
                model=model,
                platform=platform,
                order=order,
                recipe_index=recipe_index,
            )
            cells.append(cell)
    return {
        "version": 1,
        "catalog": str(catalog_path),
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
    platform: str,
    order: list[str],
    recipe_index: dict[str, set[str]],
) -> dict[str, Any]:
    preferred = order[0]
    blocker = blocked_platform_reason(model, platform)
    if blocker:
        return {
            "model_id": model_id,
            "family": model.get("family"),
            "platform": platform,
            "preferred_engine": preferred,
            "state": "blocked",
            "reason": blocker,
        }

    artifacts = model.get("artifacts") or {}
    if not isinstance(artifacts, dict):
        artifacts = {}

    missing: list[str] = []
    for engine in order:
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
            }
        missing.append(engine)

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
    return f"hf://{artifact['repo']}/{artifact.get('artifact') or '.'}"


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


def _read_yaml(path: Path) -> dict[str, Any]:
    decoded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return decoded if isinstance(decoded, dict) else {}


if __name__ == "__main__":
    main()
