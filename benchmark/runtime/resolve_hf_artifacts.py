from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml
from converters.hf_download import DEFAULT_FALLBACK_ENDPOINT, snapshot_download_with_fallback

from matrix_config import (
    blocked_engine_reason,
    blocked_platform_reason,
    blocked_platforms,
    engine_order_for_platform,
    production_platforms,
)


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = Path(__file__).resolve().parent
DEFAULT_CATALOG = RUNTIME_DIR / "hf_artifacts.yaml"
DEFAULT_OUT = RUNTIME_DIR / "artifacts.local.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve Hugging Face runtime artifacts into a local artifact map "
            "consumable by benchmark/runtime/run_all.py."
        )
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--model-id", action="append")
    parser.add_argument("--platform", action="append")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT"))
    parser.add_argument(
        "--fallback-endpoint",
        default=os.environ.get("HF_FALLBACK_ENDPOINT") or DEFAULT_FALLBACK_ENDPOINT,
    )
    parser.add_argument(
        "--engine",
        action="append",
        choices=["mlx", "coreml", "onnx", "litert"],
        help="Restrict resolution to one or more engines.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected HF sources without downloading or writing YAML.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only files already present in the Hugging Face cache.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip unavailable model/platform cells instead of failing.",
    )
    args = parser.parse_args()

    catalog = _read_yaml(args.catalog)
    resolver = HuggingFaceArtifactResolver(
        catalog=catalog,
        catalog_path=args.catalog,
        out_path=args.out,
        cache_root=args.cache_root,
        model_filter=set(args.model_id or []),
        platform_filter=set(args.platform or []),
        engine_filter=set(args.engine or []),
        local_files_only=args.local_files_only,
        allow_missing=args.allow_missing,
        endpoint=args.endpoint,
        fallback_endpoint=args.fallback_endpoint,
    )
    if args.dry_run:
        print(json.dumps(resolver.plan(), indent=2, ensure_ascii=False))
        return

    artifact_map = resolver.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        yaml.safe_dump(artifact_map, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(json.dumps({"out": str(args.out), "models": len(artifact_map["models"])}, indent=2))


class HuggingFaceArtifactResolver:
    def __init__(
        self,
        *,
        catalog: dict[str, Any],
        catalog_path: Path,
        out_path: Path,
        cache_root: Path | None,
        model_filter: set[str],
        platform_filter: set[str],
        engine_filter: set[str],
        local_files_only: bool,
        allow_missing: bool,
        endpoint: str | None = None,
        fallback_endpoint: str | None = None,
    ) -> None:
        self.catalog = catalog
        self.catalog_path = catalog_path
        self.out_path = out_path
        self.cache_root = _resolve_path(
            cache_root or Path(str(catalog.get("cache_root") or "benchmark/artifacts/huggingface"))
        )
        self.model_filter = model_filter
        self.platform_filter = platform_filter
        self.engine_filter = engine_filter
        self.local_files_only = local_files_only
        self.allow_missing = allow_missing
        self.endpoint = endpoint
        self.fallback_endpoint = fallback_endpoint

    def plan(self) -> dict[str, Any]:
        cells = []
        for model_id, model in self._models():
            for platform in self._platforms(model):
                blocker = blocked_platform_reason(model, platform)
                if blocker:
                    cells.append(
                        {
                            "model_id": model_id,
                            "platform": platform,
                            "state": "blocked",
                            "reason": blocker,
                        }
                    )
                    continue
                selection = self._select(model, platform)
                cells.append(
                    {
                        "model_id": model_id,
                        "platform": platform,
                        "state": "ready" if selection else "missing",
                        **(selection.plan if selection else {}),
                    }
                )
        missing = [cell for cell in cells if cell["state"] == "missing"]
        blocked = [cell for cell in cells if cell["state"] == "blocked"]
        if missing and not self.allow_missing:
            return {
                "version": 1,
                "catalog": str(self.catalog_path),
                "cache_root": str(self.cache_root),
                "ready_count": len(cells) - len(missing) - len(blocked),
                "missing_count": len(missing),
                "blocked_count": len(blocked),
                "cells": cells,
            }
        return {
            "version": 1,
            "catalog": str(self.catalog_path),
            "cache_root": str(self.cache_root),
            "ready_count": len(cells) - len(missing) - len(blocked),
            "missing_count": len(missing),
            "blocked_count": len(blocked),
            "cells": cells,
        }

    def resolve(self) -> dict[str, Any]:
        output = {
            "version": 1,
            "source_catalog": _rel(self.catalog_path),
            "cache_root": str(self.cache_root),
            "defaults": self.catalog.get("defaults") or {},
            "models": {},
        }
        missing: list[dict[str, Any]] = []
        for model_id, model in self._models():
            model_out: dict[str, Any] = {
                key: value
                for key, value in model.items()
                if key
                in {
                    "family",
                    "source_model",
                    "task",
                    "required_fixtures",
                    "tools_file",
                    "embedding_query_file",
                    "embedding_dim",
                    "artifact_coverage",
                    "engine_order",
                    "migrated_platforms",
                    "blocked_platforms",
                }
            }
            platforms: dict[str, Any] = {}
            for platform in self._platforms(model):
                if blocked_platform_reason(model, platform):
                    continue
                selection = self._select(model, platform)
                if selection is None:
                    missing.append({"model_id": model_id, "platform": platform})
                    continue
                cell = self._resolve_cell(model, platform, selection)
                platforms[platform] = cell
            if platforms:
                model_out["platforms"] = platforms
                output["models"][model_id] = model_out
        if missing and not self.allow_missing:
            details = ", ".join(f"{m['model_id']}:{m['platform']}" for m in missing)
            raise SystemExit(f"Missing HF artifact coverage for {details}")
        return output

    def _resolve_cell(
        self,
        model: dict[str, Any],
        platform: str,
        selection: "Selection",
    ) -> dict[str, Any]:
        artifact_path = self._download(selection.artifact)
        component_paths = self._component_paths(
            selection.artifact,
            artifact_path=artifact_path,
        )
        pipeline = selection.artifact.get("pipeline")
        if isinstance(pipeline, dict):
            artifact_path = self._write_pipeline_spec(
                selection,
                artifact_path=artifact_path,
                component_paths=component_paths,
            )
        cell: dict[str, Any] = {
            "engine": selection.engine,
            "artifact": str(artifact_path),
            "source_uri": selection.source_uri,
        }
        cell.update(self._platform_defaults(platform))
        if selection.fallback_from:
            cell["fallback_from"] = selection.fallback_from
        for key in ("component_artifacts", "pipeline"):
            if selection.artifact.get(key) is not None:
                cell[key] = selection.artifact[key]
        if component_paths:
            cell["component_artifacts_resolved"] = {
                key: str(value) for key, value in component_paths.items()
            }
        for key in (
            "task",
            "tools_file",
            "embedding_query_file",
            "embedding_dim",
            "image_file",
            "audio_file",
        ):
            if model.get(key) is not None:
                cell[key] = model[key]

        baseline_engine = self._baseline_engine(model, platform, selection.engine)
        if baseline_engine:
            cell["baseline_engine"] = baseline_engine
        baseline_artifact = self._baseline_artifact(model, platform, selection)
        if baseline_artifact is not None:
            cell["baseline_artifact"] = str(self._download(baseline_artifact))
            cell["baseline_source_uri"] = _source_uri(baseline_artifact)
        return cell

    def _baseline_engine(
        self,
        model: dict[str, Any],
        platform: str,
        engine: str,
    ) -> str | None:
        default = self._platform_defaults(platform).get("baseline_engine")
        if platform in {"ios", "macos"}:
            if model.get("coreml_llm_baseline") is False:
                return "mlx"
            return "coreml-llm" if engine == "coreml" else "mlx"
        if default and default != "external":
            return str(default)
        return engine

    def _baseline_artifact(
        self,
        model: dict[str, Any],
        platform: str,
        selection: "Selection",
    ) -> dict[str, Any] | None:
        baseline_engine = self._baseline_engine(model, platform, selection.engine)
        if baseline_engine == "coreml-llm":
            return selection.artifact
        if baseline_engine == "mlx":
            mlx = (model.get("artifacts") or {}).get("mlx")
            if isinstance(mlx, dict):
                return mlx
        return None

    def _download(self, artifact: dict[str, Any]) -> Path:
        repo = str(artifact["repo"])
        patterns = _allow_patterns(artifact)
        snapshot = snapshot_download_with_fallback(
            repo_id=repo,
            allow_patterns=patterns,
            cache_dir=str(self.cache_root),
            local_files_only=self.local_files_only,
            endpoint=self.endpoint,
            fallback_endpoint=self.fallback_endpoint,
        )
        artifact_name = str(artifact.get("artifact") or ".")
        if artifact_name == ".":
            return Path(snapshot)
        return Path(snapshot) / artifact_name

    def _component_paths(
        self,
        artifact: dict[str, Any],
        *,
        artifact_path: Path,
    ) -> dict[str, Path]:
        raw = artifact.get("component_artifacts")
        if not isinstance(raw, dict):
            return {}
        root = _artifact_root(artifact_path, artifact)
        return {str(name): root / str(path) for name, path in raw.items()}

    def _write_pipeline_spec(
        self,
        selection: "Selection",
        *,
        artifact_path: Path,
        component_paths: dict[str, Path],
    ) -> Path:
        raw = selection.artifact.get("pipeline")
        if not isinstance(raw, dict):
            return artifact_path
        spec = _expand_pipeline(raw, artifact_path, component_paths)
        digest = _safe_component(selection.artifact["repo"])
        artifact_name = _safe_component(str(selection.artifact.get("artifact") or "."))
        out = self.cache_root / "_pipelines" / digest / f"{artifact_name}.pipeline.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return out

    def _select(self, model: dict[str, Any], platform: str) -> "Selection | None":
        artifacts = model.get("artifacts") or {}
        if not isinstance(artifacts, dict):
            return None
        order = self._engine_order(model, platform)
        if self.engine_filter:
            order = [engine for engine in order if engine in self.engine_filter]
        fallback_from: list[str] = []
        for engine in order:
            if blocked_engine_reason(model, platform, engine):
                fallback_from.append(engine)
                continue
            artifact = artifacts.get(engine)
            if not isinstance(artifact, dict) or not self._artifact_supports(
                engine,
                artifact,
                platform,
            ):
                if self._engine_supports(engine, platform):
                    fallback_from.append(engine)
                continue
            return Selection(
                engine=engine,
                artifact=artifact,
                fallback_from=fallback_from,
            )
        return None

    def _models(self) -> list[tuple[str, dict[str, Any]]]:
        raw = self.catalog.get("models") or {}
        selected: list[tuple[str, dict[str, Any]]] = []
        for model_id, model in raw.items():
            if self.model_filter and model_id not in self.model_filter:
                continue
            if isinstance(model, dict):
                selected.append((str(model_id), model))
        return selected

    def _platforms(self, model: dict[str, Any] | None = None) -> list[str]:
        if self.platform_filter:
            return sorted(self.platform_filter)
        policy_platforms = self._policy_platforms()
        model_platforms = (model or {}).get("platforms")
        if isinstance(model_platforms, list) and model_platforms:
            selected = {str(platform) for platform in model_platforms}
            selected.update(blocked_platforms(model or {}).keys())
            ordered = [platform for platform in policy_platforms if platform in selected]
            return ordered
        return policy_platforms

    def _policy_platforms(self) -> list[str]:
        return production_platforms(self.catalog)

    def _platform_defaults(self, platform: str) -> dict[str, Any]:
        defaults = (self.catalog.get("defaults") or {}).get("platforms") or {}
        value = defaults.get(platform)
        return dict(value) if isinstance(value, dict) else {}

    def _engine_order(self, model: dict[str, Any], platform: str) -> list[str]:
        return engine_order_for_platform(self.catalog, model, platform)

    def _artifact_supports(
        self,
        engine: str,
        artifact: dict[str, Any],
        platform: str,
    ) -> bool:
        platforms = artifact.get("platforms")
        if isinstance(platforms, list):
            return platform in {str(item) for item in platforms}
        defaults = self.catalog.get("engine_platforms") or {}
        engine_platforms = defaults.get(engine)
        if isinstance(engine_platforms, list):
            return platform in {str(item) for item in engine_platforms}
        return True

    def _engine_supports(self, engine: str, platform: str) -> bool:
        defaults = self.catalog.get("engine_platforms") or {}
        engine_platforms = defaults.get(engine)
        if isinstance(engine_platforms, list):
            return platform in {str(item) for item in engine_platforms}
        return True


class Selection:
    def __init__(
        self,
        *,
        engine: str,
        artifact: dict[str, Any],
        fallback_from: list[str] | None = None,
    ) -> None:
        self.engine = engine
        self.artifact = artifact
        self.fallback_from = fallback_from or []

    @property
    def source_uri(self) -> str:
        return _source_uri(self.artifact)

    @property
    def plan(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "source_uri": self.source_uri,
            "repo": self.artifact.get("repo"),
            "artifact": self.artifact.get("artifact"),
            "allow_patterns": _allow_patterns(self.artifact),
            **(
                {"component_artifacts": self.artifact["component_artifacts"]}
                if self.artifact.get("component_artifacts") is not None
                else {}
            ),
            **(
                {"pipeline": self.artifact["pipeline"]}
                if self.artifact.get("pipeline") is not None
                else {}
            ),
            **(
                {"fallback_from": self.fallback_from}
                if self.fallback_from
                else {}
            ),
        }


def _allow_patterns(artifact: dict[str, Any]) -> list[str]:
    raw = artifact.get("allow_patterns")
    if isinstance(raw, list) and raw:
        return [str(item) for item in raw]
    name = str(artifact.get("artifact") or ".")
    if name == ".":
        return ["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*", "vocab*"]
    if name.endswith((".mlmodelc", ".mlpackage")):
        return [f"{name}/**", "*.json", "tokenizer*"]
    if name.endswith(".onnx"):
        return [name, f"{name}*", "*.json", "*.jinja", "tokenizer*"]
    return [name, "*.json", "tokenizer*"]


def _artifact_root(artifact_path: Path, artifact: dict[str, Any]) -> Path:
    artifact_name = str(artifact.get("artifact") or ".")
    if artifact_name == ".":
        return artifact_path
    root = artifact_path
    for _ in Path(artifact_name).parts:
        root = root.parent
    return root


def _expand_pipeline(
    pipeline: dict[str, Any],
    artifact_path: Path,
    component_paths: dict[str, Path],
) -> dict[str, Any]:
    context = {
        "artifact": str(artifact_path),
        "artifact_dir": str(artifact_path.parent),
        **{
            f"component:{name}": str(path)
            for name, path in component_paths.items()
        },
    }

    def expand(value: Any) -> Any:
        if isinstance(value, str):
            return _expand_text(value, context)
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, dict):
            return {str(key): expand(item) for key, item in value.items()}
        return value

    return expand(pipeline)


def _expand_text(value: str, context: dict[str, str]) -> str:
    out = value
    for key, replacement in context.items():
        out = out.replace("{" + key + "}", replacement)
    return out


def _safe_component(value: str) -> str:
    return "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in value
    ).strip("_") or "artifact"


def _source_uri(artifact: dict[str, Any]) -> str:
    return f"hf://{artifact['repo']}/{artifact.get('artifact') or '.'}"


def _read_yaml(path: Path) -> dict[str, Any]:
    decoded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return decoded if isinstance(decoded, dict) else {}


def _resolve_path(path: Path) -> Path:
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else ROOT / expanded


def _rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


if __name__ == "__main__":
    main()
