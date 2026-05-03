from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import yaml
from huggingface_hub import HfApi


RUNTIME_DIR = Path(__file__).resolve().parent
DEFAULT_CATALOG = RUNTIME_DIR / "hf_artifacts.yaml"
DEFAULT_FALLBACK_ENDPOINT = "https://hf-mirror.com"

RUNTIME_SUFFIXES = {
    "mlx": (".safetensors",),
    "coreml": (".mlmodelc", ".mlpackage"),
    "onnx": (".onnx",),
    "litert": (".tflite", ".task", ".litertlm"),
}

QUERY_SUFFIXES = {
    "mlx": (" MLX", " 4bit", " 8bit"),
    "coreml": (" CoreML", " mlmodelc", " mlpackage"),
    "onnx": (" ONNX",),
    "litert": (" LiteRT", " tflite", " litert-lm"),
}


class HubApi(Protocol):
    def model_info(self, repo_id: str, **kwargs: Any) -> Any: ...

    def list_models(self, **kwargs: Any) -> Any: ...

    def list_repo_tree(self, repo_id: str, **kwargs: Any) -> Any: ...


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit current runtime catalog against Hugging Face for newer source "
            "models and runtime artifacts."
        )
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--model-id", action="append")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT"))
    parser.add_argument(
        "--fallback-endpoint",
        default=os.environ.get("HF_FALLBACK_ENDPOINT") or DEFAULT_FALLBACK_ENDPOINT,
    )
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--tree-limit", type=int, default=1500)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    catalog = _read_yaml(args.catalog)
    api = _api(args.endpoint)
    fallback_api = _fallback_api(args.endpoint, args.fallback_endpoint)
    report = build_report(
        catalog,
        catalog_path=args.catalog,
        api=api,
        fallback_api=fallback_api,
        model_filter=set(args.model_id or []),
        limit=args.limit,
        max_candidates=args.max_candidates,
        tree_limit=args.tree_limit,
    )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


def build_report(
    catalog: dict[str, Any],
    *,
    catalog_path: Path,
    api: HubApi,
    fallback_api: HubApi | None,
    model_filter: set[str],
    limit: int,
    max_candidates: int,
    tree_limit: int,
) -> dict[str, Any]:
    models = []
    for model_id, model in _models(catalog):
        if model_filter and model_id not in model_filter:
            continue
        models.append(
            _model_report(
                model_id=model_id,
                model=model,
                api=api,
                fallback_api=fallback_api,
                limit=limit,
                max_candidates=max_candidates,
                tree_limit=tree_limit,
            )
        )
    return {
        "version": 1,
        "catalog": str(catalog_path),
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "model_count": len(models),
        "models": models,
    }


def _model_report(
    *,
    model_id: str,
    model: dict[str, Any],
    api: HubApi,
    fallback_api: HubApi | None,
    limit: int,
    max_candidates: int,
    tree_limit: int,
) -> dict[str, Any]:
    source_model = str(model.get("source_model") or "")
    artifacts = model.get("artifacts") if isinstance(model.get("artifacts"), dict) else {}
    artifact_reports = {
        str(engine): _configured_artifact_report(
            source_model=source_model,
            artifact=artifact,
            api=api,
            fallback_api=fallback_api,
            tree_limit=tree_limit,
        )
        for engine, artifact in artifacts.items()
        if isinstance(artifact, dict)
    }
    return {
        "id": model_id,
        "family": model.get("family"),
        "source_model": source_model,
        "source": _repo_report(source_model, api=api, fallback_api=fallback_api),
        "configured_artifacts": artifact_reports,
        "latest_candidates": {
            engine: _engine_candidates(
                model_id=model_id,
                model=model,
                engine=engine,
                api=api,
                fallback_api=fallback_api,
                limit=limit,
                max_candidates=max_candidates,
                tree_limit=tree_limit,
            )
            for engine in ["mlx", "coreml", "onnx", "litert"]
        },
    }


def _configured_artifact_report(
    *,
    source_model: str,
    artifact: dict[str, Any],
    api: HubApi,
    fallback_api: HubApi | None,
    tree_limit: int,
) -> dict[str, Any]:
    repo = str(artifact.get("repo") or "")
    path = str(artifact.get("artifact") or ".")
    tree = _repo_tree(repo, api=api, fallback_api=fallback_api, tree_limit=tree_limit)
    repo_report = _repo_report(repo, api=api, fallback_api=fallback_api)
    base_models = _base_models(repo_report.get("tags") or [])
    return {
        **repo_report,
        "artifact": path,
        "artifact_found": _artifact_exists(tree["paths"], path),
        "base_models": base_models,
        **(
            {"base_model_matches_source": source_model in base_models}
            if source_model and base_models
            else {}
        ),
        "runtime_paths": _runtime_roots(tree["paths"]),
        **({"tree_error": tree["error"]} if tree["error"] else {}),
    }


def _engine_candidates(
    *,
    model_id: str,
    model: dict[str, Any],
    engine: str,
    api: HubApi,
    fallback_api: HubApi | None,
    limit: int,
    max_candidates: int,
    tree_limit: int,
) -> list[dict[str, Any]]:
    if max_candidates <= 0:
        return []
    candidates: dict[str, dict[str, Any]] = {}
    for query in _candidate_queries(model_id, model, engine):
        for info in _list_models(query, api=api, fallback_api=fallback_api, limit=limit):
            repo = str(getattr(info, "modelId", "") or getattr(info, "id", ""))
            if not repo:
                continue
            candidates.setdefault(repo, {"repo": repo, "queries": set()})
            candidates[repo]["queries"].add(query)

    results = []
    for repo, record in sorted(candidates.items()):
        tree = _repo_tree(
            repo,
            api=api,
            fallback_api=fallback_api,
            tree_limit=tree_limit,
        )
        runtime_paths = _runtime_roots(tree["paths"]).get(engine, [])
        if not runtime_paths:
            continue
        repo_info = _repo_report(repo, api=api, fallback_api=fallback_api)
        results.append(
            {
                **repo_info,
                "queries": sorted(record["queries"]),
                "runtime_paths": runtime_paths[:20],
            }
        )
    return sorted(
        results,
        key=lambda item: (
            _candidate_rank(item, source_model=str(model.get("source_model") or "")),
            str(item.get("repo") or ""),
        ),
    )[:max_candidates]


def _candidate_queries(model_id: str, model: dict[str, Any], engine: str) -> list[str]:
    terms = []
    for value in [
        model.get("source_model"),
        model.get("family"),
        model_id.replace("_", "-"),
    ]:
        text = str(value or "").split("/")[-1].strip()
        if text and text not in terms:
            terms.append(text)
    queries = []
    for term in terms:
        for suffix in ("", *QUERY_SUFFIXES.get(engine, ())):
            query = f"{term}{suffix}".strip()
            if query and query not in queries:
                queries.append(query)
    return queries


def _candidate_rank(item: dict[str, Any], *, source_model: str) -> tuple[int, int]:
    tags = {str(tag) for tag in item.get("tags") or []}
    exact_base = f"base_model:{source_model}" in tags
    downloads = int(item.get("downloads") or 0)
    return (0 if exact_base else 1, -downloads)


def _base_models(tags: list[Any]) -> list[str]:
    models = []
    seen = set()
    for tag in tags:
        text = str(tag)
        if not text.startswith("base_model:"):
            continue
        value = text.removeprefix("base_model:")
        if value.startswith("quantized:") or value.startswith("finetune:"):
            value = value.split(":", 1)[1]
        if value and value not in seen:
            models.append(value)
            seen.add(value)
    return models


def _repo_report(
    repo: str,
    *,
    api: HubApi,
    fallback_api: HubApi | None,
) -> dict[str, Any]:
    if not repo:
        return {"repo": repo, "exists": False}
    try:
        info = api.model_info(repo, files_metadata=False)
    except Exception as error:  # noqa: BLE001 - retry mirror and report both.
        if fallback_api is None:
            return {"repo": repo, "exists": False, "error": str(error)}
        try:
            info = fallback_api.model_info(repo, files_metadata=False)
        except Exception as fallback_error:  # noqa: BLE001
            return {
                "repo": repo,
                "exists": False,
                "error": str(error),
                "fallback_error": str(fallback_error),
            }
    return {
        "repo": repo,
        "exists": True,
        "sha": getattr(info, "sha", None),
        "last_modified": _date_text(
            getattr(info, "lastModified", None)
            or getattr(info, "last_modified", None)
        ),
        "downloads": getattr(info, "downloads", None),
        "tags": list(getattr(info, "tags", []) or []),
    }


def _list_models(
    query: str,
    *,
    api: HubApi,
    fallback_api: HubApi | None,
    limit: int,
) -> list[Any]:
    try:
        return list(api.list_models(search=query, limit=limit))
    except Exception:
        if fallback_api is None:
            return []
        try:
            return list(fallback_api.list_models(search=query, limit=limit))
        except Exception:
            return []


def _repo_tree(
    repo: str,
    *,
    api: HubApi,
    fallback_api: HubApi | None,
    tree_limit: int,
) -> dict[str, Any]:
    if not repo:
        return {"paths": [], "error": ""}
    try:
        return {"paths": _repo_tree_paths(api, repo, tree_limit), "error": ""}
    except Exception as error:  # noqa: BLE001
        if fallback_api is None:
            return {"paths": [], "error": str(error)}
        try:
            return {
                "paths": _repo_tree_paths(fallback_api, repo, tree_limit),
                "error": "",
            }
        except Exception as fallback_error:  # noqa: BLE001
            return {
                "paths": [],
                "error": f"{error}; fallback: {fallback_error}",
            }


def _repo_tree_paths(api: HubApi, repo: str, tree_limit: int) -> list[str]:
    paths = []
    for index, item in enumerate(api.list_repo_tree(repo, recursive=True)):
        if index >= tree_limit:
            break
        path = str(getattr(item, "path", "") or "")
        if path:
            paths.append(path)
    return paths


def _runtime_roots(paths: list[str]) -> dict[str, list[str]]:
    result = {engine: [] for engine in RUNTIME_SUFFIXES}
    seen = {engine: set() for engine in RUNTIME_SUFFIXES}
    for path in paths:
        lower = path.lower()
        for engine, suffixes in RUNTIME_SUFFIXES.items():
            for suffix in suffixes:
                root = _artifact_root(path, lower, suffix)
                if root and root not in seen[engine]:
                    result[engine].append(root)
                    seen[engine].add(root)
    return {engine: sorted(values) for engine, values in result.items() if values}


def _artifact_root(path: str, lower: str, suffix: str) -> str | None:
    marker = suffix + "/"
    if marker in lower:
        return path[: lower.index(marker) + len(suffix)]
    if lower.endswith(suffix):
        return path
    return None


def _artifact_exists(paths: list[str], artifact: str) -> bool:
    normalized = artifact.strip().strip("/")
    if not normalized or normalized == ".":
        return bool(paths)
    return any(
        path == normalized or path.startswith(normalized.rstrip("/") + "/")
        for path in paths
    )


def _models(catalog: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    raw = catalog.get("models") or {}
    return [
        (str(model_id), model)
        for model_id, model in raw.items()
        if isinstance(model, dict)
    ]


def _date_text(value: Any) -> str | None:
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _api(endpoint: str | None) -> HfApi:
    return HfApi(endpoint=endpoint) if endpoint else HfApi()


def _fallback_api(endpoint: str | None, fallback_endpoint: str | None) -> HfApi | None:
    fallback = str(fallback_endpoint or "").strip()
    if not fallback:
        return None
    primary = str(endpoint or "https://huggingface.co").rstrip("/")
    if fallback.rstrip("/") == primary:
        return None
    return HfApi(endpoint=fallback)


def _read_yaml(path: Path) -> dict[str, Any]:
    decoded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return decoded if isinstance(decoded, dict) else {}


if __name__ == "__main__":
    main()
