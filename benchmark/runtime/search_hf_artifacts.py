from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi


RUNTIME_DIR = Path(__file__).resolve().parent
DEFAULT_CATALOG = RUNTIME_DIR / "hf_artifacts.yaml"

RUNTIME_SUFFIXES = {
    "coreml": (".mlmodelc", ".mlpackage"),
    "onnx": (".onnx",),
    "litert": (".tflite", ".task", ".litertlm"),
}

COMPONENT_MARKERS = (
    "campplus",
    "cosyvoice",
    "flow.decoder",
    "speech_tokenizer",
    "token2wav",
    "vision/",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search Hugging Face for runtime artifacts missing from catalog."
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--model-id", action="append")
    parser.add_argument("--include-full", action="store_true")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--tree-limit", type=int, default=2000)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    catalog = _read_yaml(args.catalog)
    report = search_catalog(
        catalog,
        catalog_path=args.catalog,
        model_filter=set(args.model_id or []),
        include_full=args.include_full,
        limit=args.limit,
        tree_limit=args.tree_limit,
    )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


def search_catalog(
    catalog: dict[str, Any],
    *,
    catalog_path: Path,
    model_filter: set[str],
    include_full: bool,
    limit: int,
    tree_limit: int,
) -> dict[str, Any]:
    api = HfApi()
    records = []
    for model_id, model in _models(catalog):
        if model_filter and model_id not in model_filter:
            continue
        if not include_full and model.get("artifact_coverage") != "partial":
            continue
        records.append(
            _search_model(
                api,
                model_id=model_id,
                model=model,
                limit=limit,
                tree_limit=tree_limit,
            )
        )
    return {
        "version": 1,
        "catalog": str(catalog_path),
        "model_count": len(records),
        "runtime_candidate_count": sum(
            len(record["runtime_candidates"]) for record in records
        ),
        "component_candidate_count": sum(
            len(record["component_candidates"]) for record in records
        ),
        "models": records,
    }


def _search_model(
    api: HfApi,
    *,
    model_id: str,
    model: dict[str, Any],
    limit: int,
    tree_limit: int,
) -> dict[str, Any]:
    repos: dict[str, set[str]] = {}
    for query in _queries(model_id, model):
        for match in api.list_models(search=query, limit=limit):
            repos.setdefault(match.modelId, set()).add(query)

    runtime_candidates = []
    component_candidates = []
    for repo, queries in sorted(repos.items()):
        paths = _artifact_paths(api, repo, tree_limit=tree_limit)
        if not paths:
            continue
        record = {"repo": repo, "queries": sorted(queries), "paths": paths}
        if any(_is_component_path(path) for path in paths):
            component_candidates.append(record)
        else:
            runtime_candidates.append(record)

    return {
        "id": model_id,
        "family": model.get("family"),
        "source_model": model.get("source_model"),
        "blocked_platforms": model.get("blocked_platforms") or {},
        "runtime_candidates": runtime_candidates,
        "component_candidates": component_candidates,
    }


def _queries(model_id: str, model: dict[str, Any]) -> list[str]:
    raw_terms = model.get("search_terms")
    if isinstance(raw_terms, list) and raw_terms:
        terms = [str(term) for term in raw_terms]
    else:
        terms = [
            str(model.get("source_model") or "").split("/")[-1],
            str(model.get("family") or ""),
            model_id.replace("_", "-"),
        ]
    suffixes = ["", " ONNX", " LiteRT", " tflite", " CoreML", " mlmodelc"]
    queries = []
    seen = set()
    for term in terms:
        term = term.strip()
        if not term:
            continue
        for suffix in suffixes:
            query = f"{term}{suffix}".strip()
            if query not in seen:
                queries.append(query)
                seen.add(query)
    return queries


def _artifact_paths(api: HfApi, repo: str, *, tree_limit: int) -> list[str]:
    paths: list[str] = []
    seen = set()
    try:
        for index, item in enumerate(api.list_repo_tree(repo, recursive=True)):
            if index >= tree_limit:
                break
            path = getattr(item, "path", "")
            root = _artifact_root(path)
            if root is None or root in seen:
                continue
            seen.add(root)
            paths.append(root)
    except Exception as error:  # noqa: BLE001 - report search failures in JSON.
        return [f"<tree-error: {error}>"]
    return sorted(paths)


def _artifact_root(path: str) -> str | None:
    lower = path.lower()
    for suffixes in RUNTIME_SUFFIXES.values():
        for suffix in suffixes:
            marker = suffix + "/"
            if marker in lower:
                return path[: lower.index(marker) + len(suffix)]
            if lower.endswith(suffix):
                return path
    return None


def _is_component_path(path: str) -> bool:
    lower = path.lower()
    return any(marker in lower for marker in COMPONENT_MARKERS)


def _models(catalog: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    raw = catalog.get("models") or {}
    return [
        (str(model_id), model)
        for model_id, model in raw.items()
        if isinstance(model, dict)
    ]


def _read_yaml(path: Path) -> dict[str, Any]:
    decoded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return decoded if isinstance(decoded, dict) else {}


if __name__ == "__main__":
    main()
