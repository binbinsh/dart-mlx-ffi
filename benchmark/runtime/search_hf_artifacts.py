from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import UTC, datetime
import json
import os
import signal
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi


RUNTIME_DIR = Path(__file__).resolve().parent
DEFAULT_CATALOG = RUNTIME_DIR / "hf_artifacts.yaml"
DEFAULT_FALLBACK_ENDPOINT = "https://hf-mirror.com"

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
    parser.add_argument(
        "--request-timeout-seconds",
        type=int,
        default=30,
        help="Per query/tree request timeout; 0 disables timeout.",
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT"),
        help="Primary Hugging Face-compatible endpoint.",
    )
    parser.add_argument(
        "--fallback-endpoint",
        default=os.environ.get("HF_FALLBACK_ENDPOINT") or DEFAULT_FALLBACK_ENDPOINT,
        help="Endpoint retried when the primary endpoint is rate limited.",
    )
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
        endpoint=args.endpoint,
        fallback_endpoint=args.fallback_endpoint,
        request_timeout_seconds=args.request_timeout_seconds,
    )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        out_path = _prepare_out_path(args.out)
        out_path.write_text(text + "\n", encoding="utf-8")
    print(text)


def search_catalog(
    catalog: dict[str, Any],
    *,
    catalog_path: Path,
    model_filter: set[str],
    include_full: bool,
    limit: int,
    tree_limit: int,
    endpoint: str | None = None,
    fallback_endpoint: str | None = DEFAULT_FALLBACK_ENDPOINT,
    request_timeout_seconds: int = 30,
) -> dict[str, Any]:
    api = _hf_api(endpoint)
    fallback_apis = _fallback_apis(
        endpoint=endpoint,
        fallback_endpoint=fallback_endpoint,
    )
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
                fallback_apis=fallback_apis,
                fallback_endpoint=fallback_endpoint,
                request_timeout_seconds=request_timeout_seconds,
            )
        )
    return {
        "version": 1,
        "catalog": str(catalog_path),
        "endpoint": endpoint or "https://huggingface.co",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        **(
            {"fallback_endpoints": [endpoint for endpoint, _api in fallback_apis]}
            if fallback_apis
            else {}
        ),
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
    fallback_api: HfApi | None = None,
    fallback_apis: list[tuple[str, HfApi]] | None = None,
    fallback_endpoint: str | None = None,
    request_timeout_seconds: int = 30,
) -> dict[str, Any]:
    repos: dict[str, dict[str, Any]] = {}
    search_errors = []
    search_fallbacks = []
    for query in _queries(model_id, model):
        try:
            with _request_timeout(request_timeout_seconds):
                matches = list(api.list_models(search=query, limit=limit))
        except Exception as error:  # noqa: BLE001 - keep partial search evidence.
            fallback_entries = fallback_apis or (
                [(str(fallback_endpoint or ""), fallback_api)]
                if fallback_api is not None
                else []
            )
            if not fallback_entries or not _is_rate_limited_error(error):
                search_errors.append({"query": query, "error": str(error)})
                continue
            fallback_failures = []
            for candidate_endpoint, candidate_api in fallback_entries:
                try:
                    with _request_timeout(request_timeout_seconds):
                        matches = list(
                            candidate_api.list_models(search=query, limit=limit)
                        )
                except Exception as fallback_error:  # noqa: BLE001 - try next mirror.
                    fallback_failures.append(
                        {
                            "endpoint": candidate_endpoint,
                            "error": str(fallback_error),
                        }
                    )
                    continue
                search_fallbacks.append(
                    {"query": query, "endpoint": candidate_endpoint}
                )
                query_api = candidate_api
                break
            else:
                search_errors.append(
                    {
                        "query": query,
                        "error": str(error),
                        "fallback_errors": fallback_failures,
                    }
                )
                continue
        else:
            query_api = api
        for match in matches:
            entry = repos.setdefault(
                match.modelId,
                {"queries": set(), "api": query_api},
            )
            entry["queries"].add(query)

    runtime_candidates = []
    component_candidates = []
    tree_errors = []
    for repo, info in sorted(repos.items()):
        queries = info["queries"]
        repo_api = info["api"]
        paths = _artifact_paths(
            repo_api,
            repo,
            tree_limit=tree_limit,
            request_timeout_seconds=request_timeout_seconds,
        )
        if not paths:
            continue
        if len(paths) == 1 and paths[0].startswith("<tree-error:"):
            tree_errors.append({"repo": repo, "error": paths[0]})
            continue
        runtime_paths = [path for path in paths if not _is_component_path(path)]
        component_paths = [path for path in paths if _is_component_path(path)]
        if runtime_paths:
            runtime_candidates.append(
                {
                    "repo": repo,
                    "queries": sorted(queries),
                    "paths": runtime_paths,
                    **({"component_paths": component_paths} if component_paths else {}),
                }
            )
            continue
        component_candidates.append(
            {
                "repo": repo,
                "queries": sorted(queries),
                "paths": component_paths or paths,
            }
        )

    return {
        "id": model_id,
        "family": model.get("family"),
        "source_model": model.get("source_model"),
        "blocked_platforms": model.get("blocked_platforms") or {},
        "search_errors": search_errors,
        "search_fallbacks": search_fallbacks,
        "tree_errors": tree_errors,
        "runtime_candidates": runtime_candidates,
        "component_candidates": component_candidates,
    }


def _hf_api(endpoint: str | None) -> HfApi:
    return HfApi(endpoint=endpoint) if endpoint else HfApi()


def _fallback_api(
    *,
    endpoint: str | None,
    fallback_endpoint: str | None,
) -> HfApi | None:
    fallback = (fallback_endpoint or "").strip()
    if not fallback:
        return None
    primary = (endpoint or "https://huggingface.co").rstrip("/")
    if fallback.rstrip("/") == primary:
        return None
    return HfApi(endpoint=fallback)


def _fallback_apis(
    *,
    endpoint: str | None,
    fallback_endpoint: str | None,
) -> list[tuple[str, HfApi]]:
    primary = (endpoint or "https://huggingface.co").rstrip("/")
    result = []
    for fallback in _split_endpoints(fallback_endpoint):
        if fallback.rstrip("/") == primary:
            continue
        result.append((fallback, HfApi(endpoint=fallback)))
    return result


def _split_endpoints(value: str | None) -> list[str]:
    result = []
    seen = set()
    for raw in str(value or "").split(","):
        endpoint = raw.strip()
        if not endpoint or endpoint in seen:
            continue
        result.append(endpoint)
        seen.add(endpoint)
    return result


def _is_rate_limited_error(error: Exception) -> bool:
    text = str(error).lower()
    return (
        "429" in text
        or "too many requests" in text
        or "rate limit" in text
        or "rate limited" in text
    )


@contextmanager
def _request_timeout(seconds: int):
    if seconds <= 0:
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)

    def _raise_timeout(_signum, _frame) -> None:
        raise TimeoutError(f"request timed out after {seconds}s")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


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


def _artifact_paths(
    api: HfApi,
    repo: str,
    *,
    tree_limit: int,
    request_timeout_seconds: int = 30,
) -> list[str]:
    paths: list[str] = []
    seen = set()
    try:
        with _request_timeout(request_timeout_seconds):
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


def _prepare_out_path(requested: Path) -> Path:
    expanded = requested.expanduser()
    out_path = expanded if expanded.is_absolute() else Path.cwd() / expanded
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return out_path
    except OSError:
        fallback = _fallback_out_path(out_path)
        fallback.parent.mkdir(parents=True, exist_ok=True)
        return fallback


def _fallback_out_path(requested: Path) -> Path:
    benchmark_root = RUNTIME_DIR.parent
    out_root = benchmark_root / "out"
    try:
        relative = requested.relative_to(out_root)
        return benchmark_root / "out_local" / relative
    except ValueError:
        return benchmark_root / "out_local" / "runtime" / requested.name


if __name__ == "__main__":
    main()
