from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from gguf_coreml_bridge import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TOOLS_ROOT,
    build_plan as build_gguf_plan,
    coreml_llm_command,
    _safe_name,
)


RUNTIME_DIR = Path(__file__).resolve().parent
ROOT = RUNTIME_DIR.parents[1]


COREML_SUFFIXES = (".mlmodelc", ".mlpackage")
DEFAULT_FALLBACK_ENDPOINT = "https://hf-mirror.com"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Try the convenient Core ML acquisition paths: existing HF Core ML "
            "artifact first, then CoreML-LLM conversion from the original HF source."
        )
    )
    parser.add_argument("--source-model")
    parser.add_argument("--gguf", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tools-root", type=Path, default=DEFAULT_TOOLS_ROOT)
    parser.add_argument("--context-length", default="2048")
    parser.add_argument("--quantize", default="int4")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--tree-limit", type=int, default=2000)
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT"))
    parser.add_argument(
        "--fallback-endpoint",
        default=os.environ.get("HF_FALLBACK_ENDPOINT") or DEFAULT_FALLBACK_ENDPOINT,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compatibility flag; this tool plans by default unless --execute-conversion is set.",
    )
    parser.add_argument("--execute-conversion", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    payload = build_acquisition_plan(
        source_model=args.source_model,
        gguf=args.gguf,
        output_dir=args.output_dir,
        tools_root=args.tools_root,
        context_length=args.context_length,
        quantize=args.quantize,
        trust_remote_code=args.trust_remote_code,
        limit=args.limit,
        tree_limit=args.tree_limit,
        endpoint=args.endpoint,
        fallback_endpoint=args.fallback_endpoint,
    )
    if args.execute_conversion and payload["recommended_action"] == "convert_coreml_llm":
        completed = subprocess.run(payload["conversion_command"], cwd=ROOT, check=False)
        payload["conversion_returncode"] = completed.returncode
        payload["state"] = "converted" if completed.returncode == 0 else "failed"
    _write_payload(payload, args.out)
    if args.execute_conversion and payload.get("conversion_returncode", 0) != 0:
        raise SystemExit(payload["conversion_returncode"])


def build_acquisition_plan(
    *,
    source_model: str | None,
    gguf: Path | None,
    output_dir: Path | None,
    tools_root: Path,
    context_length: str,
    quantize: str,
    trust_remote_code: bool,
    limit: int,
    tree_limit: int,
    endpoint: str | None = None,
    fallback_endpoint: str | None = DEFAULT_FALLBACK_ENDPOINT,
) -> dict[str, Any]:
    gguf_plan = None
    if gguf is not None:
        gguf_plan = build_gguf_plan(
            gguf=gguf,
            source_model=source_model,
            output_dir=output_dir,
            tools_root=tools_root,
            context_length=context_length,
            quantize=quantize,
            trust_remote_code=trust_remote_code,
        )
        source_model = str(gguf_plan.get("source_model") or source_model or "")
    if not source_model:
        return {
            "state": "blocked",
            "recommended_action": "provide_source_model",
            "reason": "No source HF model was provided or inferred from GGUF metadata.",
            "gguf_plan": _compact_gguf_plan(gguf_plan),
        }
    existing = search_existing_coreml(
        source_model,
        limit=limit,
        tree_limit=tree_limit,
        endpoint=endpoint,
        fallback_endpoint=fallback_endpoint,
    )
    if existing:
        return {
            "state": "ready",
            "source_model": source_model,
            "recommended_action": "use_existing_hf_coreml",
            "reason": "A Hugging Face repository already exposes Core ML artifacts.",
            "coreml_candidates": existing,
            "gguf_plan": _compact_gguf_plan(gguf_plan),
        }
    out_dir = output_dir or DEFAULT_OUTPUT_ROOT / _safe_name(source_model) / "coreml"
    command = coreml_llm_command(
        tool_dir=tools_root / "coreml-llm",
        source_model=source_model,
        output_dir=out_dir,
        context_length=context_length,
        quantize=quantize,
        trust_remote_code=trust_remote_code,
    )
    return {
        "state": "planned",
        "source_model": source_model,
        "recommended_action": "convert_coreml_llm",
        "reason": "No ready HF Core ML artifact was found; convert from source HF weights.",
        "coreml_candidates": [],
        "conversion_command": command,
        "output_dir": str(out_dir),
        "gguf_plan": _compact_gguf_plan(gguf_plan),
    }


def search_existing_coreml(
    source_model: str,
    *,
    limit: int,
    tree_limit: int,
    endpoint: str | None = None,
    fallback_endpoint: str | None = DEFAULT_FALLBACK_ENDPOINT,
) -> list[dict[str, Any]]:
    api = _hf_api(endpoint)
    fallback_apis = _fallback_apis(
        endpoint=endpoint,
        fallback_endpoint=fallback_endpoint,
    )
    candidates = []
    seen = set()
    for query in _queries(source_model):
        try:
            matches = list(api.list_models(search=query, limit=limit))
            query_api = api
            query_endpoint = endpoint or "https://huggingface.co"
        except Exception as error:  # noqa: BLE001 - retry mirror on primary rate limit.
            if not fallback_apis or not _is_rate_limited_error(error):
                continue
            for fallback_endpoint_value, fallback_api in fallback_apis:
                try:
                    matches = list(fallback_api.list_models(search=query, limit=limit))
                except Exception:  # noqa: BLE001 - try the next mirror.
                    continue
                query_api = fallback_api
                query_endpoint = fallback_endpoint_value
                break
            else:
                continue
        for model in matches:
            repo = model.modelId
            if repo in seen:
                continue
            seen.add(repo)
            paths = _coreml_paths(query_api, repo, tree_limit=tree_limit)
            if paths:
                candidates.append(
                    {
                        "repo": repo,
                        "query": query,
                        "paths": paths,
                        "endpoint": query_endpoint,
                    }
                )
    return candidates


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


def _queries(source_model: str) -> list[str]:
    leaf = source_model.split("/")[-1]
    return [
        f"{leaf} CoreML",
        f"{leaf} mlmodelc",
        f"{leaf} mlpackage",
        f"{source_model} CoreML",
    ]


def _coreml_paths(api: HfApi, repo: str, *, tree_limit: int) -> list[str]:
    paths = []
    seen = set()
    try:
        for index, item in enumerate(api.list_repo_tree(repo, recursive=True)):
            if index >= tree_limit:
                break
            root = _coreml_root(str(getattr(item, "path", "")))
            if root is None or root in seen:
                continue
            seen.add(root)
            paths.append(root)
    except Exception as error:  # noqa: BLE001 - report failures as candidates.
        return [f"<tree-error: {error}>"]
    return sorted(paths)


def _coreml_root(path: str) -> str | None:
    lower = path.lower()
    for suffix in COREML_SUFFIXES:
        marker = suffix + "/"
        if marker in lower:
            return path[: lower.index(marker) + len(suffix)]
        if lower.endswith(suffix):
            return path
    return None


def _compact_gguf_plan(plan: dict[str, Any] | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    return {
        "gguf": plan.get("gguf"),
        "state": plan.get("state"),
        "direct_gguf_to_coreml": plan.get("direct_gguf_to_coreml"),
        "source_model": plan.get("source_model"),
        "architecture": plan.get("architecture"),
        "tensors": plan.get("tensors"),
    }


def _write_payload(payload: dict[str, Any], out: Path | None) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
