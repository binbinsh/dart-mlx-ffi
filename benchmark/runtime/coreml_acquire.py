from __future__ import annotations

import argparse
import json
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
) -> list[dict[str, Any]]:
    api = HfApi()
    candidates = []
    seen = set()
    for query in _queries(source_model):
        for model in api.list_models(search=query, limit=limit):
            repo = model.modelId
            if repo in seen:
                continue
            seen.add(repo)
            paths = _coreml_paths(api, repo, tree_limit=tree_limit)
            if paths:
                candidates.append({"repo": repo, "query": query, "paths": paths})
    return candidates


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
