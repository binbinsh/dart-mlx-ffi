from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOOLS_ROOT = ROOT / "benchmark" / "artifacts" / "tools"
DEFAULT_OUTPUT_ROOT = ROOT / "benchmark" / "artifacts" / "converted"
COREML_LLM_REPO = "https://github.com/john-rocky/CoreML-LLM.git"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a GGUF file and route Core ML conversion through the "
            "original Hugging Face source model when available."
        )
    )
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--source-model")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tools-root", type=Path, default=DEFAULT_TOOLS_ROOT)
    parser.add_argument("--context-length", default="2048")
    parser.add_argument("--quantize", default="int4")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    payload = build_plan(
        gguf=args.gguf,
        source_model=args.source_model,
        output_dir=args.output_dir,
        tools_root=args.tools_root,
        context_length=args.context_length,
        quantize=args.quantize,
        trust_remote_code=args.trust_remote_code,
    )
    if args.dry_run:
        _write_payload(payload, args.out)
        return
    if not payload["source_model"]:
        _write_payload(payload, args.out)
        raise SystemExit(
            "GGUF cannot be converted directly to Core ML here. Pass "
            "--source-model <hf_org/hf_repo> so CoreML-LLM can convert from "
            "the original safetensors/PyTorch source."
        )
    _ensure_coreml_llm(Path(payload["tool_dir"]))
    completed = subprocess.run(payload["command"], cwd=ROOT, check=False)
    payload["returncode"] = completed.returncode
    payload["state"] = "converted" if completed.returncode == 0 else "failed"
    _write_payload(payload, args.out)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def build_plan(
    *,
    gguf: Path,
    source_model: str | None,
    output_dir: Path | None,
    tools_root: Path,
    context_length: str,
    quantize: str,
    trust_remote_code: bool,
) -> dict[str, Any]:
    metadata, tensors = read_gguf_summary(gguf)
    inferred_source = source_model or infer_source_model(metadata)
    architecture = str(metadata.get("general.architecture") or "")
    out_dir = output_dir or DEFAULT_OUTPUT_ROOT / _safe_name(
        inferred_source or gguf.stem
    ) / "coreml"
    tool_dir = tools_root / "coreml-llm"
    command = coreml_llm_command(
        tool_dir=tool_dir,
        source_model=inferred_source,
        output_dir=out_dir,
        context_length=context_length,
        quantize=quantize,
        trust_remote_code=trust_remote_code,
    )
    return {
        "gguf": str(gguf),
        "state": "planned" if inferred_source else "blocked",
        "direct_gguf_to_coreml": False,
        "reason": (
            "Core ML conversion is routed through the original HF model; GGUF "
            "is used only for metadata/source discovery."
        ),
        "source_model": inferred_source,
        "architecture": architecture,
        "metadata": summarize_metadata(metadata),
        "tensors": tensors,
        "tool_repo": COREML_LLM_REPO,
        "tool_dir": str(tool_dir),
        "output_dir": str(out_dir),
        "command": command,
    }


def read_gguf_summary(gguf: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        from gguf import GGUFReader
    except ImportError as error:
        raise SystemExit(
            "The gguf package is required. Run with "
            "`uv run --group gguf-tools python benchmark/runtime/gguf_coreml_bridge.py ...`."
        ) from error
    reader = GGUFReader(str(gguf))
    metadata = {
        str(field.name): _field_value(field)
        for field in getattr(reader, "fields", {}).values()
    }
    tensor_type_counts: dict[str, int] = {}
    tensor_count = 0
    for tensor in getattr(reader, "tensors", []):
        tensor_count += 1
        raw_type = getattr(tensor, "tensor_type", None)
        type_name = getattr(raw_type, "name", None) or str(raw_type)
        tensor_type_counts[type_name] = tensor_type_counts.get(type_name, 0) + 1
    return metadata, {"count": tensor_count, "type_counts": tensor_type_counts}


def infer_source_model(metadata: dict[str, Any]) -> str | None:
    candidates: list[tuple[int, str]] = []
    for key, value in metadata.items():
        text = _scalar_text(value)
        if not text:
            continue
        lowered = key.lower()
        if "repo_url" in lowered and "base_model" in lowered:
            candidates.append((0, text))
        elif "source" in lowered and ("repo" in lowered or "url" in lowered):
            candidates.append((1, text))
        elif "huggingface" in lowered and "repo" in lowered:
            candidates.append((2, text))
        elif "repo_url" in lowered:
            candidates.append((3, text))
    for _, value in sorted(candidates, key=lambda item: item[0]):
        repo = _repo_from_text(value)
        if repo:
            return repo
    return None


def summarize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: _summarize_value(value) for key, value in metadata.items()}


def coreml_llm_command(
    *,
    tool_dir: Path,
    source_model: str | None,
    output_dir: Path,
    context_length: str,
    quantize: str,
    trust_remote_code: bool,
) -> list[str]:
    if not source_model:
        return []
    cmd = [
        "uv",
        "run",
        "--with-requirements",
        str(tool_dir / "conversion" / "requirements.txt"),
        "python",
        str(tool_dir / "conversion" / "convert.py"),
        "--model",
        source_model,
        "--context-length",
        str(context_length),
        "--quantize",
        str(quantize),
        "--output",
        str(output_dir),
    ]
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    return cmd


def _field_value(field: Any) -> Any:
    contents = getattr(field, "contents", None)
    if callable(contents):
        return contents()
    parts = []
    for part in getattr(field, "parts", []):
        try:
            value = part.tolist()
        except AttributeError:
            value = part
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        parts.append(value)
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return parts


def _scalar_text(value: Any) -> str | None:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        for item in value:
            text = _scalar_text(item)
            if text:
                return text
    return None


def _summarize_value(value: Any) -> Any:
    if isinstance(value, str):
        if len(value) > 1000:
            return {
                "type": "string",
                "length": len(value),
                "preview": value[:1000],
            }
        return value
    if isinstance(value, list):
        if len(value) > 32:
            return {
                "type": "list",
                "length": len(value),
                "preview": [_summarize_value(item) for item in value[:16]],
            }
        return [_summarize_value(item) for item in value]
    return value


def _repo_from_text(value: str) -> str | None:
    parsed = urlparse(value)
    if parsed.netloc.endswith("huggingface.co"):
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
    match = re.search(r"(?<![\w.-])([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)(?![\w.-])", value)
    if match:
        return match.group(1)
    return None


def _ensure_coreml_llm(tool_dir: Path) -> None:
    if tool_dir.exists():
        return
    if shutil.which("git") is None:
        raise SystemExit("git is required to fetch CoreML-LLM.")
    tool_dir.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        ["git", "clone", "--depth", "1", COREML_LLM_REPO, str(tool_dir)],
        cwd=ROOT,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "gguf"


def _write_payload(payload: dict[str, Any], out: Path | None) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
