from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from hf_download import DEFAULT_FALLBACK_ENDPOINT, snapshot_download_with_fallback


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export ONNX from a local snapshot after patching config.json "
            "model_type for models that lag current exporter support."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--task", default="text-generation-with-past")
    parser.add_argument("--opset", default="18")
    parser.add_argument("--model-type-to", required=True)
    parser.add_argument("--model-type-from")
    parser.add_argument("--revision")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--allow-pattern", action="append", default=[])
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT"))
    parser.add_argument(
        "--fallback-endpoint",
        default=os.environ.get("HF_FALLBACK_ENDPOINT") or DEFAULT_FALLBACK_ENDPOINT,
    )
    args, passthrough = parser.parse_known_args()

    output_dir = args.output_dir.expanduser().resolve()
    source_root = output_dir / "_patched_source"
    source_dir = source_root / "source_model"
    allow_patterns = [str(item) for item in args.allow_pattern if str(item).strip()]
    snapshot_download_with_fallback(
        repo_id=args.model,
        revision=args.revision,
        local_dir=str(source_dir),
        allow_patterns=allow_patterns or None,
        endpoint=args.endpoint,
        fallback_endpoint=args.fallback_endpoint,
    )
    patch_report = _patch_model_type(
        config_path=source_dir / "config.json",
        expected=args.model_type_from,
        patched=args.model_type_to,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "source_patch_report.json").write_text(
        json.dumps(
            {
                "source_model": args.model,
                "source_dir": str(source_dir),
                **patch_report,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    cmd = [
        "optimum-cli",
        "export",
        "onnx",
        "--model",
        str(source_dir),
        "--task",
        args.task,
        "--opset",
        str(args.opset),
    ]
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    cmd.extend(passthrough)
    cmd.append(str(output_dir))
    completed = subprocess.run(cmd, check=False)
    raise SystemExit(completed.returncode)


def _patch_model_type(
    *,
    config_path: Path,
    expected: str | None,
    patched: str,
) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"config.json is missing in snapshot: {config_path}")
    decoded = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError(f"config.json is not a JSON object: {config_path}")
    original = str(decoded.get("model_type") or "")
    if expected and original and original != expected:
        raise ValueError(
            f"Expected model_type '{expected}' but found '{original}' in {config_path}"
        )
    decoded["model_type"] = patched
    config_path.write_text(
        json.dumps(decoded, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "config_path": str(config_path),
        "original_model_type": original,
        "patched_model_type": patched,
    }


if __name__ == "__main__":
    main()
