from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from hf_download import DEFAULT_FALLBACK_ENDPOINT
from ming_omni_source import DEFAULT_FALLBACK_REPOS, prepare_patched_source


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export Ming-omni TTS ONNX with patched dynamic-module sources "
            "when the model repo misses auto_map Python files."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--task", default="text-to-audio")
    parser.add_argument("--opset", default="18")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT"))
    parser.add_argument(
        "--fallback-endpoint",
        default=os.environ.get("HF_FALLBACK_ENDPOINT") or DEFAULT_FALLBACK_ENDPOINT,
    )
    parser.add_argument(
        "--fallback-repo",
        action="append",
        default=[],
        help="Fallback HF repo used to source missing dynamic-module files.",
    )
    args, passthrough = parser.parse_known_args()

    output_dir = args.output_dir.expanduser().resolve()
    source_root = output_dir / "_patched_source"
    source_dir, patch_report = prepare_patched_source(
        source_model=args.model,
        work_dir=source_root,
        fallback_repos=args.fallback_repo or DEFAULT_FALLBACK_REPOS,
        endpoint=args.endpoint,
        fallback_endpoint=args.fallback_endpoint,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "source_patch_report.json").write_text(
        json.dumps(patch_report, indent=2, ensure_ascii=False) + "\n",
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


if __name__ == "__main__":
    main()
