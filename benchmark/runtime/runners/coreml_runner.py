from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_schema import base_parser


def main() -> None:
    parser = base_parser("coreml")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--warmup", default="1")
    parser.add_argument("--iters", default="5")
    parser.add_argument("--num-threads")
    parser.add_argument("--coreml-mode", choices=["decode", "prefill"])
    args = parser.parse_args()
    _run_dart(args, "coreml")


def _run_dart(args, engine: str) -> None:
    cmd = [
        "dart",
        "run",
        "benchmark/runtime/dart_runtime_runner.dart",
        "--model-id",
        args.model_id,
        "--engine",
        engine,
        "--artifact",
        args.artifact,
        "--input-json",
        args.input_json,
        "--warmup",
        args.warmup,
        "--iters",
        args.iters,
        "--platform",
        args.platform,
    ]
    if args.out:
        cmd.extend(["--out", str(args.out)])
    if args.num_threads:
        cmd.extend(["--num-threads", args.num_threads])
    if args.hf_cache_root:
        cmd.extend(["--hf-cache-root", args.hf_cache_root])
    if args.coreml_mode:
        cmd.extend(["--coreml-mode", args.coreml_mode])
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
