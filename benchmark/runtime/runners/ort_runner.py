from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flutter_smoke_common import clear_runtime_env_file, write_runtime_env_file
from report_schema import base_parser
from runtime_backend_env import prepare_runtime_environment, runtime_build_env_values


def main() -> None:
    parser = base_parser("onnx")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--warmup", default="1")
    parser.add_argument("--iters", default="5")
    parser.add_argument("--num-threads")
    parser.add_argument("--provider")
    parser.add_argument("--require-provider", action="store_true")
    args = parser.parse_args()
    cmd = [
        "dart",
        "run",
        "benchmark/runtime/dart_runtime_runner.dart",
        "--model-id",
        args.model_id,
        "--engine",
        "onnx",
        "--artifact",
        args.artifact,
        "--task",
        args.task or "tensor",
        "--input-json",
        args.input_json,
        "--warmup",
        args.warmup,
        "--iters",
        args.iters,
        "--max-tokens",
        args.max_tokens,
        "--platform",
        args.platform,
    ]
    if args.out:
        cmd.extend(["--out", str(args.out)])
    if args.num_threads:
        cmd.extend(["--num-threads", args.num_threads])
    if args.hf_cache_root:
        cmd.extend(["--hf-cache-root", args.hf_cache_root])
    if args.provider:
        cmd.extend(["--provider", args.provider])
    if args.require_provider:
        cmd.append("--require-provider")
    run_env, _ = prepare_runtime_environment(
        engine="onnx",
        platform=args.platform,
        base_env=dict(os.environ),
    )
    runtime_env_file = write_runtime_env_file(runtime_build_env_values(run_env))
    run_env["DART_MLX_RUNTIME_ENV_FILE"] = str(runtime_env_file)
    try:
        subprocess.run(cmd, check=True, env=run_env)
    finally:
        clear_runtime_env_file(runtime_env_file)


if __name__ == "__main__":
    main()
