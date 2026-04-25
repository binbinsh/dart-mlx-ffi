from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_schema import base_parser, normalize_report, read_report, write_report


RUNTIME_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = RUNTIME_DIR / "fixtures" / "text_prompt.txt"
DEFAULT_PACKAGE_DIR = RUNTIME_DIR.parent / "coreml-llm" / "swift_baseline"


def main() -> None:
    parser = base_parser("coreml-llm")
    parser.add_argument(
        "--raw-report",
        type=Path,
        help="JSON emitted by benchmark/coreml-llm baseline tooling.",
    )
    parser.add_argument(
        "--baseline-bin",
        default=os.environ.get("COREML_LLM_RUNNER"),
        help=(
            "Optional external CoreML-LLM-compatible runner. Defaults to "
            "`swift run coreml-llm-baseline` in benchmark/coreml-llm/swift_baseline."
        ),
    )
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=DEFAULT_PACKAGE_DIR,
        help="SwiftPM package directory used when --baseline-bin is omitted.",
    )
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument(
        "--task",
        choices=["text", "function", "embedding", "vlm"],
        default="text",
    )
    parser.add_argument("--tools-file", type=Path)
    parser.add_argument("--tools-json")
    parser.add_argument("--embedding-query")
    parser.add_argument("--embedding-query-file", type=Path)
    parser.add_argument("--embedding-dim")
    parser.add_argument("--image-file", type=Path)
    parser.add_argument("--warmup", default="1")
    parser.add_argument("--iters", default="5")
    parser.add_argument("--max-tokens", default="64")
    parser.add_argument("--compute-units", default="cpuAndNeuralEngine")
    args = parser.parse_args()

    if args.raw_report:
        raw = read_report(args.raw_report)
        write_report(
            normalize_report(
                raw,
                model_id=args.model_id,
                platform=args.platform,
                engine="coreml-llm",
                artifact=args.artifact,
            ),
            args.out,
        )
        return

    cmd = _command(args)
    subprocess.run(cmd, check=True)


def _command(args) -> list[str]:
    if args.baseline_bin:
        cmd = [args.baseline_bin]
    else:
        cmd = [
            "swift",
            "run",
            "--package-path",
            str(args.package_dir.resolve()),
            "coreml-llm-baseline",
            "--",
        ]
    cmd.extend(
        [
            "--model-id",
            args.model_id,
            "--artifact",
            str(Path(args.artifact).resolve()),
            "--platform",
            args.platform,
            "--task",
            args.task,
            "--warmup",
            args.warmup,
            "--iters",
            args.iters,
            "--max-tokens",
            args.max_tokens,
            "--compute-units",
            args.compute_units,
        ]
    )
    if args.prompt is not None:
        cmd.extend(["--prompt", args.prompt])
    elif args.prompt_file is not None and not (
        args.task == "embedding"
        and (args.embedding_query is not None or args.embedding_query_file is not None)
    ):
        cmd.extend(["--prompt-file", str(args.prompt_file.resolve())])
    if args.tools_file is not None:
        cmd.extend(["--tools-file", str(args.tools_file.resolve())])
    if args.tools_json is not None:
        cmd.extend(["--tools-json", args.tools_json])
    if args.embedding_query is not None:
        cmd.extend(["--embedding-query", args.embedding_query])
    if args.embedding_query_file is not None:
        cmd.extend(["--embedding-query-file", str(args.embedding_query_file.resolve())])
    if args.embedding_dim is not None:
        cmd.extend(["--embedding-dim", args.embedding_dim])
    if args.image_file is not None:
        cmd.extend(["--image-file", str(args.image_file.resolve())])
    if args.out:
        cmd.extend(["--out", str(args.out.resolve())])
    return cmd


if __name__ == "__main__":
    main()
