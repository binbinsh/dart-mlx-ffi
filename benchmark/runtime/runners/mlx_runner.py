from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_schema import base_parser, normalize_report, read_report, write_report


def main() -> None:
    parser = base_parser("mlx")
    parser.add_argument("--warmup", default="1")
    parser.add_argument("--iters", default="5")
    parser.add_argument(
        "--raw-report",
        type=Path,
        help="Existing standard runtime report to normalize.",
    )
    parser.add_argument(
        "--publish-report",
        type=Path,
        default=Path("benchmark/out/publish_report.json"),
        help="Existing benchmark/publish_report.py JSON output.",
    )
    parser.add_argument(
        "--publish-model-id",
        help="Optional model id key used to locate one row in publish_report.json.",
    )
    parser.add_argument(
        "--implementation",
        choices=["dart", "python"],
        default="dart",
        help="Which timing column from publish_report.py to expose.",
    )
    args = parser.parse_args()

    if args.raw_report:
        report = normalize_report(
            read_report(args.raw_report),
            model_id=args.model_id,
            platform=args.platform,
            engine="mlx",
            artifact=args.artifact,
        )
        report.setdefault("task", args.task)
        report.setdefault("run_config", _run_config(args))
        _patch_metrics_counts(report, args)
        write_report(report, args.out)
        return

    publish_model_id = args.publish_model_id or args.model_id
    item = _find_publish_item(args.publish_report, publish_model_id)
    ms_key = "dart_ms" if args.implementation == "dart" else "python_ms"
    metrics = {
        "end_to_end_ms": item.get(ms_key),
        "iteration_count": int(args.iters),
        "warmup_count": int(args.warmup),
    }
    if "peak_memory_bytes" in item:
        metrics["peak_memory_bytes"] = item["peak_memory_bytes"]
    correctness = {
        key: item[key]
        for key in ("max_abs_diff", "mean_abs_diff", "comparison", "input_desc")
        if key in item
    }
    write_report(
        {
            "model_id": args.model_id,
            "platform": args.platform,
            "engine": "mlx",
            "artifact": args.artifact,
            "task": args.task,
            "run_config": _run_config(args),
            "correctness": correctness,
            "metrics": metrics,
            "device_profile": {
                "runtime": f"mlx-{args.implementation}",
                "source_report": str(args.publish_report),
            },
        },
        args.out,
    )


def _find_publish_item(path: Path, model_id: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run benchmark/publish_report.py first or "
            "pass --raw-report."
        )
    decoded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, list):
        raise ValueError(f"Publish report must be a JSON list: {path}")
    for item in decoded:
        if isinstance(item, dict) and item.get("model_id") == model_id:
            return item
    raise ValueError(f"Model {model_id!r} was not found in {path}")


def _run_config(args) -> dict:
    return {
        "format": "dart_mlx_ffi.run_config.v1",
        "task": args.task or "tensor",
        "warmup": int(args.warmup),
        "iters": int(args.iters),
        "max_tokens": int(args.max_tokens or 64),
        "sampling_strategy": "greedy",
    }


def _patch_metrics_counts(report: dict, args) -> None:
    metrics = report.setdefault("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
        report["metrics"] = metrics
    metrics.setdefault("iteration_count", int(args.iters))
    metrics.setdefault("warmup_count", int(args.warmup))


if __name__ == "__main__":
    main()
