from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_schema import base_parser, normalize_report, read_report, write_report


def main() -> None:
    parser = base_parser("mlx")
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
        write_report(
            normalize_report(
                read_report(args.raw_report),
                model_id=args.model_id,
                platform=args.platform,
                engine="mlx",
                artifact=args.artifact,
            ),
            args.out,
        )
        return

    publish_model_id = args.publish_model_id or args.model_id
    item = _find_publish_item(args.publish_report, publish_model_id)
    ms_key = "dart_ms" if args.implementation == "dart" else "python_ms"
    metrics = {
        "end_to_end_ms": item.get(ms_key),
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


if __name__ == "__main__":
    main()
