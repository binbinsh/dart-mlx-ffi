from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="8,12,16,18")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--token-limit", type=int, default=8)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sizes = [int(item.strip()) for item in args.sizes.split(",") if item.strip()]
    reports = []
    for size in sizes:
        proc = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "private_ane/models/qwen3_5_0_8b/python/private_grouped_shard_bench.py",
                "--grouped-shard-size",
                str(size),
                "--warmup",
                str(args.warmup),
                "--iters",
                str(args.iters),
                "--token-limit",
                str(args.token_limit),
                "--json",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            reports.append(
                {
                    "grouped_shard_size": size,
                    "ok": False,
                    "error": (proc.stderr or proc.stdout).strip(),
                }
            )
            continue
        reports.append({**json.loads(proc.stdout), "ok": True})

    ok_reports = [item for item in reports if item.get("ok")]
    best = None
    if ok_reports:
        best = min(
            ok_reports,
            key=lambda item: float(item["estimated_hybrid_forward_per_iter_ms"]),
        )

    payload = {
        "runtime": "qwen35_private_grouped_sweep",
        "sizes": sizes,
        "best_grouped_shard_size": None if best is None else best["grouped_shard_size"],
        "reports": reports,
    }

    if args.json:
        print(json.dumps(payload))
        return

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
