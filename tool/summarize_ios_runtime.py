#!/usr/bin/env python3
"""Summarize the current PaddleOCR-VL iPhone runtime probe logs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

UNIT = {"KB": 1024, "MB": 1024**2, "GB": 1024**3}

DEFAULT_LOGS = {
    "baseline": Path("/tmp/ios_runtime_probe2/launch_1/photo_render_512/live.log"),
    "wired0": Path("/tmp/ios_runtime_wired_override/launch_1/photo_render_512/live.log"),
    "streamstats": Path("/tmp/ios_runtime_streamstats/launch_1/photo_render_512/live.log"),
    "cachecount": Path("/tmp/ios_runtime_cachecount/launch_1/photo_render_512/live.log"),
    "sync_each_token": Path("/tmp/ios_runtime_sync_trace/launch_1/photo_render_512/live.log"),
    "state_detach": Path("/tmp/ios_runtime_state_detach2/launch_1/photo_render_512/live.log"),
}

LINE_RE = re.compile(
    r"decoderTail offset=(?P<offset>\d+) "
    r"step=(?P<step>\w+) .*?"
    r"active=(?P<active>[0-9.]+)(?P<active_unit>[KMG]B).*?"
    r"rsrc=(?P<rsrc>\d+)/(?P<rsrc_limit>\d+)"
    r"(?: cacheCount=(?P<cache_count>\d+))?"
    r"(?: commits=(?P<commits>\d+))?"
    r"(?: pendingOut=(?P<pending>\d+))?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="optional output path; stdout when omitted",
    )
    return parser.parse_args()


def parse_log(path: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    if not path.exists():
      return {"exists": False, "path": str(path)}
    for line in path.read_text().splitlines():
        match = LINE_RE.search(line)
        if not match:
            continue
        groups = match.groupdict()
        rows.append(
            {
                "offset": int(groups["offset"]),
                "step": groups["step"],
                "active_bytes": float(groups["active"]) * UNIT[groups["active_unit"]],
                "resource_count": int(groups["rsrc"]),
                "resource_limit": int(groups["rsrc_limit"]),
                "cache_count": int(groups["cache_count"] or 0),
                "commits": int(groups["commits"] or 0),
                "pending_outputs": int(groups["pending"] or 0),
                "line": line,
            }
        )
    summary: dict[str, object] = {
        "exists": True,
        "path": str(path),
        "rows": len(rows),
        "overall_cache_count_max": max((row["cache_count"] for row in rows), default=0),
        "overall_pending_max": max((row["pending_outputs"] for row in rows), default=0),
        "steps": {},
    }
    for step in ("forward_total", "sample_token", "sync_per_token"):
        points = [row for row in rows if row["step"] == step]
        if not points:
            continue
        first = points[0]
        last = points[-1]
        summary["steps"][step] = {
            "count": len(points),
            "offset_first": first["offset"],
            "offset_last": last["offset"],
            "active_mb_first": round(first["active_bytes"] / UNIT["MB"], 1),
            "active_mb_last": round(last["active_bytes"] / UNIT["MB"], 1),
            "active_mb_delta": round(
                (last["active_bytes"] - first["active_bytes"]) / UNIT["MB"], 2
            ),
            "resource_first": first["resource_count"],
            "resource_last": last["resource_count"],
            "resource_delta": last["resource_count"] - first["resource_count"],
            "cache_count_min": min(point["cache_count"] for point in points),
            "cache_count_max": max(point["cache_count"] for point in points),
            "commits_delta": last["commits"] - first["commits"],
            "pending_max": max(point["pending_outputs"] for point in points),
            "first_line": first["line"],
            "last_line": last["line"],
        }
    return summary


def to_markdown(data: dict[str, dict[str, object]]) -> str:
    lines = [
        "# iOS Runtime Summary",
        "",
        "| Experiment | forward_total ΔMB | forward_total Δresources | cacheCount max | commits Δ | pendingOut max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, summary in data.items():
        if not summary.get("exists"):
            lines.append(f"| {name} | missing | missing | missing | missing | missing |")
            continue
        steps = summary.get("steps", {})
        ft = steps.get("forward_total")
        if not ft:
            lines.append(f"| {name} | no data | no data | no data | no data | no data |")
            continue
        lines.append(
            "| {name} | {mb} | {rsrc} | {cache_max} | {commits} | {pending} |".format(
                name=name,
                mb=ft["active_mb_delta"],
                rsrc=ft["resource_delta"],
                cache_max=summary["overall_cache_count_max"],
                commits=ft["commits_delta"],
                pending=summary["overall_pending_max"],
            )
        )
    lines.append("")
    for name, summary in data.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"- Log: `{summary['path']}`")
        if not summary.get("exists"):
            lines.append("- Status: missing")
            lines.append("")
            continue
        for step_name, step_summary in summary.get("steps", {}).items():
            lines.append(
                "- `{}`: offsets {} -> {}, active Δ {} MB, resource Δ {}, cacheCount max {}, commits Δ {}, pendingOut max {}".format(
                    step_name,
                    step_summary["offset_first"],
                    step_summary["offset_last"],
                    step_summary["active_mb_delta"],
                    step_summary["resource_delta"],
                    summary["overall_cache_count_max"],
                    step_summary["commits_delta"],
                    summary["overall_pending_max"],
                )
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    data = {name: parse_log(path) for name, path in DEFAULT_LOGS.items()}
    rendered = json.dumps(data, indent=2) if args.json else to_markdown(data)
    if args.out is not None:
        args.out.write_text(rendered, encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
