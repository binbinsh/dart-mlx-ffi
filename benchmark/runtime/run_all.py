from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from matrix_config import (
    abandoned_platforms,
    production_platforms,
)
from run_all_support import (
    _abandoned_platform_cell,
    _artifact_health_failure,
    _cell_for,
    _command_option,
    _host_platform,
    _model_readiness,
    _model_readiness_counts,
    _readiness_class,
    _readiness_counts,
    _readiness_state,
    _run,
)
from run_matrix import _prepare_out_root

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or plan the full runtime validation matrix."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=RUNTIME_DIR / "models.yaml",
    )
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--platform")
    parser.add_argument("--engine", choices=["coreml", "onnx", "litert", "mlx"])
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "benchmark" / "out" / "runtime",
    )
    parser.add_argument("--plan-out", type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--prepare-inputs",
        action="store_true",
        help=(
            "Generate per-cell input_json fixtures with prepare_inputs.py "
            "before execution."
        ),
    )
    parser.add_argument("--allow-fail", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--include-abandoned-platforms",
        action="store_true",
        help=(
            "Allow explicit planning for platforms listed under "
            "support_policy.abandoned_platforms. They are excluded by default."
        ),
    )
    parser.add_argument(
        "--path-check",
        choices=["none", "run", "always"],
        default="run",
        help=(
            "Validate local artifact/report paths. 'run' checks only real "
            "executions, 'always' also checks planning, and 'none' disables it."
        ),
    )
    parser.add_argument(
        "--execution-check",
        choices=["none", "run", "always"],
        default="run",
        help=(
            "Validate that ready cells can execute from this host. 'run' "
            "checks only real executions, 'always' also checks planning, "
            "and 'none' disables it."
        ),
    )
    parser.add_argument(
        "--artifact-health-check",
        choices=["none", "run", "always"],
        default="run",
        help=(
            "Run native artifact load checks before execution. 'run' checks "
            "only real executions, 'always' includes the check command in "
            "plans, and 'none' disables it."
        ),
    )
    args = parser.parse_args()

    plan = build_plan(args)
    text = json.dumps(plan, indent=2, ensure_ascii=False)
    if args.plan_out:
        args.plan_out.parent.mkdir(parents=True, exist_ok=True)
        args.plan_out.write_text(text + "\n", encoding="utf-8")
    print(text)

    if args.run:
        result = run_plan(plan, allow_fail=args.allow_fail, dry_run=args.dry_run)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if not args.allow_fail and (result["failed_count"] or result["blocked_count"]):
            raise SystemExit(1)
    elif not args.allow_fail and plan["blocked_count"]:
        raise SystemExit(1)


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    config = _read_yaml(args.config)
    artifacts = _read_yaml(args.artifacts)
    resolved_out_root, _ = _prepare_out_root(args.out_root)
    required_platforms = production_platforms(config)
    abandoned = set(abandoned_platforms(config))
    models = config.get("first_wave") or []
    if args.model_id:
        models = [model for model in models if model.get("id") == args.model_id]
    check_paths = args.path_check == "always" or (
        args.path_check == "run" and args.run and not args.dry_run
    )
    check_execution = args.execution_check == "always" or (
        args.execution_check == "run" and args.run and not args.dry_run
    )
    artifact_health_check = getattr(args, "artifact_health_check", "run")
    check_artifact_health = artifact_health_check == "always" or (
        artifact_health_check == "run" and args.run and not args.dry_run
    )
    prepare_inputs = bool(getattr(args, "prepare_inputs", False))
    cells = []
    for model in models:
        model_id = model["id"]
        platforms = [args.platform] if args.platform else required_platforms
        for platform in platforms:
            if (
                platform in abandoned
                and not getattr(args, "include_abandoned_platforms", False)
            ):
                cells.append(
                    _abandoned_platform_cell(
                        model_id=model_id,
                        model_config=model,
                        platform=platform,
                        out_root=resolved_out_root,
                    )
                )
                continue
            cell = _cell_for(
                model_id=model_id,
                model_config=model,
                platform=platform,
                artifacts=artifacts,
                artifacts_path=args.artifacts,
                out_root=resolved_out_root,
                config=args.config,
                check_paths=check_paths,
                check_execution=check_execution,
                check_artifact_health=check_artifact_health,
                prepare_inputs=prepare_inputs,
            )
            if args.engine and cell.get("engine") != args.engine:
                continue
            cells.append(cell)
    for cell in cells:
        cell["readiness_state"] = _readiness_state(cell, resolved_out_root)
        cell["readiness_class"] = _readiness_class(cell)
    ready = [cell for cell in cells if cell["state"] == "ready"]
    blocked = [cell for cell in cells if cell["state"] == "blocked"]
    readiness_counts = _readiness_counts(cells)
    model_readiness = _model_readiness(cells)
    model_readiness_counts = _model_readiness_counts(model_readiness)
    artifact_ready_count = (
        readiness_counts["production-ready"]
        + readiness_counts["preferred-ready"]
        + readiness_counts["fallback-ready"]
    )
    return {
        "version": 1,
        "config": str(args.config),
        "artifacts": str(args.artifacts),
        "out_root": str(resolved_out_root),
        "path_check": args.path_check,
        "execution_check": args.execution_check,
        "artifact_health_check": artifact_health_check,
        "prepare_inputs": prepare_inputs,
        "host_platform": _host_platform(),
        "production_platforms": required_platforms,
        "abandoned_platforms": sorted(abandoned),
        "cell_count": len(cells),
        "readiness_counts": readiness_counts,
        "artifact_ready_count": artifact_ready_count,
        "ready_count": len(ready),
        "model_readiness_counts": model_readiness_counts,
        "production_model_count": model_readiness_counts["production-ready"],
        "production_ready_count": sum(
            1 for cell in cells if cell["readiness_state"] == "production_ready"
        ),
        "preferred_ready_count": sum(
            1 for cell in cells if cell["readiness_state"] == "preferred_ready"
        ),
        "fallback_ready_count": sum(
            1 for cell in cells if cell["readiness_state"] == "fallback_ready"
        ),
        "blocked_count": len(blocked),
        "model_readiness": model_readiness,
        "cells": cells,
    }

def run_plan(
    plan: dict[str, Any],
    *,
    allow_fail: bool,
    dry_run: bool,
) -> dict[str, Any]:
    results = []
    failed = 0
    for cell in plan["cells"]:
        if cell["state"] != "ready":
            continue
        cmd = list(cell["command"])
        prepare_cmd = list(cell.get("prepare_input_command") or [])
        health_cmd = list(cell.get("artifact_health_command") or [])
        if health_cmd and not dry_run:
            completed = _run(health_cmd, check=False)
            if completed.returncode != 0:
                failed += 1
                report_path = _command_option(health_cmd, "--out")
                failure = _artifact_health_failure(report_path)
                result = {
                    "model_id": cell["model_id"],
                    "platform": cell["platform"],
                    "engine": cell["engine"],
                    "passed": False,
                    "returncode": completed.returncode,
                    "stage": "artifact_health",
                    "report": report_path,
                }
                if failure:
                    if failure.get("failure_class"):
                        result["failure_class"] = failure["failure_class"]
                    if failure.get("failure_reason"):
                        result["failure_reason"] = failure["failure_reason"]
                results.append(result)
                if not allow_fail:
                    break
                continue
        if cell.get("device_smoke_only"):
            results.append(
                {
                    "model_id": cell["model_id"],
                    "platform": cell["platform"],
                    "engine": cell["engine"],
                    "passed": True,
                    "returncode": 0,
                    "stage": "device_smoke_only",
                    "report": _command_option(health_cmd, "--out"),
                }
            )
            continue
        if prepare_cmd and not dry_run:
            completed = _run(prepare_cmd, check=False)
            if completed.returncode != 0:
                failed += 1
                results.append(
                    {
                        "model_id": cell["model_id"],
                        "platform": cell["platform"],
                        "engine": cell["engine"],
                        "passed": False,
                        "returncode": completed.returncode,
                        "stage": "prepare_inputs",
                    }
                )
                if not allow_fail:
                    break
                continue
        if dry_run and "--dry-run" not in cmd:
            cmd.append("--dry-run")
        if allow_fail and "--allow-fail" not in cmd:
            cmd.append("--allow-fail")
        completed = _run(cmd, check=False)
        passed = completed.returncode == 0
        if not passed:
            failed += 1
        results.append(
            {
                "model_id": cell["model_id"],
                "platform": cell["platform"],
                "engine": cell["engine"],
                "passed": passed,
                "returncode": completed.returncode,
            }
        )
        if not allow_fail and not passed:
            break
    return {
        "ran_count": len(results),
        "failed_count": failed,
        "blocked_count": plan["blocked_count"],
        "results": results,
    }

def _read_yaml(path: Path) -> dict[str, Any]:
    decoded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return decoded if isinstance(decoded, dict) else {}


if __name__ == "__main__":
    main()
