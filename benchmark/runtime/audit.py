from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from matrix_config import (
    artifact_coverage,
    artifact_unblocks_platform,
    blocked_platform_reason,
    fallback_reason,
    preferred_engine_for_platform,
    production_platforms,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit runtime matrix evidence required for production support."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("models.yaml"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("benchmark/out/runtime"),
        help="Directory containing <model>/<platform>/verdict.json reports.",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        help=(
            "Optional artifact map. Converted platform entries with "
            "artifact_source: converted unblock catalog-level HF gaps."
        ),
    )
    parser.add_argument("--model-id")
    parser.add_argument(
        "--search-report",
        type=Path,
        help="Optional Hugging Face search report used as blocker evidence.",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    payload = audit(
        config_path=args.config,
        out_root=args.out_root,
        artifacts_path=args.artifacts,
        model_id=args.model_id,
        search_report_path=args.search_report,
    )
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


def audit(
    *,
    config_path: Path,
    out_root: Path,
    artifacts_path: Path | None = None,
    model_id: str | None = None,
    search_report_path: Path | None = None,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    artifacts = _read_yaml(artifacts_path) if artifacts_path else {}
    search_report = _read_json(search_report_path) if search_report_path else {}
    required_platforms = production_platforms(config)
    models = config.get("first_wave") or []
    if model_id:
        models = [model for model in models if model.get("id") == model_id]
    records = [
        _audit_model(
            model,
            platforms=required_platforms,
            out_root=out_root,
            artifact_model=((artifacts.get("models") or {}).get(model.get("id")) or {}),
            search_record=_search_record(search_report, str(model.get("id") or "")),
            search_report_path=search_report_path,
        )
        for model in models
    ]
    missing = sum(item["missing_count"] for item in records)
    failed = sum(item["failed_count"] for item in records)
    passed = sum(item["passed_count"] for item in records)
    blocked = sum(item["blocked_count"] for item in records)
    return {
        "config": str(config_path),
        "out_root": str(out_root),
        "required_platforms": required_platforms,
        "model_count": len(records),
        "passed_evidence_count": passed,
        "failed_evidence_count": failed,
        "missing_evidence_count": missing,
        "blocked_evidence_count": blocked,
        "remaining_to_production": missing + failed + blocked,
        "models": records,
    }


def _audit_model(
    model: dict[str, Any],
    *,
    platforms: list[str],
    out_root: Path,
    artifact_model: dict[str, Any],
    search_record: dict[str, Any] | None,
    search_report_path: Path | None,
) -> dict[str, Any]:
    model_id = model["id"]
    platform_records = []
    for platform in platforms:
        path = out_root / model_id / platform / "verdict.json"
        state = "missing"
        passed = False
        gates: dict[str, bool] = {}
        failed_checks: list[str] = []
        blocker = None
        preferred = preferred_engine_for_platform(model, platform)
        fallback = fallback_reason(
            artifact_model,
            platform,
            preferred_engine=preferred,
        )
        if not artifact_unblocks_platform(artifact_model, platform):
            blocker = blocked_platform_reason(
                artifact_model, platform
            ) or blocked_platform_reason(model, platform)
        if fallback:
            state = "blocked"
            failed_checks = [f"artifact_fallback:{fallback}"]
        elif blocker:
            state = "blocked"
            failed_checks = [f"artifact_coverage:{blocker}"]
        elif path.exists():
            verdict = _read_verdict(path)
            gates = _gate_status(verdict)
            failed_checks = _failed_checks(verdict)
            passed = bool(verdict.get("passed")) and all(gates.values())
            state = "passed" if passed else "failed"
        record = {
            "platform": platform,
            "state": state,
            "path": str(path),
            "gates": gates,
            "failed_checks": failed_checks,
        }
        if fallback or blocker:
            record["blocked_reason"] = fallback or blocker
            evidence = _search_evidence(search_record, search_report_path)
            if evidence:
                record["search_evidence"] = evidence
        platform_records.append(record)
    passed_count = sum(1 for item in platform_records if item["state"] == "passed")
    failed_count = sum(1 for item in platform_records if item["state"] == "failed")
    missing_count = sum(1 for item in platform_records if item["state"] == "missing")
    blocked_count = sum(1 for item in platform_records if item["state"] == "blocked")
    return {
        "id": model_id,
        "family": model.get("family"),
        "support_level": model.get("support_level"),
        "artifact_coverage": _effective_coverage(model, artifact_model),
        "production_ready": (
            failed_count == 0 and missing_count == 0 and blocked_count == 0
        ),
        "passed_count": passed_count,
        "failed_count": failed_count,
        "missing_count": missing_count,
        "blocked_count": blocked_count,
        "platforms": platform_records,
    }


def _read_verdict(path: Path) -> dict[str, Any]:
    decoded = json.loads(path.read_text(encoding="utf-8"))
    if "verdict" in decoded and isinstance(decoded["verdict"], dict):
        return decoded["verdict"]
    if isinstance(decoded, dict):
        return decoded
    return {"passed": False}


def _read_yaml(path: Path) -> dict[str, Any]:
    decoded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return decoded if isinstance(decoded, dict) else {}


def _read_json(path: Path) -> dict[str, Any]:
    decoded = json.loads(path.read_text(encoding="utf-8"))
    return decoded if isinstance(decoded, dict) else {}


def _search_record(
    report: dict[str, Any],
    model_id: str,
) -> dict[str, Any] | None:
    for record in report.get("models") or []:
        if isinstance(record, dict) and str(record.get("id") or "") == model_id:
            return record
    return None


def _search_evidence(
    record: dict[str, Any] | None,
    report_path: Path | None,
) -> dict[str, Any] | None:
    if not record:
        return None
    runtime_candidates = record.get("runtime_candidates") or []
    component_candidates = record.get("component_candidates") or []
    evidence: dict[str, Any] = {
        "runtime_candidate_count": len(runtime_candidates),
        "component_candidate_count": len(component_candidates),
    }
    if report_path is not None:
        evidence["report"] = str(report_path)
    if runtime_candidates:
        evidence["runtime_repos"] = _candidate_repos(runtime_candidates)
    if component_candidates:
        evidence["component_repos"] = _candidate_repos(component_candidates)
    return evidence


def _candidate_repos(candidates: list[Any]) -> list[str]:
    repos = []
    seen = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        repo = str(candidate.get("repo") or "")
        if repo and repo not in seen:
            repos.append(repo)
            seen.add(repo)
    return repos


def _effective_coverage(
    model: dict[str, Any],
    artifact_model: dict[str, Any],
) -> str:
    coverage = artifact_coverage(artifact_model)
    return artifact_coverage(model) if coverage == "full" else coverage


def _gate_status(verdict: dict[str, Any]) -> dict[str, bool]:
    return {
        "identity": bool((verdict.get("identity") or {}).get("passed")),
        "correctness": bool((verdict.get("correctness") or {}).get("passed")),
        "speed": bool((verdict.get("speed") or {}).get("passed")),
        "peak_memory": bool((verdict.get("peak_memory") or {}).get("passed")),
        "device_profile": bool(
            (verdict.get("device_profile") or {}).get("passed")
        ),
    }


def _failed_checks(verdict: dict[str, Any]) -> list[str]:
    failed = []
    for section in ("identity", "correctness", "speed", "peak_memory", "device_profile"):
        data = verdict.get(section) or {}
        for check in data.get("checks", []):
            if isinstance(check, dict) and check.get("passed") is False:
                failed.append(f"{section}:{check.get('name', 'unknown')}")
    return failed


if __name__ == "__main__":
    main()
