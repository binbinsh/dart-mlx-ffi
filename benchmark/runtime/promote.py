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
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build manifest-compatible promotion metadata from verdicts."
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
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fail-on-blocked", action="store_true")
    args = parser.parse_args()

    payload = build_promotion_patch(
        config_path=args.config,
        out_root=args.out_root,
        artifacts_path=args.artifacts,
        model_id=args.model_id,
    )
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)

    if args.fail_on_blocked and payload["blocked_count"] > 0:
        raise SystemExit(1)


def build_promotion_patch(
    *,
    config_path: Path,
    out_root: Path,
    artifacts_path: Path | None = None,
    model_id: str | None = None,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    artifacts = _read_yaml(artifacts_path) if artifacts_path else {}
    required_platforms = (
        (config.get("support_policy") or {})
        .get("production_requires", {})
        .get("platforms", [])
    )
    models = config.get("first_wave") or []
    if model_id:
        models = [model for model in models if model.get("id") == model_id]

    records = [
        _model_patch(
            model,
            required_platforms=required_platforms,
            out_root=out_root,
            artifact_model=((artifacts.get("models") or {}).get(model.get("id")) or {}),
        )
        for model in models
    ]
    promoted = [record for record in records if record["supportLevel"] == "production"]
    return {
        "version": 1,
        "source": {
            "config": str(config_path),
            "outRoot": str(out_root),
            "requiredPlatforms": required_platforms,
        },
        "promoted_count": len(promoted),
        "blocked_count": len(records) - len(promoted),
        "models": records,
    }


def _model_patch(
    model: dict[str, Any],
    *,
    required_platforms: list[str],
    out_root: Path,
    artifact_model: dict[str, Any],
) -> dict[str, Any]:
    model_id = model["id"]
    statuses: dict[str, Any] = {}
    missing: list[str] = []
    failed: list[str] = []
    blocked: dict[str, str] = {}
    for platform in required_platforms:
        blocker = None
        fallback = fallback_reason(artifact_model, platform)
        if not artifact_unblocks_platform(artifact_model, platform):
            blocker = blocked_platform_reason(model, platform)
        if fallback or blocker:
            blocked[platform] = fallback or blocker or ""
            continue
        verdict_path = out_root / model_id / platform / "verdict.json"
        if not verdict_path.exists():
            missing.append(platform)
            continue
        status = _status_from_verdict(verdict_path, model_id, platform)
        statuses[platform] = status
        if not (
            status["promotionPassed"]
            and status["correctnessPassed"]
            and status["speedPassed"]
            and status["peakMemoryPassed"]
            and status["deviceProfilePassed"]
        ):
            failed.append(platform)

    production_ready = not missing and not failed and not blocked
    notes = []
    if blocked:
        notes.append(
            "Blocked artifact platforms: "
            + ", ".join(f"{platform} ({reason})" for platform, reason in blocked.items())
        )
    if missing:
        notes.append(f"Missing verdicts: {', '.join(missing)}")
    if failed:
        notes.append(f"Failed verdicts: {', '.join(failed)}")

    record = {
        "id": model_id,
        "family": model.get("family"),
        "supportLevel": "production" if production_ready else "staging",
        "productionReady": production_ready,
        "artifactCoverage": _effective_coverage(model, artifact_model),
        "validationStatus": statuses,
    }
    if blocked:
        record["blockedPlatforms"] = blocked
    if notes:
        record["notes"] = notes
    return record


def _status_from_verdict(
    verdict_path: Path,
    model_id: str,
    platform: str,
) -> dict[str, Any]:
    payload = _read_json(verdict_path)
    verdict = payload.get("verdict") or {}
    candidate = payload.get("candidate") or {}
    baseline = payload.get("baseline") or {}
    report_path = verdict_path.with_name("report.json")
    status = {
        "platform": platform,
        "engine": _engine_for(candidate),
        "correctnessPassed": bool((verdict.get("correctness") or {}).get("passed")),
        "speedPassed": bool((verdict.get("speed") or {}).get("passed")),
        "peakMemoryPassed": bool((verdict.get("peak_memory") or {}).get("passed")),
        "deviceProfilePassed": bool(
            (verdict.get("device_profile") or {}).get("passed")
        ),
        "promotionPassed": bool(verdict.get("passed")),
        "reportPath": str(report_path if report_path.exists() else verdict_path),
        "deviceProfile": candidate.get("device_profile") or {},
    }
    speed_ratio = _first_check_value(
        verdict,
        "speed",
        [
            "prefill_tokens_per_second_ratio",
            "decode_tokens_per_second_ratio",
        ],
    )
    if speed_ratio is not None:
        status["speedRatio"] = speed_ratio
    ttft_ratio = _first_check_value(verdict, "speed", ["ttft_ratio"])
    if ttft_ratio is not None:
        status["ttftRatio"] = ttft_ratio
    peak_ratio = _first_check_value(verdict, "peak_memory", ["peak_memory_ratio"])
    if peak_ratio is not None:
        status["peakMemoryRatio"] = peak_ratio

    candidate_peak = (candidate.get("metrics") or {}).get("peak_memory_bytes")
    baseline_peak = (baseline.get("metrics") or {}).get("peak_memory_bytes")
    if candidate_peak is not None:
        status["peakMemoryBytes"] = int(candidate_peak)
    if baseline_peak is not None:
        status["baselinePeakMemoryBytes"] = int(baseline_peak)

    notes = _status_notes(payload, model_id, platform)
    if notes:
        status["notes"] = notes
    return status


def _engine_for(candidate: dict[str, Any]) -> str:
    engine = candidate.get("engine")
    if engine == "onnx":
        return "onnx"
    if engine == "litert":
        return "litert"
    if engine == "coreml":
        return "coreml"
    if engine == "mlx":
        return "mlx"
    return "mlx"


def _first_check_value(
    verdict: dict[str, Any],
    section: str,
    names: list[str],
) -> float | None:
    checks = ((verdict.get(section) or {}).get("checks") or [])
    for name in names:
        for check in checks:
            if isinstance(check, dict) and check.get("name") == name:
                value = check.get("value")
                return float(value) if value is not None else None
    return None


def _status_notes(payload: dict[str, Any], model_id: str, platform: str) -> list[str]:
    notes = []
    mismatches = payload.get("mismatches") or []
    if mismatches:
        notes.append(f"Report identity mismatch for {model_id}/{platform}")
    verdict = payload.get("verdict") or {}
    for section in ("correctness", "speed", "peak_memory", "device_profile"):
        data = verdict.get(section) or {}
        if data.get("passed") is False:
            failed = [
                check.get("name", "unknown")
                for check in data.get("checks", [])
                if isinstance(check, dict) and check.get("passed") is False
            ]
            notes.append(f"{section} failed: {', '.join(failed) or 'unknown'}")
    return notes


def _read_json(path: Path) -> dict[str, Any]:
    decoded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return decoded


def _read_yaml(path: Path) -> dict[str, Any]:
    decoded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return decoded if isinstance(decoded, dict) else {}


def _effective_coverage(
    model: dict[str, Any],
    artifact_model: dict[str, Any],
) -> str:
    coverage = artifact_coverage(artifact_model)
    return artifact_coverage(model) if coverage == "full" else coverage


if __name__ == "__main__":
    main()
