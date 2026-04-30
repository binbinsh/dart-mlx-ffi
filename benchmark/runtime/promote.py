from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from matrix_config import (
    artifact_coverage,
    artifact_unblocks_platform,
    blocked_engine_reason,
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
    platform_artifacts = _platform_artifacts_patch(
        model_id=model_id,
        artifact_model=artifact_model,
    )
    statuses: dict[str, Any] = {}
    missing: list[str] = []
    failed: list[str] = []
    blocked: dict[str, str] = {}
    for platform in required_platforms:
        blocker = None
        fallback = fallback_reason(artifact_model, platform)
        platform_cell = ((artifact_model.get("platforms") or {}).get(platform) or {})
        engine = platform_cell.get("engine") if isinstance(platform_cell, dict) else None
        if not artifact_unblocks_platform(artifact_model, platform):
            blocker = blocked_platform_reason(model, platform)
        if not blocker and isinstance(engine, str):
            blocker = blocked_engine_reason(artifact_model, platform, engine) or (
                blocked_engine_reason(model, platform, engine)
            )
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
            and status["identityPassed"]
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
    if platform_artifacts:
        record["platformArtifacts"] = platform_artifacts
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
        "identityPassed": bool((verdict.get("identity") or {}).get("passed")),
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
    end_to_end_ratio = _first_check_value(verdict, "speed", ["end_to_end_ratio"])
    if end_to_end_ratio is not None:
        status["endToEndRatio"] = end_to_end_ratio
    peak_ratio = _first_check_value(verdict, "peak_memory", ["peak_memory_ratio"])
    if peak_ratio is not None:
        status["peakMemoryRatio"] = peak_ratio

    candidate_peak = (candidate.get("metrics") or {}).get("peak_memory_bytes")
    baseline_peak = (baseline.get("metrics") or {}).get("peak_memory_bytes")
    if candidate_peak is not None:
        status["peakMemoryBytes"] = int(candidate_peak)
    if baseline_peak is not None:
        status["baselinePeakMemoryBytes"] = int(baseline_peak)

    candidate_metrics = candidate.get("metrics") or {}
    baseline_metrics = baseline.get("metrics") or {}
    if candidate_metrics.get("iteration_count") is not None:
        status["iterationCount"] = int(candidate_metrics["iteration_count"])
    if candidate_metrics.get("warmup_count") is not None:
        status["warmupCount"] = int(candidate_metrics["warmup_count"])
    latency = _latency_summary(candidate_metrics)
    if latency:
        status["latencyMs"] = latency
    baseline_latency = _latency_summary(baseline_metrics)
    if baseline_latency:
        status["baselineLatencyMs"] = baseline_latency
    if candidate.get("run_config") is not None:
        status["runConfig"] = candidate["run_config"]
    input_signature = _input_signature_for_status(candidate)
    if input_signature is not None:
        status["inputSignature"] = input_signature

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


def _platform_artifacts_patch(
    *,
    model_id: str,
    artifact_model: dict[str, Any],
) -> dict[str, Any]:
    platforms = artifact_model.get("platforms")
    if not isinstance(platforms, dict):
        return {}

    grouped: dict[str, dict[str, Any]] = {}
    for platform, raw in sorted(platforms.items()):
        if not isinstance(platform, str):
            continue
        if not isinstance(raw, dict):
            continue
        engine = raw.get("engine")
        artifact = raw.get("artifact")
        if not isinstance(engine, str) or not engine:
            continue
        if not isinstance(artifact, str) or not artifact:
            continue
        source_uri = raw.get("source_uri")
        if not isinstance(source_uri, str) or not source_uri:
            source_uri = artifact

        state = grouped.setdefault(
            engine,
            {
                "path": artifact,
                "sourceUri": source_uri,
                "targetPlatforms": [],
                "platformEntries": [],
            },
        )
        state["targetPlatforms"].append(platform)
        state["platformEntries"].append(
            {
                "platform": platform,
                "artifact": artifact,
                "sourceUri": source_uri,
                "artifactSource": raw.get("artifact_source"),
                "fallbackFrom": raw.get("fallback_from"),
            }
        )

    patch: dict[str, Any] = {}
    for engine, state in grouped.items():
        target_platforms = sorted({*state["targetPlatforms"]})
        patch[engine] = {
            "engine": engine,
            "path": state["path"],
            "sourceUri": state["sourceUri"],
            "format": _artifact_format(engine, str(state["path"])),
            "targetPlatforms": target_platforms,
            "accelerators": _accelerators_for_engine(engine),
            "metadata": {
                "source": "runtime_matrix",
                "modelId": model_id,
                "platformEntries": state["platformEntries"],
            },
        }
    return patch


def _artifact_format(engine: str, path: str) -> str:
    lower = path.lower()
    if lower.endswith(".json"):
        payload = _read_json_if_exists(Path(path))
        format_name = payload.get("format")
        if isinstance(format_name, str):
            if format_name == "dart_mlx_ffi.coreml_pipeline.v1":
                return "coreml-pipeline"
            if format_name == "dart_mlx_ffi.onnx_pipeline.v1":
                return "onnx-pipeline"
            if format_name == "dart_mlx_ffi.litert_pipeline.v1":
                return "litert-pipeline"
        if engine == "coreml":
            return "coreml-pipeline"
        if engine == "onnx":
            return "onnx-pipeline"
        if engine == "litert":
            return "litert-pipeline"
    if engine == "mlx":
        return "mlx-safetensors"
    if engine == "coreml":
        return "coreml-bundle"
    if engine == "onnx":
        return "onnx"
    if engine == "litert":
        return "tflite"
    return "unknown"


def _accelerators_for_engine(engine: str) -> list[str]:
    if engine == "coreml":
        return ["ane", "gpu", "cpu"]
    if engine == "onnx":
        return ["gpu", "cpu"]
    if engine == "litert":
        return ["npu", "gpu", "cpu"]
    if engine == "mlx":
        return ["gpu", "cpu"]
    return ["cpu"]


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(data, dict):
        return data
    return {}


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


def _latency_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    latency = metrics.get("latency_ms")
    if not isinstance(latency, dict):
        return {}
    summary: dict[str, Any] = {}
    for key in ("mean", "p50", "p95"):
        value = latency.get(key)
        if value is not None:
            summary[key] = float(value)
    values = latency.get("values")
    if isinstance(values, list):
        summary["sampleCount"] = len(values)
    return summary


def _input_signature_for_status(candidate: dict[str, Any]) -> Any:
    signature = candidate.get("input_signature")
    if signature is not None:
        return signature
    digest = candidate.get("input_digest")
    if digest is not None:
        return {"digest": digest}
    return None


def _status_notes(payload: dict[str, Any], model_id: str, platform: str) -> list[str]:
    notes = []
    mismatches = payload.get("mismatches") or []
    if mismatches:
        notes.append(f"Report identity mismatch for {model_id}/{platform}")
    verdict = payload.get("verdict") or {}
    for section in ("identity", "correctness", "speed", "peak_memory", "device_profile"):
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
