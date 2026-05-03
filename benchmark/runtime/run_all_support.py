from __future__ import annotations

import json
import os
import platform as host_platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from matrix_config import (
    artifact_coverage,
    artifact_unblocks_platform,
    blocked_engine_reason,
    blocked_platform_reason,
    preferred_engine_for_platform,
)

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = Path(__file__).resolve().parent
ENGINE_BY_PLATFORM = {
    "ios": "coreml",
    "macos": "coreml",
    "linux": "onnx",
    "android": "litert",
}
READINESS_LABELS = {
    "production_ready": "production-ready",
    "preferred_ready": "preferred-ready",
    "fallback_ready": "fallback-ready",
    "blocked": "blocked",
}


def _cell_for(
    *,
    model_id: str,
    model_config: dict[str, Any],
    platform: str,
    artifacts: dict[str, Any],
    artifacts_path: Path,
    out_root: Path,
    config: Path,
    check_paths: bool,
    check_execution: bool,
    check_artifact_health: bool,
    prepare_inputs: bool,
) -> dict[str, Any]:
    defaults = artifacts.get("defaults") or {}
    platform_defaults = (defaults.get("platforms") or {}).get(platform) or {}
    config_platform = (model_config.get("platforms") or {}).get(platform) or {}
    model = (artifacts.get("models") or {}).get(model_id) or {}
    model_platforms = model.get("platforms") or {}
    model_platform = (
        model_platforms.get(platform) if isinstance(model_platforms, dict) else {}
    ) or {}
    merged = {
        **_global_defaults(defaults),
        **platform_defaults,
        **_global_defaults(model_config),
        **config_platform,
        **{key: value for key, value in model.items() if key != "platforms"},
        **model_platform,
    }
    if _should_prepare_inputs(merged, prepare_inputs):
        merged["input_json"] = str(out_root / model_id / platform / "input.json")
        merged["_prepare_inputs"] = True
    engine = merged.get("engine") or ENGINE_BY_PLATFORM.get(platform)
    preferred_engine = preferred_engine_for_platform(merged, platform)
    blocker = None
    if not artifact_unblocks_platform(model, platform):
        blocker = (
            blocked_platform_reason(model, platform)
            or blocked_platform_reason(model_config, platform)
        )
    if not blocker and engine:
        blocker = (
            blocked_engine_reason(model, platform, str(engine))
            or blocked_engine_reason(model_config, platform, str(engine))
        )
    baseline_engine = merged.get("baseline_engine")
    if baseline_engine == "coreml-llm" and merged.get("coreml_llm_baseline") is False:
        baseline_engine = None
    baseline_engine = baseline_engine or _baseline_engine(platform, engine, merged)
    coverage = artifact_coverage(model)
    if coverage == "full":
        coverage = artifact_coverage(model_config)
    if blocker:
        fallback_from = _fallback_from(merged, engine, preferred_engine)
        return {
            "model_id": model_id,
            "platform": platform,
            "engine": engine,
            "preferred_engine": preferred_engine,
            "baseline_engine": baseline_engine,
            "artifact_coverage": coverage,
            "fallback_from": fallback_from,
            "path_check": check_paths,
            "execution_check": check_execution,
            "executor": merged.get("executor", "local"),
            "state": "blocked",
            "reasons": [f"Blocked by artifact coverage: {blocker}"],
            "command": [],
        }
    reasons = []
    if not engine:
        reasons.append(f"No engine default for platform {platform}")
    if not merged.get("artifact") and not merged.get("candidate_report"):
        reasons.append("Missing candidate artifact")
    has_mlx_publish_baseline = (
        baseline_engine == "mlx" and merged.get("baseline_publish_report")
    )
    has_coreml_llm_baseline_artifact = (
        baseline_engine == "coreml-llm"
        and (merged.get("baseline_artifact") or merged.get("artifact"))
    )
    if (
        not merged.get("baseline_report")
        and not merged.get("baseline_artifact")
        and not merged.get("raw_baseline_report")
        and not has_mlx_publish_baseline
        and not has_coreml_llm_baseline_artifact
    ):
        if baseline_engine == "coreml-llm":
            reasons.append("Missing CoreML-LLM baseline artifact or raw report")
        elif baseline_engine in {"mlx", "coreml", "onnx", "litert"} and merged.get("artifact"):
            pass
        else:
            reasons.append("Missing baseline report")
    if check_paths:
        reasons.extend(_path_reasons(merged, baseline_engine, platform))
    reasons.extend(_fixture_reasons(merged))
    if check_execution:
        reasons.extend(_execution_reasons(merged, platform))
    command = _cell_command(
        model_id=model_id,
        platform=platform,
        engine=engine or "",
        merged=merged,
        out_root=out_root,
        config=config,
        baseline_engine=baseline_engine,
    )
    prepare_command = _prepare_input_command(
        model_id=model_id,
        platform=platform,
        engine=engine or "",
        merged=merged,
        out_root=out_root,
        artifacts_path=artifacts_path,
    )
    artifact_health_command = _artifact_health_command(
        model_id=model_id,
        engine=engine or "",
        platform=platform,
        merged=merged,
        enabled=check_artifact_health,
        out_root=out_root,
    )
    host = _host_platform()
    device_smoke_only = bool(artifact_health_command) and str(
        merged.get("executor") or "local"
    ) == "local" and host != platform and (
        (platform == "ios" and engine == "coreml" and bool(merged.get("ios_device_smoke")))
        or (
            platform == "android"
            and engine in {"onnx", "litert"}
            and bool(merged.get("android_device_smoke"))
        )
    )
    fallback_from = _fallback_from(merged, engine, preferred_engine)
    return {
        "model_id": model_id,
        "platform": platform,
        "engine": engine,
        "preferred_engine": preferred_engine,
        "baseline_engine": baseline_engine,
        "artifact_coverage": coverage,
        "fallback_from": fallback_from,
        "path_check": check_paths,
        "execution_check": check_execution,
        "artifact_health_check": check_artifact_health,
        "executor": merged.get("executor", "local"),
        "state": "blocked" if reasons else "ready",
        "reasons": reasons,
        "device_smoke_only": device_smoke_only,
        "artifact_health_command": artifact_health_command,
        "prepare_input_command": prepare_command,
        "command": command,
    }


def _abandoned_platform_cell(
    *,
    model_id: str,
    model_config: dict[str, Any],
    platform: str,
    out_root: Path,
) -> dict[str, Any]:
    del out_root
    engine = ENGINE_BY_PLATFORM.get(platform)
    return {
        "model_id": model_id,
        "platform": platform,
        "engine": engine,
        "preferred_engine": engine,
        "baseline_engine": None,
        "artifact_coverage": artifact_coverage(model_config),
        "fallback_from": [],
        "path_check": False,
        "execution_check": False,
        "executor": "none",
        "state": "blocked",
        "reasons": [
            f"Platform {platform} is abandoned by support_policy.abandoned_platforms."
        ],
        "command": [],
    }


def _fallback_from(
    merged: dict[str, Any],
    engine: object,
    preferred_engine: str | None,
) -> list[str]:
    fallback = _string_list(merged.get("fallback_from"))
    if fallback and (not preferred_engine or not engine or str(engine) != preferred_engine):
        return fallback
    if preferred_engine and engine and str(engine) != preferred_engine:
        return [preferred_engine]
    return []


def _readiness_state(cell: dict[str, Any], out_root: Path) -> str:
    if cell.get("state") != "ready":
        return "blocked"
    if cell.get("fallback_from"):
        return "fallback_ready"
    verdict_path = (
        out_root
        / str(cell.get("model_id") or "")
        / str(cell.get("platform") or "")
        / "verdict.json"
    )
    if _verdict_is_production_ready(verdict_path):
        return "production_ready"
    return "preferred_ready"


def _readiness_class(cell: dict[str, Any]) -> str:
    state = str(cell.get("readiness_state") or "blocked")
    return READINESS_LABELS.get(state, "blocked")


def _readiness_counts(cells: list[dict[str, Any]]) -> dict[str, int]:
    counts = {label: 0 for label in READINESS_LABELS.values()}
    for cell in cells:
        label = _readiness_class(cell)
        counts[label] = counts.get(label, 0) + 1
    return counts


def _model_readiness(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        model_id = str(cell.get("model_id") or "")
        grouped.setdefault(model_id, []).append(cell)
    records = []
    for model_id, model_cells in grouped.items():
        counts = _readiness_counts(model_cells)
        if counts["blocked"]:
            readiness = "blocked"
        elif counts["fallback-ready"]:
            readiness = "fallback-ready"
        elif counts["production-ready"] == len(model_cells):
            readiness = "production-ready"
        else:
            readiness = "preferred-ready"
        records.append(
            {
                "model_id": model_id,
                "readiness_class": readiness,
                "cell_count": len(model_cells),
                "readiness_counts": counts,
            }
        )
    return records


def _model_readiness_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {label: 0 for label in READINESS_LABELS.values()}
    for record in records:
        label = str(record.get("readiness_class") or "blocked")
        counts[label] = counts.get(label, 0) + 1
    return counts


def _verdict_is_production_ready(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    verdict = payload.get("verdict") if isinstance(payload, dict) else None
    if not isinstance(verdict, dict):
        verdict = payload if isinstance(payload, dict) else {}
    if verdict.get("passed") is not True:
        return False
    for section in ("identity", "correctness", "speed", "peak_memory", "device_profile"):
        data = verdict.get(section) or {}
        if not isinstance(data, dict) or data.get("passed") is not True:
            return False
    return True


def _cell_command(
    *,
    model_id: str,
    platform: str,
    engine: str,
    merged: dict[str, Any],
    out_root: Path,
    config: Path,
    baseline_engine: str,
) -> list[str]:
    if str(merged.get("executor") or "") == "adb" and platform == "android":
        return _adb_command(
            model_id=model_id,
            platform=platform,
            engine=engine,
            baseline_engine=baseline_engine,
            merged=merged,
            out_root=out_root,
            config=config,
        )
    return _run_matrix_command(
        model_id=model_id,
        platform=platform,
        engine=engine,
        merged=merged,
        out_root=out_root,
        config=config,
        baseline_engine=baseline_engine,
    )

def _run_matrix_command(
    *,
    model_id: str,
    platform: str,
    engine: str,
    merged: dict[str, Any],
    out_root: Path,
    config: Path,
    baseline_engine: str,
) -> list[str]:
    cmd = [
        sys.executable,
        str(RUNTIME_DIR / "run_matrix.py"),
        "--model-id",
        model_id,
        "--platform",
        platform,
        "--engine",
        engine,
        "--artifact",
        str(merged.get("artifact") or ""),
        "--baseline-engine",
        baseline_engine,
        "--out-root",
        str(out_root),
        "--config",
        str(config),
    ]
    _add_optional(cmd, "--baseline-artifact", merged.get("baseline_artifact"))
    _add_optional(cmd, "--baseline-report", merged.get("baseline_report"))
    if baseline_engine == "mlx":
        _add_optional(
            cmd,
            "--baseline-publish-report",
            merged.get("baseline_publish_report"),
        )
        _add_optional(
            cmd,
            "--baseline-publish-model-id",
            merged.get("baseline_publish_model_id"),
        )
    _add_optional(cmd, "--candidate-report", merged.get("candidate_report"))
    _add_optional(cmd, "--raw-baseline-report", merged.get("raw_baseline_report"))
    _add_optional(cmd, "--input-json", merged.get("input_json"))
    _add_optional(cmd, "--prompt-file", merged.get("prompt_file"))
    _add_optional(cmd, "--prompt", merged.get("prompt"))
    _add_optional(cmd, "--task", merged.get("task"))
    _add_optional(cmd, "--tools-file", merged.get("tools_file"))
    _add_optional(cmd, "--tools-json", merged.get("tools_json"))
    _add_optional(cmd, "--embedding-query", merged.get("embedding_query"))
    _add_optional(cmd, "--embedding-query-file", merged.get("embedding_query_file"))
    _add_optional(cmd, "--embedding-dim", merged.get("embedding_dim"))
    _add_optional(cmd, "--image-file", merged.get("image_file"))
    _add_optional(cmd, "--audio-file", merged.get("audio_file"))
    _add_optional(cmd, "--warmup", merged.get("warmup"))
    _add_optional(cmd, "--iters", merged.get("iters"))
    _add_optional(cmd, "--max-tokens", merged.get("max_tokens"))
    _add_optional(cmd, "--num-threads", merged.get("num_threads"))
    _add_optional(cmd, "--provider", merged.get("provider"))
    _add_optional(cmd, "--delegate", merged.get("delegate"))
    _add_optional(cmd, "--coreml-mode", merged.get("coreml_mode"))
    _add_optional(cmd, "--coreml-compute-units", merged.get("coreml_compute_units"))
    _add_optional(cmd, "--litert-section-index", merged.get("litert_section_index"))
    _add_optional(cmd, "--hf-cache-root", merged.get("hf_cache_root"))
    if merged.get("require_provider"):
        cmd.append("--require-provider")
    if merged.get("require_delegate"):
        cmd.append("--require-delegate")
    return cmd
def _adb_command(
    *,
    model_id: str,
    platform: str,
    engine: str,
    baseline_engine: str,
    merged: dict[str, Any],
    out_root: Path,
    config: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(RUNTIME_DIR / "adb_runner.py"),
        "--model-id",
        model_id,
        "--platform",
        platform,
        "--engine",
        engine,
        "--artifact",
        str(merged.get("artifact") or ""),
        "--out-root",
        str(out_root),
        "--config",
        str(config),
    ]
    _add_optional(cmd, "--baseline-report", merged.get("baseline_report"))
    _add_optional(cmd, "--baseline-engine", baseline_engine)
    _add_optional(cmd, "--baseline-artifact", merged.get("baseline_artifact"))
    _add_optional(
        cmd,
        "--remote-baseline-report",
        merged.get("remote_baseline_report"),
    )
    _add_optional(
        cmd,
        "--baseline-publish-model-id",
        merged.get("baseline_publish_model_id"),
    )
    _add_optional(cmd, "--input-json", merged.get("input_json"))
    _add_optional(cmd, "--prompt-file", merged.get("prompt_file"))
    _add_optional(cmd, "--task", merged.get("task"))
    _add_optional(cmd, "--tools-file", merged.get("tools_file"))
    _add_optional(cmd, "--tools-json", merged.get("tools_json"))
    _add_optional(cmd, "--embedding-query", merged.get("embedding_query"))
    _add_optional(cmd, "--embedding-query-file", merged.get("embedding_query_file"))
    _add_optional(cmd, "--embedding-dim", merged.get("embedding_dim"))
    _add_optional(cmd, "--image-file", merged.get("image_file"))
    _add_optional(cmd, "--audio-file", merged.get("audio_file"))
    _add_optional(cmd, "--candidate-report", merged.get("candidate_report"))
    _add_optional(
        cmd,
        "--remote-candidate-report",
        merged.get("remote_candidate_report"),
    )
    _add_optional(cmd, "--device-id", merged.get("device_id"))
    _add_optional(cmd, "--remote-dir", merged.get("remote_dir"))
    _add_optional(cmd, "--device-runner", merged.get("device_runner"))
    _add_optional(cmd, "--device-command", merged.get("device_command"))
    for value in merged.get("push") or []:
        _add_optional(cmd, "--push", value)
    for value in merged.get("pull") or []:
        _add_optional(cmd, "--pull", value)
    _add_optional(cmd, "--warmup", merged.get("warmup"))
    _add_optional(cmd, "--iters", merged.get("iters"))
    _add_optional(cmd, "--max-tokens", merged.get("max_tokens"))
    _add_optional(cmd, "--num-threads", merged.get("num_threads"))
    _add_optional(cmd, "--provider", merged.get("provider"))
    _add_optional(cmd, "--delegate", merged.get("delegate"))
    _add_optional(cmd, "--coreml-mode", merged.get("coreml_mode"))
    _add_optional(cmd, "--coreml-compute-units", merged.get("coreml_compute_units"))
    _add_optional(cmd, "--litert-section-index", merged.get("litert_section_index"))
    _add_optional(cmd, "--hf-cache-root", merged.get("hf_cache_root"))
    if merged.get("require_provider"):
        cmd.append("--require-provider")
    if merged.get("require_delegate"):
        cmd.append("--require-delegate")
    return cmd
def _prepare_input_command(
    *,
    model_id: str,
    platform: str,
    engine: str,
    merged: dict[str, Any],
    out_root: Path,
    artifacts_path: Path,
) -> list[str]:
    if not merged.get("_prepare_inputs"):
        return []
    task = str(merged.get("task") or "text")
    if task == "vlm":
        cmd = ["uv", "run", "--group", "vlm-prepare", "python"]
    else:
        cmd = [sys.executable]
    cmd.extend(
        [
            str(RUNTIME_DIR / "prepare_inputs.py"),
            "--model-id",
            model_id,
            "--artifacts",
            str(artifacts_path),
            "--task",
            task,
            "--engine",
            engine,
            "--out",
            str(out_root / model_id / platform / "input.json"),
        ]
    )
    _add_optional(cmd, "--source-model", merged.get("source_model"))
    _add_optional(cmd, "--prompt", merged.get("prompt"))
    _add_optional(cmd, "--prompt-file", merged.get("prompt_file"))
    _add_optional(cmd, "--embedding-query", merged.get("embedding_query"))
    _add_optional(cmd, "--embedding-query-file", merged.get("embedding_query_file"))
    _add_optional(cmd, "--image-file", merged.get("image_file"))
    _add_optional(cmd, "--audio-file", merged.get("audio_file"))
    _add_optional(cmd, "--hf-cache-root", merged.get("hf_cache_root"))
    max_length = (
        merged.get("prepare_max_length")
        or merged.get("input_max_length")
        or (None if task == "vlm" else merged.get("max_tokens"))
    )
    _add_optional(cmd, "--max-length", max_length)
    if merged.get("trust_remote_code"):
        cmd.append("--trust-remote-code")
    artifact = merged.get("artifact")
    if engine == "onnx" and artifact and not _is_hf_uri(artifact):
        path = _local_path(artifact)
        if path.exists() and _is_onnx_input_artifact(path):
            cmd.extend(["--onnx-artifact", str(path)])
    if engine == "coreml" and artifact and not _is_hf_uri(artifact):
        path = _local_path(artifact)
        if path.exists():
            cmd.extend(["--coreml-artifact", str(path)])
    if engine == "litert" and artifact and not _is_hf_uri(artifact):
        path = _local_path(artifact)
        if path.exists():
            cmd.extend(["--litert-artifact", str(path)])
    return cmd

def _should_prepare_inputs(merged: dict[str, Any], requested: bool) -> bool:
    if requested:
        return True
    task = str(merged.get("task") or "text").lower()
    if task == "text":
        return False
    input_json = str(merged.get("input_json") or "")
    if not input_json:
        return True
    return Path(input_json).name == "tiny_input.json"

def _artifact_health_command(
    *,
    model_id: str,
    engine: str,
    platform: str,
    merged: dict[str, Any],
    enabled: bool,
    out_root: Path,
) -> list[str]:
    if not enabled:
        return []

    if (
        platform == "ios"
        and engine == "coreml"
        and bool(merged.get("ios_device_smoke"))
    ):
        artifact = merged.get("artifact")
        cmd = [
            sys.executable,
            str(RUNTIME_DIR / "ios_flutter_smoke.py"),
            "--model-id",
            model_id,
            "--engine",
            "coreml",
            "--out",
            str(
                out_root
                / "_artifact_health"
                / f"{_safe_name(model_id)}_ios_device_smoke.json"
            ),
        ]
        if artifact:
            cmd.extend(["--artifact", str(artifact)])
        _add_optional(cmd, "--device-id", merged.get("device_id"))
        _add_optional(cmd, "--build-mode", merged.get("ios_smoke_build_mode"))
        _add_optional(cmd, "--timeout-seconds", merged.get("ios_smoke_timeout_seconds"))
        _add_optional(cmd, "--coreml-compute-units", merged.get("coreml_compute_units"))
        _add_optional(
            cmd,
            "--wait-for-artifact-seconds",
            merged.get("ios_smoke_wait_for_artifact_seconds"),
        )
        if merged.get("allow_ios_smoke_fail"):
            cmd.append("--allow-fail")
        return cmd

    if (
        platform == "android"
        and engine in {"onnx", "litert"}
        and bool(merged.get("android_device_smoke"))
    ):
        artifact = merged.get("artifact")
        cmd = [
            sys.executable,
            str(RUNTIME_DIR / "android_flutter_smoke.py"),
            "--model-id",
            model_id,
            "--engine",
            engine,
            "--out",
            str(
                out_root
                / "_artifact_health"
                / f"{_safe_name(model_id)}_android_device_smoke.json"
            ),
        ]
        if artifact:
            cmd.extend(["--artifact", str(artifact)])
        _add_optional(cmd, "--device-id", merged.get("device_id"))
        _add_optional(
            cmd,
            "--build-mode",
            merged.get("android_smoke_build_mode"),
        )
        _add_optional(
            cmd,
            "--timeout-seconds",
            merged.get("android_smoke_timeout_seconds"),
        )
        _add_optional(cmd, "--provider", merged.get("provider"))
        _add_optional(cmd, "--delegate", merged.get("delegate"))
        _add_optional(cmd, "--litert-section-index", merged.get("litert_section_index"))
        if engine == "onnx" and merged.get("require_provider"):
            cmd.append("--require-provider")
        if engine == "litert" and merged.get("require_delegate"):
            cmd.append("--require-delegate")
        if merged.get("allow_android_smoke_fail"):
            cmd.append("--allow-fail")
        return cmd

    if engine not in {"onnx", "coreml", "litert"}:
        return []
    executor = str(merged.get("executor") or "local")
    if executor != "local":
        return []
    artifact = merged.get("artifact")
    if not artifact or _is_hf_uri(artifact) or _is_device_path(artifact, platform, executor):
        return []
    path = _local_path(artifact)
    if not path.exists():
        return []
    if engine == "onnx":
        cmd = ["uv", "run", "--group", "onnx-convert", "python"]
    else:
        cmd = [sys.executable]
    cmd.extend(
        [
            str(RUNTIME_DIR / "artifact_health.py"),
            "--engine",
            engine,
            "--platform",
            platform,
            "--artifact",
            str(path),
            "--out",
            str(
                out_root
                / "_artifact_health"
                / f"{_safe_name(str(merged.get('source_model') or path.stem))}_{engine}.json"
            ),
        ]
    )
    if engine == "onnx" and merged.get("provider"):
        cmd.extend(["--provider", str(merged["provider"])])
    if engine == "onnx" and merged.get("require_provider"):
        cmd.append("--require-provider")
    if engine == "litert" and merged.get("delegate"):
        cmd.extend(["--delegate", str(merged["delegate"])])
    if engine == "litert" and merged.get("litert_section_index"):
        cmd.extend(["--litert-section-index", str(merged["litert_section_index"])])
    if engine == "litert" and merged.get("require_delegate"):
        cmd.append("--require-delegate")
    if engine == "coreml" and merged.get("coreml_mode"):
        cmd.extend(["--coreml-mode", str(merged["coreml_mode"])])
    if engine == "coreml" and merged.get("coreml_compute_units"):
        cmd.extend(["--coreml-compute-units", str(merged["coreml_compute_units"])])
    return cmd
def _global_defaults(defaults: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in defaults.items() if key != "platforms"}

def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]

def _safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)

def _artifact_health_failure(report: str | None) -> dict[str, str] | None:
    if not report:
        return None
    path = Path(report)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    runtime_smoke = payload.get("runtime_smoke")
    if isinstance(runtime_smoke, dict) and runtime_smoke.get("passed") is False:
        engine = str(payload.get("engine") or "")
        error = str(runtime_smoke.get("error") or "")
        failure_class = _classify_runtime_smoke_error(engine, error)
        reason = _tail_message(error) or _tail_message(runtime_smoke.get("stack"))
        result: dict[str, str] = {}
        if failure_class:
            result["failure_class"] = failure_class
        if reason:
            result["failure_reason"] = reason
        return result or None
    checks = payload.get("checks")
    if not isinstance(checks, list) or not checks or not isinstance(checks[0], dict):
        return None
    check = checks[0]
    failure_class = check.get("failure_class")
    reason = (
        check.get("failure_reason")
        or check.get("reason")
        or _tail_message(check.get("stderr"))
        or _tail_message(check.get("stdout"))
    )
    result: dict[str, str] = {}
    if isinstance(failure_class, str) and failure_class:
        result["failure_class"] = failure_class
    if isinstance(reason, str) and reason:
        result["failure_reason"] = reason
    return result or None

def _tail_message(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if not lines:
        return None
    return lines[-1][:400]

def _classify_runtime_smoke_error(engine: str, error: str) -> str:
    text = error.lower()
    if "socketexception" in text or "failed host lookup" in text:
        return "network_unavailable"
    if "http error 401" in text or "unauthorized" in text:
        return "artifact_auth_required"
    if engine == "litert":
        if "unable to load litert/tflite c library" in text:
            return "runtime_library_missing"
        if "litert container has multiple tflite sections" in text:
            return "section_index_required"
        if "builtin_code out of range" in text:
            return "runtime_version_mismatch"
        if "tfliteinterpretercreate failed" in text:
            if "with delegates" in text:
                return "delegate_interpreter_create_failed"
            if "no optional support libraries loaded" in text:
                return "missing_optional_support_libraries"
            return "interpreter_create_failed"
        if "tflitemodelcreatefromfile failed" in text:
            return "model_load_failed"
    if engine == "onnx":
        if "unable to load onnx runtime shared library" in text:
            return "runtime_library_missing"
        if "requested onnx runtime provider is unavailable" in text:
            return "provider_unavailable"
    return "execution_failed"


def _baseline_engine(platform: str, engine: Any, merged: dict[str, Any]) -> str:
    if platform in {"ios", "macos"}:
        return "coreml-llm" if merged.get("coreml_llm_baseline") is not False else "mlx"
    return str(engine or ENGINE_BY_PLATFORM.get(platform) or "external")


def _execution_reasons(merged: dict[str, Any], platform: str) -> list[str]:
    executor = str(merged.get("executor") or "local")
    if executor == "adb":
        if platform != "android":
            return ["executor: adb is only valid for Android cells."]
        if not _has_online_adb_device(merged.get("device_id")):
            return ["executor: adb requires an online adb device."]
        has_way_to_collect = (
            merged.get("candidate_report")
            or merged.get("remote_candidate_report")
            or merged.get("device_runner")
            or merged.get("device_command")
        )
        if not has_way_to_collect:
            return [
                "executor: adb requires candidate_report, "
                "remote_candidate_report, device_runner, or device_command."
            ]
        if not merged.get("baseline_report"):
            has_way_to_collect_baseline = (
                merged.get("remote_baseline_report")
                or merged.get("device_runner")
                or merged.get("device_command")
            )
            if not has_way_to_collect_baseline:
                return [
                    "executor: adb requires baseline_report, "
                    "remote_baseline_report, device_runner, or device_command."
                ]
        return []
    if executor in {"remote", "device", "external"}:
        if merged.get("candidate_report"):
            return []
        return [
            f"executor: {executor} requires candidate_report because this "
            "harness does not launch remote hosts or devices directly."
        ]
    host = _host_platform()
    if platform == host:
        return []
    if host == "macos" and platform == "ios" and bool(merged.get("ios_device_smoke")):
        return []
    if host == "macos" and platform == "android" and bool(merged.get("android_device_smoke")):
        return []
    return [
        "Platform "
        f"{platform} cannot be executed by local runner on host {host}. "
        "Run this cell on the target host with --platform, or mark the cell "
        "executor: remote/device and provide pre-collected reports."
    ]


def _path_reasons(
    merged: dict[str, Any],
    baseline_engine: str,
    platform: str,
) -> list[str]:
    executor = str(merged.get("executor") or "local")
    checks: list[tuple[str, Any]] = []
    if not merged.get("_prepare_inputs"):
        checks.append(("input JSON", merged.get("input_json")))
    if merged.get("candidate_report") is not None and executor != "adb":
        checks.append(("candidate report", merged.get("candidate_report")))
    elif executor != "adb":
        checks.append(("candidate artifact", merged.get("artifact")))
    if merged.get("baseline_report") is not None:
        checks.append(("baseline report", merged.get("baseline_report")))
    if baseline_engine == "mlx" and merged.get("baseline_publish_report") is not None:
        checks.append(
            ("baseline publish report", merged.get("baseline_publish_report"))
        )
    if merged.get("raw_baseline_report") is not None:
        checks.append(("raw baseline report", merged.get("raw_baseline_report")))
    if baseline_engine == "coreml-llm" and merged.get("baseline_artifact") is not None:
        checks.append(("CoreML-LLM baseline artifact", merged.get("baseline_artifact")))
    if baseline_engine in {"coreml", "onnx", "litert"} and merged.get("baseline_artifact") is not None:
        checks.append(("baseline artifact", merged.get("baseline_artifact")))
    if _needs_coreml_llm_prompt_file(merged, baseline_engine):
        checks.append(("prompt file", merged.get("prompt_file")))
    for label, key in (
        ("tool schema file", "tools_file"),
        ("embedding query file", "embedding_query_file"),
        ("image file", "image_file"),
        ("audio file", "audio_file"),
    ):
        if merged.get(key) is not None:
            checks.append((label, merged.get(key)))

    reasons = []
    for label, value in checks:
        if value is None or str(value) == "":
            continue
        if _is_device_path(value, platform, executor):
            continue
        if "artifact" in label and _is_hf_uri(value):
            continue
        path = _local_path(value)
        if not path.exists():
            reasons.append(f"Missing {label}: {path}")
    input_json = merged.get("input_json")
    if (
        input_json is not None
        and not merged.get("_prepare_inputs")
        and not _is_device_path(input_json, platform, executor)
    ):
        input_path = _local_path(input_json)
        if input_path.exists():
            reasons.extend(_input_sidecar_reasons(input_path, platform, executor))
    if executor == "adb":
        artifact = merged.get("artifact")
        if (
            artifact
            and not _is_device_path(artifact, platform, executor)
            and not _is_hf_uri(artifact)
        ):
            path = _local_path(artifact)
            if not path.exists():
                reasons.append(f"Missing candidate artifact: {path}")
        for label, key in (
            ("device runner", "device_runner"),
        ):
            value = merged.get(key)
            if value is not None and not _is_device_path(value, platform, executor):
                path = _local_path(value)
                if not path.exists():
                    reasons.append(f"Missing {label}: {path}")
    if _needs_coreml_llm_image(merged, baseline_engine):
        reasons.append("Missing image_file for CoreML-LLM VLM baseline")
    return reasons


def _needs_coreml_llm_prompt_file(
    merged: dict[str, Any],
    baseline_engine: str,
) -> bool:
    if baseline_engine != "coreml-llm":
        return False
    if merged.get("baseline_report") is not None or merged.get("raw_baseline_report") is not None:
        return False
    if merged.get("prompt") is not None:
        return False
    task = str(merged.get("task") or "text")
    if task == "embedding" and (
        merged.get("embedding_query") is not None
        or merged.get("embedding_query_file") is not None
    ):
        return False
    return True


def _fixture_reasons(merged: dict[str, Any]) -> list[str]:
    required = set(str(value) for value in merged.get("required_fixtures") or [])
    reasons = []
    if "text_prompt" in required and not (
        merged.get("prompt") or merged.get("prompt_file")
    ):
        reasons.append("Missing required text_prompt fixture")
    if "image" in required and not merged.get("image_file"):
        reasons.append("Missing required image fixture")
    if "audio" in required and not merged.get("audio_file"):
        reasons.append("Missing required audio fixture")
    if "tool_schema" in required and not (
        merged.get("tools_file") or merged.get("tools_json")
    ):
        reasons.append("Missing required tool_schema fixture")
    if "embedding_query" in required and not (
        merged.get("embedding_query") or merged.get("embedding_query_file")
    ):
        reasons.append("Missing required embedding_query fixture")
    return reasons


def _needs_coreml_llm_image(
    merged: dict[str, Any],
    baseline_engine: str,
) -> bool:
    if baseline_engine != "coreml-llm":
        return False
    if merged.get("baseline_report") is not None or merged.get("raw_baseline_report") is not None:
        return False
    return str(merged.get("task") or "text") == "vlm" and merged.get("image_file") is None


def _local_path(value: Any) -> Path:
    text = str(value)
    if len(text) > 2 and text[1] == ":" and text[2] in {"/", "\\"}:
        return Path(text)
    path = Path(text).expanduser()
    if path.is_absolute():
        return path
    return ROOT / path


def _is_device_path(value: Any, platform: str, executor: str) -> bool:
    if platform != "android" or executor != "adb":
        return False
    text = str(value)
    return text.startswith(("/data/", "/sdcard/", "/storage/", "/mnt/", "content://"))


def _is_hf_uri(value: Any) -> bool:
    return str(value).startswith("hf://")


def _is_onnx_input_artifact(path: Path) -> bool:
    if path.suffix.lower() == ".onnx":
        return True
    if path.suffix.lower() != ".json":
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return data.get("format") in {
        "dart_inference.onnx_pipeline.v1",
        "dart_mlx_ffi.onnx_pipeline.v1",
    }


def _input_sidecar_reasons(
    input_json: Path,
    platform: str,
    executor: str,
) -> list[str]:
    try:
        payload = json.loads(input_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict):
        return []
    inputs = payload.get("inputs", payload)
    if not isinstance(inputs, dict):
        return []
    reasons = []
    for name, spec in inputs.items():
        if not isinstance(spec, dict):
            continue
        raw = spec.get("file")
        if not isinstance(raw, str):
            raw = spec.get("path")
        if not isinstance(raw, str):
            continue
        if _is_device_path(raw, platform, executor):
            continue
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = input_json.parent / path
        if not path.exists():
            reasons.append(f"Missing input sidecar for {name}: {path}")
    return reasons


def _has_online_adb_device(device_id: Any) -> bool:
    if shutil.which("adb") is None:
        return False
    cmd = ["adb"]
    if device_id:
        cmd.extend(["-s", str(device_id)])
    cmd.extend(["get-state"])
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return result.stdout.strip() == "device"


def _add_optional(cmd: list[str], flag: str, value: Any) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def _run(cmd: list[str], *, check: bool) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    return subprocess.run(cmd, cwd=ROOT, check=check, env=env)


def _command_option(cmd: list[str], name: str) -> str | None:
    try:
        index = cmd.index(name)
    except ValueError:
        return None
    if index + 1 >= len(cmd):
        return None
    return cmd[index + 1]


def _host_platform() -> str:
    system = host_platform.system().lower()
    if system == "darwin":
        return "macos"
    if system == "windows":
        return "windows"
    if system == "linux":
        return "linux"
    return system or "unknown"
