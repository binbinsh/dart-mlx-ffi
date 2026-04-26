from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate that native runtime artifacts can be loaded safely."
    )
    parser.add_argument("--engine", required=True, choices=["onnx", "coreml", "litert"])
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--platform")
    parser.add_argument("--provider", default="cpu")
    parser.add_argument("--delegate")
    parser.add_argument("--coreml-mode")
    parser.add_argument("--litert-section-index")
    parser.add_argument("--require-delegate", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--probe-child", choices=["onnx"], help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.probe_child == "onnx":
        payload = _probe_onnx_child(args.artifact, args.provider)
        print(json.dumps(payload, ensure_ascii=False))
        if not payload["passed"]:
            raise SystemExit(1)
        return

    payload = validate_artifact(
        engine=args.engine,
        artifact=args.artifact,
        platform=args.platform,
        provider=args.provider,
        delegate=args.delegate,
        coreml_mode=args.coreml_mode,
        litert_section_index=args.litert_section_index,
        require_delegate=args.require_delegate,
        timeout=args.timeout,
    )
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not payload["passed"]:
        raise SystemExit(1)


def validate_artifact(
    *,
    engine: str,
    artifact: Path,
    platform: str | None = None,
    provider: str = "cpu",
    delegate: str | None = None,
    coreml_mode: str | None = None,
    litert_section_index: str | None = None,
    require_delegate: bool = False,
    timeout: float = 120.0,
) -> dict[str, Any]:
    artifact = artifact.expanduser()
    if engine == "onnx":
        return _validate_onnx_artifact(artifact, provider=provider, timeout=timeout)
    return _validate_dart_runtime_artifact(
        engine=engine,
        artifact=artifact,
        platform=platform,
        delegate=delegate,
        coreml_mode=coreml_mode,
        litert_section_index=litert_section_index,
        require_delegate=require_delegate,
        timeout=timeout,
    )


def _validate_onnx_artifact(
    artifact: Path,
    *,
    provider: str,
    timeout: float,
) -> dict[str, Any]:
    checks = _onnx_checks(artifact)
    results = []
    for check in checks:
        if check.get("passed") is False:
            results.append(check)
            continue
        if check["kind"] == "op":
            results.append(
                {
                    "name": check["name"],
                    "kind": "op",
                    "op": check["op"],
                    "passed": True,
                }
            )
            continue
        results.append(
            _run_onnx_probe(
                name=check["name"],
                path=Path(check["path"]),
                provider=provider,
                timeout=timeout,
            )
        )
    return {
        "engine": "onnx",
        "artifact": str(artifact),
        "pipeline": artifact.suffix.lower() == ".json",
        "passed": all(item.get("passed") for item in results),
        "checks": results,
    }


def _validate_dart_runtime_artifact(
    *,
    engine: str,
    artifact: Path,
    platform: str | None,
    delegate: str | None,
    coreml_mode: str | None,
    litert_section_index: str | None,
    require_delegate: bool,
    timeout: float,
) -> dict[str, Any]:
    check = _run_dart_runtime_probe(
        engine=engine,
        artifact=artifact,
        platform=platform or _default_platform(engine),
        delegate=delegate,
        coreml_mode=coreml_mode,
        litert_section_index=litert_section_index,
        require_delegate=require_delegate,
        timeout=timeout,
    )
    return {
        "engine": engine,
        "artifact": str(artifact),
        "passed": bool(check.get("passed")),
        "checks": [check],
    }


def _onnx_checks(artifact: Path) -> list[dict[str, Any]]:
    if artifact.suffix.lower() != ".json":
        return [{"name": artifact.name, "kind": "model", "path": str(artifact)}]
    spec = json.loads(artifact.read_text(encoding="utf-8"))
    if spec.get("format") != "dart_inference.onnx_pipeline.v1":
        return [{"name": artifact.name, "kind": "model", "path": str(artifact)}]
    checks: list[dict[str, Any]] = []
    for index, stage in enumerate(spec.get("stages") or []):
        if not isinstance(stage, dict):
            checks.append(
                {
                    "name": f"stage_{index}",
                    "kind": "invalid",
                    "passed": False,
                    "state": "invalid",
                    "reason": "Pipeline stage must be an object.",
                }
            )
            continue
        name = str(stage.get("name") or f"stage_{index}")
        if stage.get("op"):
            checks.append({"name": name, "kind": "op", "op": str(stage["op"])})
            continue
        model = stage.get("model")
        if not isinstance(model, str):
            checks.append(
                {
                    "name": name,
                    "kind": "invalid",
                    "passed": False,
                    "state": "invalid",
                    "reason": "ONNX pipeline model stage has no model path.",
                }
            )
            continue
        checks.append(
            {
                "name": name,
                "kind": "model",
                "path": str(_resolve_pipeline_path(artifact, model)),
            }
        )
    return checks


def _run_onnx_probe(
    *,
    name: str,
    path: Path,
    provider: str,
    timeout: float,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--engine",
        "onnx",
        "--artifact",
        str(path),
        "--provider",
        provider,
        "--probe-child",
        "onnx",
    ]
    try:
        completed = subprocess.run(
            cmd,
            cwd=ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        return {
            "name": name,
            "kind": "model",
            "path": str(path),
            "passed": False,
            "state": "timeout",
            "timeout_seconds": timeout,
            "stderr": error.stderr or "",
        }
    result = {
        "name": name,
        "kind": "model",
        "path": str(path),
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    if completed.returncode < 0:
        result["state"] = "crashed"
        result["signal"] = _signal_name(-completed.returncode)
    elif completed.returncode != 0:
        result["state"] = "failed"
    else:
        result["state"] = "loaded"
        try:
            result["diagnostics"] = json.loads(completed.stdout)
        except json.JSONDecodeError:
            result["passed"] = False
            result["state"] = "invalid_output"
    return result


def _run_dart_runtime_probe(
    *,
    engine: str,
    artifact: Path,
    platform: str,
    delegate: str | None,
    coreml_mode: str | None,
    litert_section_index: str | None,
    require_delegate: bool,
    timeout: float,
) -> dict[str, Any]:
    cmd = [
        "dart",
        "run",
        "benchmark/runtime/dart_runtime_runner.dart",
        "--health-check",
        "--model-id",
        "artifact_health",
        "--platform",
        platform,
        "--engine",
        engine,
        "--artifact",
        str(artifact),
    ]
    if delegate:
        cmd.extend(["--delegate", delegate])
    if coreml_mode:
        cmd.extend(["--coreml-mode", coreml_mode])
    if litert_section_index:
        cmd.extend(["--litert-section-index", litert_section_index])
    if require_delegate:
        cmd.append("--require-delegate")
    try:
        completed = subprocess.run(
            cmd,
            cwd=ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        return {
            "name": "dart_runtime_health",
            "kind": "model",
            "path": str(artifact),
            "passed": False,
            "state": "timeout",
            "failure_class": "timeout",
            "failure_reason": f"Health-check timed out after {timeout} seconds.",
            "timeout_seconds": timeout,
            "stderr": error.stderr or "",
        }
    result = {
        "name": "dart_runtime_health",
        "kind": "model",
        "path": str(artifact),
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    if completed.returncode < 0:
        result["state"] = "crashed"
        result["signal"] = _signal_name(-completed.returncode)
        result["failure_class"] = "crashed"
        result["failure_reason"] = (
            f"Health-check process crashed with signal {result['signal']}."
        )
    elif completed.returncode != 0:
        result["state"] = "failed"
        failure = _classify_dart_runtime_failure(
            engine=engine,
            stdout=result["stdout"],
            stderr=result["stderr"],
        )
        result["failure_class"] = failure["failure_class"]
        result["failure_reason"] = failure["failure_reason"]
    else:
        result["state"] = "loaded"
        try:
            result["diagnostics"] = _extract_json(completed.stdout)
        except json.JSONDecodeError:
            result["passed"] = False
            result["state"] = "invalid_output"
            result["failure_class"] = "invalid_output"
            result["failure_reason"] = (
                "Health-check completed but did not emit a valid JSON payload."
            )
    return result


def _classify_dart_runtime_failure(
    *,
    engine: str,
    stdout: str,
    stderr: str,
) -> dict[str, str]:
    text = "\n".join(item for item in [stderr, stdout] if item).lower()
    if engine == "litert":
        checks = [
            (
                "unable to load litert/tflite c library",
                "runtime_library_missing",
                "LiteRT runtime library is missing or not discoverable.",
            ),
            (
                "missing litert/tflite symbol",
                "runtime_symbol_missing",
                "LiteRT runtime library is present but missing required C API symbols.",
            ),
            (
                "litert container has multiple tflite sections",
                "section_index_required",
                "LiteRT container has multiple TFLite sections; set litertSectionIndex.",
            ),
            (
                "litert container does not contain raw tflite flatbuffer sections",
                "container_requires_litertlm_runtime",
                "Artifact needs higher-level LiteRT-LM/MediaPipe runtime integration.",
            ),
            (
                "tflitemodelcreatefromfile failed",
                "model_load_failed",
                "LiteRT runtime failed to load the model file.",
            ),
            (
                "builtin_code out of range",
                "runtime_version_mismatch",
                "LiteRT model requires newer TensorFlow Lite runtime operator support.",
            ),
            (
                "no optional support libraries loaded",
                "missing_optional_support_libraries",
                "Optional LiteRT side libraries (for example flex/select-ops) were not loaded.",
            ),
            (
                "tfliteinterpretercreate failed",
                "interpreter_create_failed",
                "LiteRT runtime loaded the model but failed to create an interpreter.",
            ),
            (
                "tflite qnn delegate is not available",
                "delegate_unavailable",
                "Requested LiteRT delegate is not available in the bundled runtime.",
            ),
        ]
        for needle, failure_class, failure_reason in checks:
            if needle in text:
                return {
                    "failure_class": failure_class,
                    "failure_reason": failure_reason,
                }
    if engine == "onnx":
        checks = [
            (
                "unable to load onnx runtime shared library",
                "runtime_library_missing",
                "ONNX Runtime shared library is missing or not discoverable.",
            ),
            (
                "requested onnx runtime provider is unavailable",
                "provider_unavailable",
                "Requested ONNX Runtime execution provider is unavailable.",
            ),
            (
                "onnxruntime load failed",
                "session_create_failed",
                "ONNX Runtime failed to create an inference session.",
            ),
        ]
        for needle, failure_class, failure_reason in checks:
            if needle in text:
                return {
                    "failure_class": failure_class,
                    "failure_reason": failure_reason,
                }
    if engine == "coreml":
        checks = [
            (
                "coreml backend is unavailable",
                "runtime_unavailable",
                "Core ML backend is unavailable on this host.",
            ),
            (
                "mlmodel",
                "model_load_failed",
                "Core ML runtime failed to load the model artifact.",
            ),
        ]
        for needle, failure_class, failure_reason in checks:
            if needle in text:
                return {
                    "failure_class": failure_class,
                    "failure_reason": failure_reason,
                }
    return {
        "failure_class": "execution_failed",
        "failure_reason": (
            "Native runtime health-check failed; inspect stdout/stderr for details."
        ),
    }


def _probe_onnx_child(artifact: Path, provider: str) -> dict[str, Any]:
    try:
        import onnxruntime as ort
    except ImportError as error:
        return {
            "passed": False,
            "reason": f"onnxruntime is not installed: {error}",
        }
    providers = [_canonical_provider(provider)]
    session = ort.InferenceSession(str(artifact), providers=providers)
    return {
        "passed": True,
        "provider": session.get_providers()[0] if session.get_providers() else None,
        "inputs": [_io_meta(item) for item in session.get_inputs()],
        "outputs": [_io_meta(item) for item in session.get_outputs()],
    }


def _io_meta(value: Any) -> dict[str, Any]:
    return {
        "name": value.name,
        "type": value.type,
        "shape": list(value.shape or []),
    }


def _canonical_provider(provider: str) -> str:
    normalized = provider.lower()
    if normalized in {"", "cpu"}:
        return "CPUExecutionProvider"
    if normalized in {"coreml", "ane"}:
        return "CoreMLExecutionProvider"
    if normalized in {"cuda"}:
        return "CUDAExecutionProvider"
    if normalized in {"dml", "directml"}:
        return "DmlExecutionProvider"
    if normalized == "openvino":
        return "OpenVINOExecutionProvider"
    if normalized == "qnn":
        return "QNNExecutionProvider"
    return provider


def _extract_json(text: str) -> dict[str, Any]:
    start = text.find("{")
    if start < 0:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    value, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(value, dict):
        raise json.JSONDecodeError("JSON payload is not an object", text, start)
    return value


def _default_platform(engine: str) -> str:
    if engine == "coreml":
        return "macos"
    if engine == "litert":
        return "android"
    return "linux"


def _resolve_pipeline_path(spec_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return spec_path.parent / path


def _signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"SIG{signum}"


if __name__ == "__main__":
    main()
