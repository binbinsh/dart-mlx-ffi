from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import PurePosixPath
from pathlib import Path
from urllib.parse import unquote, urlparse

from flutter_smoke_common import (
    MarkerParser,
    capture_runtime_smoke,
    extract_marker_payload,
    terminate_process,
)


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = ROOT / "example"
DEFAULT_IOS_BUNDLE_ID = "com.example.dartMlxFfiProbe"
DEFAULT_IOS_ARTIFACT_DIR = "Documents/dart_mlx_ffi_runtime_smoke"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Flutter example on an iOS device in runtime-smoke mode "
            "and capture the structured result marker from app logs."
        )
    )
    parser.add_argument("--device-id")
    parser.add_argument("--model-id", default="silero_vad")
    parser.add_argument("--engine", default="coreml", choices=["coreml", "onnx", "litert"])
    parser.add_argument(
        "--build-mode",
        choices=["debug", "profile", "release"],
        default="release",
    )
    parser.add_argument(
        "--artifact",
        help="Optional artifact override passed to Flutter runtime-smoke mode.",
    )
    parser.add_argument(
        "--bundle-id",
        default=DEFAULT_IOS_BUNDLE_ID,
        help=(
            "Bundle ID used for iOS app data container file copy when pushing "
            "local artifacts."
        ),
    )
    parser.add_argument(
        "--device-artifact-dir",
        default=DEFAULT_IOS_ARTIFACT_DIR,
        help=(
            "Destination directory inside the iOS app data container for "
            "pushed local artifacts."
        ),
    )
    parser.add_argument(
        "--no-device-push",
        action="store_true",
        help=(
            "Disable local artifact copy to iOS app container and pass "
            "--artifact value directly."
        ),
    )
    parser.add_argument(
        "--device-copy-retries",
        type=int,
        default=2,
        help="Retry count for transient devicectl copy failures.",
    )
    parser.add_argument(
        "--allow-direct-fallback-on-push-failure",
        action="store_true",
        help=(
            "If local artifact push fails, continue with the original "
            "--artifact path instead of failing fast."
        ),
    )
    parser.add_argument(
        "--wait-for-artifact-seconds",
        type=int,
        default=180,
        help=(
            "When local artifacts are pushed to iOS after app launch, wait at "
            "most this many seconds for the artifact to appear before loading."
        ),
    )
    parser.add_argument(
        "--coreml-compute-units",
        help=(
            "Optional Core ML compute units passed through to the Flutter "
            "runtime smoke backend (for example cpuAndNeuralEngine or "
            "cpuAndGPU)."
        ),
    )
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--no-debug-retry-on-timeout",
        action="store_true",
        help=(
            "Do not retry a release-mode timeout in debug mode. Wireless iOS "
            "release runs can suppress Flutter stdout markers even when the "
            "runtime load succeeds."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
    )
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()

    device_id = args.device_id or detect_ios_device_id()
    if not device_id:
        raise SystemExit("No available iOS device found via `flutter devices --machine`.")
    started = time.time()
    out_path = _requested_out_path(
        model_id=args.model_id,
        out=args.out,
    )
    try:
        resolved_artifact, artifact_prepare = resolve_ios_artifact(
            artifact=args.artifact,
            device_id=device_id,
            bundle_id=args.bundle_id,
            model_id=args.model_id,
            engine=args.engine,
            device_artifact_dir=args.device_artifact_dir,
            device_push=not args.no_device_push,
            device_copy_retries=max(0, args.device_copy_retries),
            direct_fallback_on_push_failure=args.allow_direct_fallback_on_push_failure,
        )
    except Exception as exc:
        finished = time.time()
        result = {
            "platform": "ios",
            "device_id": device_id,
            "bundle_id": args.bundle_id,
            "model_id": args.model_id,
            "engine": args.engine,
            "build_mode": args.build_mode,
            "artifact": args.artifact,
            "command": None,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(finished)),
            "duration_seconds": finished - started,
            "timed_out": False,
            "runtime_smoke": None,
            "passed": False,
            "status": "artifact_prepare_failed",
            "error": str(exc),
        }
        _write_result(
            out_path=out_path,
            model_id=args.model_id,
            result=result,
        )
        if not args.allow_fail:
            raise SystemExit(1)
        return

    result = _run_flutter_smoke_attempt(
        started=started,
        device_id=device_id,
        bundle_id=args.bundle_id,
        model_id=args.model_id,
        engine=args.engine,
        build_mode=args.build_mode,
        artifact=args.artifact,
        resolved_artifact=resolved_artifact,
        artifact_prepare=artifact_prepare,
        wait_for_artifact_seconds=args.wait_for_artifact_seconds,
        coreml_compute_units=args.coreml_compute_units,
        device_copy_retries=args.device_copy_retries,
        direct_fallback_on_push_failure=args.allow_direct_fallback_on_push_failure,
        timeout_seconds=args.timeout_seconds,
    )
    if _should_retry_debug_after_release_timeout(
        result,
        build_mode=args.build_mode,
        disabled=args.no_debug_retry_on_timeout,
    ):
        release_attempt = result
        debug_prepare = dict(artifact_prepare)
        debug_result = _run_flutter_smoke_attempt(
            started=time.time(),
            device_id=device_id,
            bundle_id=args.bundle_id,
            model_id=args.model_id,
            engine=args.engine,
            build_mode="debug",
            artifact=args.artifact,
            resolved_artifact=resolved_artifact,
            artifact_prepare=debug_prepare,
            wait_for_artifact_seconds=args.wait_for_artifact_seconds,
            coreml_compute_units=args.coreml_compute_units,
            device_copy_retries=args.device_copy_retries,
            direct_fallback_on_push_failure=args.allow_direct_fallback_on_push_failure,
            timeout_seconds=args.timeout_seconds,
        )
        if debug_result.get("passed") is True:
            result = debug_result
            result["debug_retry_after_release_timeout"] = True
            result["release_attempt"] = release_attempt
        else:
            result["debug_retry"] = debug_result

    _write_result(
        out_path=out_path,
        model_id=args.model_id,
        result=result,
    )
    if not args.allow_fail and not result["passed"]:
        raise SystemExit(1)


def _run_flutter_smoke_attempt(
    *,
    started: float,
    device_id: str,
    bundle_id: str,
    model_id: str,
    engine: str,
    build_mode: str,
    artifact: str | None,
    resolved_artifact: str | None,
    artifact_prepare: dict[str, object],
    wait_for_artifact_seconds: int,
    coreml_compute_units: str | None,
    device_copy_retries: int,
    direct_fallback_on_push_failure: bool,
    timeout_seconds: int,
) -> dict[str, object]:
    deferred_push = None
    artifact_wait = 0
    if _is_deferred_push_mode(str(artifact_prepare.get("mode") or "")):
        artifact_wait = max(0, wait_for_artifact_seconds)
        deferred_push = _DeferredIosArtifactPush(
            device_id=device_id,
            bundle_id=bundle_id,
            source_path=Path(str(artifact_prepare["copy_source"])),
            destination=str(artifact_prepare["copy_destination"]),
            retries=max(0, device_copy_retries),
            runtime_artifact=resolved_artifact,
            direct_fallback_on_push_failure=direct_fallback_on_push_failure,
        )

    command = flutter_command(
        device_id=device_id,
        model_id=model_id,
        engine=engine,
        build_mode=build_mode,
        artifact=resolved_artifact,
        wait_for_artifact_seconds=artifact_wait,
        coreml_compute_units=coreml_compute_units,
    )
    result: dict[str, object] = {
        "platform": "ios",
        "device_id": device_id,
        "bundle_id": bundle_id,
        "model_id": model_id,
        "engine": engine,
        "build_mode": build_mode,
        "artifact": artifact,
        "resolved_artifact": resolved_artifact,
        "coreml_compute_units": coreml_compute_units,
        "artifact_prepare": artifact_prepare,
        "command": command,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
    }
    marker_parser = MarkerParser()
    process = subprocess.Popen(
        command,
        cwd=EXAMPLE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=dict(os.environ),
    )
    try:
        marker_payload, timed_out, lines = capture_runtime_smoke(
            process,
            timeout_seconds=timeout_seconds,
            parser=marker_parser,
            line_observer=deferred_push.observe_line if deferred_push else None,
        )
    finally:
        terminate_process(process)
    flutter_exit_code = process.returncode

    finished = time.time()
    result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(finished))
    result["duration_seconds"] = finished - started
    result["timed_out"] = timed_out
    result["flutter_exit_code"] = flutter_exit_code
    result["captured_line_count"] = len(lines)
    result["captured_tail"] = lines[-200:]
    result["runtime_smoke"] = marker_payload
    if deferred_push is not None:
        artifact_prepare["deferred_push"] = deferred_push.meta()
        if deferred_push.error:
            artifact_prepare["deferred_push_error"] = deferred_push.error
    passed = bool(marker_payload and marker_payload.get("passed") is True) and not timed_out
    result["passed"] = passed
    status, error = classify_run_result(
        marker_payload=marker_payload,
        timed_out=timed_out,
        flutter_exit_code=flutter_exit_code,
        captured_lines=lines,
    )
    result["status"] = status
    if error:
        result["error"] = error
    if (
        not result["passed"]
        and deferred_push is not None
        and deferred_push.error
        and result.get("status") == "runtime_failed"
    ):
        result["status"] = "artifact_prepare_failed"
        result["error"] = deferred_push.error
    return result


def _should_retry_debug_after_release_timeout(
    result: dict[str, object],
    *,
    build_mode: str,
    disabled: bool,
) -> bool:
    if disabled or build_mode != "release":
        return False
    return result.get("status") == "timeout" and result.get("runtime_smoke") is None


def _write_result(
    *,
    out_path: Path,
    model_id: str,
    result: dict[str, object],
) -> None:
    fallback_out_path = (
        ROOT
        / "benchmark"
        / "out_local"
        / "runtime"
        / model_id
        / "ios"
        / out_path.name
    )
    resolved_out_path, out_path_meta = prepare_out_path(
        out_path=out_path,
        fallback_out_path=fallback_out_path,
    )
    result["out_path_requested"] = str(out_path)
    if out_path_meta["mode"] == "fallback":
        result["out_path_fallback"] = out_path_meta
    result["out_path"] = str(resolved_out_path)
    resolved_out_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


def _requested_out_path(*, model_id: str, out: Path | None) -> Path:
    return (
        out
        if out is not None
        else ROOT / "benchmark" / "out" / "runtime" / model_id / "ios" / "device_smoke.json"
    )


class _DeferredIosArtifactPush:
    def __init__(
        self,
        *,
        device_id: str,
        bundle_id: str,
        source_path: Path,
        destination: str,
        retries: int,
        runtime_artifact: str | None,
        direct_fallback_on_push_failure: bool,
    ) -> None:
        self._device_id = device_id
        self._bundle_id = bundle_id
        self._source_path = source_path
        self._destination = destination
        self._retries = retries
        self._runtime_artifact = runtime_artifact
        self._direct_fallback_on_push_failure = direct_fallback_on_push_failure
        self.triggered = False
        self.attempted = False
        self.pushed = False
        self.error: str | None = None
        self.trigger_line: str | None = None
        self.local_copy_source: str | None = None
        self.materialized_copy_source: str | None = None
        self.remote_copy_destination: str | None = None
        self.triggered_at: float | None = None
        self.completed_at: float | None = None

    def observe_line(self, line: str) -> None:
        if self.attempted:
            return
        if not _is_runtime_smoke_boot_line(line):
            return
        self.attempted = True
        self.triggered = True
        self.triggered_at = time.time()
        self.trigger_line = line
        cleanup_dir = None
        try:
            copy_source = self._source_path
            if _contains_symlink_entry(copy_source):
                cleanup_dir = Path(tempfile.mkdtemp(prefix="dmf_ios_copy_src_"))
                copy_source = cleanup_dir / self._source_path.name
                _copy_dereferencing_symlinks(self._source_path, copy_source)
                self.materialized_copy_source = str(copy_source)
            self.local_copy_source = str(copy_source)
            self.remote_copy_destination = _devicectl_copy_to(
                device_id=self._device_id,
                bundle_id=self._bundle_id,
                source_path=copy_source,
                destination=self._destination,
                retries=self._retries,
            )
            self.pushed = True
        except Exception as exc:
            if self._direct_fallback_on_push_failure:
                self.error = (
                    "deferred push failed and direct fallback was requested: "
                    f"{exc}"
                )
            else:
                self.error = f"deferred push failed: {exc}"
        finally:
            if cleanup_dir is not None:
                shutil.rmtree(cleanup_dir, ignore_errors=True)
            self.completed_at = time.time()

    def meta(self) -> dict[str, object]:
        return {
            "attempted": self.attempted,
            "triggered": self.triggered,
            "pushed": self.pushed,
            "trigger_line": self.trigger_line,
            "source_path": str(self._source_path),
            "local_copy_source": self.local_copy_source,
            "materialized_copy_source": self.materialized_copy_source,
            "copy_destination": self._destination,
            "copy_destination_resolved": self.remote_copy_destination,
            "runtime_artifact": self._runtime_artifact,
            "triggered_at": _isoformat(self.triggered_at),
            "completed_at": _isoformat(self.completed_at),
        }


def flutter_command(
    *,
    device_id: str,
    model_id: str,
    engine: str,
    build_mode: str,
    artifact: str | None,
    wait_for_artifact_seconds: int = 0,
    coreml_compute_units: str | None = None,
) -> list[str]:
    command = [
        "flutter",
        "run",
        "-d",
        device_id,
        f"--{build_mode}",
        "--dart-define=DMF_RUNTIME_SMOKE=true",
        f"--dart-define=DMF_RUNTIME_SMOKE_MODEL={model_id}",
        f"--dart-define=DMF_RUNTIME_SMOKE_ENGINE={engine}",
    ]
    if wait_for_artifact_seconds > 0:
        command.append(
            "--dart-define="
            f"DMF_RUNTIME_SMOKE_WAIT_FOR_ARTIFACT_SECONDS={wait_for_artifact_seconds}"
        )
    if coreml_compute_units:
        command.append(
            "--dart-define="
            f"DMF_RUNTIME_SMOKE_COREML_COMPUTE_UNITS={coreml_compute_units}"
        )
    if artifact:
        command.append(f"--dart-define=DMF_RUNTIME_SMOKE_ARTIFACT={artifact}")
    return command


def detect_ios_device_id() -> str | None:
    command = ["flutter", "devices", "--machine"]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )
    if completed.returncode != 0:
        return None
    try:
        devices = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(devices, list):
        return None
    for item in devices:
        if not isinstance(item, dict):
            continue
        target_platform = str(item.get("targetPlatform") or "")
        if target_platform != "ios":
            continue
        if item.get("isSupported") is False:
            continue
        identifier = item.get("id")
        if isinstance(identifier, str) and identifier:
            return identifier
    return None


def resolve_ios_artifact(
    *,
    artifact: str | None,
    device_id: str,
    bundle_id: str,
    model_id: str,
    engine: str,
    device_artifact_dir: str,
    device_push: bool,
    device_copy_retries: int,
    direct_fallback_on_push_failure: bool,
) -> tuple[str | None, dict[str, object]]:
    if not artifact:
        return None, {"mode": "none"}
    if not device_push:
        return artifact, {"mode": "direct", "reason": "device_push_disabled"}
    if _is_hf_uri(artifact):
        return artifact, {"mode": "direct", "reason": "hf_uri"}
    local_artifact = _resolve_local_artifact_path(artifact)
    if local_artifact is None:
        return artifact, {"mode": "direct", "reason": "not_hf_or_local_path"}
    copy_plan = _build_ios_copy_plan(
        model_id=model_id,
        engine=engine,
        local_artifact=local_artifact,
        device_artifact_dir=device_artifact_dir,
    )
    mode = "local_push_deferred"
    if direct_fallback_on_push_failure:
        mode = "local_push_deferred_fallback"
    return str(copy_plan["runtime_artifact"]), {
        "mode": mode,
        "source_path": str(local_artifact),
        "runtime_artifact": str(copy_plan["runtime_artifact"]),
        "copy_source": str(copy_plan["copy_source"]),
        "copy_destination": str(copy_plan["copy_destination"]),
        "copy_retries": int(device_copy_retries),
        "pipeline_layout": bool(copy_plan["pipeline_layout"]),
    }


def push_artifact_to_ios_device(
    *,
    device_id: str,
    bundle_id: str,
    model_id: str,
    engine: str,
    local_artifact: Path,
    device_artifact_dir: str,
    copy_retries: int,
) -> tuple[str, dict[str, object]]:
    copy_plan = _build_ios_copy_plan(
        model_id=model_id,
        engine=engine,
        local_artifact=local_artifact,
        device_artifact_dir=device_artifact_dir,
    )
    copy_source = Path(str(copy_plan["copy_source"]))
    destination = str(copy_plan["copy_destination"])
    runtime_artifact = str(copy_plan["runtime_artifact"])
    pipeline_layout = bool(copy_plan["pipeline_layout"])
    destination_path = _devicectl_copy_to(
        device_id=device_id,
        bundle_id=bundle_id,
        source_path=copy_source,
        destination=destination,
        retries=copy_retries,
    )
    if pipeline_layout:
        runtime_name = PurePosixPath(runtime_artifact).name
        remote_artifact = str(PurePosixPath(destination_path) / runtime_name)
    else:
        remote_artifact = destination_path
    return remote_artifact, {
        "copy_source": str(copy_source),
        "copy_destination": destination,
        "runtime_artifact": runtime_artifact,
    }


def _build_ios_copy_plan(
    *,
    model_id: str,
    engine: str,
    local_artifact: Path,
    device_artifact_dir: str,
) -> dict[str, object]:
    resolved = local_artifact.resolve()
    remote_base = PurePosixPath(device_artifact_dir) / model_id / engine
    pipeline_layout = resolved.is_file() and _is_pipeline_artifact(resolved)
    if pipeline_layout:
        copy_source = resolved.parent
        copy_destination = str(remote_base)
        runtime_artifact = str(remote_base / resolved.name)
    elif resolved.is_dir():
        copy_source = resolved
        copy_destination = str(remote_base / resolved.name)
        runtime_artifact = copy_destination
    else:
        copy_source = resolved
        copy_destination = str(remote_base / resolved.name)
        runtime_artifact = copy_destination
    return {
        "copy_source": copy_source,
        "copy_destination": copy_destination,
        "runtime_artifact": runtime_artifact,
        "pipeline_layout": pipeline_layout,
    }


def _devicectl_copy_to(
    *,
    device_id: str,
    bundle_id: str,
    source_path: Path,
    destination: str,
    retries: int = 2,
) -> str:
    with tempfile.NamedTemporaryFile(
        prefix="dmf_ios_copy_",
        suffix=".json",
        delete=False,
    ) as temp:
        json_output = Path(temp.name)
    command = [
        "xcrun",
        "devicectl",
        "device",
        "copy",
        "to",
        "--device",
        device_id,
        "--source",
        str(source_path),
        "--destination",
        destination,
        "--domain-type",
        "appDataContainer",
        "--domain-identifier",
        bundle_id,
        "--json-output",
        str(json_output),
    ]
    attempts = 0
    while True:
        attempts += 1
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            env=dict(os.environ),
        )
        if completed.returncode == 0:
            break
        message = completed.stderr.strip() or completed.stdout.strip()
        retryable = _is_retryable_devicectl_error(message)
        if attempts > retries or not retryable:
            raise RuntimeError(
                "devicectl copy failed: "
                f"{message or f'exit code {completed.returncode}'}"
            )
        time.sleep(min(10.0, 2.0 * attempts))
    try:
        payload = json.loads(json_output.read_text(encoding="utf-8"))
    finally:
        json_output.unlink(missing_ok=True)
    result = payload.get("result") if isinstance(payload, dict) else None
    destination_url = result.get("destination") if isinstance(result, dict) else None
    if not isinstance(destination_url, str) or not destination_url:
        raise RuntimeError("devicectl copy returned no destination path.")
    parsed = urlparse(destination_url)
    if parsed.scheme != "file":
        raise RuntimeError(f"Unsupported devicectl destination URL: {destination_url}")
    return unquote(parsed.path)


def _contains_symlink_entry(path: Path) -> bool:
    if path.is_symlink():
        return True
    if not path.is_dir():
        return False
    try:
        return any(child.is_symlink() for child in path.rglob("*"))
    except OSError:
        return False


def _copy_dereferencing_symlinks(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination, symlinks=False)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination, follow_symlinks=True)


def _is_hf_uri(value: str) -> bool:
    return value.startswith("hf://")


def _is_pipeline_artifact(path: Path) -> bool:
    if path.suffix.lower() != ".json":
        return False
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    if not isinstance(decoded, dict):
        return False
    return decoded.get("format") in {
        "dart_mlx_ffi.coreml_pipeline.v1",
        "dart_mlx_ffi.onnx_pipeline.v1",
    }


def classify_run_result(
    *,
    marker_payload: dict[str, object] | None,
    timed_out: bool,
    flutter_exit_code: int | None,
    captured_lines: list[str],
) -> tuple[str, str | None]:
    if marker_payload and marker_payload.get("passed") is True and not timed_out:
        return "passed", None
    if timed_out:
        return "timeout", "Timed out waiting for runtime smoke marker."
    if marker_payload is not None:
        marker_error = marker_payload.get("error")
        if isinstance(marker_error, str) and marker_error.strip():
            return "runtime_failed", marker_error.strip()
        return "runtime_failed", "Runtime smoke marker reported failure."
    if flutter_exit_code not in (None, 0):
        launch_error = _extract_flutter_launch_error(captured_lines)
        if launch_error:
            return "launch_failed", launch_error
        return (
            "launch_failed",
            f"flutter run exited with code {flutter_exit_code} before runtime marker.",
        )
    return "no_runtime_marker", "Runtime smoke marker was not emitted."


def _extract_flutter_launch_error(lines: list[str]) -> str | None:
    keywords = (
        "Error running application",
        "Could not run ",
        "Failed to",
        "error:",
        "Exception",
    )
    for line in reversed(lines):
        text = line.strip()
        if not text:
            continue
        if any(keyword in text for keyword in keywords):
            return text
    return None


def _is_retryable_devicectl_error(message: str) -> bool:
    lowered = message.lower()
    retry_tokens = (
        "socket was closed unexpectedly",
        "failed to write data to the network socket",
        "no provider was found",
        "the specified file could not be transferred",
        "timed out",
        "connection reset by peer",
    )
    return any(token in lowered for token in retry_tokens)


def _is_runtime_smoke_boot_line(line: str) -> bool:
    return "DMF_RUNTIME_SMOKE:BOOT" in line


def _isoformat(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))


def _resolve_local_artifact_path(path: str) -> Path | None:
    local = Path(path).expanduser()
    if local.exists():
        return local
    rooted = (ROOT / path).expanduser()
    if rooted.exists():
        return rooted
    return None


def _is_deferred_push_mode(mode: str) -> bool:
    return mode.startswith("local_push_deferred")


def prepare_out_path(
    *,
    out_path: Path,
    fallback_out_path: Path,
) -> tuple[Path, dict[str, str]]:
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return out_path, {"mode": "requested"}
    except OSError as exc:
        fallback_out_path.parent.mkdir(parents=True, exist_ok=True)
        return fallback_out_path, {
            "mode": "fallback",
            "reason": str(exc),
            "requested_out_path": str(out_path),
            "resolved_out_path": str(fallback_out_path),
        }


if __name__ == "__main__":
    main()
