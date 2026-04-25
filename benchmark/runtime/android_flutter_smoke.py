from __future__ import annotations

import argparse
import json
import os
from pathlib import PurePosixPath
import selectors
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from flutter_smoke_common import MarkerParser, terminate_process
from litert_env import resolve_litert_environment
from ort_env import resolve_ort_environment


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = ROOT / "example"
ANDROID_LOG_TAG = "DMF_RUNTIME_SMOKE"
DEFAULT_ANDROID_PACKAGE = "com.example.dartmlxffiexample"
DEFAULT_HF_CACHE_DIR = ROOT / "benchmark" / "out" / "runtime" / "_hf_cache"
RUNTIME_ENV_FILE = ROOT / ".dart_mlx_runtime_env.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Flutter example on an Android device in runtime-smoke "
            "mode and capture the structured result marker from app logs."
        )
    )
    parser.add_argument("--device-id")
    parser.add_argument(
        "--device-user",
        default="0",
        help="Android user/profile ID for launch (default: 0).",
    )
    parser.add_argument("--model-id", default="silero_vad")
    parser.add_argument("--engine", default="litert", choices=["coreml", "onnx", "litert"])
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
        "--package-name",
        default=DEFAULT_ANDROID_PACKAGE,
        help=(
            "Android package used for app-specific external storage when "
            "prefetching HF artifacts."
        ),
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=DEFAULT_HF_CACHE_DIR,
        help="Host cache root for downloaded Hugging Face artifacts.",
    )
    parser.add_argument(
        "--device-artifact-dir",
        default=None,
        help=(
            "Override remote directory for pushed artifacts. Defaults to "
            "/sdcard/Android/data/<package>/files/dart_mlx_ffi_runtime_smoke."
        ),
    )
    parser.add_argument(
        "--no-host-prefetch",
        action="store_true",
        help="Disable host-side HF artifact download/push and pass artifact URI directly.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
    )
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()

    device_id = args.device_id or detect_android_device_id()
    if not device_id:
        raise SystemExit("No available Android device found via `flutter devices --machine`.")

    resolved_artifact, artifact_prepare = resolve_android_artifact(
        artifact=args.artifact,
        device_id=device_id,
        package_name=args.package_name,
        model_id=args.model_id,
        engine=args.engine,
        hf_cache_dir=args.hf_cache_dir,
        device_artifact_dir=args.device_artifact_dir,
        host_prefetch=not args.no_host_prefetch,
    )

    command = flutter_command(
        device_id=device_id,
        device_user=args.device_user,
        model_id=args.model_id,
        engine=args.engine,
        artifact=resolved_artifact,
        build_mode=args.build_mode,
    )
    started = time.time()
    result = {
        "platform": "android",
        "device_id": device_id,
        "device_user": args.device_user,
        "model_id": args.model_id,
        "engine": args.engine,
        "build_mode": args.build_mode,
        "package_name": args.package_name,
        "artifact": args.artifact,
        "resolved_artifact": resolved_artifact,
        "artifact_prepare": artifact_prepare,
        "command": command,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
    }
    marker_parser = MarkerParser()
    flutter_env = _flutter_env()
    android_abi = detect_android_abi(device_id)
    result["android_abi"] = android_abi
    runtime_env_values: dict[str, str] = {}
    if args.engine == "onnx":
        ort_env = resolve_ort_environment(
            fetch_headers=True,
            target_os="android",
            target_arch=android_abi,
        )
        result["ort_env"] = ort_env.to_json()
        if ort_env.ready:
            values = ort_env.to_env()
            runtime_env_values.update(values)
            flutter_env.update(values)
    if args.engine == "litert":
        litert_env = resolve_litert_environment(
            fetch_library=True,
            target_os="android",
            target_arch=android_abi,
        )
        result["litert_env"] = litert_env.to_json()
        if litert_env.ready:
            values = litert_env.to_env()
            runtime_env_values.update(values)
            flutter_env.update(values)
    runtime_env_file = write_runtime_env_file(runtime_env_values)
    flutter_env["DART_MLX_RUNTIME_ENV_FILE"] = str(runtime_env_file)
    result["runtime_env_file"] = str(runtime_env_file)
    result["runtime_env"] = runtime_env_values
    logcat_reset(device_id)
    logcat_command = logcat_follow_command(device_id=device_id)
    result["logcat_command"] = logcat_command
    process = subprocess.Popen(
        command,
        cwd=EXAMPLE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=flutter_env,
    )
    logcat_process = subprocess.Popen(
        logcat_command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=dict(os.environ),
    )
    try:
        marker_payload, timed_out, lines, logcat_lines = capture_runtime_smoke_android(
            flutter_process=process,
            logcat_process=logcat_process,
            timeout_seconds=args.timeout_seconds,
            parser=marker_parser,
        )
    finally:
        terminate_process(process)
        terminate_process(logcat_process)
        clear_runtime_env_file(runtime_env_file)

    finished = time.time()
    result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(finished))
    result["duration_seconds"] = finished - started
    result["timed_out"] = timed_out
    result["captured_line_count"] = len(lines)
    result["captured_tail"] = lines[-200:]
    result["captured_logcat_line_count"] = len(logcat_lines)
    result["captured_logcat_tail"] = logcat_lines[-200:]
    result["runtime_smoke"] = marker_payload
    result["passed"] = (
        bool(marker_payload and marker_payload.get("passed") is True) and not timed_out
    )

    out_path = (
        args.out
        if args.out is not None
        else ROOT / "benchmark" / "out" / "runtime" / args.model_id / "android" / "device_smoke.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if not args.allow_fail and not result["passed"]:
        raise SystemExit(1)


def flutter_command(
    *,
    device_id: str,
    device_user: str | None,
    model_id: str,
    engine: str,
    artifact: str | None,
    build_mode: str,
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
    if device_user:
        command.append(f"--device-user={device_user}")
    if artifact:
        command.append(f"--dart-define=DMF_RUNTIME_SMOKE_ARTIFACT={artifact}")
    return command


def resolve_android_artifact(
    *,
    artifact: str | None,
    device_id: str,
    package_name: str,
    model_id: str,
    engine: str,
    hf_cache_dir: Path,
    device_artifact_dir: str | None,
    host_prefetch: bool,
) -> tuple[str | None, dict[str, Any]]:
    if not artifact:
        return None, {"mode": "none"}
    if not host_prefetch:
        return artifact, {"mode": "direct", "reason": "host_prefetch_disabled"}
    if not _is_hf_uri(artifact):
        return artifact, {"mode": "direct", "reason": "not_hf_uri"}
    try:
        local_path = download_hf_artifact(artifact, hf_cache_dir)
        remote_path = push_artifact_to_device(
            device_id=device_id,
            local_path=local_path,
            package_name=package_name,
            model_id=model_id,
            engine=engine,
            device_artifact_dir=device_artifact_dir,
        )
        return remote_path, {
            "mode": "host_prefetch",
            "source_uri": artifact,
            "local_path": str(local_path),
            "remote_path": remote_path,
        }
    except Exception as exc:
        return artifact, {
            "mode": "direct_fallback",
            "source_uri": artifact,
            "error": str(exc),
        }


def download_hf_artifact(uri: str, cache_dir: Path) -> Path:
    repo_id, artifact_path = parse_hf_uri(uri)
    local_path = cache_dir / repo_id / artifact_path
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = local_path.with_suffix(local_path.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    url = f"https://huggingface.co/{repo_id}/resolve/main/{artifact_path}"
    request = urllib.request.Request(url, headers={"User-Agent": "dart-mlx-ffi-runtime-smoke"})
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with temp_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to download HF artifact {uri}: {exc}") from exc
    temp_path.replace(local_path)
    return local_path


def push_artifact_to_device(
    *,
    device_id: str,
    local_path: Path,
    package_name: str,
    model_id: str,
    engine: str,
    device_artifact_dir: str | None,
) -> str:
    remote_root = (
        device_artifact_dir
        if device_artifact_dir
        else f"/sdcard/Android/data/{package_name}/files/dart_mlx_ffi_runtime_smoke"
    )
    remote_dir = f"{remote_root.rstrip('/')}/{model_id}/{engine}"
    _run_adb(device_id, ["shell", "mkdir", "-p", remote_dir], check=True)
    for path in _artifact_permission_paths(
        package_name=package_name,
        remote_root=remote_root,
        remote_dir=remote_dir,
    ):
        _chmod_device_path(device_id=device_id, path=path, mode="0775")
    remote_path = f"{remote_dir}/{local_path.name}"
    _run_adb(device_id, ["push", str(local_path), remote_path], check=True)
    _chmod_device_path(device_id=device_id, path=remote_path, mode="0664")
    return remote_path


def parse_hf_uri(uri: str) -> tuple[str, str]:
    if not _is_hf_uri(uri):
        raise ValueError(f"Not an HF URI: {uri}")
    raw = uri[len("hf://") :]
    parts = [part for part in raw.split("/") if part]
    if len(parts) < 3:
        raise ValueError(f"Invalid HF URI, expected hf://<org>/<repo>/<path>: {uri}")
    repo_id = f"{parts[0]}/{parts[1]}"
    artifact_path = "/".join(parts[2:])
    return repo_id, artifact_path


def _is_hf_uri(value: str) -> bool:
    return value.startswith("hf://")


def _run_adb(device_id: str, args: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
    command = ["adb", "-s", device_id, *args]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"adb command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout: {completed.stdout}\nstderr: {completed.stderr}"
        )
    return completed


def _artifact_permission_paths(
    *,
    package_name: str,
    remote_root: str,
    remote_dir: str,
) -> list[str]:
    paths: list[PurePosixPath] = []
    root = PurePosixPath(remote_root)
    paths.append(root)
    package_root = PurePosixPath(f"/sdcard/Android/data/{package_name}")
    if _is_same_or_child(root, package_root):
        current = package_root
        while True:
            paths.append(current)
            if current == root:
                break
            next_path = current / root.relative_to(current).parts[0]
            current = next_path
    current = PurePosixPath(remote_dir)
    while True:
        paths.append(current)
        if current == root or str(current) in {"/", ".", ""}:
            break
        current = current.parent
    deduped = {
        str(path): path
        for path in paths
        if str(path) not in {"/", "", "."}
        and str(path).startswith(("/sdcard/", "/storage/"))
    }
    return [str(path) for path in deduped.values()]


def _is_same_or_child(path: PurePosixPath, parent: PurePosixPath) -> bool:
    if path == parent:
        return True
    return parent in path.parents


def _chmod_device_path(*, device_id: str, path: str, mode: str) -> None:
    _run_adb(
        device_id,
        ["shell", "chmod", mode, path],
        check=False,
    )


def write_runtime_env_file(values: dict[str, str]) -> Path:
    payload = dict(values)
    RUNTIME_ENV_FILE.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return RUNTIME_ENV_FILE


def clear_runtime_env_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def logcat_reset(device_id: str) -> None:
    subprocess.run(
        ["adb", "-s", device_id, "logcat", "-c"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )


def logcat_follow_command(*, device_id: str) -> list[str]:
    return [
        "adb",
        "-s",
        device_id,
        "logcat",
        "-v",
        "brief",
        f"{ANDROID_LOG_TAG}:I",
        "flutter:I",
        "*:S",
    ]


def capture_runtime_smoke_android(
    *,
    flutter_process: subprocess.Popen[str],
    logcat_process: subprocess.Popen[str],
    timeout_seconds: int,
    parser: MarkerParser,
) -> tuple[dict[str, Any] | None, bool, list[str], list[str]]:
    flutter_lines: list[str] = []
    logcat_lines: list[str] = []
    marker_payload: dict[str, Any] | None = None
    flutter_parser = parser
    logcat_parser = MarkerParser()
    timed_out = False
    deadline = time.time() + timeout_seconds
    selector = selectors.DefaultSelector()
    if flutter_process.stdout is not None:
        selector.register(flutter_process.stdout, selectors.EVENT_READ, ("flutter", flutter_process))
    if logcat_process.stdout is not None:
        selector.register(logcat_process.stdout, selectors.EVENT_READ, ("logcat", logcat_process))
    try:
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                timed_out = True
                break
            events = selector.select(timeout=min(1.0, remaining))
            if not events:
                if flutter_process.poll() is not None and logcat_process.poll() is not None:
                    break
                continue
            for key, _ in events:
                source, owner = key.data
                line = key.fileobj.readline()
                if line == "":
                    if owner.poll() is not None:
                        try:
                            selector.unregister(key.fileobj)
                        except Exception:
                            pass
                    continue
                line = line.rstrip("\n")
                print(line, flush=True)
                if source == "flutter":
                    flutter_lines.append(line)
                    marker_payload = flutter_parser.feed(line)
                else:
                    logcat_lines.append(line)
                    marker_payload = logcat_parser.feed(line)
                if marker_payload is not None:
                    return marker_payload, timed_out, flutter_lines, logcat_lines
            if not selector.get_map():
                break
    finally:
        selector.close()
    return marker_payload, timed_out, flutter_lines, logcat_lines


def detect_android_device_id() -> str | None:
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
        if not target_platform.startswith("android"):
            continue
        if item.get("isSupported") is False:
            continue
        identifier = item.get("id")
        if isinstance(identifier, str) and identifier:
            return identifier
    return None


def detect_android_abi(device_id: str) -> str | None:
    completed = _run_adb(
        device_id,
        ["shell", "getprop", "ro.product.cpu.abi"],
        check=False,
    )
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value if value else None


def _flutter_env() -> dict[str, str]:
    env = dict(os.environ)
    java_home = env.get("JAVA_HOME")
    if java_home and Path(java_home).exists():
        return env
    if sys.platform != "darwin":
        return env
    completed = subprocess.run(
        ["/usr/libexec/java_home"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if completed.returncode != 0:
        return env
    resolved = completed.stdout.strip()
    if resolved and Path(resolved).exists():
        env["JAVA_HOME"] = resolved
    return env


if __name__ == "__main__":
    main()
