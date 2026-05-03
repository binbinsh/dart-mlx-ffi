from __future__ import annotations

import argparse
import json
import os
import selectors
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from flutter_smoke_common import (
    MarkerParser,
    capture_runtime_smoke,
    extract_marker_payload,
    terminate_process,
)


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = ROOT / "example"
EXAMPLE_TARGET = "lib/main.dart"


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
        default="debug",
    )
    parser.add_argument(
        "--artifact",
        help="Optional artifact override passed to Flutter runtime-smoke mode.",
    )
    parser.add_argument("--wait-for-artifact-seconds", type=int)
    parser.add_argument("--coreml-compute-units")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--no-syslog",
        action="store_true",
        help="Disable idevicesyslog capture and only parse flutter run output.",
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

    command = flutter_command(
        device_id=device_id,
        model_id=args.model_id,
        engine=args.engine,
        artifact=args.artifact,
        build_mode=args.build_mode,
        wait_for_artifact_seconds=args.wait_for_artifact_seconds,
        coreml_compute_units=args.coreml_compute_units,
    )
    started = time.time()
    result = {
        "platform": "ios",
        "device_id": device_id,
        "model_id": args.model_id,
        "engine": args.engine,
        "build_mode": args.build_mode,
        "artifact": args.artifact,
        "command": command,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
    }
    marker_parser = MarkerParser()
    syslog_command = ios_syslog_command(device_id=device_id) if not args.no_syslog else None
    result["syslog_command"] = syslog_command
    process = subprocess.Popen(
        command,
        cwd=EXAMPLE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=dict(os.environ),
    )
    syslog_process = (
        subprocess.Popen(
            syslog_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=dict(os.environ),
        )
        if syslog_command is not None
        else None
    )
    try:
        if syslog_process is None:
            marker_payload, timed_out, lines = capture_runtime_smoke(
                process,
                timeout_seconds=args.timeout_seconds,
                parser=marker_parser,
            )
            syslog_lines: list[str] = []
        else:
            marker_payload, timed_out, lines, syslog_lines = capture_runtime_smoke_ios(
                flutter_process=process,
                syslog_process=syslog_process,
                timeout_seconds=args.timeout_seconds,
                parser=marker_parser,
            )
    finally:
        terminate_process(process)
        if syslog_process is not None:
            terminate_process(syslog_process)

    finished = time.time()
    result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(finished))
    result["duration_seconds"] = finished - started
    result["timed_out"] = timed_out
    result["captured_line_count"] = len(lines)
    result["captured_tail"] = lines[-200:]
    result["captured_syslog_line_count"] = len(syslog_lines)
    result["captured_syslog_tail"] = syslog_lines[-200:]
    result["runtime_smoke"] = marker_payload
    result["passed"] = (
        bool(marker_payload and marker_payload.get("passed") is True) and not timed_out
    )

    out_path = (
        args.out
        if args.out is not None
        else ROOT / "benchmark" / "out" / "runtime" / args.model_id / "ios" / "device_smoke.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if not args.allow_fail and not result["passed"]:
        raise SystemExit(1)


def flutter_command(
    *,
    device_id: str,
    model_id: str,
    engine: str,
    artifact: str | None,
    build_mode: str = "debug",
    wait_for_artifact_seconds: int | None = None,
    coreml_compute_units: str | None = None,
) -> list[str]:
    command = [
        "flutter",
        "run",
        "-d",
        device_id,
        f"--{build_mode}",
        "--target",
        EXAMPLE_TARGET,
        "--dart-define=DINF_RUNTIME_SMOKE=true",
        "--dart-define=DMF_RUNTIME_SMOKE=true",
        f"--dart-define=DINF_RUNTIME_SMOKE_MODEL={model_id}",
        f"--dart-define=DMF_RUNTIME_SMOKE_MODEL={model_id}",
        f"--dart-define=DINF_RUNTIME_SMOKE_ENGINE={engine}",
        f"--dart-define=DMF_RUNTIME_SMOKE_ENGINE={engine}",
    ]
    if artifact:
        command.append(f"--dart-define=DINF_RUNTIME_SMOKE_ARTIFACT={artifact}")
        command.append(f"--dart-define=DMF_RUNTIME_SMOKE_ARTIFACT={artifact}")
    if wait_for_artifact_seconds is not None:
        command.append(
            "--dart-define="
            f"DINF_RUNTIME_SMOKE_WAIT_FOR_ARTIFACT_SECONDS={wait_for_artifact_seconds}"
        )
        command.append(
            "--dart-define="
            f"DMF_RUNTIME_SMOKE_WAIT_FOR_ARTIFACT_SECONDS={wait_for_artifact_seconds}"
        )
    if coreml_compute_units:
        command.append(
            f"--dart-define=DINF_RUNTIME_SMOKE_COREML_COMPUTE_UNITS={coreml_compute_units}"
        )
        command.append(
            f"--dart-define=DMF_RUNTIME_SMOKE_COREML_COMPUTE_UNITS={coreml_compute_units}"
        )
    return command


def ios_syslog_command(*, device_id: str) -> list[str] | None:
    if shutil.which("idevicesyslog") is None:
        return None
    return ["idevicesyslog", "-u", device_id]


def capture_runtime_smoke_ios(
    *,
    flutter_process: subprocess.Popen[str],
    syslog_process: subprocess.Popen[str],
    timeout_seconds: int,
    parser: MarkerParser,
    post_flutter_exit_grace_seconds: float = 15.0,
) -> tuple[dict[str, Any] | None, bool, list[str], list[str]]:
    flutter_lines: list[str] = []
    syslog_lines: list[str] = []
    marker_payload: dict[str, Any] | None = None
    flutter_parser = parser
    syslog_parser = MarkerParser()
    timed_out = False
    deadline = time.time() + timeout_seconds
    flutter_exited_at: float | None = None
    selector = selectors.DefaultSelector()
    if flutter_process.stdout is not None:
        selector.register(
            flutter_process.stdout,
            selectors.EVENT_READ,
            ("flutter", flutter_process),
        )
    if syslog_process.stdout is not None:
        selector.register(
            syslog_process.stdout,
            selectors.EVENT_READ,
            ("syslog", syslog_process),
        )
    try:
        while True:
            if flutter_exited_at is None and flutter_process.poll() is not None:
                flutter_exited_at = time.time()
            if (
                flutter_exited_at is not None
                and time.time() - flutter_exited_at >= post_flutter_exit_grace_seconds
            ):
                break
            remaining = deadline - time.time()
            if remaining <= 0:
                timed_out = True
                break
            events = selector.select(timeout=min(1.0, remaining))
            if not events:
                if flutter_process.poll() is not None and syslog_process.poll() is not None:
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
                    syslog_lines.append(line)
                    marker_payload = syslog_parser.feed(line)
                if marker_payload is not None:
                    return marker_payload, timed_out, flutter_lines, syslog_lines
            if not selector.get_map():
                break
    finally:
        selector.close()
    return marker_payload, timed_out, flutter_lines, syslog_lines


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


if __name__ == "__main__":
    main()
