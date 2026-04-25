from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from flutter_smoke_common import (
    MarkerParser,
    capture_runtime_smoke,
    extract_marker_payload,
    terminate_process,
)


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = ROOT / "example"


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
        "--artifact",
        help="Optional artifact override passed to Flutter runtime-smoke mode.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=1800)
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
    )
    started = time.time()
    result = {
        "platform": "ios",
        "device_id": device_id,
        "model_id": args.model_id,
        "engine": args.engine,
        "artifact": args.artifact,
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
            timeout_seconds=args.timeout_seconds,
            parser=marker_parser,
        )
    finally:
        terminate_process(process)

    finished = time.time()
    result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(finished))
    result["duration_seconds"] = finished - started
    result["timed_out"] = timed_out
    result["captured_line_count"] = len(lines)
    result["captured_tail"] = lines[-200:]
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
) -> list[str]:
    command = [
        "flutter",
        "run",
        "-d",
        device_id,
        "--dart-define=DMF_RUNTIME_SMOKE=true",
        f"--dart-define=DMF_RUNTIME_SMOKE_MODEL={model_id}",
        f"--dart-define=DMF_RUNTIME_SMOKE_ENGINE={engine}",
    ]
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


if __name__ == "__main__":
    main()
