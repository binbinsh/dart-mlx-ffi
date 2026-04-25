from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="List benchmark target devices.")
    parser.add_argument(
        "--platform",
        choices=["android", "ios", "all"],
        default="all",
    )
    parser.add_argument("--out")
    args = parser.parse_args()

    devices: list[dict[str, Any]] = []
    if args.platform in {"android", "all"}:
        devices.extend(android_devices())
    if args.platform in {"ios", "all"}:
        devices.extend(apple_devices(args.platform))

    payload = {"devices": devices}
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as file:
            file.write(text + "\n")
    print(text)


def android_devices() -> list[dict[str, Any]]:
    adb = shutil.which("adb")
    if adb is None:
        return [
            {
                "platform": "android",
                "available": False,
                "state": "adb_missing",
            }
        ]
    result = subprocess.run(
        [adb, "devices", "-l"],
        check=False,
        capture_output=True,
        text=True,
    )
    devices = []
    for line in result.stdout.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        serial = parts[0]
        state = parts[1] if len(parts) > 1 else "unknown"
        fields = _key_values(parts[2:])
        devices.append(
            {
                "platform": "android",
                "id": serial,
                "state": state,
                "available": state == "device",
                "model": fields.get("model"),
                "product": fields.get("product"),
                "device": fields.get("device"),
                "transport_id": fields.get("transport_id"),
            }
        )
    return devices


def apple_devices(platform_filter: str) -> list[dict[str, Any]]:
    flutter = flutter_ios_devices(platform_filter)
    if flutter:
        return flutter
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        return [
            {
                "platform": "ios",
                "available": False,
                "state": "xcrun_missing",
            }
        ]
    result = subprocess.run(
        [xcrun, "xctrace", "list", "devices"],
        check=False,
        capture_output=True,
        text=True,
    )
    devices = []
    section = ""
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("== ") and line.endswith(" =="):
            section = line.strip("= ").lower()
            continue
        if section not in {"devices", "devices offline"}:
            continue
        parsed = _parse_apple_device(line)
        if parsed is None:
            continue
        name, os_version, identifier = parsed
        platform = _apple_platform(name)
        if platform_filter != "all" and platform != platform_filter:
            continue
        devices.append(
            {
                "platform": platform,
                "id": identifier,
                "name": name,
                "os_version": os_version,
                "state": "offline" if section == "devices offline" else "device",
                "available": section == "devices",
            }
        )
    return devices


def flutter_ios_devices(platform_filter: str) -> list[dict[str, Any]]:
    flutter = shutil.which("flutter")
    if flutter is None:
        return []
    result = subprocess.run(
        [flutter, "devices", "--machine"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    try:
        decoded = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    if not isinstance(decoded, list):
        return []
    devices: list[dict[str, Any]] = []
    for item in decoded:
        if not isinstance(item, dict):
            continue
        target_platform = str(item.get("targetPlatform") or "")
        if target_platform not in {"ios", "darwin"}:
            continue
        name = str(item.get("name") or item.get("id") or "")
        platform = _apple_platform(name)
        if platform_filter != "all" and platform != platform_filter:
            continue
        device_id = item.get("id")
        if not isinstance(device_id, str) or not device_id:
            continue
        devices.append(
            {
                "platform": platform,
                "id": device_id,
                "name": name,
                "os_version": item.get("sdk"),
                "state": "device",
                "available": bool(item.get("isSupported", True)),
                "via": "flutter",
            }
        )
    return devices


def _apple_platform(name: str) -> str:
    lower = name.lower()
    if "mac" in lower:
        return "macos"
    if "watch" in lower:
        return "watchos"
    return "ios"


def _parse_apple_device(line: str) -> tuple[str, str | None, str] | None:
    match = re.match(r"^(?P<name>.+?) \((?P<version>[^()]+)\) \((?P<id>[^()]+)\)$", line)
    if match:
        return match.group("name"), match.group("version"), match.group("id")
    match = re.match(r"^(?P<name>.+?) \((?P<id>[^()]+)\)$", line)
    if match:
        return match.group("name"), None, match.group("id")
    return None


def _key_values(parts: list[str]) -> dict[str, str]:
    values = {}
    for part in parts:
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        values[key] = value
    return values


if __name__ == "__main__":
    main()
