from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def base_parser(engine: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"{engine} runtime benchmark runner")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--hf-cache-root")
    return parser


def emit_report(
    *,
    model_id: str,
    platform: str,
    engine: str,
    artifact: str,
    correctness: dict[str, Any],
    metrics: dict[str, Any],
    device_profile: dict[str, Any] | None = None,
    out: Path | None = None,
) -> None:
    payload = {
        "model_id": model_id,
        "platform": platform,
        "engine": engine,
        "artifact": artifact,
        "correctness": correctness,
        "metrics": metrics,
        "device_profile": device_profile or {},
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)


def read_report(path: Path) -> dict[str, Any]:
    decoded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError(f"Report must be a JSON object: {path}")
    return decoded


def normalize_report(
    raw: dict[str, Any],
    *,
    model_id: str,
    platform: str,
    engine: str,
    artifact: str,
) -> dict[str, Any]:
    return {
        "model_id": raw.get("model_id") or model_id,
        "platform": raw.get("platform") or platform,
        "engine": raw.get("engine") or engine,
        "artifact": raw.get("artifact") or artifact,
        "correctness": raw.get("correctness") or {},
        "metrics": raw.get("metrics") or {},
        "device_profile": raw.get("device_profile") or {},
    }


def write_report(payload: dict[str, Any], out: Path | None = None) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
