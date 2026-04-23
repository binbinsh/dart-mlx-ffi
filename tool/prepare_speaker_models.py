#!/usr/bin/env python3
"""Prepare speaker model bundles for vendor parity tests and local runs.

This mirrors the root app's speaker asset preparation but targets
`models/` inside `dart-mlx-ffi` so parity tests can run without checking the
large safetensors bundles into git.
"""

from __future__ import annotations

import argparse
import os
import shutil
import urllib.parse
import urllib.request
from pathlib import Path


_BUNDLES = {
    "ecapa_tdnn": [
        "cmdspace_mlx_ecapa_tdnn.json",
        "weights.safetensors",
    ],
    "pyannote_seg": [
        "cmdspace_mlx_pyannote_seg.json",
        "weights.safetensors",
    ],
}


def _is_url(value: str) -> bool:
    return value.startswith("https://") or value.startswith("http://")


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url) as response, temp.open("wb") as sink:
        shutil.copyfileobj(response, sink)
    temp.replace(destination)


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".part")
    shutil.copyfile(source, temp)
    temp.replace(destination)


def _resolve_source(bundle_name: str, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    env_key = {
        "ecapa_tdnn": "CMDSPACE_ECAPA_BUNDLE_SOURCE",
        "pyannote_seg": "CMDSPACE_PYANNOTE_SEG_BUNDLE_SOURCE",
    }[bundle_name]
    if env_key in os.environ and os.environ[env_key].strip():
        return os.environ[env_key].strip()
    base = os.environ.get("CMDSPACE_SPEAKER_BUNDLE_BASE_URL", "").strip()
    if base:
        return f"{base.rstrip('/')}/{bundle_name}"
    local_fallback = Path.home() / ".cmdspace" / "model_bundles" / "speaker" / bundle_name
    if local_fallback.is_dir():
        return str(local_fallback)
    return None


def _ensure_bundle(bundle_name: str, target_root: Path, source: str | None, force: bool) -> None:
    target_dir = target_root / bundle_name
    target_dir.mkdir(parents=True, exist_ok=True)
    required = _BUNDLES[bundle_name]
    existing = all((target_dir / name).exists() for name in required)
    if existing and not force:
        return
    if not source:
        missing = ", ".join(str(target_dir / name) for name in required if not (target_dir / name).exists())
        raise SystemExit(
            f"Missing speaker bundle files for {bundle_name}: {missing}. "
            "Provide --ecapa-source/--pyannote-source or set CMDSPACE_SPEAKER_BUNDLE_BASE_URL."
        )

    if _is_url(source):
        for name in required:
            _download(f"{source.rstrip('/')}/{urllib.parse.quote(name)}", target_dir / name)
        return

    source_dir = Path(source).expanduser().resolve()
    if not source_dir.is_dir():
        raise SystemExit(f"Speaker bundle source is not a directory or URL: {source}")
    for name in required:
        src = source_dir / name
        if not src.exists():
            raise SystemExit(f"Speaker bundle source is missing {src}")
        _copy(src, target_dir / name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare speaker bundles for dart-mlx-ffi")
    parser.add_argument("--target-root", default="models")
    parser.add_argument("--ecapa-source")
    parser.add_argument("--pyannote-source")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_root = Path(args.target_root).expanduser().resolve()
    ecapa_source = _resolve_source("ecapa_tdnn", args.ecapa_source)
    pyannote_source = _resolve_source("pyannote_seg", args.pyannote_source)
    _ensure_bundle("ecapa_tdnn", target_root, ecapa_source, args.force)
    _ensure_bundle("pyannote_seg", target_root, pyannote_source, args.force)
    print(f"Prepared speaker bundles under {target_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
