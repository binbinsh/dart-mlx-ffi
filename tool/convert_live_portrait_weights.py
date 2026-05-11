#!/usr/bin/env python3
"""Download + convert Ditto LivePortrait weights for the Dart MLX engine.

**Status: skeleton.** Downloads the 7 ONNX modules from HuggingFace, copies
them into ``<output>/onnx/``, and writes a ``manifest.json`` describing each
module so ``LivePortraitSnapshot.open`` (Dart side) can locate them. Tensor →
safetensors conversion for the modules we plan to MLX-port (appearance,
motion, stitch) is wired but disabled by default since the actual port lives
in ``../lib/src/models/live_portrait/`` and is not yet implemented.

Mirrors the spirit of ``generate_flow_support.py``: small, declarative,
single-purpose, no PyTorch unless ``--convert-mlx`` is passed.

Usage::

    uv run tool/convert_live_portrait_weights.py \\
        --output ../cmdspace-app/.cache/live_portrait \\
        --modules all

Source repos::

    https://huggingface.co/digital-avatar/ditto-talkinghead
    https://github.com/antgroup/ditto-talkinghead

License: Apache-2.0 (Ditto). BlazeFace face detector pulled separately from
``onnx-community/mediapipe-blazeface-onnx`` (Apache-2.0).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

# ----------------------------------------------------------------------------
# Module manifest. Sizes/SHAs are populated lazily on first download.
# ----------------------------------------------------------------------------

DITTO_REPO = "digital-avatar/ditto-talkinghead"
DITTO_SUBDIR = "ditto_onnx"
# BlazeFace ships in-repo as ditto_onnx/blaze_face.onnx — no need for the
# external mediapipe-blazeface mirror.

SCHEMA_VERSION = "cmdspace.live_portrait.snapshot.v1"


@dataclass(frozen=True)
class OnnxModule:
    """One ONNX module shipped by Ditto."""

    key: str
    """Stable key used by Dart's ``LivePortraitModule`` enum."""

    remote: str
    """Path within the Ditto HuggingFace repo (relative to repo root),
    OR a fully-qualified ``https://`` URL when ``direct_url`` is True."""

    local: str
    """Filename under ``<output>/onnx/``."""

    description: str

    convert_to_mlx: bool = False
    """Whether this module is a candidate for safetensors+MLX conversion.

    True for modules small enough and op-compatible with MLX today
    (appearance, motion, stitch). The big ones (warp, decoder) stay as ONNX
    until 3D grid_sample + spatially-adaptive normalization Metal kernels
    exist — see ``docs/live_portrait_integration.md`` Phase 4.
    """

    direct_url: bool = False
    """When True, ``remote`` is a fully-qualified URL fetched via ``httpx``
    instead of HuggingFace. Used for modules hosted outside HF (e.g.
    YuNet on the OpenCV Zoo GitHub LFS)."""


MODULES: tuple[OnnxModule, ...] = (
    OnnxModule(
        key="appearance",
        remote=f"{DITTO_SUBDIR}/appearance_extractor.onnx",
        local="appearance_extractor.onnx",
        description="3D appearance feature extractor (~3.3 MB).",
        convert_to_mlx=True,
    ),
    OnnxModule(
        key="motion",
        remote=f"{DITTO_SUBDIR}/motion_extractor.onnx",
        local="motion_extractor.onnx",
        description="Canonical kp + R + t + δ + scale extractor (~108 MB).",
        convert_to_mlx=True,
    ),
    OnnxModule(
        key="warp",
        remote=f"{DITTO_SUBDIR}/warp_network.onnx",
        local="warp_network.onnx",
        description=(
            "Warping module (~174 MB). Stays as ONNX in v1: needs custom "
            "Metal 3D grid_sample kernel before MLX port is viable."
        ),
        convert_to_mlx=False,
    ),
    OnnxModule(
        key="decoder",
        remote=f"{DITTO_SUBDIR}/decoder.onnx",
        local="decoder.onnx",
        description=(
            "SPADE generator (~212 MB). Stays as ONNX in v1: spatially-"
            "adaptive normalization needs hand-fused Metal pass."
        ),
        convert_to_mlx=False,
    ),
    OnnxModule(
        key="stitch",
        remote=f"{DITTO_SUBDIR}/stitch_network.onnx",
        local="stitch_network.onnx",
        description="Stitching MLP (~2.3 MB). Tiny; safe to MLX-port.",
        convert_to_mlx=True,
    ),
    OnnxModule(
        key="hubert",
        remote=f"{DITTO_SUBDIR}/hubert.onnx",
        local="hubert.onnx",
        description=(
            "HuBERT audio encoder. Stays as ONNX initially; CoreML EP gives "
            "good Apple Silicon throughput. Native Dart/MLX port is a "
            "Phase 5 deepening, not a v1 requirement."
        ),
        convert_to_mlx=False,
    ),
    OnnxModule(
        key="lmdm",
        remote=f"{DITTO_SUBDIR}/lmdm_v0.4_hubert.onnx",
        local="lmdm.onnx",
        description=(
            "Latent Motion Diffusion Model. ONNX in v1; streaming sampler "
            "wraps it (see audio_motion.dart). Default 50 DDIM steps; cut "
            "to ~10 for realtime."
        ),
        convert_to_mlx=False,
    ),
    OnnxModule(
        key="face_detector",
        remote=(
            "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
            "main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ),
        local="yunet.onnx",
        description=(
            "YuNet face detector (OpenCV Zoo, Apache-2.0). Replaces "
            "BlazeFace short-range -- handles the small-face-in-large-frame "
            "regime our full-body buddy portraits hit. 232 KB; 5 keypoints "
            "(eyes/nose/mouth corners) ready for LivePortrait alignment."
        ),
        convert_to_mlx=False,
        direct_url=True,
    ),
)

# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "Output snapshot directory. Will contain onnx/, mlx/ (if "
            "--convert-mlx), and manifest.json."
        ),
    )
    parser.add_argument(
        "--modules",
        nargs="+",
        default=["all"],
        help=(
            "Module keys to fetch. Use 'all' (default), or any subset of: "
            + ", ".join(m.key for m in MODULES)
        ),
    )
    parser.add_argument(
        "--convert-mlx",
        action="store_true",
        help=(
            "Also convert MLX-eligible modules to safetensors. Requires "
            "torch + onnx + onnx2pytorch + mlx (uv add as dev deps). "
            "Currently a no-op stub — Phase 2 work."
        ),
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help=(
            "HuggingFace token. Falls back to env HF_TOKEN. Ditto repo is "
            "public so this is optional."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if local file exists with matching size.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be fetched, don't download.",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------------
# Download
# ----------------------------------------------------------------------------


def _selected_modules(keys: list[str]) -> list[OnnxModule]:
    if keys == ["all"]:
        return list(MODULES)
    available = {m.key: m for m in MODULES}
    bad = [k for k in keys if k not in available]
    if bad:
        raise SystemExit(
            f"unknown module(s) {bad!r}; valid keys: {sorted(available)}"
        )
    return [available[k] for k in keys]


def _hf_download(repo: str, remote: str, dest: Path, token: str | None) -> Path:
    """Pull a single file via huggingface_hub. Caches per-file."""
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required: `uv add huggingface_hub`"
        ) from exc

    cached = hf_hub_download(
        repo_id=repo,
        filename=remote,
        token=token,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    # Hard-copy out of HF cache so the snapshot dir is self-contained
    # and survives `huggingface-cli scan-cache --delete`.
    shutil.copy2(cached, dest)
    return dest


def _direct_download(url: str, dest: Path) -> Path:
    """Pull a single file from a fully-qualified URL via stdlib urllib.

    Used for modules hosted outside HuggingFace (e.g. the OpenCV Zoo
    on GitHub LFS). No auth.
    """
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "cmdspace-live-portrait-converter/1"},
    )
    with urllib.request.urlopen(req) as resp:
        if resp.status != 200:
            raise SystemExit(
                f"_direct_download {url}: HTTP {resp.status}"
            )
        with dest.open("wb") as f:
            shutil.copyfileobj(resp, f)
    return dest


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------
# Manifest
# ----------------------------------------------------------------------------


def _write_manifest(snapshot_dir: Path, fetched: list[tuple[OnnxModule, Path]]) -> None:
    by_key = {m.key: (m, p) for (m, p) in fetched}

    def _rel(key: str) -> str:
        if key not in by_key:
            # Allow partial snapshots (e.g. user fetched only some modules).
            # Dart side's LivePortraitWeightPaths requires all 7 entries; we
            # fill the path string anyway so manifest stays parseable, and
            # LivePortraitSnapshot.pathFor will raise on actual access.
            return f"onnx/__missing__/{key}.onnx"
        _, path = by_key[key]
        return f"{path.parent.name}/{path.name}"

    modules_meta = []
    for module, path in fetched:
        modules_meta.append(
            {
                "key": module.key,
                "filename": path.name,
                "subdir": path.parent.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "description": module.description,
                "convert_to_mlx": module.convert_to_mlx,
                "format": "onnx",
            }
        )

    manifest = {
        # Schema kind matches `kLivePortraitSchemaVersion` on the Dart side
        # (../lib/src/models/live_portrait/config.dart).
        "kind": SCHEMA_VERSION,
        "version": 1,
        "source": "ditto-talkinghead",
        "source_repo": DITTO_REPO,
        "weights": {
            "appearance": _rel("appearance"),
            "motion": _rel("motion"),
            "warp": _rel("warp"),
            "decoder": _rel("decoder"),
            "stitch": _rel("stitch"),
            "hubert": _rel("hubert"),
            "lmdm": _rel("lmdm"),
            # face_detector is a Phase-1 helper, not in the core 7. The Dart
            # FaceCropService picks it up by convention from <snapshot>/onnx/
            # blaze_face.onnx.
            "face_detector": _rel("face_detector"),
        },
        "audio": {
            "sampleRate": 16000,
            "hopFrames": 320,
            "featureDim": 768,
        },
        "render": {
            "frameWidth": 512,
            "frameHeight": 512,
            "internalRes": 256,
            "fpsTarget": 25,
        },
        "motion": {
            "keypointCount": 21,
            "appearanceVolume": [1, 32, 16, 64, 64],
        },
        "sampler": {
            "kind": "ddim",
            "steps": 10,
            "guidance": 1.5,
            "windowFrames": 20,
        },
        "modules": modules_meta,
    }
    (snapshot_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


# ----------------------------------------------------------------------------
# MLX conversion (stub)
# ----------------------------------------------------------------------------


def _convert_mlx(modules: list[tuple[OnnxModule, Path]], snapshot_dir: Path) -> None:
    """Convert MLX-eligible ONNX modules to safetensors.

    Currently raises NotImplementedError. Implementation plan:

      1. Use ``onnx`` to walk the graph and extract initializers as
         ``np.ndarray`` tables.
      2. Pattern-match Ditto's known module shapes (we own the schema; this
         isn't generic onnx2mlx).
      3. Emit one ``.safetensors`` per module under ``<snapshot>/mlx/``.
      4. Update manifest.json entries' ``format`` to ``"safetensors"``.

    See ``docs/live_portrait_integration.md`` Phase 2.
    """
    raise NotImplementedError(
        "MLX conversion is Phase 2 work — see "
        "docs/live_portrait_integration.md."
    )


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> None:
    import os

    args = parse_args()
    selected = _selected_modules(args.modules)
    snapshot = args.output.expanduser().resolve()
    onnx_dir = snapshot / "onnx"
    token = args.hf_token or os.environ.get("HF_TOKEN")

    print(f"snapshot dir: {snapshot}")
    print(f"modules     : {[m.key for m in selected]}")
    if args.dry_run:
        for m in selected:
            print(f"  would fetch {m.key:14s} <- {m.remote}")
        return

    snapshot.mkdir(parents=True, exist_ok=True)
    onnx_dir.mkdir(parents=True, exist_ok=True)

    fetched: list[tuple[OnnxModule, Path]] = []
    for module in selected:
        dest = onnx_dir / module.local
        if dest.exists() and not args.force:
            print(f"[{module.key}] {dest} (cached, --force to refetch)")
            fetched.append((module, dest))
            continue
        if module.direct_url:
            print(f"[{module.key}] {module.remote} -> {dest}")
            path = _direct_download(module.remote, dest)
        else:
            print(f"[{module.key}] {DITTO_REPO}/{module.remote} -> {dest}")
            path = _hf_download(DITTO_REPO, module.remote, dest, token)
        fetched.append((module, path))

    _write_manifest(snapshot, fetched)
    print(f"wrote {snapshot / 'manifest.json'}")

    if args.convert_mlx:
        eligible = [(m, p) for (m, p) in fetched if m.convert_to_mlx]
        if not eligible:
            print("no MLX-eligible modules in selection; skipping conversion")
            return
        _convert_mlx(eligible, snapshot)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
