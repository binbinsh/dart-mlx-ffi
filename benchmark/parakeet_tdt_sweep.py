from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

try:
    from .common import compare_lists, parse_last_json, slug
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import compare_lists, parse_last_json, slug

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_URL = (
    "https://docs-assets.developer.apple.com/ml-research/datasets/"
    "spatial-librispeech/v1/ambisonics/000000.flac"
)
SAMPLE_ROOT = Path("/tmp/cmdspace-audio")


def _ensure_audio() -> Path:
    SAMPLE_ROOT.mkdir(parents=True, exist_ok=True)
    audio_path = SAMPLE_ROOT / "000000.flac"
    if not audio_path.exists():
        urllib.request.urlretrieve(SAMPLE_URL, audio_path)
    return audio_path


def _ensure_pcm(audio_path: Path) -> Path:
    pcm_path = SAMPLE_ROOT / "000000_pcm.f32"
    if pcm_path.exists():
        return pcm_path
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-i",
            str(audio_path),
            "-threads",
            "0",
            "-f",
            "f32le",
            "-ac",
            "1",
            "-acodec",
            "pcm_f32le",
            "-ar",
            "16000",
            str(pcm_path),
        ],
        check=True,
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return pcm_path


def _ensure_mel(bundle_path: Path) -> tuple[Path, Path]:
    mel_path = SAMPLE_ROOT / "000000_mel.f32"
    meta_path = SAMPLE_ROOT / "000000_mel.json"
    if mel_path.exists() and meta_path.exists():
        return mel_path, meta_path
    audio_path = _ensure_audio()
    pcm_path = _ensure_pcm(audio_path)
    subprocess.run(
        [
            "dart",
            "run",
            "benchmark/parakeet_tdt/mel_dump.dart",
            f"--samples={pcm_path}",
            f"--out-data={mel_path}",
            f"--out-meta={meta_path}",
            f"--bundle={bundle_path}",
        ],
        cwd=ROOT,
        check=True,
        text=True,
    )
    return mel_path, meta_path


def _load_mel(meta_path: Path, mel_path: Path):
    shape = tuple(json.loads(meta_path.read_text())["shape"])
    return shape


def asr_bench(model_id: str, *, warmup: int = 3, iters: int = 10) -> dict[str, object]:
    bundle_path = Path.home() / ".cmdspace" / "models" / "parakeet-tdt" / "default"
    mel_path, meta_path = _ensure_mel(bundle_path)
    shape = _load_mel(meta_path, mel_path)

    py_raw = subprocess.check_output(
        [
            "uv",
            "run",
            "--no-project",
            "--with",
            "parakeet-mlx",
            "--with",
            "numpy",
            "python",
            "benchmark/parakeet_tdt/python_ref.py",
            f"--model-id={model_id}",
            f"--mel={mel_path}",
            f"--meta={meta_path}",
            f"--warmup={warmup}",
            f"--iters={iters}",
        ],
        cwd=ROOT,
        text=True,
    )
    py_payload = parse_last_json(py_raw)
    py_token = [float(v) for v in py_payload["token_preview"]]
    py_duration = [float(v) for v in py_payload["duration_logits"]]

    raw = subprocess.check_output(
        [
            "dart",
            "run",
            "benchmark/parakeet_tdt/model_bench.dart",
            f"--mel={mel_path}",
            f"--meta={meta_path}",
            f"--bundle={bundle_path}",
            f"--warmup={warmup}",
            f"--iters={iters}",
        ],
        cwd=ROOT,
        text=True,
    )
    payload = parse_last_json(raw)
    dart_token = [float(v) for v in payload["token_preview"]]
    dart_duration = [float(v) for v in payload["duration_logits"]]
    max_diff, mean_diff = compare_lists(
        py_token + py_duration,
        dart_token + dart_duration,
    )
    return {
        "model_id": model_id,
        "kind": "asr",
        "input_desc": f"fixed mel {shape[1]}x{shape[2]} from 000000.flac",
        "comparison": "first-step token_logits[:16] + duration_logits",
        "python_ms": float(py_payload["python_ms"]),
        "dart_ms": float(payload["per_iter_ms"]),
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "python_text": py_payload["text"],
        "dart_text": payload["text"],
    }
