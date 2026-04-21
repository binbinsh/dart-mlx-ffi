from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

try:
    from .common import compare_lists, parse_last_json
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import compare_lists, parse_last_json

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


def vad_audio_bench(*, warmup: int = 3, iters: int = 10, max_samples: int = 160000) -> dict[str, object]:
    bundle_path = Path.home() / ".cmdspace" / "models" / "fsmn-vad" / "default"
    audio_path = _ensure_audio()
    pcm_path = _ensure_pcm(audio_path)

    py_raw = subprocess.check_output(
        [
            "uv",
            "run",
            "--no-project",
            "--with",
            "torch",
            "--with",
            "safetensors",
            "--with",
            "numpy",
            "python",
            "benchmark/fsmn_vad/python_audio_ref.py",
            f"--bundle={bundle_path}",
            f"--pcm={pcm_path}",
            f"--warmup={warmup}",
            f"--iters={iters}",
            f"--max-samples={max_samples}",
        ],
        cwd=ROOT,
        text=True,
    )
    py_payload = parse_last_json(py_raw)

    dart_raw = subprocess.check_output(
        [
            "dart",
            "run",
            "benchmark/fsmn_vad/audio_bench.dart",
            f"--bundle={bundle_path}",
            f"--pcm={pcm_path}",
            f"--warmup={warmup}",
            f"--iters={iters}",
            f"--max-samples={max_samples}",
        ],
        cwd=ROOT,
        text=True,
    )
    dart_payload = parse_last_json(dart_raw)

    max_diff, mean_diff = compare_lists(
        [float(v) for v in py_payload["speech_preview"]],
        [float(v) for v in dart_payload["speech_preview"]],
    )
    return {
        "model_id": "funasr/fsmn-vad",
        "kind": "vad",
        "python_backend": "pytorch",
        "dart_backend": "dart_mlx_ffi",
        "input_desc": f"real audio {float(py_payload['samples']) / 16000.0:.2f}s from 000000.flac",
        "comparison": "speech_probs[:16] after audio frontend",
        "python_ms": float(py_payload["python_ms"]),
        "dart_ms": float(dart_payload["per_iter_ms"]),
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
    }


if __name__ == "__main__":
    print(json.dumps(vad_audio_bench(), indent=2))
