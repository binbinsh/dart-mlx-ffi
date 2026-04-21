from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

try:
    from .common import compare_lists, parse_last_json
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import compare_lists, parse_last_json

ROOT = Path(__file__).resolve().parents[1]


def vad_bench(*, warmup: int = 3, iters: int = 10, frames: int = 30) -> dict[str, object]:
    bundle_path = Path.home() / ".cmdspace" / "models" / "fsmn-vad" / "default"
    py_raw = subprocess.check_output(
        [
            "uv",
            "run",
            "--no-project",
            "--with",
            "torch",
            "--with",
            "safetensors",
            "python",
            "benchmark/fsmn_vad/python_ref.py",
            f"--bundle={bundle_path}",
            f"--warmup={warmup}",
            f"--iters={iters}",
            f"--frames={frames}",
        ],
        cwd=ROOT,
        text=True,
    )
    py_payload = parse_last_json(py_raw)

    dart_raw = subprocess.check_output(
        [
            "dart",
            "run",
            "benchmark/fsmn_vad/model_bench.dart",
            f"--bundle={bundle_path}",
            f"--warmup={warmup}",
            f"--iters={iters}",
            f"--frames={frames}",
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
        "input_desc": f"fixed features {frames}x400",
        "comparison": "speech_probs[:16]",
        "python_ms": float(py_payload["python_ms"]),
        "dart_ms": float(dart_payload["per_iter_ms"]),
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
    }


if __name__ == "__main__":
    print(json.dumps(vad_bench(), indent=2))
