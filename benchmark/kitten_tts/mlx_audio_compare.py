from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

try:
    from .mlx_audio_models import TTS_TEXT, pick_voice, _configure_espeak
    from ..common import (
        add_vendor_to_path,
        cleanup_mlx,
        parse_last_json,
        preview,
        resolve_model_path,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mlx_audio_models import TTS_TEXT, pick_voice, _configure_espeak
    from common import (
        add_vendor_to_path,
        cleanup_mlx,
        parse_last_json,
        preview,
        resolve_model_path,
    )

add_vendor_to_path("mlx-audio")

import mlx.core as mx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default=TTS_TEXT)
    parser.add_argument("--voice", default="expr-voice-5-m")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iters", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _configure_espeak()
    from mlx_audio.tts.utils import get_model_path, load

    snapshot_path = resolve_model_path(
        "mlx-community/kitten-tts-nano-0.8-6bit",
        get_model_path,
    )
    model = load(str(snapshot_path), lazy=False)
    selected_voice = pick_voice(model, requested=args.voice)
    input_ids, ref_s, _ = model._prepare_inputs(args.text, selected_voice, 1.0, True)
    input_ids_json = json.dumps(input_ids.tolist())
    ref_s_json = json.dumps(ref_s.tolist())

    def python_forward() -> tuple[mx.array, mx.array]:
        mx.random.seed(0)
        output = model(input_ids, ref_s, return_output=True)
        mx.eval(output.audio, output.pred_dur)
        mx.synchronize()
        return output.audio, output.pred_dur

    try:
        for _ in range(args.warmup):
            audio, pred_dur = python_forward()
            del audio, pred_dur

        started = time.perf_counter()
        py_audio = None
        py_dur = None
        for _ in range(args.iters):
            py_audio, py_dur = python_forward()
        assert py_audio is not None and py_dur is not None
        py_ms = (time.perf_counter() - started) * 1000.0 / args.iters

        env = dict(os.environ)
        env["KITTEN_WARMUP"] = str(args.warmup)
        env["KITTEN_ITERS"] = str(args.iters)
        temp_dir = Path(tempfile.mkdtemp())
        values_path = temp_dir / "kitten_dart_output.safetensors"
        env["KITTEN_VALUES_PATH"] = str(values_path)
        stdout_path = temp_dir / "kitten_runner.stdout"
        stderr_path = temp_dir / "kitten_runner.stderr"
        with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_file:
            subprocess.run(
                [
                    "dart",
                    "run",
                    "benchmark/kitten_tts/kitten_run.dart",
                    str(snapshot_path),
                    input_ids_json,
                    ref_s_json,
                ],
                cwd=Path(__file__).resolve().parents[2],
                env=env,
                check=True,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
            )
        payload = parse_last_json(stdout_path.read_text(encoding="utf-8"))
        dart_tensors = mx.load(str(values_path))
        dart_values = [float(v) for v in dart_tensors["audio"].reshape([-1]).tolist()]
        py_values = [float(v) for v in py_audio.reshape([-1]).tolist()]
        if len(py_values) != len(dart_values):
            raise RuntimeError(
                f"Audio length mismatch: python={len(py_values)} dart={len(dart_values)}"
            )
        diffs = [abs(a - b) for a, b in zip(py_values, dart_values)]
        report = {
            "snapshot_path": str(snapshot_path),
            "text": args.text,
            "voice": selected_voice,
            "input_shape": list(input_ids.shape),
            "ref_shape": list(ref_s.shape),
            "python_ms": py_ms,
            "dart_ms": float(payload["per_iter_ms"]),
            "python_shape": list(py_audio.shape),
            "dart_shape": payload["shape"],
            "pred_dur_shape": payload["pred_dur_shape"],
            "pred_dur_preview": preview([int(v) for v in py_dur[:16].tolist()]),
            "output_preview": preview(py_values, limit=16),
            "max_abs_diff": max(diffs) if diffs else 0.0,
            "mean_abs_diff": (sum(diffs) / len(diffs)) if diffs else 0.0,
        }
        print(json.dumps(report, indent=2))
    finally:
        del model
        cleanup_mlx(mx)


if __name__ == "__main__":
    main()
