from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

try:
    from .mlx_audio_models import MODEL_SPECS, pick_voice, _configure_espeak
    from ..common import add_vendor_to_path, cleanup_mlx, preview, resolve_model_path
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mlx_audio_models import MODEL_SPECS, pick_voice, _configure_espeak
    from common import add_vendor_to_path, cleanup_mlx, preview, resolve_model_path

add_vendor_to_path("mlx-audio")

import mlx.core as mx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice", default="expr-voice-5-m")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    spec = next(
        (item for item in MODEL_SPECS if item["name"] == args.model_name),
        None,
    )
    if spec is None:
        raise SystemExit(f"Unknown --model-name={args.model_name!r}")

    payload = synthesize(
        spec,
        text=args.text,
        voice=args.voice,
        output_path=args.output_path,
    )

    if args.json:
        print(json.dumps(payload))
        return

    print(payload["output_path"])


def synthesize(
    spec: dict[str, object],
    *,
    text: str,
    voice: str,
    output_path: str,
) -> dict[str, object]:
    _configure_espeak()
    from mlx_audio.tts.utils import get_model_path, load

    model_path = resolve_model_path(str(spec["model_id"]), get_model_path)
    model = load(str(model_path), lazy=False)
    selected_voice = pick_voice(model, requested=voice)

    try:
        mx.random.seed(0)
        parts = [
            segment.audio.astype(mx.float32)
            for segment in model.generate(
                text,
                voice=selected_voice,
                clean_text=True,
                chunk_size=400,
            )
        ]
        if not parts:
            raise RuntimeError("KittenTTS produced no audio segments.")
        audio = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)
        mx.eval(audio)
        mx.synchronize()

        samples = int(audio.shape[0])
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        audio_np = np.array(audio.tolist(), dtype=np.float32)
        sf.write(output, audio_np, int(model.sample_rate))

        return {
            "name": spec["name"],
            "model_id": spec["model_id"],
            "snapshot_path": str(model_path.resolve()),
            "text": text,
            "voice": selected_voice,
            "output_path": str(output.resolve()),
            "sample_rate": int(model.sample_rate),
            "samples": samples,
            "duration_seconds": samples / float(model.sample_rate),
            "output_preview": preview(audio_np[:16].tolist()),
            "peak_memory": float(mx.get_peak_memory() / 1e9),
        }
    finally:
        del model
        cleanup_mlx(mx)


if __name__ == "__main__":
    main()
