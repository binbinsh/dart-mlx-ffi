from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


DEFAULT_TEXT = "Hello from Qwen ASR. This is a speech test."
DEFAULT_VOICE = "Eddy (英语（美国）)"
DEFAULT_MODEL = "Qwen/Qwen3-ASR-1.7B"


def normalize_text(value: str) -> str:
    lowered = value.lower()
    collapsed = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(collapsed.split())


def synthesize_with_say(*, text: str, voice: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["say", "-v", voice, "-o", str(out_path), text],
        check=True,
    )


def load_model(model_name: str):
    import torch
    from qwen_asr import Qwen3ASRModel

    attempts: list[tuple[str, object]] = []
    if torch.backends.mps.is_available():
        attempts.append(("mps", torch.float16))
    attempts.append(("cpu", torch.float32))

    last_error: Exception | None = None
    for device_map, dtype in attempts:
        try:
            model = Qwen3ASRModel.from_pretrained(
                model_name,
                dtype=dtype,
                device_map=device_map,
                max_inference_batch_size=1,
                max_new_tokens=64,
            )
            return model, device_map, str(dtype).replace("torch.", "")
        except Exception as error:  # pragma: no cover - runtime fallback only
            last_error = error

    assert last_error is not None
    raise last_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--audio")
    parser.add_argument("--audio-out", default="output/speech/qwen3_asr_ref.aiff")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--language", default="English")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.audio:
        audio_path = Path(args.audio)
    else:
        audio_path = Path(args.audio_out)
        synthesize_with_say(text=args.text, voice=args.voice, out_path=audio_path)

    model, device_map, dtype = load_model(args.model)
    results = model.transcribe(
        audio=str(audio_path),
        language=args.language,
    )
    result = results[0]
    transcript = result.text

    report = {
        "model": args.model,
        "device_map": device_map,
        "dtype": dtype,
        "audio": str(audio_path),
        "expected_text": args.text,
        "transcript": transcript,
        "expected_normalized": normalize_text(args.text),
        "transcript_normalized": normalize_text(transcript),
    }
    report["match"] = (
        report["expected_normalized"] == report["transcript_normalized"]
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
