from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx

try:
    from .kitten_tts.mlx_audio_models import MODEL_SPECS as AUDIO_MODELS
    from .kitten_tts.mlx_audio_models import benchmark_model as benchmark_audio_model
    from .qwen3_5.mlx_vlm_models import MODEL_SPECS as VLM_MODELS
    from .qwen3_5.mlx_vlm_models import benchmark_model as benchmark_vlm_model
except ImportError:
    from kitten_tts.mlx_audio_models import MODEL_SPECS as AUDIO_MODELS
    from kitten_tts.mlx_audio_models import benchmark_model as benchmark_audio_model
    from qwen3_5.mlx_vlm_models import MODEL_SPECS as VLM_MODELS
    from qwen3_5.mlx_vlm_models import benchmark_model as benchmark_vlm_model

MODELS = [*VLM_MODELS, *AUDIO_MODELS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--voice", default=None)
    parser.add_argument(
        "--output",
        default="benchmark/out/hub_models/report.json",
        help="Output path for the JSON report.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    selected = [
        spec for spec in MODELS if args.model_name is None or spec["name"] == args.model_name
    ]
    if not selected:
        raise SystemExit(f"No model matched --model-name={args.model_name!r}")

    report = {
        "runtime": "python_vendor_refs",
        "mlx_version": getattr(mx, "__version__", "unknown"),
        "device": str(mx.default_device()),
        "metal": bool(mx.metal.is_available()),
        "warmup": args.warmup,
        "iters": args.iters,
        "seq_len": args.seq_len,
        "models": [
            benchmark_model(
                spec,
                warmup=args.warmup,
                iters=args.iters,
                seq_len=args.seq_len,
                voice=args.voice,
            )
            for spec in selected
        ],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report))
        return

    print(f"report_path: {output_path}")
    print(f"mlx_version: {report['mlx_version']}")
    print(f"device: {report['device']}")
    print(f"metal: {report['metal']}")
    print(f"warmup: {report['warmup']}")
    print(f"iters: {report['iters']}")
    for model in report["models"]:
        print()
        print(f"{model['name']}: {model['model_id']}")
        print(f"  backend: {model['backend']}")
        print(f"  output_kind: {model['output_kind']}")
        print(f"  output_shape: {model['output_shape']}")
        print(f"  output_preview: {model['output_preview']}")
        print(f"  per_iter_ms: {model['per_iter_ms']:.4f}")
        print(f"  peak_bytes_delta: {model['peak_bytes_delta']}")
        if "voice" in model:
            print(f"  voice: {model['voice']}")
            print(f"  sample_rate: {model['sample_rate']}")


def benchmark_model(
    spec: dict[str, object],
    *,
    warmup: int,
    iters: int,
    seq_len: int,
    voice: str | None,
) -> dict[str, object]:
    if spec["backend"] == "mlx_vlm":
        return benchmark_vlm_model(
            spec,
            warmup=warmup,
            iters=iters,
            seq_len=seq_len,
        )
    if spec["backend"] == "mlx_audio":
        return benchmark_audio_model(
            spec,
            warmup=warmup,
            iters=iters,
            voice=voice,
        )
    raise ValueError(f"Unsupported backend: {spec['backend']}")


if __name__ == "__main__":
    main()
