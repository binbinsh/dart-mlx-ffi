from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
_RUNNERS: dict[str, object] | None = None
_EXTRA_MODELS = [
    {
        "model_id": "mlx-community/Qwen3.5-9B-MLX-4bit",
        "kind": "text",
    },
    {
        "model_id": "mlx-community/Qwen3.5-35B-A3B-4bit",
        "kind": "text",
    },
    {
        "model_id": "mlx-community/kitten-tts-nano-0.8-6bit",
        "kind": "tts",
        "runner": "kitten",
        "name": "kitten_tts_nano_08",
    },
]


def _load_model_specs() -> list[dict[str, object]]:
    specs = json.loads(
        (ROOT / "benchmark" / "publish_model_list.json").read_text(encoding="utf-8"),
    )
    merged = []
    for item in [*specs, *_EXTRA_MODELS]:
        spec = dict(item)
        spec.setdefault("name", str(spec["model_id"]).split("/", 1)[-1])
        merged.append(spec)
    return merged


def _matches_model(spec: dict[str, object], query: str | None) -> bool:
    if query is None:
        return True
    needle = query.lower()
    haystacks = {
        str(spec["model_id"]).lower(),
        str(spec.get("name", "")).lower(),
        str(spec["model_id"]).split("/", 1)[-1].lower(),
    }
    return any(needle == value or needle in value for value in haystacks if value)


def _load_runners() -> dict[str, object]:
    global _RUNNERS
    if _RUNNERS is not None:
        return _RUNNERS

    benchmark_root = str(Path(__file__).resolve().parent)
    try:
        from .parakeet_tdt_sweep import asr_bench
        from .publish_report import kitten_bench, ming_tts_bench, text_bench, vlm_bench
    except ImportError:
        if benchmark_root not in sys.path:
            sys.path.insert(0, benchmark_root)
        from parakeet_tdt_sweep import asr_bench
        from publish_report import kitten_bench, ming_tts_bench, text_bench, vlm_bench

    _RUNNERS = {
        "text": text_bench,
        "vlm": vlm_bench,
        "tts": ming_tts_bench,
        "asr": asr_bench,
        "kitten": kitten_bench,
    }
    return _RUNNERS


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

    if args.seq_len != 48:
        print(
            "warning: --seq-len is ignored by the current export-based benchmark runners.",
            file=sys.stderr,
        )
    if args.voice is not None:
        print(
            "warning: --voice is ignored by the current export-based benchmark runners.",
            file=sys.stderr,
        )

    import mlx.core as mx

    models = _load_model_specs()
    selected = [
        spec for spec in models if _matches_model(spec, args.model_name)
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
        "voice": args.voice,
        "models": [
            benchmark_model(
                spec,
                warmup=args.warmup,
                iters=args.iters,
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
        print(model["model_id"])
        print(f"  kind: {model['kind']}")
        print(f"  input_desc: {model['input_desc']}")
        print(f"  comparison: {model['comparison']}")
        print(f"  python_ms: {model['python_ms']:.4f}")
        print(f"  dart_ms: {model['dart_ms']:.4f}")
        print(f"  max_abs_diff: {model['max_abs_diff']}")
        if "python_text" in model:
            print(f"  python_text: {model['python_text']}")
            print(f"  dart_text: {model['dart_text']}")


def benchmark_model(
    spec: dict[str, object],
    *,
    warmup: int,
    iters: int,
) -> dict[str, object]:
    runners = _load_runners()
    if spec.get("runner") == "kitten":
        return runners["kitten"](warmup=warmup, iters=iters)

    model_id = str(spec["model_id"])
    match spec["kind"]:
        case "text":
            return runners["text"](model_id, warmup=warmup, iters=iters)
        case "vlm":
            return runners["vlm"](model_id, warmup=warmup, iters=iters)
        case "tts":
            return runners["tts"](model_id, warmup=warmup, iters=iters)
        case "asr":
            return runners["asr"](model_id, warmup=warmup, iters=iters)
        case _:
            raise ValueError(f"Unsupported kind: {spec['kind']}")


if __name__ == "__main__":
    main()
