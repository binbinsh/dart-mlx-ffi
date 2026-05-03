from __future__ import annotations

from pathlib import Path


def _fallback_catalog() -> dict:
    return {
        "support_policy": {
            "production_requires": {"platforms": ["ios", "android"]}
        },
        "engine_platforms": {
            "mlx": ["ios", "macos"],
            "coreml": ["ios", "macos"],
            "onnx": ["ios", "macos", "windows", "linux", "android"],
            "litert": ["android"],
        },
        "engine_order": {
            "ios": ["coreml", "mlx", "onnx"],
            "android": ["litert", "onnx"],
        },
        "models": {
            "paddle_ocr_vl": {
                "family": "PaddleOCR-VL",
                "source_model": "PaddlePaddle/PaddleOCR-VL-1.5",
                "artifacts": {
                    "mlx": {
                        "repo": "mlx-community/PaddleOCR-VL-1.5-8bit",
                        "artifact": ".",
                    },
                    "onnx": {
                        "repo": "lbm364dl/PaddleOCR-VL-1.5-ONNX",
                        "artifact": "onnx/decoder_model_merged.onnx",
                        "component_artifacts": {
                            "embed_tokens": "onnx/embed_tokens.onnx",
                            "vision_encoder": "onnx/vision_encoder.onnx",
                            "decoder": "onnx/decoder_model_merged.onnx",
                        },
                    },
                },
            }
        },
    }


def _write_passing_verdict(
    out_root: Path,
    model_id: str,
    platform: str,
    engine: str,
) -> None:
    path = out_root / model_id / platform / "verdict.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json_dump(
            {
                "verdict": {
                    "passed": True,
                    "identity": {"passed": True, "checks": []},
                    "correctness": {"passed": True, "checks": []},
                    "speed": {
                        "passed": True,
                        "checks": [
                            {"name": "end_to_end_ratio", "passed": True, "value": 1.0}
                        ],
                    },
                    "peak_memory": {"passed": True, "checks": []},
                    "device_profile": {"passed": True, "checks": []},
                },
                "candidate": {
                    "model_id": model_id,
                    "platform": platform,
                    "engine": engine,
                    "run_config": {
                        "format": "dart_mlx_ffi.run_config.v1",
                        "task": "tensor",
                        "warmup": 1,
                        "iters": 5,
                        "max_tokens": 64,
                        "sampling_strategy": "greedy",
                    },
                    "input_signature": {"digest": "same-input"},
                    "metrics": {
                        "peak_memory_bytes": 100,
                        "iteration_count": 5,
                        "warmup_count": 1,
                        "latency_ms": {
                            "values": [10.0, 10.0, 10.0, 10.0, 10.0],
                            "mean": 10.0,
                            "p50": 10.0,
                            "p95": 10.0,
                        },
                    },
                    "device_profile": {},
                },
                "baseline": {
                    "model_id": model_id,
                    "platform": platform,
                    "engine": engine,
                    "run_config": {
                        "format": "dart_mlx_ffi.run_config.v1",
                        "task": "tensor",
                        "warmup": 1,
                        "iters": 5,
                        "max_tokens": 64,
                        "sampling_strategy": "greedy",
                    },
                    "input_signature": {"digest": "same-input"},
                    "metrics": {
                        "peak_memory_bytes": 100,
                        "iteration_count": 5,
                        "warmup_count": 1,
                        "latency_ms": {
                            "values": [10.0, 10.0, 10.0, 10.0, 10.0],
                            "mean": 10.0,
                            "p50": 10.0,
                            "p95": 10.0,
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def json_dump(value: dict) -> str:
    import json

    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"
