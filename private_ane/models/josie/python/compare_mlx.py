from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "private_ane" / "models" / "josie" / "export"))
sys.path.insert(0, str(ROOT / "private_ane" / "shared" / "benchmark"))

from ane_private_mlprogram_bench import bench_one, build_helper
from make_private_blocks import (
    API_URL,
    CONFIG_URL,
    LOCAL_CONFIG,
    build_specs,
    compute_expected,
    emit_block,
    fetch_json_with_fallback,
    make_input,
    make_weights,
)


def bench_mlx_block(spec: dict[str, object], *, warmup: int, iters: int) -> dict[str, object]:
    sample = make_input(dim=int(spec["dim"]), lane=int(spec["lane"]))
    expected = None
    x = mx.array(sample, dtype=mx.float32)
    w1_np, w3_np, w2_np = make_weights(spec)
    expected = compute_expected(sample, w1=w1_np, w3=w3_np, w2=w2_np)
    w1 = mx.array(w1_np[:, :, 0, 0], dtype=mx.float32)
    w3 = mx.array(w3_np[:, :, 0, 0], dtype=mx.float32)
    w2 = mx.array(w2_np[:, :, 0, 0], dtype=mx.float32)

    def forward():
        h1 = mx.matmul(w1, x)
        h3 = mx.matmul(w3, x)
        gate = (h1 * mx.sigmoid(h1)) * h3
        y = mx.matmul(w2, gate)
        out = x + y
        mx.eval(out)
        mx.synchronize()
        return out

    for _ in range(warmup):
        forward()
    started = time.perf_counter()
    last = None
    for _ in range(iters):
        last = forward()
    per_iter_ms = (time.perf_counter() - started) * 1000.0 / iters
    values = np.array(last, copy=False)
    diffs = np.abs(values - expected)
    return {
        "name": spec["name"],
        "per_iter_ms": per_iter_ms,
        "max_abs_diff_vs_cpu": float(np.max(diffs)) if diffs.size else 0.0,
        "mean_abs_diff_vs_cpu": float(np.mean(diffs)) if diffs.size else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config = fetch_json_with_fallback(
        CONFIG_URL,
        fallback=str(LOCAL_CONFIG) if LOCAL_CONFIG.exists() else None,
    )
    api = fetch_json_with_fallback(API_URL, default={"sha": None})
    specs = build_specs(config, api.get("sha"))

    mlx_models = [
        bench_mlx_block(spec, warmup=args.warmup, iters=args.iters) for spec in specs
    ]

    work_dir = Path(tempfile.mkdtemp(prefix="josie_compare_"))
    try:
        helper = build_helper(work_dir)
        artifacts_dir = work_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        ane_models = []
        for spec in specs:
            artifact = emit_block(artifacts_dir, spec)
            result = bench_one(
                helper,
                Path(artifact["dir"]),
                channels=int(spec["dim"]),
                spatial=int(spec["lane"]),
                iters=args.iters,
            )
            if not result.get("ok"):
                raise SystemExit(
                    f"{spec['name']} failed at stage={result.get('stage')} error={result.get('error', '')}"
                )
            ane_models.append(
                {
                    "name": spec["name"],
                    "per_iter_ms": float(result["per_iter_ms"]),
                }
            )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    mlx_by_name = {model["name"]: model for model in mlx_models}
    ane_by_name = {model["name"]: model for model in ane_models}
    models = []
    for spec in specs:
        name = spec["name"]
        mlx_model = mlx_by_name[name]
        ane_model = ane_by_name[name]
        ane_speedup = (
            mlx_model["per_iter_ms"] / ane_model["per_iter_ms"]
            if ane_model["per_iter_ms"]
            else math.inf
        )
        models.append(
            {
                "name": name,
                "mlx_per_iter_ms": mlx_model["per_iter_ms"],
                "ane_per_iter_ms": ane_model["per_iter_ms"],
                "ane_speedup_vs_mlx": ane_speedup,
                "mlx_max_abs_diff_vs_cpu": mlx_model["max_abs_diff_vs_cpu"],
                "mlx_mean_abs_diff_vs_cpu": mlx_model["mean_abs_diff_vs_cpu"],
            }
        )

    report = {
        "runtime": "josie_private_vs_mlx",
        "mlx_version": mx.__version__,
        "mlx_device": str(mx.default_device()),
        "warmup": args.warmup,
        "iters": args.iters,
        "models": models,
    }

    if args.json:
        print(json.dumps(report))
        return

    print("josie private ane vs mlx")
    print(f"mlx: {report['mlx_version']} | device: {report['mlx_device']}")
    for model in models:
        print(
            f"{model['name']}: "
            f"ANE {model['ane_per_iter_ms']:.4f} ms | "
            f"MLX {model['mlx_per_iter_ms']:.4f} ms | "
            f"ANE vs MLX {model['ane_speedup_vs_mlx']:.3f}x"
        )


if __name__ == "__main__":
    main()
