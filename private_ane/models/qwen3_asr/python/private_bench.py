from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "private_ane" / "models" / "qwen3_asr" / "export"))
sys.path.insert(0, str(ROOT / "private_ane" / "shared" / "benchmark"))

from ane_private_mlprogram_bench import bench_one, build_helper
from make_private_blocks import (
    API_URL,
    CONFIG_URL,
    build_specs,
    emit_block,
    fetch_json_with_fallback,
)


LOCAL_MODEL_DIR = ROOT / "tmp" / "Qwen3-ASR-1.7B"
LOCAL_CONFIG = LOCAL_MODEL_DIR / "config.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    work_dir = Path(tempfile.mkdtemp(prefix="qwen3_asr_private_"))
    try:
        helper = build_helper(work_dir)
        artifacts_dir = work_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        config = fetch_json_with_fallback(
            CONFIG_URL,
            fallback=str(LOCAL_CONFIG) if LOCAL_CONFIG.exists() else None,
        )
        api = fetch_json_with_fallback(
            API_URL,
            default={"sha": None},
        )
        specs = build_specs(config, api.get("sha"))
        report: dict[str, object] = {
            "runtime": "qwen3_asr_private_blocks",
            "model_sha": api.get("sha"),
            "models": [],
        }
        for spec in specs:
            artifact = emit_block(artifacts_dir, spec)
            result = bench_one(
                helper,
                Path(artifact["dir"]),
                channels=int(spec["dim"]),
                spatial=int(spec["lane"]),
                iters=args.iters,
            )
            result.update(
                {
                    "name": spec["name"],
                    "block_type": spec["block_type"],
                    "dim": spec["dim"],
                    "hidden": spec["hidden"],
                    "lane": spec["lane"],
                    "max_ms": spec["max_ms"],
                }
            )
            report["models"].append(result)
            if not result.get("ok"):
                raise SystemExit(
                    f"{spec['name']} failed at stage={result.get('stage')} error={result.get('error', '')}"
                )
            if float(result["per_iter_ms"]) > float(spec["max_ms"]):
                raise SystemExit(
                    f"{spec['name']} exceeded max ms/iter: {result['per_iter_ms']} > {spec['max_ms']}"
                )

        if args.json:
            print(json.dumps(report))
            return

        print("qwen3-asr private ane benchmark")
        for model in report["models"]:
            print(
                f"{model['name']}: {model['per_iter_ms']:.4f} ms/iter "
                f"(limit {model['max_ms']:.4f})"
            )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
