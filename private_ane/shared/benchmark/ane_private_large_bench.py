from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from ane_private_make_large_models import LARGE_MODEL_SPECS, make_deep_identity_model
from ane_private_mlprogram_bench import bench_one, build_helper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    work_dir = Path(tempfile.mkdtemp(prefix="ane_private_large_"))
    try:
        helper = build_helper(work_dir)
        report: dict[str, object] = {"runtime": "private_ane_large_mlprogram", "models": []}
        for spec in LARGE_MODEL_SPECS:
            compiled = make_deep_identity_model(
                work_dir,
                channels=spec["channels"],
                spatial=spec["spatial"],
                depth=spec["depth"],
            )
            result = bench_one(
                helper,
                compiled,
                channels=spec["channels"],
                spatial=spec["spatial"],
                iters=args.iters,
            )
            result.update(
                {
                    "name": spec["name"],
                    "depth": spec["depth"],
                    "channels": spec["channels"],
                    "spatial": spec["spatial"],
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

        print("private ane large benchmark")
        for model in report["models"]:
            print(
                f"{model['name']}: {model['per_iter_ms']:.4f} ms/iter "
                f"(limit {model['max_ms']:.4f})"
            )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
