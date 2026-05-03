from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and run a tiny native LiteRT pipeline smoke test."
    )
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="dmf_litert_smoke_"))
    build_dir = args.build_dir or work_dir / "build"
    try:
        result = run_smoke(build_dir=build_dir, work_dir=work_dir)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        if not args.keep and args.work_dir is None:
            shutil.rmtree(work_dir, ignore_errors=True)


def run_smoke(*, build_dir: Path, work_dir: Path) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = work_dir / "scatter_pipeline.json"
    input_path = work_dir / "scatter_input.json"
    report_path = work_dir / "scatter_report.json"

    pipeline_path.write_text(
        json.dumps(
            {
                "format": "dart_mlx_ffi.litert_pipeline.v1",
                "stages": [
                    {
                        "name": "merge",
                        "op": "scatter_embeddings",
                        "inputs": {
                            "base": "inputs_embeds",
                            "updates": "image_embeds",
                            "indices": "image_token_indices",
                        },
                        "outputs": {"output": "inputs_embeds"},
                    }
                ],
                "outputs": ["inputs_embeds"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    input_path.write_text(
        json.dumps(
            {
                "inputs_embeds": {
                    "dtype": "float32",
                    "shape": [1, 3, 2],
                    "values": [[[1, 1], [2, 2], [3, 3]]],
                },
                "image_embeds": {
                    "dtype": "float32",
                    "shape": [1, 2],
                    "values": [[9, 9]],
                },
                "image_token_indices": {
                    "dtype": "int64",
                    "shape": [1],
                    "values": [1],
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    _run(
        [
            "cmake",
            "-S",
            str(ROOT / "native" / "runtime"),
            "-B",
            str(build_dir),
            "-DDMF_BUILD_CLI=ON",
        ]
    )
    _run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            "dart_mlx_ffi_runtime_runner",
            "-j2",
        ]
    )
    runner = build_dir / "dart_mlx_ffi_runtime_runner"
    _run(
        [
            str(runner),
            "--model-id",
            "tiny_litert_scatter",
            "--platform",
            "android",
            "--engine",
            "litert",
            "--artifact",
            str(pipeline_path),
            "--input-json",
            str(input_path),
            "--warmup",
            "1",
            "--iters",
            "2",
            "--out",
            str(report_path),
        ]
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    values = report["correctness"]["output_values"]["inputs_embeds"]["values"]
    if values != [1.0, 1.0, 9.0, 9.0, 3.0, 3.0]:
        raise RuntimeError(f"Unexpected LiteRT pipeline output: {values}")
    diagnostics = report["device_profile"]["runtime_diagnostics"]
    if diagnostics.get("pipeline") is not True:
        raise RuntimeError(f"LiteRT diagnostics did not report pipeline: {diagnostics}")
    return {
        "passed": True,
        "build_dir": str(build_dir),
        "work_dir": str(work_dir),
        "report": str(report_path),
        "values": values,
        "pipeline_stage_count": diagnostics.get("stage_count"),
    }


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
