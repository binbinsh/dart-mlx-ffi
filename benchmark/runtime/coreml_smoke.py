from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and run a tiny native Core ML smoke test."
    )
    parser.add_argument("--build-dir", type=Path, default=Path("/tmp/dmf_runtime_build"))
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="dmf_coreml_smoke_"))
    try:
        result = run_smoke(build_dir=args.build_dir, work_dir=work_dir)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        if not args.keep and args.work_dir is None:
            shutil.rmtree(work_dir, ignore_errors=True)


def run_smoke(*, build_dir: Path, work_dir: Path) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    model_path = work_dir / "identity.mlmodel"
    add_one_path = work_dir / "add_one.mlmodel"
    compiled_dir = work_dir / "compiled"
    add_compiled_dir = work_dir / "add_compiled"
    pipeline_path = work_dir / "add_twice_pipeline.json"
    scatter_path = work_dir / "scatter_pipeline.json"
    scatter_input_path = work_dir / "scatter_input.json"
    input_path = work_dir / "input.json"
    report_path = work_dir / "report.json"
    health_report_path = work_dir / "health_report.json"
    pipeline_report_path = work_dir / "pipeline_report.json"
    scatter_report_path = work_dir / "scatter_report.json"
    _write_linear_model(model_path, bias=0.0)
    _write_linear_model(add_one_path, bias=1.0)
    _run(["xcrun", "coremlcompiler", "compile", str(model_path), str(compiled_dir)])
    _run(["xcrun", "coremlcompiler", "compile", str(add_one_path), str(add_compiled_dir)])
    compiled_model = compiled_dir / "identity.mlmodelc"
    compiled_add_one = add_compiled_dir / "add_one.mlmodelc"
    pipeline_path.write_text(
        json.dumps(
            {
                "format": "dart_mlx_ffi.coreml_pipeline.v1",
                "stages": [
                    {
                        "name": "first",
                        "model": str(compiled_add_one),
                        "outputs": {"output": "middle"},
                    },
                    {
                        "name": "second",
                        "model": str(compiled_add_one),
                        "inputs": {"input": "middle"},
                        "outputs": {"output": "output"},
                    },
                ],
                "outputs": ["output"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    input_path.write_text(
        json.dumps(
            {
                "input": {
                    "dtype": "float32",
                    "shape": [4],
                    "values": [1, 2, 3, 4],
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    scatter_path.write_text(
        json.dumps(
            {
                "format": "dart_mlx_ffi.coreml_pipeline.v1",
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
    scatter_input_path.write_text(
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
                    "dtype": "int32",
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
            sys.executable,
            str(ROOT / "benchmark" / "runtime" / "artifact_health.py"),
            "--engine",
            "coreml",
            "--platform",
            "macos",
            "--artifact",
            str(compiled_model),
            "--out",
            str(health_report_path),
        ]
    )
    health_report = json.loads(health_report_path.read_text(encoding="utf-8"))
    if not health_report.get("passed"):
        raise RuntimeError(f"Unexpected Core ML health report: {health_report}")
    _run(
        [
            str(runner),
            "--model-id",
            "tiny_coreml",
            "--platform",
            "macos",
            "--engine",
            "coreml",
            "--artifact",
            str(compiled_model),
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
    values = _first_values(report)
    if values != [1.0, 2.0, 3.0, 4.0]:
        raise RuntimeError(f"Unexpected Core ML smoke output: {values}")
    _run(
        [
            str(runner),
            "--model-id",
            "tiny_coreml_pipeline",
            "--platform",
            "macos",
            "--engine",
            "coreml",
            "--artifact",
            str(pipeline_path),
            "--input-json",
            str(input_path),
            "--warmup",
            "1",
            "--iters",
            "2",
            "--out",
            str(pipeline_report_path),
        ]
    )
    pipeline_report = json.loads(pipeline_report_path.read_text(encoding="utf-8"))
    pipeline_values = _first_values(pipeline_report)
    if pipeline_values != [3.0, 4.0, 5.0, 6.0]:
        raise RuntimeError(f"Unexpected Core ML pipeline output: {pipeline_values}")
    _run(
        [
            str(runner),
            "--model-id",
            "tiny_coreml_scatter",
            "--platform",
            "macos",
            "--engine",
            "coreml",
            "--artifact",
            str(scatter_path),
            "--input-json",
            str(scatter_input_path),
            "--warmup",
            "1",
            "--iters",
            "2",
            "--out",
            str(scatter_report_path),
        ]
    )
    scatter_report = json.loads(scatter_report_path.read_text(encoding="utf-8"))
    scatter_values = _first_values(scatter_report)
    if scatter_values != [1.0, 1.0, 9.0, 9.0, 3.0, 3.0]:
        raise RuntimeError(f"Unexpected Core ML scatter output: {scatter_values}")
    diagnostics = report["device_profile"]["runtime_diagnostics"]
    pipeline_diagnostics = pipeline_report["device_profile"]["runtime_diagnostics"]
    return {
        "passed": True,
        "build_dir": str(build_dir),
        "work_dir": str(work_dir),
        "artifact": str(compiled_model),
        "health_report": str(health_report_path),
        "report": str(report_path),
        "pipeline_report": str(pipeline_report_path),
        "scatter_report": str(scatter_report_path),
        "values": values,
        "pipeline_values": pipeline_values,
        "scatter_values": scatter_values,
        "pipeline_loaded_models": pipeline_diagnostics.get("loaded_models"),
        "compute_units": diagnostics.get("compute_units"),
        "input_names": diagnostics.get("input_names"),
        "output_names": diagnostics.get("output_names"),
    }


def _write_linear_model(path: Path, *, bias: float) -> None:
    try:
        import coremltools as ct
        from coremltools.models import datatypes
        from coremltools.models.neural_network import NeuralNetworkBuilder
    except ImportError as error:
        raise RuntimeError(
            "coremltools is required. Run with "
            "`uv run --with coremltools python benchmark/runtime/coreml_smoke.py`."
        ) from error

    input_features = [("input", datatypes.Array(4))]
    output_features = [("output", datatypes.Array(4))]
    builder = NeuralNetworkBuilder(input_features, output_features)
    builder.add_activation(
        name="identity",
        non_linearity="LINEAR",
        input_name="input",
        output_name="output",
        params=[1.0, bias],
    )
    model = ct.models.MLModel(builder.spec)
    model.save(str(path))


def _first_values(report: dict[str, Any]) -> list[float]:
    outputs = report["correctness"]["output_values"]
    first = next(iter(outputs.values()))
    return [float(value) for value in first["values"]]


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
