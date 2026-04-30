from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ort_env import resolve_ort_environment


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and run a tiny native ONNX Runtime smoke test."
    )
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--no-fetch-headers", action="store_true")
    parser.add_argument("--target-os", choices=["host", "android"], default="host")
    parser.add_argument("--target-arch", default=None)
    parser.add_argument("--android-ndk-home", type=Path)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="dmf_ort_smoke_"))
    build_dir = args.build_dir or work_dir / "build"
    try:
        result = run_smoke(
            build_dir=build_dir,
            work_dir=work_dir,
            fetch_headers=not args.no_fetch_headers,
            target_os=args.target_os,
            target_arch=args.target_arch,
            android_ndk_home=args.android_ndk_home,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        if not args.keep and args.work_dir is None:
            shutil.rmtree(work_dir, ignore_errors=True)


def run_smoke(
    *,
    build_dir: Path,
    work_dir: Path,
    fetch_headers: bool,
    target_os: str = "host",
    target_arch: str | None = None,
    android_ndk_home: Path | None = None,
) -> dict[str, Any]:
    env = resolve_ort_environment(
        fetch_headers=fetch_headers,
        target_os=target_os,
        target_arch=target_arch,
    )
    if not env.ready or env.include_dir is None or env.library is None:
        raise RuntimeError(f"ONNX Runtime C API environment is incomplete: {env}")
    work_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    model_path = work_dir / "add_one.onnx"
    pipeline_path = work_dir / "add_twice_pipeline.json"
    scatter_path = work_dir / "scatter_pipeline.json"
    scatter_input_path = work_dir / "scatter_input.json"
    input_path = work_dir / "input.json"
    report_path = work_dir / "report.json"
    pipeline_report_path = work_dir / "pipeline_report.json"
    scatter_report_path = work_dir / "scatter_report.json"
    _write_add_one_model(model_path)
    pipeline_path.write_text(
        json.dumps(
            {
                "format": "dart_mlx_ffi.onnx_pipeline.v1",
                "stages": [
                    {
                        "name": "first",
                        "model": str(model_path.resolve()),
                        "outputs": {"output": "middle"},
                    },
                    {
                        "name": "second",
                        "model": str(model_path.resolve()),
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
                    "shape": [1, 4],
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
                "format": "dart_mlx_ffi.onnx_pipeline.v1",
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
        _cmake_configure_command(
            env=env,
            build_dir=build_dir,
            target_os=target_os,
            target_arch=target_arch,
            android_ndk_home=android_ndk_home,
        )
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
    if env.runtime_library is not None:
        shutil.copy2(env.runtime_library, build_dir / env.runtime_library.name)

    runner = build_dir / "dart_mlx_ffi_runtime_runner"
    if target_os == "android":
        return {
            "passed": True,
            "target_os": "android",
            "target_arch": target_arch or "arm64-v8a",
            "build_only": True,
            "build_dir": str(build_dir),
            "runner": str(runner),
            "library": str(build_dir / "libdart_mlx_ffi_runtime.so"),
            "ort": env.to_json(),
        }
    _run(
        [
            str(runner),
            "--model-id",
            "tiny_onnx",
            "--platform",
            "macos",
            "--engine",
            "onnx",
            "--artifact",
            str(model_path),
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
    values = report["correctness"]["output_values"]["output"]["values"]
    if values != [2.0, 3.0, 4.0, 5.0]:
        raise RuntimeError(f"Unexpected ONNX smoke output: {values}")
    _run(
        [
            str(runner),
            "--model-id",
            "tiny_onnx_pipeline",
            "--platform",
            "macos",
            "--engine",
            "onnx",
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
    pipeline_values = pipeline_report["correctness"]["output_values"]["output"][
        "values"
    ]
    if pipeline_values != [3.0, 4.0, 5.0, 6.0]:
        raise RuntimeError(f"Unexpected ONNX pipeline output: {pipeline_values}")
    _run(
        [
            str(runner),
            "--model-id",
            "tiny_onnx_scatter",
            "--platform",
            "macos",
            "--engine",
            "onnx",
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
    scatter_values = scatter_report["correctness"]["output_values"]["inputs_embeds"][
        "values"
    ]
    if scatter_values != [1.0, 1.0, 9.0, 9.0, 3.0, 3.0]:
        raise RuntimeError(f"Unexpected ONNX scatter output: {scatter_values}")
    diagnostics = report["device_profile"]["runtime_diagnostics"]
    pipeline_diagnostics = pipeline_report["device_profile"]["runtime_diagnostics"]
    return {
        "passed": True,
        "build_dir": str(build_dir),
        "work_dir": str(work_dir),
        "report": str(report_path),
        "pipeline_report": str(pipeline_report_path),
        "scatter_report": str(scatter_report_path),
        "values": values,
        "pipeline_values": pipeline_values,
        "scatter_values": scatter_values,
        "pipeline_stage_count": pipeline_diagnostics.get("stage_count"),
        "provider": diagnostics.get("provider"),
        "input_names": diagnostics.get("input_names"),
        "output_names": diagnostics.get("output_names"),
        "ort": env.to_json(),
    }


def _cmake_configure_command(
    *,
    env: Any,
    build_dir: Path,
    target_os: str,
    target_arch: str | None,
    android_ndk_home: Path | None,
) -> list[str]:
    command = [
        "cmake",
        "-S",
        str(ROOT / "native" / "runtime"),
        "-B",
        str(build_dir),
        "-DDMF_BUILD_CLI=ON",
        "-DDMF_ENABLE_ORT=ON",
        f"-DDMF_ORT_INCLUDE_DIR={env.include_dir}",
        f"-DDMF_ORT_LIBRARY={env.library}",
    ]
    if target_os == "android":
        toolchain = _android_toolchain_file(android_ndk_home)
        command.extend(
            [
                f"-DCMAKE_TOOLCHAIN_FILE={toolchain}",
                f"-DANDROID_ABI={target_arch or 'arm64-v8a'}",
                "-DANDROID_PLATFORM=android-26",
            ]
        )
    return command


def _android_toolchain_file(android_ndk_home: Path | None) -> Path:
    candidates: list[Path] = []
    if android_ndk_home is not None:
        candidates.append(android_ndk_home)
    for env_name in ("ANDROID_NDK_HOME", "ANDROID_NDK_ROOT"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value))
    for sdk_env in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        sdk = os.environ.get(sdk_env)
        if sdk:
            candidates.extend(sorted((Path(sdk) / "ndk").glob("*"), reverse=True))
    candidates.extend(
        sorted((Path.home() / "Library" / "Android" / "sdk" / "ndk").glob("*"), reverse=True)
    )
    for candidate in candidates:
        toolchain = candidate / "build" / "cmake" / "android.toolchain.cmake"
        if toolchain.exists():
            return toolchain
    raise RuntimeError(
        "Android NDK toolchain not found. Set --android-ndk-home or ANDROID_NDK_HOME."
    )


def _write_add_one_model(path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper

    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    one = helper.make_tensor("one", TensorProto.FLOAT, [1, 4], [1, 1, 1, 1])
    node = helper.make_node("Add", ["input", "one"], ["output"])
    graph = helper.make_graph([node], "add_one", [input_info], [output_info], [one])
    model = helper.make_model(
        graph,
        producer_name="dart_mlx_ffi_smoke",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 10
    onnx.save(model, path)


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
