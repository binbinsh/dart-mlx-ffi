from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from hf_download import DEFAULT_FALLBACK_ENDPOINT
from ming_omni_components import (
    extract_llm_hf,
    prepare_source,
    split_component_weights,
)
from ming_omni_native_components import export_native_components
from ming_omni_source import DEFAULT_FALLBACK_REPOS


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export Ming-omni TTS LiteRT with patched dynamic-module sources "
            "when the model repo misses auto_map Python files."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--task", default="text_to_audio")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--official-source", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--skip-llm-litert-export", action="store_true")
    parser.add_argument("--export-small-litert-components", action="store_true")
    parser.add_argument("--export-audio-decoder-litert", action="store_true")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT"))
    parser.add_argument(
        "--fallback-endpoint",
        default=os.environ.get("HF_FALLBACK_ENDPOINT") or DEFAULT_FALLBACK_ENDPOINT,
    )
    parser.add_argument(
        "--fallback-repo",
        action="append",
        default=[],
        help="Fallback HF repo used to source missing dynamic-module files.",
    )
    args, passthrough = parser.parse_known_args()

    output_dir = args.output_dir.expanduser().resolve()
    if _is_text_to_audio_task(args.task):
        _export_text_to_audio_components(
            model=args.model,
            output_dir=output_dir,
            official_source=args.official_source,
            source_dir=args.source_dir,
            endpoint=args.endpoint,
            fallback_endpoint=args.fallback_endpoint,
            fallback_repos=args.fallback_repo or DEFAULT_FALLBACK_REPOS,
            skip_llm_litert_export=args.skip_llm_litert_export,
            export_small_litert_components=args.export_small_litert_components,
            export_audio_decoder_litert=args.export_audio_decoder_litert,
        )
        return

    source_dir, patch_report = prepare_source(
        model=args.model,
        output_dir=output_dir,
        official_source=args.official_source,
        fallback_repos=args.fallback_repo or DEFAULT_FALLBACK_REPOS,
        endpoint=args.endpoint,
        fallback_endpoint=args.fallback_endpoint,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "source_patch_report.json").write_text(
        json.dumps(patch_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    cmd = [
        sys.executable,
        "benchmark/runtime/converters/litert_hf_export.py",
        "--model",
        str(source_dir),
        "--output-dir",
        str(output_dir),
        "--task",
        args.task,
    ]
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    cmd.extend(passthrough)
    completed = subprocess.run(cmd, check=False)
    raise SystemExit(completed.returncode)


def _export_text_to_audio_components(
    *,
    model: str,
    output_dir: Path,
    official_source: Path | None,
    source_dir: Path | None,
    endpoint: str | None,
    fallback_endpoint: str | None,
    fallback_repos: list[str],
    skip_llm_litert_export: bool,
    export_small_litert_components: bool,
    export_audio_decoder_litert: bool,
) -> None:
    if source_dir is None:
        prepared_source_dir, source_report = prepare_source(
            model=model,
            output_dir=output_dir,
            official_source=official_source,
            fallback_repos=fallback_repos,
            endpoint=endpoint,
            fallback_endpoint=fallback_endpoint,
        )
        source_report["fallback_repos"] = fallback_repos
    else:
        from ming_omni_components import copy_official_source

        prepared_source_dir = source_dir.expanduser().resolve()
        copied = copy_official_source(
            official_source=official_source,
            target_dir=prepared_source_dir,
        )
        source_report = {
            "source_model": model,
            "source_dir": str(prepared_source_dir),
            "used_existing_source_dir": True,
            "official_source": None if official_source is None else str(official_source),
            "official_source_copied": copied,
            "fallback_repos": fallback_repos,
        }
    llm_hf_dir = output_dir / "components" / "llm_hf"
    llm_litert_dir = output_dir / "components" / "llm_litert"
    torch_components_dir = output_dir / "components" / "torch"
    torch_components = split_component_weights(
        source_dir=prepared_source_dir,
        output_dir=torch_components_dir,
    )
    onnx_components = export_native_components(
        source_dir=prepared_source_dir,
        components_dir=torch_components_dir,
        output_dir=output_dir / "components" / "onnx",
        components=[
            "flowloss_dit_step",
            "linear_proj_audio",
            "stop_head",
            "audio_decode_chunk",
        ],
    )
    missing_native_components = [
        "audio decoder LiteRT runner",
        "streaming TTS sampler/orchestration",
    ]
    report: dict[str, object] = {
        "format": "dart_mlx_ffi.ming_omni_tts_litert_component_export.v1",
        "source": source_report,
        "components": {
            "llm_hf": extract_llm_hf(
                source_dir=prepared_source_dir,
                output_dir=llm_hf_dir,
            ),
            "torch": torch_components,
            "onnx": onnx_components,
        },
        "full_tts_runtime_ready": False,
        "missing_native_components": missing_native_components,
    }
    if export_small_litert_components:
        report["components"]["litert_small"] = _export_small_litert_components(
            onnx_components=onnx_components,
            output_dir=output_dir / "components" / "litert",
        )
    if export_audio_decoder_litert:
        audio_decoder_result = _export_litert_component(
            onnx_components=onnx_components,
            output_dir=output_dir / "components" / "litert",
            name="audio_decode_chunk",
            timeout_seconds=1200,
            onnx2tf_extra_args=["-kat", "sampled_latent_patch"],
            tflite_preference="float16",
        )
        report["components"]["litert_audio_decoder"] = audio_decoder_result
        if audio_decoder_result.get("returncode") == 0:
            missing_native_components.remove("audio decoder LiteRT runner")
    if not skip_llm_litert_export:
        cmd = [
            sys.executable,
            "benchmark/runtime/converters/litert_hf_export.py",
            "--model",
            str(llm_hf_dir),
            "--output-dir",
            str(llm_litert_dir),
            "--task",
            "text_generation",
            "--prefill-lengths",
            "64",
            "--cache-length",
            "512",
            "--quantization-recipe",
            "dynamic_wi8_afp32",
            "--bundle-litert-lm",
        ]
        completed = subprocess.run(cmd, check=False)
        report["components"]["llm_litert"] = {
            "command": cmd,
            "returncode": completed.returncode,
            "artifact": str(llm_litert_dir / "model.litertlm"),
        }
        _write_component_report(output_dir, report)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
    else:
        _write_component_report(output_dir, report)

    print(
        "Ming Omni TTS LiteRT component export is incomplete: "
        "LLM, flowloss, linear projection, stop-head, and audio decoder ONNX "
        "exports succeeded, but remaining components are required for a full "
        f"TTS artifact: {', '.join(missing_native_components)}."
    )
    raise SystemExit(2)


def _export_small_litert_components(
    *,
    onnx_components: dict[str, object],
    output_dir: Path,
) -> dict[str, object]:
    results: dict[str, object] = {}
    components = onnx_components.get("components")
    if not isinstance(components, dict):
        components = {}
    for name in ("flowloss_dit_step", "linear_proj_audio", "stop_head"):
        results[name] = _export_litert_component(
            onnx_components=onnx_components,
            output_dir=output_dir,
            name=name,
            timeout_seconds=300,
            onnx2tf_extra_args=[],
            tflite_preference="dynamic_range",
        )
    return results


def _export_litert_component(
    *,
    onnx_components: dict[str, object],
    output_dir: Path,
    name: str,
    timeout_seconds: int,
    onnx2tf_extra_args: list[str],
    tflite_preference: str,
) -> dict[str, object]:
    components = onnx_components.get("components")
    if not isinstance(components, dict):
        components = {}
    component = components.get(name)
    if not isinstance(component, dict):
        return {"status": "missing_onnx_component"}
    artifact = component.get("artifact")
    if not isinstance(artifact, str) or not artifact:
        return {"status": "missing_onnx_artifact"}
    target_dir = output_dir / name
    cmd = [
        "uv",
        "run",
        "--group",
        "onnx2tf-convert",
        "python",
        "benchmark/runtime/converters/onnx_to_litert.py",
        "--repo",
        "local",
        "--artifact",
        artifact,
        "--output-dir",
        str(target_dir),
        "--attempt-timeout-seconds",
        str(timeout_seconds),
        "--tflite-preference",
        tflite_preference,
        "--no-fallback-isolated-onnx2tf2",
    ]
    for extra_arg in onnx2tf_extra_args:
        cmd.extend(["--onnx2tf-extra-arg", extra_arg])
    completed = subprocess.run(cmd, check=False)
    return {
        "command": cmd,
        "returncode": completed.returncode,
        "artifact": str(target_dir / "model.tflite"),
        "report": str(target_dir / "onnx_to_litert_report.json"),
        "tflite_preference": tflite_preference,
    }


def _write_component_report(output_dir: Path, report: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "component_export_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _is_text_to_audio_task(task: str) -> bool:
    return task.replace("-", "_").lower() in {"text_to_audio", "tts"}


if __name__ == "__main__":
    main()
