from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from hf_download import DEFAULT_FALLBACK_ENDPOINT, snapshot_download_with_fallback


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export a Hugging Face text-generation model with ONNX Runtime GenAI "
            "Model Builder, optionally patching config.json model_type first."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--precision", default="int4")
    parser.add_argument("--execution-provider", default="cpu")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--model-type-to")
    parser.add_argument("--model-type-from")
    parser.add_argument("--revision")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT"))
    parser.add_argument(
        "--fallback-endpoint",
        default=os.environ.get("HF_FALLBACK_ENDPOINT") or DEFAULT_FALLBACK_ENDPOINT,
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=[],
        help="Optional snapshot allow pattern, useful for config-only probes.",
    )
    parser.add_argument(
        "--extra-option",
        action="append",
        default=[],
        help="ONNX Runtime GenAI builder --extra_options item, e.g. key=value.",
    )
    args, passthrough = parser.parse_known_args()

    output_dir = args.output_dir.expanduser().resolve()
    cache_dir = (args.cache_dir or (output_dir / "_hf_cache")).expanduser().resolve()
    source = _prepare_source(
        model=args.model,
        output_dir=output_dir,
        revision=args.revision,
        model_type_from=args.model_type_from,
        model_type_to=args.model_type_to,
        endpoint=args.endpoint,
        fallback_endpoint=args.fallback_endpoint,
        allow_patterns=args.allow_pattern,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_builder_command(
        source=source,
        output_dir=output_dir,
        precision=args.precision,
        execution_provider=args.execution_provider,
        cache_dir=cache_dir,
        extra_options=args.extra_option,
        passthrough=passthrough,
    )
    completed = subprocess.run(cmd, check=False)
    raise SystemExit(completed.returncode)


def _prepare_source(
    *,
    model: str,
    output_dir: Path,
    revision: str | None,
    model_type_from: str | None,
    model_type_to: str | None,
    endpoint: str | None,
    fallback_endpoint: str | None,
    allow_patterns: list[str] | None = None,
) -> Path | str:
    allow_patterns = _clean_patterns(allow_patterns)
    if not model_type_to and not allow_patterns:
        return model
    source_root = output_dir / "_patched_source"
    source_dir = source_root / "source_model"
    snapshot_download_with_fallback(
        repo_id=model,
        revision=revision,
        local_dir=str(source_dir),
        endpoint=endpoint,
        fallback_endpoint=fallback_endpoint,
        allow_patterns=allow_patterns,
    )
    patch_report: dict[str, Any] = {"config_path": str(source_dir / "config.json")}
    if model_type_to:
        patch_report.update(
            _patch_model_type(
                config_path=source_dir / "config.json",
                expected=model_type_from,
                patched=model_type_to,
            )
        )
        patch_report.update(
            _patch_nested_text_config_for_ort_genai(source_dir / "config.json")
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "source_patch_report.json").write_text(
        json.dumps(
            {"source_model": model, "source_dir": str(source_dir), **patch_report},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return source_dir


def _clean_patterns(patterns: list[str] | None) -> list[str] | None:
    cleaned = [str(item).strip() for item in patterns or [] if str(item).strip()]
    return cleaned or None


def _patch_nested_text_config_for_ort_genai(config_path: Path) -> dict[str, Any]:
    decoded = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError(f"config.json is not a JSON object: {config_path}")
    text_config = decoded.get("text_config")
    if not isinstance(text_config, dict):
        return {"flattened_text_config_keys": []}

    cleaned_text_config = {
        str(key): value for key, value in text_config.items() if value is not None
    }
    dropped_null_keys = sorted(
        str(key) for key, value in text_config.items() if value is None
    )
    decoded["text_config"] = cleaned_text_config

    flattened = []
    for key, value in cleaned_text_config.items():
        if key not in decoded or decoded[key] is None:
            decoded[key] = value
            flattened.append(str(key))

    if decoded.get("pad_token_id") is None and decoded.get("eos_token_id") is not None:
        decoded["pad_token_id"] = decoded["eos_token_id"]
        flattened.append("pad_token_id")

    rope_parameters = cleaned_text_config.get("rope_parameters")
    if isinstance(rope_parameters, dict):
        rope_scaling = dict(rope_parameters)
        if decoded.get("rope_scaling") is None:
            decoded["rope_scaling"] = rope_scaling
        for key in ["rope_theta", "partial_rotary_factor"]:
            if key in rope_scaling and decoded.get(key) is None:
                decoded[key] = rope_scaling[key]

    config_path.write_text(
        json.dumps(decoded, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "flattened_text_config_keys": sorted(flattened),
        "dropped_null_text_config_keys": dropped_null_keys,
        "rope_parameters_to_rope_scaling": isinstance(rope_parameters, dict),
    }


def _build_builder_command(
    *,
    source: Path | str,
    output_dir: Path,
    precision: str,
    execution_provider: str,
    cache_dir: Path,
    extra_options: list[str],
    passthrough: list[str],
) -> list[str]:
    cmd = [
        "python",
        "-m",
        "onnxruntime_genai.models.builder",
        "-o",
        str(output_dir),
        "-p",
        precision,
        "-e",
        execution_provider,
        "-c",
        str(cache_dir),
    ]
    if isinstance(source, Path):
        cmd.extend(["-i", str(source)])
    else:
        cmd.extend(["-m", source])
    if extra_options:
        cmd.append("--extra_options")
        cmd.extend(extra_options)
    cmd.extend(passthrough)
    return cmd


def _patch_model_type(
    *,
    config_path: Path,
    expected: str | None,
    patched: str,
) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"config.json is missing in snapshot: {config_path}")
    decoded = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError(f"config.json is not a JSON object: {config_path}")
    original = str(decoded.get("model_type") or "")
    if expected and original and original != expected:
        raise ValueError(
            f"Expected model_type '{expected}' but found '{original}' in {config_path}"
        )
    decoded["model_type"] = patched
    config_path.write_text(
        json.dumps(decoded, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "config_path": str(config_path),
        "original_model_type": original,
        "patched_model_type": patched,
    }


if __name__ == "__main__":
    main()
