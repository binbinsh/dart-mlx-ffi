from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a Hugging Face generative model with litert-torch."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--task",
        default="text_generation",
        help="litert-torch export task, for example text_generation or image_text_to_text.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--prefill-lengths")
    parser.add_argument("--cache-length", type=int)
    parser.add_argument("--quantization-recipe")
    parser.add_argument("--externalize-embedder", action="store_true")
    parser.add_argument("--single-token-embedder", action="store_true")
    parser.add_argument("--split-cache", action="store_true")
    parser.add_argument("--cache-implementation")
    parser.add_argument("--auto-model-override")
    parser.add_argument("--use-jinja-template", action="store_true")
    parser.add_argument("--bundle-litert-lm", action="store_true")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--export-vision-encoder", action="store_true")
    parser.add_argument("--vision-encoder-quantization-recipe")
    parser.add_argument("--litert-lm-model-type")
    parser.add_argument("--litert-lm-metadata")
    parser.add_argument("--lightweight-conversion", action="store_true")
    parser.add_argument("--model-type-to")
    parser.add_argument("--model-type-from")
    parser.add_argument("--patch-minicpm-optional-deps", action="store_true")
    parser.add_argument("--revision")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT"))
    parser.add_argument(
        "--fallback-endpoint",
        default=os.environ.get("HF_FALLBACK_ENDPOINT") or "https://hf-mirror.com",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=[],
        help="Optional snapshot allow pattern, useful for config-only probes.",
    )
    args = parser.parse_args()

    model = _prepare_source(
        model=args.model,
        output_dir=Path(args.output_dir),
        revision=args.revision,
        model_type_from=args.model_type_from,
        model_type_to=args.model_type_to,
        endpoint=args.endpoint,
        fallback_endpoint=args.fallback_endpoint,
        allow_patterns=args.allow_pattern,
    )
    if args.patch_minicpm_optional_deps:
        _patch_minicpm_optional_deps(
            model=model,
            repo_id=args.model,
            output_dir=Path(args.output_dir),
            revision=args.revision,
            endpoint=args.endpoint,
            fallback_endpoint=args.fallback_endpoint,
        )

    if args.auto_model_override:
        import transformers

        getattr(transformers, args.auto_model_override)

    from litert_torch.generative.export_hf import export as export_module
    if args.auto_model_override:
        from litert_torch.generative.export_hf.core import export_lib

        getattr(export_lib.transformers, args.auto_model_override)
    if args.trust_remote_code:
        from litert_torch.generative.export_hf.core import export_lib

        _patch_trust_remote_code(export_lib.transformers.AutoImageProcessor)
        _patch_trust_remote_code(export_lib.transformers.AutoTokenizer)

    export_fn = getattr(export_module, "export", export_module)
    export_fn(
        model=str(model),
        output_dir=args.output_dir,
        task=args.task,
        trust_remote_code=args.trust_remote_code,
        prefill_lengths=_int_list(args.prefill_lengths),
        cache_length=args.cache_length,
        quantization_recipe=args.quantization_recipe,
        externalize_embedder=args.externalize_embedder or None,
        single_token_embedder=args.single_token_embedder or None,
        split_cache=args.split_cache or None,
        cache_implementation=args.cache_implementation,
        auto_model_override=args.auto_model_override,
        use_jinja_template=args.use_jinja_template or None,
        bundle_litert_lm=args.bundle_litert_lm or None,
        experimental_use_mixed_precision=args.mixed_precision or None,
        export_vision_encoder=args.export_vision_encoder or None,
        vision_encoder_quantization_recipe=args.vision_encoder_quantization_recipe,
        litert_lm_model_type_override=args.litert_lm_model_type,
        litert_lm_llm_metadata_override=args.litert_lm_metadata,
        experimental_lightweight_conversion=args.lightweight_conversion,
    )


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
    if not model_type_to:
        return model

    from hf_download import snapshot_download_with_fallback

    output_dir = output_dir.expanduser().resolve()
    source_root = output_dir / "_patched_source"
    source_dir = source_root / "source_model"
    snapshot_download_with_fallback(
        repo_id=model,
        revision=revision,
        local_dir=str(source_dir),
        endpoint=endpoint,
        fallback_endpoint=fallback_endpoint,
        allow_patterns=_clean_patterns(allow_patterns),
    )
    patch_report = _patch_model_type(
        config_path=source_dir / "config.json",
        expected=model_type_from,
        patched=model_type_to,
    )
    patch_report.update(
        _patch_nested_text_config_for_flat_exporter(source_dir / "config.json")
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


def _patch_nested_text_config_for_flat_exporter(config_path: Path) -> dict[str, Any]:
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


def _patch_minicpm_optional_deps(
    *,
    model: str | Path,
    repo_id: str,
    output_dir: Path,
    revision: str | None,
    endpoint: str | None,
    fallback_endpoint: str | None,
) -> dict[str, Any]:
    candidates: list[Path] = []
    model_path = Path(model).expanduser()
    if model_path.exists():
        candidates.append(model_path / "utils.py")
    else:
        from hf_download import hf_hub_download_with_fallback

        path = hf_hub_download_with_fallback(
            repo_id=repo_id,
            filename="utils.py",
            revision=revision,
            endpoint=endpoint,
            fallback_endpoint=fallback_endpoint,
        )
        candidates.append(Path(path))
    candidates.extend(_cached_minicpm_utils_files())
    patched = []
    for path in candidates:
        if _patch_minicpm_utils_file(path):
            patched.append(str(path))
    report = {
        "source_model": repo_id,
        "patched_minicpm_optional_dep_files": sorted(set(patched)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "minicpm_optional_deps_patch_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def _cached_minicpm_utils_files() -> list[Path]:
    roots = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home).expanduser() / "modules" / "transformers_modules")
    roots.append(
        Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"
    )
    result = []
    for root in roots:
        if not root.exists():
            continue
        result.extend(path for path in root.rglob("utils.py") if _is_minicpm_path(path))
    return result


def _is_minicpm_path(path: Path) -> bool:
    lowered = str(path).lower()
    return "minicpm" in lowered or "minicpm_hyphen" in lowered


def _patch_minicpm_utils_file(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    marker = "from minicpmo.utils import get_video_frame_audio_segments"
    if marker not in text:
        return False
    replacement = (
        "import importlib\n"
        "            get_video_frame_audio_segments = importlib.import_module(\n"
        "                \"minicpmo.utils\"\n"
        "            ).get_video_frame_audio_segments"
    )
    path.write_text(text.replace(marker, replacement), encoding="utf-8")
    return True


def _int_list(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _patch_trust_remote_code(auto_cls: object) -> None:
    original = getattr(auto_cls, "from_pretrained")
    if getattr(original, "_dinf_trust_remote_code", False):
        return

    def from_pretrained_with_trust(*args: object, **kwargs: object) -> object:
        kwargs.setdefault("trust_remote_code", True)
        return original(*args, **kwargs)

    from_pretrained_with_trust._dinf_trust_remote_code = True  # type: ignore[attr-defined]
    setattr(auto_cls, "from_pretrained", from_pretrained_with_trust)


if __name__ == "__main__":
    main()
