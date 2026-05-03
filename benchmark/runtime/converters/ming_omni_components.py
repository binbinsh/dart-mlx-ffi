from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

from safetensors import safe_open
from safetensors.torch import save_file

from ming_omni_source import DEFAULT_FALLBACK_REPOS, prepare_patched_source


OFFICIAL_SOURCE_FILES = (
    "chat_format.py",
    "configuration_bailing_moe.py",
    "configuration_bailingmm.py",
    "modeling_bailing_moe.py",
    "modeling_bailingmm.py",
    "spkemb_extractor.py",
    "tokenization_bailing.py",
)
OFFICIAL_SOURCE_DIRS = (
    "audio_tokenizer",
    "fm",
    "sentence_manager",
)
TOKENIZER_FILES = (
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
COMPONENT_PREFIXES = {
    "audio": "audio.",
    "flowloss": "flowloss.",
    "linear_proj_audio": "linear_proj_audio.",
    "spk_head": "spk_head.",
    "stop_head": "stop_head.",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Ming-omni TTS source components for native runtime export."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Use an already downloaded Ming model snapshot instead of downloading.",
    )
    parser.add_argument("--official-source", type=Path)
    parser.add_argument("--endpoint")
    parser.add_argument("--fallback-endpoint")
    parser.add_argument("--extract-llm-hf", action="store_true")
    parser.add_argument("--split-components", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    if args.source_dir is None:
        source_dir, source_report = prepare_source(
            model=args.model,
            output_dir=output_dir,
            official_source=args.official_source,
            fallback_repos=None,
            endpoint=args.endpoint,
            fallback_endpoint=args.fallback_endpoint,
        )
    else:
        source_dir = args.source_dir.expanduser().resolve()
        copied = copy_official_source(
            official_source=args.official_source,
            target_dir=source_dir,
        )
        source_report = {
            "source_model": args.model,
            "source_dir": str(source_dir),
            "used_existing_source_dir": True,
            "official_source": (
                None if args.official_source is None else str(args.official_source)
            ),
            "official_source_copied": copied,
        }

    report: dict[str, Any] = {
        "source": source_report,
        "source_dir": str(source_dir),
        "outputs": {},
    }
    if args.extract_llm_hf:
        report["outputs"]["llm_hf"] = extract_llm_hf(
            source_dir=source_dir,
            output_dir=output_dir / "components" / "llm_hf",
        )
    if args.split_components:
        report["outputs"]["torch_components"] = split_component_weights(
            source_dir=source_dir,
            output_dir=output_dir / "components" / "torch",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "component_export_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def prepare_source(
    *,
    model: str,
    output_dir: Path,
    official_source: Path | None,
    fallback_repos: Iterable[str] | None,
    endpoint: str | None,
    fallback_endpoint: str | None,
) -> tuple[Path, dict[str, Any]]:
    source_dir, report = prepare_patched_source(
        source_model=model,
        work_dir=output_dir / "_patched_source",
        fallback_repos=fallback_repos or DEFAULT_FALLBACK_REPOS,
        endpoint=endpoint,
        fallback_endpoint=fallback_endpoint,
    )
    copied = copy_official_source(
        official_source=official_source,
        target_dir=source_dir,
    )
    report["official_source"] = None if official_source is None else str(official_source)
    report["official_source_copied"] = copied
    return source_dir, report


def copy_official_source(
    *,
    official_source: Path | None,
    target_dir: Path,
) -> list[str]:
    if official_source is None:
        return []
    source = official_source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Ming official source not found: {source}")
    copied: list[str] = []
    for name in OFFICIAL_SOURCE_FILES:
        src = source / name
        if not src.exists():
            continue
        shutil.copy2(src, target_dir / name)
        copied.append(name)
    for name in OFFICIAL_SOURCE_DIRS:
        src = source / name
        if not src.exists():
            continue
        dst = target_dir / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        copied.append(f"{name}/")
    return copied


def extract_llm_hf(
    *,
    source_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = _read_config(source_dir)
    llm_config = dict(_require_dict(config, "llm_config"))
    llm_config["architectures"] = ["Qwen2ForCausalLM"]
    llm_config["model_type"] = "qwen2"
    llm_config.setdefault("tie_word_embeddings", True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(llm_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "generation_config.json").write_text(
        json.dumps(_generation_config(config, llm_config), indent=2) + "\n",
        encoding="utf-8",
    )
    copied_tokenizers = _copy_existing(source_dir, output_dir, TOKENIZER_FILES)
    tensors = {}
    for path in _safetensor_files(source_dir):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if not key.startswith("model."):
                    continue
                tensors[key.removeprefix("model.")] = handle.get_tensor(key)
    if not tensors:
        raise ValueError(f"No Ming LLM tensors found in {source_dir}")
    save_file(tensors, output_dir / "model.safetensors")
    return {
        "path": str(output_dir),
        "format": "huggingface_qwen2_causal_lm",
        "tensor_count": len(tensors),
        "tokenizer_files": copied_tokenizers,
    }


def split_component_weights(
    *,
    source_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for component, prefix in COMPONENT_PREFIXES.items():
        tensors = {}
        for path in _safetensor_files(source_dir):
            with safe_open(path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key.startswith(prefix):
                        tensors[key.removeprefix(prefix)] = handle.get_tensor(key)
        if not tensors:
            continue
        target = output_dir / f"{component}.safetensors"
        save_file(tensors, target)
        results[component] = {
            "path": str(target),
            "tensor_count": len(tensors),
        }
    (output_dir / "ming_omni_components.json").write_text(
        json.dumps(
            {
                "format": "dart_mlx_ffi.ming_omni_tts_torch_components.v1",
                "components": results,
                "config": str(source_dir / "config.json"),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return results


def _read_config(source_dir: Path) -> dict[str, Any]:
    config_path = source_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing Ming config: {config_path}")
    decoded = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError(f"Ming config is not a JSON object: {config_path}")
    return decoded


def _require_dict(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Ming config field {key!r} must be an object.")
    return value


def _generation_config(
    config: dict[str, Any],
    llm_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "bos_token_id": llm_config.get("bos_token_id"),
        "eos_token_id": llm_config.get("eos_token_id"),
        "pad_token_id": config.get("pad_token_id") or llm_config.get("eos_token_id"),
        "do_sample": False,
        "transformers_version": config.get("transformers_version"),
    }


def _copy_existing(
    source_dir: Path,
    output_dir: Path,
    names: Iterable[str],
) -> list[str]:
    copied: list[str] = []
    for name in names:
        src = source_dir / name
        if not src.exists():
            continue
        shutil.copy2(src, output_dir / name)
        copied.append(name)
    return copied


def _safetensor_files(source_dir: Path) -> list[Path]:
    files = sorted(source_dir.glob("*.safetensors"))
    if files:
        return files
    raise FileNotFoundError(f"No safetensors weights found in {source_dir}")


if __name__ == "__main__":
    main()
