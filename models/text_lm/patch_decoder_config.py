"""Patch ``architectures`` (and related fields) in a converted MLX snapshot's
``config.json`` so HF/MLX loaders treat it as a decoder-only checkpoint.

Used by ``convert_paddle_ocr_vl_decoder.sh`` after the multimodal
PaddleOCR-VL-1.5 source is converted with ``--skip-prefix visual.``: the
resulting snapshot still carries the original multimodal ``architectures``
entry (``["PaddleOCRVLForConditionalGeneration"]``), which causes
``mlx_lm.load`` / ``transformers.AutoModelForCausalLM`` to refuse the
file because the corresponding model class expects ``visual.*`` weights
that no longer exist.

After this patch the snapshot is loadable as a plain ERNIE-4.5 causal LM
(``Ernie4_5ForCausalLM`` / ``model_type=ernie4_5``), which is what
PaddleOCR-VL-1.5's language backbone actually is upstream.

The function is intentionally tiny and side-effect-free except for the
single ``config.json`` write so it is trivially testable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

# Default decoder-only patch for PaddleOCR-VL-1.5. The ERNIE-4.5 base LM
# (``baidu/ERNIE-4.5-0.3B-PT``) uses these exact values upstream, which
# is the architecture PaddleOCR-VL-1.5's language backbone is derived
# from. Confirmed via:
#   hf_hub_download('baidu/ERNIE-4.5-0.3B-PT', 'config.json')
#     -> architectures=['Ernie4_5ForCausalLM'], model_type='ernie4_5'
DEFAULT_DECODER_ARCHITECTURE = "Ernie4_5ForCausalLM"
DEFAULT_DECODER_MODEL_TYPE = "ernie4_5"

# Multimodal-only fields that should be dropped from a decoder-only
# config so loaders don't try to instantiate a non-existent vision tower.
MULTIMODAL_ONLY_KEYS: tuple[str, ...] = (
    "vision_config",
    "image_token_id",
    "video_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
)


def patch_config(
    config_path: Path,
    *,
    architecture: str = DEFAULT_DECODER_ARCHITECTURE,
    model_type: str | None = DEFAULT_DECODER_MODEL_TYPE,
    drop_keys: Iterable[str] = MULTIMODAL_ONLY_KEYS,
    drop_auto_map: bool = True,
) -> dict:
    """Rewrite ``config.json`` in place. Returns the new config dict."""
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    config["architectures"] = [architecture]
    if model_type is not None:
        config["model_type"] = model_type

    for key in drop_keys:
        config.pop(key, None)

    if drop_auto_map:
        # ``auto_map`` points at the multimodal trust-remote-code modules
        # that no longer match the stripped weights. Removing it forces
        # loaders to use the in-tree ``Ernie4_5ForCausalLM`` class.
        config.pop("auto_map", None)

    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patch a converted MLX snapshot's config.json to look like a "
            "decoder-only ERNIE-4.5 checkpoint."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to config.json inside the converted MLX snapshot.",
    )
    parser.add_argument(
        "--architecture",
        default=DEFAULT_DECODER_ARCHITECTURE,
        help=f"architectures[0] value (default: {DEFAULT_DECODER_ARCHITECTURE}).",
    )
    parser.add_argument(
        "--model-type",
        default=DEFAULT_DECODER_MODEL_TYPE,
        help=f"model_type value (default: {DEFAULT_DECODER_MODEL_TYPE}).",
    )
    parser.add_argument(
        "--keep-auto-map",
        action="store_true",
        help="Keep the auto_map field (default: drop).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = patch_config(
        args.config,
        architecture=args.architecture,
        model_type=args.model_type,
        drop_auto_map=not args.keep_auto_map,
    )
    print(
        f"patched_config={args.config} "
        f"architectures={config['architectures']} "
        f"model_type={config.get('model_type')}"
    )


if __name__ == "__main__":
    main()
