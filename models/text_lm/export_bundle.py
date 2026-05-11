from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_SAMPLE_PROMPT = "Explain why MLX is useful for local inference on Apple Silicon."
MANIFEST_FILE_NAME = "mlx_bundle.json"
DEFAULT_INPUT_NAMES = ["input_ids"]
DEFAULT_OUTPUT_NAMES = ["logits"]
CONTEXT_LENGTH_KEYS = (
    "max_position_embeddings",
    "model_max_length",
    "max_seq_len",
    "seq_length",
    "n_positions",
    "context_length",
)
TEXT_CONFIG_KEYS = (
    "text_config",
    "llm_config",
    "language_config",
    "model_config",
)


def extract_logits(output):
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, tuple):
        return output[0]
    return output


def build_bundle_manifest(
    *,
    bundle_id: str,
    name: str,
    kind: str,
    entrypoint: str,
    sample_inputs: str,
    input_names: list[str],
    output_names: list[str],
    description: str | None = None,
    source_model_id: str | None = None,
    context_length: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "id": bundle_id,
        "name": name,
        "kind": kind,
        "entrypoint": entrypoint,
        "sample_inputs": sample_inputs,
        "input_names": input_names,
        "output_names": output_names,
    }
    if description:
        manifest["description"] = description
    if source_model_id:
        manifest["source_model_id"] = source_model_id
    if context_length is not None:
        manifest["context_length"] = context_length
    if metadata:
        manifest["metadata"] = metadata
    return manifest


def infer_context_length(snapshot_dir: Path) -> int | None:
    config_path = snapshot_dir / "config.json"
    if not config_path.exists():
        return None
    try:
        config = _read_json_object(config_path)
    except (OSError, json.JSONDecodeError, ValueError):
        return None

    for key in TEXT_CONFIG_KEYS:
        nested = config.get(key)
        if isinstance(nested, dict):
            value = _first_int_for_keys(nested, CONTEXT_LENGTH_KEYS)
            if value is not None:
                return value

    return _first_int_for_keys(config, CONTEXT_LENGTH_KEYS)


def load_extra_metadata(
    *, metadata_json: str | None, metadata_file: str | None
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if metadata_file:
        metadata.update(_read_json_object(Path(metadata_file)))
    if metadata_json:
        parsed = json.loads(metadata_json)
        if not isinstance(parsed, dict):
            raise ValueError("--metadata-json must decode to a JSON object.")
        metadata.update(parsed)
    return metadata


def _read_json_object(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return parsed


def _first_int_for_keys(data: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = _as_int(data.get(key))
        if value is not None and value > 0:
            return value
    return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a shapeless next-token MLX function for a local mlx-lm snapshot.",
    )
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--sample-prompt",
        default=DEFAULT_SAMPLE_PROMPT,
        help=(
            "Example text used to generate sample input_ids for export. "
            "This only seeds the example input tensor; it does not define the "
            "runtime prompt format for your app."
        ),
    )
    parser.add_argument(
        "--sample-prompt-file",
        help="Optional text file whose contents override --sample-prompt.",
    )
    parser.add_argument(
        "--bundle-id",
        help="Stable bundle id written to mlx_bundle.json. Defaults to the output directory name.",
    )
    parser.add_argument(
        "--bundle-name",
        help="Human-readable bundle name. Defaults to --source-model-id or the snapshot directory name.",
    )
    parser.add_argument(
        "--bundle-kind",
        default="text",
        help='Bundle kind written to mlx_bundle.json. Defaults to "text".',
    )
    parser.add_argument("--description", help="Optional bundle description.")
    parser.add_argument(
        "--source-model-id",
        help="Source model id or repository, for example mlx-community/Qwen3.6-27B-4bit.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        help="Context length written to mlx_bundle.json. Defaults to config.json detection.",
    )
    parser.add_argument(
        "--metadata-json",
        help="Optional JSON object merged into the manifest metadata field.",
    )
    parser.add_argument(
        "--metadata-file",
        help="Optional JSON object file merged into the manifest metadata field before --metadata-json.",
    )
    args = parser.parse_args()

    snapshot_dir = Path(args.snapshot_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_prompt = (
        Path(args.sample_prompt_file).read_text(encoding="utf-8")
        if args.sample_prompt_file
        else args.sample_prompt
    )

    import mlx.core as mx
    from mlx_lm import load

    model, tokenizer = load(str(snapshot_dir), lazy=False)
    token_ids = tokenizer.encode(sample_prompt)
    tokens = mx.array([token_ids], dtype=mx.int32)

    def forward(input_ids):
        output = extract_logits(model(input_ids))
        return output[:, -1, :].astype(mx.float32)

    export_path = output_dir / "function.mlxfn"
    sample_inputs_path = output_dir / "inputs.safetensors"
    sample_inputs_json_path = output_dir / "inputs.json"
    manifest_path = output_dir / MANIFEST_FILE_NAME
    if export_path.exists():
        export_path.unlink()
    if sample_inputs_path.exists():
        sample_inputs_path.unlink()
    if sample_inputs_json_path.exists():
        sample_inputs_json_path.unlink()
    if manifest_path.exists():
        manifest_path.unlink()

    mx.export_function(str(export_path), forward, tokens, shapeless=True)
    mx.save_safetensors(str(sample_inputs_path), {"input_ids": tokens})
    sample_inputs_json_path.write_text(
        json.dumps(
            {
                "input_order": ["input_ids"],
                "inputs": {
                    "input_ids": {
                        "dtype": "int32",
                        "shape": [1, len(token_ids)],
                        "values": token_ids,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    metadata = {
        "schema_version": 1,
        "exporter": "dart-inference/models/text_lm/export_bundle.py",
        "inputs_json": sample_inputs_json_path.name,
        "sample_prompt": sample_prompt,
        **load_extra_metadata(
            metadata_json=args.metadata_json,
            metadata_file=args.metadata_file,
        ),
    }
    manifest = build_bundle_manifest(
        bundle_id=args.bundle_id or output_dir.name,
        name=args.bundle_name
        or args.source_model_id
        or snapshot_dir.name.replace("-", " "),
        kind=args.bundle_kind,
        description=args.description,
        source_model_id=args.source_model_id,
        context_length=args.context_length or infer_context_length(snapshot_dir),
        entrypoint=export_path.name,
        sample_inputs=sample_inputs_path.name,
        input_names=DEFAULT_INPUT_NAMES,
        output_names=DEFAULT_OUTPUT_NAMES,
        metadata=metadata,
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"exported={export_path}")
    print(f"inputs={sample_inputs_path}")
    print(f"inputs_json={sample_inputs_json_path}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
