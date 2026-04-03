from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from huggingface_hub import hf_hub_download, snapshot_download

DEFAULT_IMATRIX_FILE = "imatrix_unsloth.gguf"
DEFAULT_MLX_NODE_PACKAGE = "@mlx-node/cli"


@dataclass(frozen=True)
class ConvertPlan:
    model_source: str
    input_path: Path
    output_dir: Path
    imatrix_source: str | None
    imatrix_path: Path | None
    command: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a standard Hugging Face text checkpoint into an "
            "Unsloth-optimized MLX snapshot."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Local model directory or Hugging Face model id containing the "
            "source SafeTensors checkpoint."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the converted MLX snapshot will be written.",
    )
    parser.add_argument(
        "--imatrix-path",
        help="Local path to an imatrix GGUF file.",
    )
    parser.add_argument(
        "--imatrix-repo",
        help=(
            "Hugging Face repo id that contains the imatrix GGUF file. "
            "Used when --imatrix-path is not provided."
        ),
    )
    parser.add_argument(
        "--imatrix-file",
        default=DEFAULT_IMATRIX_FILE,
        help=(
            "Filename inside --imatrix-repo to download. "
            f"Defaults to {DEFAULT_IMATRIX_FILE!r}."
        ),
    )
    parser.add_argument(
        "--model-type",
        help=(
            "Optional model type for the converter, for example "
            "'qwen3_5' or 'qwen3_5_moe'."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        help="Optional output dtype override for non-quantized tensors.",
    )
    parser.add_argument(
        "--q-bits",
        type=int,
        help="Optional quantization bit override passed through to the converter.",
    )
    parser.add_argument(
        "--q-group-size",
        type=int,
        help="Optional quantization group size override passed through to the converter.",
    )
    parser.add_argument(
        "--mlx-cli",
        default="mlx",
        help=(
            "Command prefix for the mlx converter. Defaults to 'mlx'. If that "
            "binary is missing, the script falls back to "
            f"'npx --yes {DEFAULT_MLX_NODE_PACKAGE}'."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        help="Optional Hugging Face cache dir for model and imatrix downloads.",
    )
    parser.add_argument(
        "--revision",
        help="Optional revision for the source Hugging Face model id.",
    )
    parser.add_argument(
        "--imatrix-revision",
        help="Optional revision for --imatrix-repo.",
    )
    parser.add_argument(
        "--token",
        help=(
            "Optional Hugging Face token. If omitted, the script uses HF_TOKEN "
            "from the environment when available."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command and inputs without running conversion.",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help=(
            "Skip quantization and only convert the input into an MLX snapshot "
            "using the selected dtype and model-type mapping."
        ),
    )
    return parser.parse_args()


def resolve_cli_prefix(cli: str) -> list[str]:
    parts = shlex.split(cli)
    if not parts:
        raise ValueError("--mlx-cli resolved to an empty command.")
    if shutil.which(parts[0]) is not None:
        return parts
    if cli == "mlx":
        if shutil.which("npx") is None:
            raise FileNotFoundError(
                "Neither 'mlx' nor 'npx' was found in PATH. Install "
                f"{DEFAULT_MLX_NODE_PACKAGE} or pass --mlx-cli explicitly."
            )
        return ["npx", "--yes", DEFAULT_MLX_NODE_PACKAGE]
    raise FileNotFoundError(f"Unable to find converter command: {parts[0]!r}")


def resolve_input_path(
    source: str,
    *,
    cache_dir: str | None,
    revision: str | None,
    token: str | None,
) -> tuple[Path, str]:
    local = Path(source).expanduser()
    if local.exists():
        return local.resolve(), "local"
    downloaded = snapshot_download(
        repo_id=source,
        repo_type="model",
        cache_dir=cache_dir,
        revision=revision,
        token=token or os.environ.get("HF_TOKEN"),
    )
    return Path(downloaded).resolve(), "huggingface"


def resolve_imatrix_path(args: argparse.Namespace) -> tuple[Path, str]:
    token = args.token or os.environ.get("HF_TOKEN")
    if args.imatrix_path:
        path = Path(args.imatrix_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"imatrix file not found: {path}")
        return path.resolve(), "local"
    if not args.imatrix_repo:
        raise ValueError(
            "Provide either --imatrix-path or --imatrix-repo/--imatrix-file."
        )
    downloaded = hf_hub_download(
        repo_id=args.imatrix_repo,
        filename=args.imatrix_file,
        repo_type="model",
        cache_dir=args.cache_dir,
        revision=args.imatrix_revision,
        token=token,
    )
    return Path(downloaded).resolve(), f"hf:{args.imatrix_repo}"


def build_convert_command(
    *,
    cli_prefix: Sequence[str],
    input_path: Path,
    output_dir: Path,
    imatrix_path: Path | None,
    model_type: str | None,
    dtype: str | None,
    q_bits: int | None,
    q_group_size: int | None,
    quantize: bool,
) -> list[str]:
    command = list(cli_prefix) + [
        "convert",
        "--input",
        str(input_path),
        "--output",
        str(output_dir),
    ]
    if model_type:
        command.extend(["--model-type", model_type])
    if dtype:
        command.extend(["--dtype", dtype])
    if quantize:
        if imatrix_path is None:
            raise ValueError("Quantized conversion requires an imatrix path.")
        command.extend(
            [
                "--quantize",
                "--q-recipe",
                "unsloth",
                "--imatrix-path",
                str(imatrix_path),
            ]
        )
        if q_bits is not None:
            command.extend(["--q-bits", str(q_bits)])
        if q_group_size is not None:
            command.extend(["--q-group-size", str(q_group_size)])
    return command


def verify_output_dir(output_dir: Path, *, quantized: bool) -> None:
    config_path = output_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing converted config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if quantized and "quantization" not in config:
        raise ValueError("Converted snapshot is missing quantization metadata.")
    tensor_files = sorted(output_dir.glob("*.safetensors"))
    if not tensor_files:
        raise FileNotFoundError(
            f"No .safetensors files found in converted snapshot: {output_dir}"
        )


def normalize_output_config(
    output_dir: Path,
    *,
    quantized: bool,
    q_bits: int | None,
    q_group_size: int | None,
) -> None:
    if quantized:
        return
    config_path = output_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "quantization" not in config:
        config["quantization"] = {
            "bits": q_bits or 4,
            "group_size": q_group_size or 64,
            "mode": "affine",
        }
        config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def copy_sidecar_files(input_path: Path, output_dir: Path) -> None:
    if not input_path.is_dir():
        return
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.jinja",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "image_processor_config.json",
    ):
        src = input_path / name
        dst = output_dir / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def write_manifest(plan: ConvertPlan) -> Path:
    manifest_path = plan.output_dir / "conversion_manifest.json"
    manifest = {
        "kind": "unsloth_mlx_conversion",
        "model_source": plan.model_source,
        "input_path": str(plan.input_path),
        "output_dir": str(plan.output_dir),
        "imatrix_source": plan.imatrix_source,
        "imatrix_path": str(plan.imatrix_path),
        "command": plan.command,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest_path


def make_plan(args: argparse.Namespace) -> ConvertPlan:
    cli_prefix = resolve_cli_prefix(args.mlx_cli)
    input_path, model_source = resolve_input_path(
        args.input,
        cache_dir=args.cache_dir,
        revision=args.revision,
        token=args.token,
    )
    quantize = not args.no_quantize
    if quantize:
        imatrix_path, imatrix_source = resolve_imatrix_path(args)
    else:
        imatrix_path = None
        imatrix_source = None
    output_dir = Path(args.output_dir).expanduser().resolve()
    command = build_convert_command(
        cli_prefix=cli_prefix,
        input_path=input_path,
        output_dir=output_dir,
        imatrix_path=imatrix_path,
        model_type=args.model_type,
        dtype=args.dtype,
        q_bits=args.q_bits,
        q_group_size=args.q_group_size,
        quantize=quantize,
    )
    return ConvertPlan(
        model_source=model_source,
        input_path=input_path,
        output_dir=output_dir,
        imatrix_source=imatrix_source,
        imatrix_path=imatrix_path,
        command=command,
    )


def run_plan(
    plan: ConvertPlan,
    *,
    quantized: bool,
    q_bits: int | None,
    q_group_size: int | None,
) -> Path:
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(plan.command, check=True)
    normalize_output_config(
        plan.output_dir,
        quantized=quantized,
        q_bits=q_bits,
        q_group_size=q_group_size,
    )
    copy_sidecar_files(plan.input_path, plan.output_dir)
    verify_output_dir(plan.output_dir, quantized=quantized)
    return write_manifest(plan)


def main() -> None:
    args = parse_args()
    plan = make_plan(args)
    print(f"resolved_input={plan.input_path}")
    print(f"resolved_imatrix={plan.imatrix_path}")
    print(f"command={' '.join(shlex.quote(part) for part in plan.command)}")
    if args.dry_run:
        return
    manifest_path = run_plan(
        plan,
        quantized=not args.no_quantize,
        q_bits=args.q_bits,
        q_group_size=args.q_group_size,
    )
    print(f"manifest={manifest_path}")
    print(f"output_dir={plan.output_dir}")


if __name__ == "__main__":
    main()
