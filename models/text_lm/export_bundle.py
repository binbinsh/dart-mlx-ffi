from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

DEFAULT_SAMPLE_PROMPT = "Explain why MLX is useful for local inference on Apple Silicon."


def extract_logits(output):
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, tuple):
        return output[0]
    return output


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
    args = parser.parse_args()

    snapshot_dir = Path(args.snapshot_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_prompt = (
        Path(args.sample_prompt_file).read_text(encoding="utf-8")
        if args.sample_prompt_file
        else args.sample_prompt
    )

    model, tokenizer = load(str(snapshot_dir), lazy=False)
    token_ids = tokenizer.encode(sample_prompt)
    tokens = mx.array([token_ids], dtype=mx.int32)

    def forward(input_ids):
        output = extract_logits(model(input_ids))
        return output[:, -1, :].astype(mx.float32)

    export_path = output_dir / "function.mlxfn"
    sample_inputs_path = output_dir / "inputs.safetensors"
    if export_path.exists():
        export_path.unlink()
    if sample_inputs_path.exists():
        sample_inputs_path.unlink()

    mx.export_function(str(export_path), forward, tokens, shapeless=True)
    mx.save_safetensors(str(sample_inputs_path), {"input_ids": tokens})
    print(f"exported={export_path}")
    print(f"inputs={sample_inputs_path}")


if __name__ == "__main__":
    main()
