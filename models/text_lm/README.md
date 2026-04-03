# text lm

Helpers in this directory target text-generation models loaded through
`mlx_lm`.

Current entrypoints:

- [`convert_unsloth_mlx.py`](convert_unsloth_mlx.py): convert a standard
  Hugging Face SafeTensors checkpoint into an Unsloth-style MLX snapshot
- [`export_bundle.py`](export_bundle.py): export a local `mlx-lm` snapshot into
  a shapeless `.mlxfn` bundle plus example `input_ids`

Use cases:

1. prepare a local MLX snapshot from a Hugging Face checkpoint
2. export a reusable next-token MLX function once
3. run the exported artifact from Dart with
   [`../common/import_run.dart`](../common/import_run.dart)
4. use `benchmark/` scripts when you want MLX parity or performance reports

For `Gemma 4`, publish-time parity currently uses the official
`unsloth/gemma-4-E2B-it-UD-MLX-4bit` MLX snapshot directly instead of trying to
rebuild Unsloth Dynamic quantization locally.

### Prerequisites

- `uv sync`
- `node` / `npx` available in `PATH`

### Example

```sh
uv run python models/text_lm/convert_unsloth_mlx.py \
  --input Qwen/Qwen3.5-9B \
  --output-dir /tmp/qwen3.5-9b-unsloth-mlx \
  --model-type qwen3_5
```

Then export a reusable bundle:

```sh
uv run python models/text_lm/export_bundle.py \
  --snapshot-dir /tmp/qwen3.5-9b-unsloth-mlx \
  --output-dir /tmp/qwen3.5-9b-bundle
```
