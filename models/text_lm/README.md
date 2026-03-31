# text lm

Helpers in this directory target text-generation models loaded through
`mlx_lm`.

Current entrypoints:

- [`convert_unsloth_mlx.py`](convert_unsloth_mlx.py): convert a standard
  Hugging Face SafeTensors checkpoint into an Unsloth-optimized MLX snapshot
  by wrapping `@mlx-node/cli convert`
- [`export_bundle.py`](export_bundle.py): export a local `mlx-lm` snapshot into
  a shapeless `.mlxfn` bundle plus example `input_ids`

## Convert Hugging Face Checkpoints To Unsloth MLX

This repository now includes a complete wrapper toolchain for:

1. resolving a local model dir or Hugging Face model id
2. resolving a local or Hub-hosted `imatrix` GGUF file
3. invoking the upstream Unsloth-aware MLX converter
4. verifying that the output directory is a valid quantized MLX snapshot

Model-family support follows the upstream converter. Today that means the
wrapper is suitable for standard Hugging Face SafeTensors checkpoints and any
special model types that `@mlx-node/cli convert` already understands.

The wrapper uses the upstream Node CLI:

- `@mlx-node/cli`

and specifically its native support for:

- `mlx convert --q-recipe unsloth`
- `--imatrix-path`

### Prerequisites

- `uv sync`
- `node` / `npx` available in `PATH`

You do not need to globally install the converter. The wrapper will fall back
to:

```sh
npx --yes @mlx-node/cli
```

when `mlx` is not already installed locally.

### Example

```sh
uv run python models/text_lm/convert_unsloth_mlx.py \
  --input Qwen/Qwen3.5-9B \
  --output-dir /tmp/qwen3.5-9b-unsloth-mlx \
  --model-type qwen3_5 \
  --imatrix-repo unsloth/Qwen3.5-9B-GGUF \
  --imatrix-file imatrix_unsloth.gguf
```

The script prints:

- the resolved input snapshot path
- the resolved imatrix path
- the exact converter command
- the output directory and conversion manifest path

It also writes:

- `conversion_manifest.json`

inside the output directory.

### Local imatrix Example

```sh
uv run python models/text_lm/convert_unsloth_mlx.py \
  --input /path/to/hf-model \
  --output-dir /tmp/model-unsloth-mlx \
  --imatrix-path /path/to/imatrix_unsloth.gguf
```

### Dry Run

```sh
uv run python models/text_lm/convert_unsloth_mlx.py \
  --input Qwen/Qwen3.5-9B \
  --output-dir /tmp/qwen3.5-9b-unsloth-mlx \
  --model-type qwen3_5 \
  --imatrix-repo unsloth/Qwen3.5-9B-GGUF \
  --dry-run
```

Typical flow:

1. Download or prepare a local MLX snapshot.
2. Export a reusable next-token function with [`export_bundle.py`](export_bundle.py).
3. Run the exported artifact from Dart with
   [`../common/import_run.dart`](../common/import_run.dart).
4. Use `benchmark/` scripts only when you want parity or performance reports.
