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
3. run the exported artifact from Dart through `RuntimeEngine.mlx`, which now
   enters the Dart FFI/native MLX imported-function path
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
  --output-dir /tmp/qwen3.5-9b-bundle \
  --bundle-id qwen3_5_9b \
  --bundle-name "Qwen3.5 9B" \
  --source-model-id Qwen/Qwen3.5-9B
```

The exporter writes `function.mlxfn`, `inputs.safetensors`, `inputs.json`, and
`mlx_bundle.json`. Use `inputs.json` with
`benchmark/runtime/dart_runtime_runner.dart` when running the bundle through
the Dart-facing native runtime ABI. SuperPlanner discovers `mlx_bundle.json`
from its app models directory and can run the exported sample inputs through
`dart_inference`'s MLX FFI path.

### SuperPlanner Qwen3.6 bundle

Qwen3.6 27B is registered in the package manifest as `qwen3_6_27b`, with the
Unsloth MLX artifact `unsloth/Qwen3.6-27B-UD-MLX-4bit`. The current public artifact is a
vision-language MLX model and should be exported with VLM-aware inputs. For a
text-only `mlx-lm` snapshot, use this exporter with the same bundle id:

```sh
uv run python models/text_lm/export_bundle.py \
  --snapshot-dir /path/to/qwen3.6-27b-text-mlx \
  --output-dir "$HOME/Library/Application Support/SuperPlanner/models/qwen3_6_27b" \
  --bundle-id qwen3_6_27b \
  --bundle-name "Qwen3.6 27B" \
  --bundle-kind text \
  --source-model-id unsloth/Qwen3.6-27B-UD-MLX-4bit \
  --context-length 262144 \
  --description "Qwen3.6 27B local MLX bundle"
```

If you use `unsloth/Qwen3.6-27B-UD-MLX-4bit` directly, follow the VLM benchmark
export path so the bundle includes the image tensors expected by the model.
`benchmark/vlm_export_sweep.py` writes the same `mlx_bundle.json` schema and
uses `qwen3_6_27b` automatically for that MLX artifact.
