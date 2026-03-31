# dart_mlx_ffi

`dart_mlx_ffi` is a Dart and Flutter FFI package for Apple's [MLX C API](https://ml-explore.github.io/mlx-c/). It vendors `mlx`, `mlx-c`, and the required native build pieces, then uses a build hook to compile a local native library for the current Apple target.

## Highlights

- Stable high-level Dart API for arrays, tensor ops, scans, linalg, FFT, quantization, convolutions, streams, runtime helpers, export/import, and custom fast-kernel wrappers
- Full raw binding surface through `package:dart_mlx_ffi/raw.dart`
- macOS packaging with native MLX build hooks included
- Verified parity against Python MLX on deterministic operator suites
- Verified model parity on 14 MLX checkpoints and subgraphs for publish-time validation

## Platform

- `macOS`

This repository's published benchmark numbers are from `macOS` on Apple Silicon, and the package metadata now only advertises macOS support.

MLX is most useful on Apple Silicon with Metal available. If the local Xcode installation does not contain the `MetalToolchain` component, the build hook falls back to CPU-only MLX so the package still compiles.

To install the Metal shader toolchain on the build machine:

```sh
xcodebuild -downloadComponent MetalToolchain
```

## Installation

```sh
dart pub add dart_mlx_ffi
```

## Model Workflows

There are three main model-workflow areas in this repository:

- [`lib/src/models/`](lib/src/models/) contains the main stable Dart model implementations
- [`models/`](models/) contains reusable non-runtime export and artifact tooling
- [`benchmark/`](benchmark/) contains publish-time parity runners, benchmark orchestration, and report-generation scripts

Current stable Dart model implementations under [`lib/src/models/`](lib/src/models/) include:

- `parakeet_tdt`
- `qwen2_5`
- `qwen3_5`
- `kitten_tts`
- `shared` helpers

Current publish-time model parity coverage under [`benchmark/`](benchmark/) includes:

- `text`: `JOSIE-1.1-4B-Instruct-4bit`, `GLM-4.7-Flash-abliterated-mxfp8`, `tiny-aya-fire-4bit`, `Huihui-Qwen3.5-27B-abliterated-6bit`, `IQuest-Coder-V1-7B-Thinking-mlx_8bit`, `DASD-4B-Thinking-bfloat16`, `Qwen3.5-9B-MLX-4bit`, `Qwen3.5-35B-A3B-4bit`
- `vlm`: `MiniCPM-o-4_5-4bit`, `Gemma-SEA-LION-v4-4B-VL-mlx-3bit`, `Qwen3.5-0.8B-4bit`
- `tts`: `Ming-omni-tts-0.5B-4bit`, `kitten-tts-nano-0.8-6bit`
- `asr`: `parakeet-tdt-0.6b-v3`

Top-level [`models/`](models/) is reserved for reusable non-runtime tooling such as:

- [`models/common/`](models/common/) generic import / execution helpers
- [`models/text_lm/`](models/text_lm/) text export helpers and Hugging Face ->
  Unsloth MLX conversion wrappers

## Convert Hugging Face Models To Unsloth MLX

The repository now includes a complete wrapper script for converting a
standard Hugging Face SafeTensors checkpoint into an Unsloth-optimized MLX
snapshot:

- [`models/text_lm/convert_unsloth_mlx.py`](models/text_lm/convert_unsloth_mlx.py)

The wrapper:

- accepts either a local model directory or a Hugging Face model id
- resolves a local or Hub-hosted `imatrix` GGUF file
- invokes the upstream Unsloth-aware MLX converter
- verifies the output snapshot and writes a conversion manifest

Model-family support follows the upstream converter. In practice, this means
standard Hugging Face SafeTensors checkpoints plus any explicit model types
that `@mlx-node/cli convert` already supports.

It uses the upstream CLI implementation from `@mlx-node/cli`, which currently
exposes:

- `mlx convert --q-recipe unsloth`
- `--imatrix-path`

### Prerequisites

```sh
uv sync
```

You also need `node` / `npx` available in `PATH`. The wrapper will use an
installed `mlx` binary when present, otherwise it falls back to:

```sh
npx --yes @mlx-node/cli
```

### Example

```sh
uv run python models/text_lm/convert_unsloth_mlx.py \
  --input Qwen/Qwen3.5-9B \
  --output-dir /tmp/qwen3.5-9b-unsloth-mlx \
  --model-type qwen3_5 \
  --imatrix-repo unsloth/Qwen3.5-9B-GGUF \
  --imatrix-file imatrix_unsloth.gguf
```

This produces a quantized MLX snapshot in `/tmp/qwen3.5-9b-unsloth-mlx` plus
`conversion_manifest.json`.

## Exporting Text Model Bundles

The repository includes a Python helper for turning an `mlx-lm` text snapshot
into a shapeless `.mlxfn` artifact plus matching sample inputs:

- [`models/text_lm/export_bundle.py`](models/text_lm/export_bundle.py)

This is useful when you want to:

- download a model from Hugging Face as a normal MLX snapshot
- export a reusable next-token forward function once
- run the exported artifact from Dart with `MlxExport.importFunction(...)`

### Prepare Python Tooling

```sh
uv sync
```

### Export A Local Snapshot

```sh
uv run python models/text_lm/export_bundle.py \
  --snapshot-dir /path/to/mlx-snapshot \
  --output-dir /path/to/out-bundle
```

Outputs:

- `/path/to/out-bundle/function.mlxfn`
- `/path/to/out-bundle/inputs.safetensors`

The export is shapeless, so the imported function accepts variable-length
`input_ids` tensors.

By default, the helper uses a neutral sample prompt only to create example
`input_ids` for export. It does not lock you into any runtime chat template or
task-specific prompt format.

If you want the example input to match a specific prompt style, pass custom
text directly:

```sh
uv run python models/text_lm/export_bundle.py \
  --snapshot-dir /path/to/mlx-snapshot \
  --output-dir /path/to/out-bundle \
  --sample-prompt "Summarize why MLX matters."
```

Or load the sample prompt from a file:

```sh
uv run python models/text_lm/export_bundle.py \
  --snapshot-dir /path/to/mlx-snapshot \
  --output-dir /path/to/out-bundle \
  --sample-prompt-file /path/to/prompt.txt
```

### Example: TranslateGemma

If you already have `mlx-community/translategemma-4b-it-4bit` downloaded as a
local MLX snapshot, export it like this:

```sh
uv run python models/text_lm/export_bundle.py \
  --snapshot-dir /path/to/translategemma-4b-it-4bit \
  --output-dir /tmp/translategemma-bundle \
  --sample-prompt-file /path/to/translategemma_prompt.txt
```

### Run The Exported Bundle In Dart

```dart
import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

final imported = MlxExport.importFunction('/tmp/translategemma-bundle/function.mlxfn');
final inputIds = MlxArray.fromInt32List([2, 106, 1234], shape: [1, 3]);

try {
  final outputs = imported.call([inputIds]);
  try {
    final logits = outputs.first;
    MlxRuntime.evalAll([logits]);
    print(logits.shape);
  } finally {
    for (final output in outputs) {
      output.close();
    }
  }
} finally {
  inputIds.close();
  imported.close();
}
```

### Compare Against Python

The repository also includes a generic imported-function runner:

```sh
dart run models/common/import_run.dart \
  /tmp/translategemma-bundle/function.mlxfn \
  /tmp/translategemma-bundle/inputs.safetensors
```

## Quick Start

```dart
import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

final a = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [2, 2]);
final b = MlxArray.fromFloat32List([5, 6, 7, 8], shape: [2, 2]);
final c = mx.matmul(a, b);
final s = c.sum();

print(MlxVersion.current());
print(MlxDevice.defaultDevice());
print(c.toList());
print(s.toList());

s.close();
c.close();
b.close();
a.close();
```

## API Coverage

Stable high-level coverage includes:

- arrays and constructors
- shape transforms and indexing helpers
- unary, binary, comparison, and reduction ops
- scans and matrix construction helpers
- random generation
- FFT and linear algebra
- quantization helpers
- convolution and transpose convolution
- file, bytes, and safetensors IO
- streams, devices, memory, and Metal helpers
- function export/import and transform wrappers

For exact C-level coverage, use:

```dart
import 'package:dart_mlx_ffi/raw.dart' as raw;
```

## Validation

Deterministic operator parity currently covers `114` checks across arithmetic, tensor ops, scans, convolutions, linalg, fast ops, quantization, and random APIs, with `0` failures on the benchmark machine.

### Benchmark Environment

- Date: `2026-03-10`
- Machine: `MacBook Pro (Mac16,5)`
- Chip: `Apple M4 Max`
- CPU cores: `16` (`12` performance + `4` efficiency)
- Memory: `128 GB`
- OS: `macOS 26.4 (25E5223i)`
- Kernel: `Darwin 25.4.0`
- Dart SDK: `3.11.1`
- Python: `3.12` via `uv`
- MLX runtime: `0.31.0`

### Model Parity And Speed

Method:

- Each benchmark used `3` warmup iterations and `10` measured iterations.
- Text and VLM checkpoints compare the last-token `logits[:16]`.
- `Ming-omni-tts-0.5B-4bit` compares a deterministic `forward_with_cfg` diffusion subgraph instead of full waveform generation.
- `kitten-tts-nano-0.8-6bit` compares the full waveform for one fixed text and voice.
- `parakeet-tdt-0.6b-v3` compares the first-step `token_logits[:16] + duration_logits` on a fixed mel extracted from `000000.flac`.

Results:

| Model | Kind | Python avg ms | Dart avg ms | Max abs diff |
| --- | --- | ---: | ---: | ---: |
| `JOSIE-1.1-4B-Instruct-4bit` | `text` | `45.2020` | `44.1369` | `0` |
| `GLM-4.7-Flash-abliterated-mxfp8` | `text` | `92.5673` | `82.0703` | `0` |
| `tiny-aya-fire-4bit` | `text` | `27.3209` | `27.5509` | `0` |
| `Huihui-Qwen3.5-27B-abliterated-6bit` | `text` | `189.2214` | `193.7680` | `0` |
| `IQuest-Coder-V1-7B-Thinking-mlx_8bit` | `text` | `49.8725` | `50.2037` | `0` |
| `DASD-4B-Thinking-bfloat16` | `text` | `45.9885` | `45.4372` | `0` |
| `Qwen3.5-9B-MLX-4bit` | `text` | `63.6051` | `65.1154` | `0` |
| `Qwen3.5-35B-A3B-4bit` | `text` | `36.5219` | `36.3253` | `0` |
| `MiniCPM-o-4_5-4bit` | `vlm` | `140.6685` | `141.9633` | `0` |
| `Gemma-SEA-LION-v4-4B-VL-mlx-3bit` | `vlm` | `25.4516` | `30.4278` | `0` |
| `Qwen3.5-0.8B-4bit` | `vlm` | `5.4574` | `5.1285` | `0` |
| `Ming-omni-tts-0.5B-4bit` | `tts` | `5.1497` | `4.9890` | `0` |
| `kitten-tts-nano-0.8-6bit` | `tts` | `75.4205` | `77.4532` | `8.941e-08` |
| `parakeet-tdt-0.6b-v3` | `asr` | `31.6649` | `30.9405` | `7.629e-06` |

Summary:

- `12 / 14` model checks were exact at the chosen comparison tensor.
- `parakeet-tdt-0.6b-v3` matched transcript text and stayed within float32-scale residual drift on the compared first-step logits (`max_abs_diff = 7.62939453125e-06`).
- `kitten-tts-nano-0.8-6bit` matched shape and fixed-input waveform closely, with residual float32-scale drift on the full waveform path (`max_abs_diff = 1.1920928955078125e-07`).
- For large text and multimodal checkpoints, Dart imported-function execution is generally in the same performance class as Python MLX once startup costs are excluded.
- `Ming-omni-tts-0.5B-4bit` is now back in the same performance class as the Python reference after the imported-function fast path and benchmark-policy fixes.
- `MiniCPM-o-4_5-4bit` is now back in the same performance class as the Python reference after clearing MLX cache pressure between the Python and Dart phases of the benchmark.

### What `Max abs diff` Means

`Max abs diff` is the maximum absolute difference between the Python MLX output and the Dart MLX output for the compared tensor.

Examples:

- `0` means the compared tensor matched exactly at the chosen dtype
- `7.62939453125e-06` means the worst element differed by about `0.00000763`
- for text and VLM rows, the compared tensor is the final-token `logits[:16]`
- for `parakeet-tdt-0.6b-v3`, the compared tensor is the first-step `token_logits[:16] + duration_logits`
- for `Ming-omni-tts-0.5B-4bit`, the compared tensor is the deterministic `forward_with_cfg` subgraph output
- for `kitten-tts-nano-0.8-6bit`, the compared tensor is the full waveform

### Reproduce The 14-Model Report

Generate the publish-time report with `warmup=3` and `iters=10`:

```sh
uv sync
uv run --no-project --with mlx-lm --with pillow --with mlx-vlm --with mlx-audio --with parakeet-mlx python benchmark/publish_report.py
```

The aggregated results are written to:

- `benchmark/out/publish_report.json`

Useful focused runs:

```sh
# full-waveform KittenTTS comparison
uv run python benchmark/kitten_tts/mlx_audio_compare.py --warmup 3 --iters 10

# real-weight KittenTTS conv_post probe
uv run python benchmark/kitten_tts/conv_post_probe.py

# deterministic Ming Omni TTS subgraph comparison
uv run python - <<'PY'
from benchmark.tts_export_sweep import python_forward, export_model, dart_forward, slug
from pathlib import Path
model_id='mlx-community/Ming-omni-tts-0.5B-4bit'
root = Path('benchmark/out/publish') / slug(model_id)
export_path, input_path, input_names = export_model(model_id, root)
print(dart_forward(export_path, input_path, input_names))
PY

# fixed-mel Parakeet TDT comparison
uv run --no-project --with parakeet-mlx --with numpy python - <<'PY'
from benchmark.parakeet_tdt_sweep import asr_bench
import json
print(json.dumps(asr_bench('mlx-community/parakeet-tdt-0.6b-v3', warmup=1, iters=1), indent=2))
PY

# generic imported-function runner for exported text or VLM artifacts
dart run models/common/import_run.dart <export_path> <input_safetensors_path> [input_names_json]
```

## Development

Regenerate the raw bindings:

```sh
dart run ffigen --config ffigen.yaml
```

Typical local verification:

```sh
dart analyze
dart test
dart pub publish --dry-run
```

Benchmark tooling uses `uv`:

```sh
uv sync
```

## Notes

- This package targets Apple platforms only.
- The raw layer remains the escape hatch for the full MLX C surface.

## Private ANE

Private-ANE-specific work lives under [`private_ane/`](private_ane/). It is experimental, uses private Apple APIs, and is not part of the stable package surface described above.

If you need that path, start with:

- [`private_ane/README.md`](private_ane/README.md)
- [`private_ane/models/`](private_ane/models/)
- [`private_ane/shared/`](private_ane/shared/)
