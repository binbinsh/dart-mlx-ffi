# dart_mlx_ffi

`dart_mlx_ffi` is a Dart and Flutter FFI package for Apple's
[MLX C API](https://ml-explore.github.io/mlx-c/).

The package vendors `mlx`, `mlx-c`, and the native build pieces needed to
compile a local MLX dynamic library for the current Apple target.

## Highlights

- Stable high-level Dart API for arrays, tensor ops, scans, linalg, FFT,
  quantization, convolutions, streams, runtime helpers, export/import, and
  custom fast-kernel wrappers
- Full raw binding surface through `package:dart_mlx_ffi/raw.dart`
- Native build hooks for Apple MLX on `iOS` and `macOS`
- Canonical MLX snapshot preparation through the repository's Unsloth MLX
  wrapper
- Verified parity against Python MLX on deterministic operator suites
- Publish-time parity coverage for text, VLM, TTS, and ASR checkpoints

## Platform

- `iOS`
- `macOS`

This package targets Apple platforms only.

MLX is most useful on Apple Silicon with Metal available. If the local Xcode
installation does not contain the `MetalToolchain` component, the build hook
falls back to CPU-only MLX so the package still compiles.

To install the Metal shader toolchain on the build machine:

```sh
xcodebuild -downloadComponent MetalToolchain
```

## Installation

```sh
dart pub add dart_mlx_ffi
```

## Public Entry Points

- `package:dart_mlx_ffi/dart_mlx_ffi.dart`: stable MLX tensor/runtime API
- `package:dart_mlx_ffi/models.dart`: stable Dart model runners shipped by this
  repository
- `package:dart_mlx_ffi/raw.dart`: generated low-level `mlx-c` bindings

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

## MLX Snapshot Workflow

This repository uses a canonical MLX conversion wrapper:

- [`models/text_lm/convert_unsloth_mlx.py`](models/text_lm/convert_unsloth_mlx.py)

Use it when you want to:

- prepare a local MLX snapshot from a Hugging Face checkpoint
- standardize publish-time benchmark inputs
- keep local evaluation reproducible across machines

That wrapper produces MLX snapshots that can be used directly by:

- Dart model runners under [`lib/src/models/`](lib/src/models/)
- export/import tooling under [`models/text_lm/`](models/text_lm/)
- publish-time parity scripts under [`benchmark/`](benchmark/)

For `Gemma 4`, the current publish-time text coverage uses the official
Unsloth MLX snapshot `unsloth/gemma-4-E2B-it-UD-MLX-4bit` directly instead of
re-quantizing locally, because Unsloth currently ships `gemma4` model patches
for `mlx-lm` as a separate install step.

## Exporting Text Model Bundles

The repository includes a Python helper for turning an `mlx-lm` snapshot into a
shapeless `.mlxfn` artifact plus matching sample inputs:

- [`models/text_lm/export_bundle.py`](models/text_lm/export_bundle.py)

Example:

```sh
uv sync

uv run python models/text_lm/export_bundle.py \
  --snapshot-dir /path/to/mlx-snapshot \
  --output-dir /path/to/out-bundle
```

Outputs:

- `/path/to/out-bundle/function.mlxfn`
- `/path/to/out-bundle/inputs.safetensors`

The export is shapeless, so the imported function accepts variable-length
`input_ids` tensors.

The generic Dart runner for exported artifacts is:

```sh
dart run models/common/import_run.dart \
  /path/to/out-bundle/function.mlxfn \
  /path/to/out-bundle/inputs.safetensors
```

## Model Workflows

There are three main model-workflow areas in this repository:

- [`lib/src/models/`](lib/src/models/) contains the main stable Dart model
  implementations
- [`models/`](models/) contains reusable non-runtime export and artifact tooling
- [`benchmark/`](benchmark/) contains publish-time parity runners and report
  generation

Current stable Dart model implementations under [`lib/src/models/`](lib/src/models/)
include:

- `parakeet_tdt`
- `qwen2_5`
- `qwen3_5`
- `kitten_tts`
- `shared` helpers

Current publish-time validation under [`benchmark/`](benchmark/) is organized
as a release matrix instead of a grab bag of local experiments.

Recommended prepublish text coverage:

- `unsloth/gemma-4-E2B-it-UD-MLX-4bit`
- `mlx-community/Qwen3.5-27B-4bit-DWQ`
- `mlx-community/translategemma-27b-it-4bit`
- `mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit`
- `mlx-community/GLM-4.7-Flash-4bit`

Recommended prepublish multimodal / speech coverage:

- `mlx-community/MiniCPM-o-4_5-4bit`
- `mlx-community/Gemma-SEA-LION-v4-4B-VL-mlx-3bit`
- `mlx-community/Ming-omni-tts-0.5B-4bit`
- `mlx-community/kitten-tts-nano-0.8-6bit`
- `mlx-community/parakeet-tdt-0.6b-v3`

## Validation

Deterministic operator parity currently covers `114` checks across arithmetic,
tensor ops, scans, convolutions, linalg, fast ops, quantization, and random
APIs, with `0` failures on the benchmark machine.

### Benchmark Environment

- Date: `2026-04-04`
- Machine: `MacBook Pro (Mac16,5)`
- Chip: `Apple M4 Max`
- CPU cores: `16` (`12` performance + `4` efficiency)
- Memory: `128 GB`
- OS: `macOS 26.4 (25E5223i)`
- Kernel: `Darwin 25.4.0`
- Dart SDK: `3.11.1`
- Python: `3.12` via `uv`
- MLX runtime: `0.31.1`

### Latest Runtime Snapshot

Latest measured runtime snapshot on the benchmark machine, refreshed on
`2026-04-04`:

Text models:

| Model | Python MLX ms | Dart MLX ms | Max abs diff |
| --- | ---: | ---: | ---: |
| `gemma-4-E2B-it-UD-MLX-4bit` | `30.47` | `34.30` | `0` |
| `Qwen3.5-27B-4bit-DWQ` | `172.81` | `170.25` | `0` |
| `translategemma-27b-it-4bit` | `166.52` | `170.46` | `0` |
| `NVIDIA-Nemotron-3-Nano-30B-A3B-4bit` | `36.62` | `35.67` | `0` |
| `GLM-4.7-Flash-4bit` | `46.61` | `45.81` | `0` |

Non-text models:

| Model | Kind | Python MLX ms | Dart MLX ms | Max abs diff | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| `MiniCPM-o-4_5-4bit` | `vlm` | `130.82` | `131.58` | `0` | synthetic image + prompt |
| `Gemma-SEA-LION-v4-4B-VL-mlx-3bit` | `vlm` | `718.60` | `756.92` | `0` | synthetic image + prompt |
| `Ming-omni-tts-0.5B-4bit` | `tts` | `4.59` | `4.85` | `0` | deterministic `forward_with_cfg` |
| `kitten-tts-nano-0.8-6bit` | `tts` | `66.25` | `69.20` | `1.19e-07` | full waveform |
| `parakeet-tdt-0.6b-v3` | `asr` | `30.95` | `29.72` | `5.72e-06` | transcript matched |

### What `Max abs diff` Means

`Max abs diff` is the maximum absolute difference between the Python MLX output
and the Dart MLX output for the compared tensor.

Examples:

- `0` means the compared tensor matched exactly at the chosen dtype
- `7.62939453125e-06` means the worst element differed by about `0.00000763`
- for text and VLM rows, the compared tensor is the final-token `logits[:16]`
- for `parakeet-tdt-0.6b-v3`, the compared tensor is the first-step
  `token_logits[:16] + duration_logits`
- for `Ming-omni-tts-0.5B-4bit`, the compared tensor is the deterministic
  `forward_with_cfg` subgraph output
- for `kitten-tts-nano-0.8-6bit`, the compared tensor is the full waveform

### Reproduce The Release Matrix Report

Generate the publish-time report with `warmup=3` and `iters=10`:

```sh
uv sync
HF_HUB_DISABLE_XET=1 uv run --no-project --with mlx-lm --with pillow --with mlx-vlm --with mlx-audio --with parakeet-mlx python benchmark/publish_report.py
```

The aggregated results are written to:

- `benchmark/out/publish_report.json`

Useful focused runs:

```sh
# full-waveform KittenTTS comparison
uv run python benchmark/kitten_tts/mlx_audio_compare.py --warmup 3 --iters 10

# fixed-mel Parakeet TDT comparison
uv run --no-project --with parakeet-mlx --with numpy python - <<'PY'
from benchmark.parakeet_tdt_sweep import asr_bench
import json
print(json.dumps(asr_bench('mlx-community/parakeet-tdt-0.6b-v3', warmup=1, iters=1), indent=2))
PY
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
