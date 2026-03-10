# dart_mlx_ffi

`dart_mlx_ffi` is a Dart and Flutter FFI package for Apple's [MLX C API](https://ml-explore.github.io/mlx-c/). It vendors `mlx`, `mlx-c`, and the required native build pieces, then uses a build hook to compile a local native library for the current Apple target.

## Highlights

- Stable high-level Dart API for arrays, tensor ops, scans, linalg, FFT, quantization, convolutions, streams, runtime helpers, export/import, and custom fast-kernel wrappers
- Full raw binding surface through `package:dart_mlx_ffi/raw.dart`
- macOS packaging with native MLX build hooks included
- Verified parity against Python MLX on deterministic operator suites
- Verified model parity on 13 MLX checkpoints and subgraphs for publish-time validation

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

Results:

| Model | Kind | Python avg ms | Dart avg ms | Max abs diff |
| --- | --- | ---: | ---: | ---: |
| `JOSIE-1.1-4B-Instruct-4bit` | `text` | `33.5777` | `38.2766` | `0` |
| `GLM-4.7-Flash-abliterated-mxfp8` | `text` | `78.2033` | `77.7014` | `0` |
| `tiny-aya-fire-4bit` | `text` | `25.6706` | `26.2876` | `0` |
| `Huihui-Qwen3.5-27B-abliterated-6bit` | `text` | `176.6442` | `182.2428` | `0` |
| `IQuest-Coder-V1-7B-Thinking-mlx_8bit` | `text` | `46.7368` | `47.1135` | `0` |
| `DASD-4B-Thinking-bfloat16` | `text` | `43.8261` | `43.6674` | `0` |
| `Qwen3.5-9B-MLX-4bit` | `text` | `59.8508` | `61.8239` | `0` |
| `Qwen3.5-35B-A3B-4bit` | `text` | `34.7274` | `34.2320` | `0` |
| `MiniCPM-o-4_5-4bit` | `vlm` | `135.0798` | `135.5181` | `0` |
| `Gemma-SEA-LION-v4-4B-VL-mlx-3bit` | `vlm` | `22.2738` | `22.2928` | `0` |
| `Qwen3.5-0.8B-4bit` | `vlm` | `5.2338` | `5.5181` | `0` |
| `Ming-omni-tts-0.5B-4bit` | `tts` | `4.5822` | `4.5606` | `0` |
| `kitten-tts-nano-0.8-6bit` | `tts` | `75.3437` | `79.6089` | `3.394e-06` |

Summary:

- `12 / 13` model checks were exact at the chosen comparison tensor.
- `kitten-tts-nano-0.8-6bit` matched shape and fixed-input waveform closely, with residual float32-scale drift on the full waveform path (`max_abs_diff = 3.393739462e-06`).
- For large text and multimodal checkpoints, Dart imported-function execution is generally in the same performance class as Python MLX once startup costs are excluded.
- `Ming-omni-tts-0.5B-4bit` is notably warmup-sensitive, so the published row above uses extended warmup instead of the repository-wide default `warmup=3`. With deeper warmup (`20+`), the imported MLX subgraph settles near Python's imported-function timing (`~4.6 ms` on the benchmark machine).

### What `Max abs diff` Means

`Max abs diff` is the maximum absolute difference between the Python MLX output and the Dart MLX output for the compared tensor.

Examples:

- `0` means the compared tensor matched exactly at the chosen dtype
- `3.393739462e-06` means the worst element differed by about `0.00000339`
- for text and VLM rows, the compared tensor is the final-token `logits[:16]`
- for `Ming-omni-tts-0.5B-4bit`, the compared tensor is the deterministic `forward_with_cfg` subgraph output
- for `kitten-tts-nano-0.8-6bit`, the compared tensor is the full waveform

### Reproduce The 13-Model Report

Generate the publish-time report with `warmup=3` and `iters=10`:

```sh
uv sync
uv run python benchmark/publish_report.py
```

The aggregated results are written to:

- `benchmark/out/publish_report.json`

Useful focused runs:

```sh
# full-waveform KittenTTS comparison
uv run python benchmark/kitten_tts/mlx_audio_compare.py --warmup 3 --iters 10

# deterministic Ming Omni TTS subgraph comparison
uv run python - <<'PY'
from benchmark.recent_tts_sweep import python_forward, export_model, dart_forward, slug
from pathlib import Path
model_id='mlx-community/Ming-omni-tts-0.5B-4bit'
root = Path('benchmark/out/recent_tts') / slug(model_id)
export_path, input_path, input_names = export_model(model_id, root)
print(dart_forward(export_path, input_path, input_names))
PY

# generic imported-function runner for exported text or VLM artifacts
dart run benchmark/generic_import_run.dart <export_path> <input_safetensors_path> [input_names_json]
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
