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
- Cross-platform model runtime entry point for Core ML, ONNX Runtime, and
  LiteRT staging artifacts
- Canonical MLX snapshot preparation through the repository's Unsloth MLX
  wrapper
- Verified parity against Python MLX on deterministic operator suites
- Publish-time parity coverage for text, VLM, and TTS checkpoints

## Platform

- `iOS`
- `macOS`
- `Windows`
- `Linux`
- `Android`

The MLX tensor API targets Apple platforms. The model-level runtime API in
`package:dart_mlx_ffi/runtime.dart` adds staging support for Core ML on Apple
platforms, ONNX Runtime on desktop/server targets, and LiteRT/ONNX paths on
Android.

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
- `package:dart_mlx_ffi/runtime.dart`: cross-platform model runtime API
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
MLX snapshot `mlx-community/gemma-4-e4b-it-4bit` directly instead of
re-quantizing locally, because it already ships the current Gemma 4 E4B MLX
layout and tokenizer sidecars needed by the runtime matrix.

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

- `paddle_ocr_vl`
- `qwen2_5`
- `qwen3_5`
- `qwen3_asr`
- `kitten_tts`
- `silero_vad`
- `shared` helpers

Current runtime migration status:

| Model | Runtime status |
| --- | --- |
| `qwen2_5` | staging, artifact coverage for iOS/macOS/Windows/Linux/Android |
| `qwen3_5` | staging, artifact coverage for iOS/macOS/Windows/Linux/Android |
| `paddle_ocr_vl` | staging, iOS/macOS resolve through converted Core ML pipeline (ANE compute-plan audited), Windows/Linux resolve through the same-model ONNX pipeline, Android uses ONNX fallback, and Android LiteRT remains an engine-level blocker (`onnx_invalid_subgraph_constant_binding`, conversion report/log tracked) |
| `qwen3_asr` | staging, upgraded to Qwen3-ASR 1.7B; iOS/macOS now resolve the same-model Core ML component bundle through `Qwen3AsrCoreMlRunner` with tokenizer sidecars pulled from the ONNX tokenizer repo, Windows/Linux use the same-model ONNX int4 component bundle through `Qwen3AsrNativeRunner`, Android uses that ONNX bundle with `NnapiExecutionProvider` appended on-device, iOS and Android device load smoke have passing evidence with peak memory, Apple Core ML still fails default ANE compilation, and Android LiteRT is blocked on decoder chunking because expanding ORT int4 `MatMulNBits` exceeds TensorFlow/TFLite single-graph protobuf limits |
| `kitten_tts` | staging, upgraded to KittenTTS mini 0.8; iOS/macOS use same-model MLX/Core ML, Windows/Linux use same-model ONNX, and Android currently resolves through ONNX fallback while ONNX→LiteRT conversion is tracked as an engine blocker (`onnx2tf_attempt_timeout`) |
| `silero_vad` | staging, iOS/macOS/Windows/Linux artifacts are ready; Android now resolves through converted LiteRT (`benchmark/artifacts_local/converted/silero_vad/litert/model.tflite`) with XNNPACK; required NNAPI load is recorded as failing, so Android production stays on the validated XNNPACK path |
| `qwen3_vl` | staging, artifact coverage for iOS/macOS/Windows/Linux/Android |
| `gemma4` | staging, artifact coverage for iOS/macOS/Windows/Linux/Android |
| `function_gemma` | staging, artifact coverage for iOS/macOS/Windows/Linux/Android |
| `embedding_gemma` | staging, artifact coverage for iOS/macOS/Windows/Linux/Android |
| `qwen3_6_27b` | staging, Apple MLX artifact only; Windows/Linux ONNX Runtime GenAI and Android LiteRT patched conversion paths are implemented and config probes pass through `https://hf-mirror.com`, but no validated 27B weight artifact has been exported yet |
| `translategemma_4b_it` | staging, artifact coverage for iOS/macOS/Windows/Linux/Android; 4B is the supported TranslateGemma runtime target, not 27B |
| `minicpm_o_4_5` | staging, Apple MLX artifact only; the 2026-04-26 HF scan found only token2wav/CosyVoice ONNX components plus a GGUF repo vision-only Core ML sidecar, not a full ONNX/LiteRT runtime artifact |
| `gemma_sea_lion_v4_4b_vl` | staging, Apple MLX artifact only; Windows/Linux ONNX conversion is blocked by Optimum task support (`export_task_unsupported`), and Android LiteRT conversion now records a real `conversion_timeout` blocker at `benchmark/artifacts_local/converted/gemma_sea_lion_v4_4b_vl/litert/conversion.log` |
| `ming_omni_tts_0_5b` | staging, Apple MLX artifact only; the 2026-04-26 HF scan still finds only `campplus.onnx`, and the patched-source exporter remains blocked on full TTS architecture support (`BailingMMConfig`) |

The runtime artifact catalog is checked with:

```sh
uv run python benchmark/runtime/resolve_hf_artifacts.py --dry-run
```

As of `2026-04-26`, the merged runtime map
(`benchmark/runtime/artifacts.converted.yaml`) plans `75` platform cells:
`47` preferred-engine cells, `13` fallback-engine cells, `0` missing cells, and
`12` explicitly blocked cells. Blocked cells stay in `staging` until a directly
loadable Core ML, ONNX Runtime, or LiteRT artifact is found or a converted
artifact passes the full runtime matrix.

As of `2026-04-25`, `paddle_ocr_vl` also has a local merged runtime map at
`benchmark/runtime/artifacts.paddle_ocr_vl.full.yaml`:
iOS/macOS use converted Core ML (`benchmark/artifacts_local/converted/paddle_ocr_vl/coreml/pipeline.json`),
Windows/Linux/Android use the Hugging Face ONNX pipeline, and the Android
LiteRT conversion failure is recorded as an `onnx_invalid_subgraph_constant_binding`
engine blocker with report/log paths instead of blocking ONNX fallback execution.

As of `2026-04-26`, Qwen3.6 27B now has two real conversion entry points:
Windows/Linux use the ONNX Runtime GenAI builder with a `qwen3_5` -> `qwen3`
config adapter, and Android LiteRT uses the same patched HF source preparation.
Both config-only probes pass through `https://hf-mirror.com`; the remaining
work is the full 27B weight export plus `artifact_health.py` and runtime-matrix
validation.

As of `2026-04-26`, `benchmark/runtime/artifacts.converted.yaml` also records
conversion-attempt blocker evidence for:
`paddle_ocr_vl` (LiteRT ONNX subgraph constant binding blocker),
`kitten_tts` (ONNX→LiteRT `onnx2tf` attempt timeout),
`qwen3_asr` (ONNX model-level ASR runner is wired; Core ML stateful runner is
wired and now resolves tokenizer sidecars from the ONNX tokenizer repo; iOS and
macOS Core ML health pass with `cpuAndGPU` but default ANE compilation is still
a promotion blocker; same-model ONNX to LiteRT component conversion is wired and
still needs the decoder graph-size blocker resolved plus artifact_health and
Android matrix evidence),
`qwen3_6_27b` (ONNX Runtime GenAI/LiteRT conversion pending full weight export),
`gemma_sea_lion_v4_4b_vl` (ONNX task unsupported + LiteRT conversion timeout),
and `ming_omni_tts_0_5b` (LiteRT conversion timeout; ONNX remains component-only
until a full TTS export path is available).

`convert_artifacts.py` also imports existing
`benchmark/artifacts_local/converted/<model>/<engine>/conversion_record.json`
entries before writing `artifacts.converted.yaml`, so successful local
conversions (for example `paddle_ocr_vl/coreml` and `silero_vad/litert`) are
persisted without re-running the exporter.

As of `2026-04-25`, `silero_vad` runtime-smoke also passed on both connected
iPhone and Android devices:
`benchmark/out/runtime/silero_vad/ios/device_smoke_runtime_coreml_latest.json`
and
`benchmark/out/runtime/silero_vad/android/device_smoke_litert_localpush.json`.

For a broader current Hugging Face search over blocked models, run:

```sh
uv run python benchmark/runtime/search_hf_artifacts.py \
  --fallback-endpoint https://hf-mirror.com \
  --model-id qwen3_6_27b \
  --model-id minicpm_o_4_5 \
  --model-id gemma_sea_lion_v4_4b_vl \
  --model-id ming_omni_tts_0_5b \
  --out benchmark/out_local/runtime/hf_search_blocked_latest.json
```

The latest local scan is recorded at
`benchmark/out_local/runtime/hf_search_blocked_latest.json`: at
`2026-04-26T14:19:07Z` it found `0` complete runtime candidates and `13`
component-only candidates for the four remaining blocked model families.
The 2026-04-26 scan uses
`translategemma_4b_it` as the TranslateGemma target, not 27B. 4B has complete
Core ML, ONNX, LiteRT, and MLX artifact coverage; the remaining blocked cells
are for other model families. Component sidecars for MiniCPM-o 4.5 and
Ming-omni TTS are intentionally not exposed as platform runtime artifacts.

Current publish-time validation under [`benchmark/`](benchmark/) is organized
as a release matrix instead of a grab bag of local experiments.

Recommended prepublish text coverage:

- `mlx-community/gemma-4-e4b-it-4bit`
- `mlx-community/Qwen3-ASR-1.7B-8bit`
- `mlx-community/Qwen3.6-27B-4bit`
- `mlx-community/translategemma-4b-it-4bit`

Recommended prepublish multimodal / audio coverage:

- `mlx-community/MiniCPM-o-4_5-4bit`
- `mlx-community/Gemma-SEA-LION-v4-4B-VL-mlx-3bit`
- `mlx-community/PaddleOCR-VL-1.5-8bit`
- `mlx-community/Ming-omni-tts-0.5B-4bit`
- `mlx-community/kitten-tts-mini-0.8-8bit`

### PaddleOCR-VL Reference Workflow

For `PaddleOCR-VL-1.5`, regenerate local parity fixtures with:

```sh
uv run --no-project --with mlx-vlm --with pillow \
  python tool/dump_paddle_v15_reference.py
```

That tool now writes a canonical upstream reference:

- `full_output.txt` and `py_generated_ids.txt` come from the same fresh
  `mlx_vlm.generate.stream_generate()` run
- generation uses the same `min_pixels` / `max_pixels` bounds that were used
  to build the saved image tensors
- by default it uses the repo-local image
  `benchmark/assets/paddle_ocr_vl_test.jpg`, so the workflow works from a
  clean checkout without a sibling workspace
- `metadata.json` records the generation source and the effective resize bounds

If `test/paddle_ocr_v15_parity_test.dart` fails on metadata assertions, refresh
`/tmp/paddle_v15_ref` with that tool before trusting token mismatches.

## Validation

Deterministic operator parity currently covers `114` checks across arithmetic,
tensor ops, scans, convolutions, linalg, fast ops, quantization, and random
APIs, with `0` failures on the benchmark machine.

### Benchmark Environment

- Date: `2026-04-16`
- Machine: `MacBook Pro (Mac16,5)`
- Chip: `Apple M4 Max`
- CPU cores: `16` (`12` performance + `4` efficiency)
- Memory: `128 GB`
- OS: `macOS 26.4.1 (25E253)`
- Kernel: `Darwin 25.4.0`
- Dart SDK: `3.11.4`
- Python: `3.12.9` via `uv`
- MLX runtime: `0.31.1`

### Latest Runtime Snapshot

Latest measured runtime snapshot on the benchmark machine, refreshed on
`2026-04-16`:

Text models:

| Model | Python MLX ms | Dart MLX ms | Max abs diff |
| --- | ---: | ---: | ---: |
| `gemma-4-e4b-it-4bit` | pending refresh | pending refresh | pending refresh |
| `Qwen3.6-27B-4bit` | pending refresh | pending refresh | pending refresh |

Non-text models:

| Model | Kind | Python MLX ms | Dart MLX ms | Max abs diff | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| `MiniCPM-o-4_5-4bit` | `vlm` | `216.37` | `171.69` | `0` | synthetic image + prompt |
| `Gemma-SEA-LION-v4-4B-VL-mlx-3bit` | `vlm` | `1151.80` | `1187.94` | `0` | synthetic image + prompt |
| `PaddleOCR-VL-1.5-8bit` | `vlm` | `1056.92` | `1019.21` | `0.08848` | `benchmark/assets/paddle_ocr_vl_test.jpg` + prompt, processor-default resize bounds `(1316x728, min/max_pixels=112896/1003520)`, direct runner `warmup=3/iters=3` |
| `Ming-omni-tts-0.5B-4bit` | `tts` | `4.98` | `4.78` | `0` | deterministic `forward_with_cfg` |
| `kitten-tts-mini-0.8-8bit` | `tts` | pending refresh | pending refresh | pending refresh | full waveform |

### What `Max abs diff` Means

`Max abs diff` is the maximum absolute difference between the Python MLX output
and the Dart MLX output for the compared tensor.

Examples:

- `0` means the compared tensor matched exactly at the chosen dtype
- `7.62939453125e-06` means the worst element differed by about `0.00000763`
- for text and VLM rows, the compared tensor is the final-token `logits[:16]`
- for `Ming-omni-tts-0.5B-4bit`, the compared tensor is the deterministic
  `forward_with_cfg` subgraph output
- for `kitten-tts-mini-0.8-8bit`, the compared tensor is the full waveform

### Reproduce The Release Matrix Report

Generate the publish-time report with `warmup=3` and `iters=10`:

```sh
uv sync
HF_HUB_DISABLE_XET=1 uv run --no-project --with mlx-lm --with pillow --with mlx-vlm --with mlx-audio --with parakeet-mlx python benchmark/publish_report.py
```

`PaddleOCR-VL-1.5-8bit` is the one exception in the current matrix: its
publish-time benchmark uses the direct Dart runner path and internally clamps
to `warmup=3` / `iters=3` so the release report can complete reliably while
still letting the non-iOS vision caches settle before the timed pass.

When you want a case-specific `PaddleOCR-VL` benchmark instead of the default
processor bounds, `benchmark/paddle_ocr_vl/python_ref.py` also accepts
`--min-pixels` and `--max-pixels` so the Python payload can match a deployment
resize target such as the iPhone `501760` path.

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

- The MLX tensor/FFI layer targets Apple platforms only.
- The model-level runtime API (`RuntimeEngine.coreml/onnx/litert`) is
  cross-platform and intended for iOS, macOS, Windows, Linux, and Android
  staging/production matrix workflows.
- The raw layer remains the escape hatch for the full MLX C surface.
