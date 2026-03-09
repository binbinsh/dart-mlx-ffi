# Benchmark Report

This document records the current parity and performance checks for
`dart-mlx-ffi` against Python MLX on the same machine.

## Environment

- Date: 2026-03-09
- Device: `Device(gpu, 0)`
- Dart runtime MLX: `0.31.1`
- Python runtime MLX: `0.31.0`
- Python dependency manager: `uv`

The Python wheel for `mlx==0.31.1` was not available during this run, so the
Python baseline remained on `0.31.0`.

## Summary

- Low-level MLX parity is good.
  The three synthetic model-style workloads below matched at float32 rounding
  level and Dart was faster than Python for those small direct-op benchmarks.
- Direct Hugging Face MLX model loading is now possible from Dart for Qwen2.5
  4-bit safetensors without exporting `.mlxfn` artifacts first.
- The first direct Hugging Face implementation was correct enough but too slow.
- A compiled-graph Qwen2.5 runner now closes most of the speed gap with
  `mlx_lm` for `0.5B`, `1.5B`, and `3B`.

## Synthetic Benchmarks

Workloads:

- `tiny_mlp`: `matmul + sigmoid + softmax`
- `tiny_conv`: `conv2d + sigmoid + global average pooling + softmax`
- `tiny_attention`: `scaled dot product attention + layer norm + softmax`

Inputs and outputs:

- `tiny_mlp`: input `[32, 64]`, output `[32, 6]`
- `tiny_conv`: input `[8, 32, 32, 3]`, output `[8, 5]`
- `tiny_attention`: `q/k/v [2, 4, 16, 64]`, output `[2, 4]`

Results from:

- `dart --packages=.dart_tool/package_config.json benchmark/models.dart --warmup=20 --iters=200`
- `uv run python benchmark/models.py --warmup 20 --iters 200`

| Workload | Dart ms/iter | Python ms/iter | Dart vs Python |
| --- | ---: | ---: | ---: |
| `tiny_mlp` | `0.0660` | `0.3280` | `4.97x faster` |
| `tiny_conv` | `0.1969` | `0.3646` | `1.85x faster` |
| `tiny_attention` | `0.2665` | `0.4073` | `1.53x faster` |

Correctness notes:

- Output shapes matched exactly.
- Output previews matched within normal float32 rounding noise.
- Earlier checksum checks on the full outputs stayed within `1e-7`.

## Direct HF MLX Models (Uncompiled)

Scope:

- These runs use existing Hugging Face MLX checkpoints directly.
- No `.mlxfn` export/import artifact was used.
- Current direct-loading path is implemented only for `Qwen2`-family 4-bit
  models because they share one architecture and one quantization scheme.

Models:

- [`mlx-community/Qwen2.5-0.5B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-0.5B-Instruct-4bit)
- [`mlx-community/Qwen2.5-1.5B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-1.5B-Instruct-4bit)
- [`mlx-community/Qwen2.5-3B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-3B-Instruct-4bit)

Shared prompt:

`Summarize why MLX on Apple Silicon is useful for local inference, and mention memory efficiency and developer ergonomics.`

Token count after tokenizer truncation: `25`

Measured output:

- Last-token logits slice `[:, -1, :16]`

Results from:

- `uv run python benchmark/real_nn.py --warmup 1 --iters 1 --seq-len 32`
- `dart --packages=.dart_tool/package_config.json benchmark/real_nn.dart --manifest=benchmark/out/real_nn/real_manifest.json --output=benchmark/out/real_nn/real_dart.json --warmup=1 --iters=1`
- `uv run python benchmark/real_compare.py --python benchmark/out/real_nn/real_python.json --dart benchmark/out/real_nn/real_dart.json`

| Model | Repo Size | Python ms/iter | Dart ms/iter | Dart slower | Max abs diff | Mean abs diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Qwen2.5-0.5B-Instruct-4bit` | `276.2 MiB` | `7.6765` | `1101.2890` | `143.46x` | `0.01706` | `0.00640` |
| `Qwen2.5-1.5B-Instruct-4bit` | `839.4 MiB` | `15.2112` | `3934.0890` | `258.63x` | `0.02441` | `0.01323` |
| `Qwen2.5-3B-Instruct-4bit` | `1666.9 MiB` | `25.1107` | `8340.7140` | `332.16x` | `0.33334` | `0.13555` |

Interpretation:

- The direct HF path is functionally viable.
  Dart can load the snapshot, parse `config.json`, read quantized safetensors,
  reconstruct the Qwen2 forward pass, and emit the same logits slice shape as
  Python.
- The direct HF path is not yet efficient.
  Current Dart timings are two to three orders of magnitude slower than
  `mlx_lm`.
- Numerical drift remains acceptable for `0.5B` and `1.5B` as a first direct
  reconstruction, but `3B` is not tight enough yet.

Likely causes:

- The Dart path currently rebuilds the model forward in user code instead of
  using the optimized `mlx_lm` implementation.
- Quantized embeddings are still dequantized row-by-row in Dart.
- The final LM head currently computes a full quantized matmul and slices the
  first `16` logits afterward.
- The current Dart implementation stays on the safe side for correctness and
  debuggability rather than squeezing the last kernel-level optimization.

## Direct HF MLX Models (Compiled Graph, Steady State)

Scope:

- Same three Hugging Face MLX Qwen2.5 models as above.
- Same prompt, token count, and logits slice.
- The Dart path now wraps the Qwen forward builder in `MlxFunction.compile()`
  and executes the compiled graph rather than crossing the FFI boundary for
  every layer op.

Results from:

- `uv run python benchmark/real_nn.py --warmup 20 --iters 50 --seq-len 32 --out-dir benchmark/out/real_nn_compiled_steady`
- `dart --packages=.dart_tool/package_config.json benchmark/real_nn.dart --manifest=benchmark/out/real_nn_compiled_steady/real_manifest.json --engine=compiled --output=benchmark/out/real_nn_compiled_steady/real_dart_compiled.json --warmup=20 --iters=50`
- `uv run python benchmark/real_compare.py --python benchmark/out/real_nn_compiled_steady/real_python.json --dart benchmark/out/real_nn_compiled_steady/real_dart_compiled.json --output benchmark/out/real_nn_compiled_steady/real_compare_compiled.json`

| Model | Python ms/iter | Dart compiled ms/iter | Dart vs Python | Max abs diff | Mean abs diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Qwen2.5-0.5B-Instruct-4bit` | `7.2583` | `6.5551` | `1.11x faster` | `0.00000` | `0.00000` |
| `Qwen2.5-1.5B-Instruct-4bit` | `15.7590` | `16.5435` | `1.05x slower` | `0.00000` | `0.00000` |
| `Qwen2.5-3B-Instruct-4bit` | `26.3794` | `26.8772` | `1.02x slower` | `0.00000` | `0.00000` |

Interpretation:

- The compiled graph path now matches Python on the measured logits slice
  exactly for all three Qwen2.5 models.
- In steady state, `0.5B` is faster in Dart than in Python on this machine.
- `1.5B` and `3B` remain slightly slower in Dart, but the remaining gap is now
  small: roughly `5%` and `2%`.
- The large latency gap from the first direct HF implementation is gone.

Practical takeaway:

- If the goal is “run real Hugging Face MLX Qwen2.5 models from Dart at
  Python-like speed”, the compiled path is now viable.
- If the goal is exact slice parity with Python on these checkpoints, that is
  also now achieved for the benchmarked output slice.

## Benchmark Form And Cache Strategy

Additional experiments on the compiled Qwen2.5 runner:

- Compile mode:
  - `MLX_COMPILE_MODE_ENABLED` was best.
  - On `Qwen2.5-3B`, `ENABLED` measured about `25.92 ms`, while
    `NO_SIMPLIFY` and `NO_FUSE` regressed to about `28.61 ms` and `29.02 ms`.
- `shapeless` compile:
  - `shapeless=false` was best.
  - On `Qwen2.5-0.5B`, `shapeless=true` regressed from about `6.98 ms` to
    about `8.80 ms`.
- Warmup matters:
  - Small warmups can overstate the remaining Dart/Python gap.
  - By `warmup=20`, the compiled path reached the stable numbers listed above.
- Sequential versus isolated processes:
  - Isolated single-model runs showed noticeably higher variance, especially on
    Python.
  - For steady cross-runtime comparisons in this repository, the best current
    signal comes from fixed warmup plus fixed-iteration runs written to disk,
    not single ad-hoc process launches.

## Why Not `mlx-audio` Or `mlx-vlm`

The following Hugging Face models were considered but not included in the
direct Dart/Python parity table:

- [`mlx-community/kitten-tts-nano-0.8-6bit`](https://huggingface.co/mlx-community/kitten-tts-nano-0.8-6bit)
- [`mlx-community/Qwen3.5-9B-MLX-4bit`](https://huggingface.co/mlx-community/Qwen3.5-9B-MLX-4bit)
- [`mlx-community/Qwen3.5-35B-A3B-4bit`](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-4bit)

Reason:

- `mlx-audio` and `mlx-vlm` expose Python-package runtimes rather than a
  dedicated stable C ABI that Dart can bind directly.
- Their model execution depends on higher-level Python processors and runtime
  orchestration, not only on the core `mlx-c` operator surface.
- Reaching strict Dart/Python parity for those models would require separate
  Dart implementations for the audio or multimodal runtime stacks.

## Reproduction

Install Python tooling:

```sh
uv sync
```

Synthetic parity:

```sh
dart run benchmark/models.dart --warmup=20 --iters=200
uv run python benchmark/models.py --warmup 20 --iters 200
```

Direct HF Qwen2.5 benchmarks:

```sh
uv run python benchmark/real_nn.py --warmup 1 --iters 1 --seq-len 32
dart --packages=.dart_tool/package_config.json benchmark/real_nn.dart \
  --manifest=benchmark/out/real_nn/real_manifest.json \
  --output=benchmark/out/real_nn/real_dart.json \
  --warmup=1 \
  --iters=1
uv run python benchmark/real_compare.py \
  --python benchmark/out/real_nn/real_python.json \
  --dart benchmark/out/real_nn/real_dart.json
```

Compiled HF Qwen2.5 benchmarks:

```sh
uv run python benchmark/real_nn.py --warmup 20 --iters 50 --seq-len 32 --out-dir benchmark/out/real_nn_compiled_steady
dart --packages=.dart_tool/package_config.json benchmark/real_nn.dart \
  --manifest=benchmark/out/real_nn_compiled_steady/real_manifest.json \
  --engine=compiled \
  --output=benchmark/out/real_nn_compiled_steady/real_dart_compiled.json \
  --warmup=20 \
  --iters=50
uv run python benchmark/real_compare.py \
  --python benchmark/out/real_nn_compiled_steady/real_python.json \
  --dart benchmark/out/real_nn_compiled_steady/real_dart_compiled.json \
  --output benchmark/out/real_nn_compiled_steady/real_compare_compiled.json
```

For isolated single-model runs in separate processes, use `--model-name` on
both sides. Example:

```sh
uv run python benchmark/real_nn.py --model-name qwen25_15b --warmup 20 --iters 50 --seq-len 32 --out-dir benchmark/out/real_nn_single_15b
dart --packages=.dart_tool/package_config.json benchmark/real_nn.dart \
  --manifest=benchmark/out/real_nn_single_15b/real_manifest.json \
  --engine=compiled \
  --model-name=qwen25_15b \
  --output=benchmark/out/real_nn_single_15b/real_dart_compiled.json \
  --warmup=20 \
  --iters=50
```
