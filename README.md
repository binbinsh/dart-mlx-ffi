# dart_mlx_ffi

[![CI](https://github.com/binbinsh/dart-mlx-ffi/actions/workflows/ci.yml/badge.svg)](https://github.com/binbinsh/dart-mlx-ffi/actions/workflows/ci.yml)

`dart_mlx_ffi` is a Dart/Flutter FFI package for Apple's
[MLX C API](https://ml-explore.github.io/mlx-c/). It targets Apple platforms,
vendors `mlx`, `mlx-c`, and their native dependencies, and uses a build hook to
compile a bundled native library for the current target architecture.

## What you get

- A stable high-level Dart API for common workflows:
  - version lookup
  - default device inspection
  - array creation from Dart lists
  - array factories such as `zeros`, `ones`, `full`, `arange`
  - like-constructors such as `zerosLike`, `onesLike`, `fullLike`
  - shape transforms such as `reshape`, `transpose`
  - unary ops such as `abs`, `negative`, `exp`, `log`, `sin`, `cos`,
    `floor`, `sqrt`, `square`, `reciprocal`, `sigmoid`, `degrees`,
    `radians`, `expm1`, `erf`, `erfinv`, `log1p`, `log2`, `log10`
  - binary ops such as `add`, `subtract`, `multiply`, `divide`, `matmul`,
    `greater`, `greaterEqual`, `less`, `lessEqual`, `floorDivide`,
    `logaddexp`, `inner`
  - numerically useful ops such as `softmax`, `logSumExp`, `topK`
  - tensor algebra and indexing helpers such as `take`, `gather`, `slice`,
    `sliceDynamic`, `sliceUpdate`, `sliceUpdateDynamic`, `flatten`,
    `moveaxis`, `swapaxes`, `transposeAxes`, `tile`, `pad`, `padSymmetric`,
    `unflatten`, `gatherMm`, `diag`, `diagonal`, `kron`, `meshgrid`,
    `partition`, `scatter`, `einsum`, `tensordot`
  - numeric helpers such as `linspace`, `outer`, `isClose`, `logicalAnd`,
    `logicalOr`, `logicalNot`, `repeat`, `roll`, `median`, `maskedScatter`,
    `nanToNum`, `divmod`
  - scan and matrix construction helpers such as `cumsum`, `cumprod`,
    `cummax`, `cummin`, `logcumsumexp`, `eye`, `identity`, `tri`, `tril`,
    `triu`, `trace`
  - comparisons and selection via `equal` and `where`
  - reductions via `sum` and `mean`
  - random APIs via `key`, `split`, `uniform`, `normal`, `bernoulli`,
    `categorical`, `permutation`, `permutationArange`, `gumbel`, `laplace`,
    `randint`, `multivariateNormal`
  - convolution helpers via `conv1d`, `conv2d`, `conv3d`, `convGeneral`,
    `convTranspose1d`, `convTranspose2d`, `convTranspose3d`
  - FFT APIs via `fft`, `ifft`, `rfft`, `irfft`, `fft2`, `fftn`, `rfft2`,
    `rfftn`, `irfft2`, `irfftn`, `fftshift`, `ifftshift`
  - linear algebra APIs via `inv`, `solve`, `qr`, `eig`, `cholesky`, `eigh`,
    `svd`, `pinv`, `norm`, `cross`, `eigvals`, `eigvalsh`, `lu`,
    `luFactor`, `solveTriangular`
  - quantization helpers via `quantize`, `dequantize`, `quantizedMatmul`,
    `qqmm`, `gatherQmm`, `toFp8`, `fromFp8`
  - file IO via `load`, `save`, `loadSafetensors`, `saveSafetensors`
  - stream-aware and in-memory IO via `load(..., stream:)`, `loadBytes`,
    `saveBytes`, `loadSafetensorsBytes`, `saveSafetensorsBytes`,
    `MlxBytesReader`, and `MlxBytesWriter`
  - function export/import helpers including kwargs export and incremental
    exporters
  - tensor composition APIs such as `concatenate`, `stack`, `broadcastTo`,
    `expandDims`, `squeeze`, `clip`, `minimum`, `maximum`, `argmax`, `argmin`,
    `sort`, `argsort`
  - advanced matrix helpers such as `broadcastArrays`, `splitSections`,
    `segmentedMm`, `blockMaskedMm`
  - runtime helpers via `seed`, `evalAll`, `asyncEvalAll`, memory stats, and
    Metal availability/capture helpers
  - system helpers via `MlxStream`, `MlxDevice.info`,
    `MlxDevice.setDefault(...)`, and `mx.distributed`
  - fast helpers such as `layerNorm`, `rmsNorm`, `rope`, `ropeDynamic`,
    `scaledDotProductAttention`, plus Metal/CUDA custom-kernel wrappers
- A complete low-level raw binding layer for the MLX C surface in
  [`package:dart_mlx_ffi/raw.dart`](lib/raw.dart)
- Apple-native build integration for macOS and iOS simulator/device builds

## Platform notes

- The package is intended for Apple targets only.
- MLX itself is most useful with Metal enabled.
- If the local Xcode installation does not contain the `MetalToolchain`
  component, the build hook automatically falls back to a CPU-only MLX build so
  the package still compiles.
- On this repository's current build configuration, macOS uses Metal when the
  toolchain is present. iOS simulator builds intentionally fall back to
  `MLX_BUILD_METAL=OFF` because the simulator Metal compile path currently emits
  incompatible deployment flags.
- To enable Metal shader compilation on the build machine:

```sh
xcodebuild -downloadComponent MetalToolchain
```

## High-level API

```dart
import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

final a = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [2, 2]);
final b = MlxArray.fromFloat32List([5, 6, 7, 8], shape: [2, 2]);
final c = MlxOps.matmul(a, b);
final z = MlxArray.zeros([2, 2]);
final r = MlxArray.arange(0, 4, 1).reshape([2, 2]);
final s = MlxOps.sum(c);
final q = mx.linalg.qr(a);

print(MlxVersion.current());
print(MlxDevice.defaultDevice());
print(c.toList());
print(z.toList());
print(r.transpose().toList());
print(s.toList());
print(q.q.shape);

c.close();
z.close();
r.close();
s.close();
q.r.close();
q.q.close();
b.close();
a.close();
```

## Stream And IO

```dart
import 'dart:typed_data';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

final stream = MlxStream.defaultCpu();
final array = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [2, 2]);

final bytes = mx.io.saveBytes(array);
final loaded = mx.io.loadBytes(bytes, stream: stream);

final writer = MlxBytesWriter();
mx.io.saveWriter(writer, array);
final reader = MlxBytesReader(writer.bytes);
final loadedAgain = mx.io.loadReader(reader, stream: stream);

print(loaded.toList());
print(loadedAgain.toList());

loadedAgain.close();
loaded.close();
reader.close();
writer.close();
array.close();
stream.close();
```

## Benchmark

Run the tiny model-style benchmark from the package root:

```sh
dart run benchmark/synthetic/tiny.dart
dart run benchmark/synthetic/tiny.dart --warmup=20 --iters=100
```

The benchmark keeps work on the MLX side, uses `MlxRuntime.evalAll(...)`, and
reports total time, mean time per iteration, peak-memory delta, and the last
`topK` output.

Detailed benchmark results and direct Hugging Face MLX model comparisons:
[BENCHMARK.md](BENCHMARK.md)

Deterministic Python parity harness for comparable tensor APIs:
[PARITY.md](PARITY.md)

The parity harness supports `--groups` so CI can run smaller API slices.

For cross-runtime parity and speed checks against `python-mlx`, sync the
Python tooling with `uv` and run the dedicated model runners:

```sh
uv sync
dart run benchmark/synthetic/models.dart --warmup=20 --iters=200
uv run python benchmark/synthetic/models.py --warmup 20 --iters 200
```

These runners cover three deterministic model shapes:
`tiny_mlp`, `tiny_conv`, and `tiny_attention`.

For real Hugging Face MLX Qwen2.5 checkpoints, the benchmark runner also
supports a compiled-graph mode:

```sh
uv run python benchmark/qwen2_5/real_nn.py --warmup 20 --iters 50 --seq-len 32 --out-dir benchmark/out/real_nn_compiled_steady
dart --packages=.dart_tool/package_config.json benchmark/qwen2_5/real_nn.dart --manifest=benchmark/out/real_nn_compiled_steady/real_manifest.json --engine=compiled --output=benchmark/out/real_nn_compiled_steady/real_dart_compiled.json --warmup=20 --iters=50
```

You can also benchmark a single model in isolation with `--model-name`, for
example `qwen25_15b`.

For vendored reference-model coverage of newer multimodal and audio checkpoints,
the repository also includes a multi-backend runner for:

- `mlx-community/Qwen3.5-9B-MLX-4bit`
- `mlx-community/Qwen3.5-35B-A3B-4bit`
- `mlx-community/kitten-tts-nano-0.8-6bit`

Run it directly in Python or through the Dart wrapper:

```sh
uv sync
uv run python benchmark/run_all.py --iters 1 --warmup 1
dart run benchmark/run_all.dart --iters=1 --warmup=1
```

These three models currently use vendored Python reference backends from
`vendors/mlx-vlm` and `vendors/mlx-audio`; the native Dart Qwen2.5 runner
remains the only direct-Dart Hugging Face model path in this repository today.
The vendored `mlx-vlm` and `mlx-audio` integrations live in separate runner
files under `benchmark/`.

## Distributed

- `mx.distributed.isAvailable()` checks whether the current runtime exposes MLX
  distributed support.
- `mx.distributed.init(strict: false)` creates a group wrapper when the backend
  is available.
- Single-machine developer environments may report availability but still lack
  the launcher/runtime setup needed for actual collectives. The test suite treats
  those cases as environment-specific rather than package bugs.
- A dedicated smoke script is included for launcher-configured environments:

```sh
dart run tool/dist_smoke.dart
```

## Known Limits

- The package targets Apple platforms only.
- Complex tensors are supported by MLX, but high-level `toList()` intentionally
  remains conservative and does not try to decode every dtype.
- macOS builds use Metal when the toolchain is present.
- iOS simulator builds intentionally stay on `MLX_BUILD_METAL=OFF`.
- Qwen3.5 and KittenTTS repository support currently lives in the benchmark
  tooling via vendored Python reference implementations, not the stable Dart
  package API.
- The raw layer remains the escape hatch for the full C surface, especially for
  niche or backend-specific APIs that are not yet part of the stable Dart layer.

## Raw bindings

For full MLX C API coverage, import the raw layer directly:

```dart
import 'package:dart_mlx_ffi/raw.dart' as raw;
```

The raw layer is generated from the vendored `mlx-c` headers with `ffigen`.

## Development

Regenerate the raw bindings:

```sh
dart run ffigen --config ffigen.yaml
```

Validate the package locally:

```sh
dart analyze
dart test
cd example && flutter build macos --debug
cd example && flutter build ios --simulator --debug
uv sync
dart run benchmark/synthetic/models.dart --warmup=20 --iters=200
uv run python benchmark/synthetic/models.py --warmup 20 --iters 200
dart run benchmark/synthetic/tiny.dart --warmup=20 --iters=100
```

Release checklist: [RELEASE.md](RELEASE.md)
Release notes: [RELEASE_NOTES.md](RELEASE_NOTES.md)
API coverage matrix: [API_MATRIX.md](API_MATRIX.md)

## Publishing

The repository includes:

- CI workflow: [ci.yml](.github/workflows/ci.yml)
- pub.dev publishing workflow: [publish.yml](.github/workflows/publish.yml)

Recommended release flow:

1. Publish the first package version manually from a trusted local machine.
2. Configure pub.dev trusted publishing for `github.com/binbinsh/dart-mlx-ffi`
   with tag pattern `v{{version}}`.
3. Use the package version format `YY.MMDD.HHMM`.
4. Push a matching tag such as `v26.0308.1557`.

The publish workflow is configured for tags matching:

```text
vYY.MMDD.HHMM
```

For example:

```sh
git tag v26.0308.1557
git push origin v26.0308.1557
```
