## 26.0310.0203

- Moved the ongoing `mlx-vlm` and `mlx-audio` model migration code fully back
  under `benchmark/`, keeping `dart-mlx-ffi` core directories focused on the
  reusable FFI and stable API surface.

## 26.0310.0157

- Reorganized the native migration layout to follow `mlx-vlm` and `mlx-audio`
  more closely, including shared `vlm` tensor-map loading and shared audio
  preprocessing/text-cleaner helpers outside model-specific directories.

## 26.0309.1956

- Restructured the ongoing native VLM and audio migration work into
  internal source-style modules as a staging step before later moving the
  model-specific code back under `benchmark/`.

## 26.0309.1950

- Split the native Qwen3.5 benchmark implementation into multiple
  under-800-line files and added a synthetic snapshot regression test for the
  dense text path.
- Added pure Dart offline KittenTTS input helpers for phoneme-token
  preparation.

## 26.0309.1936

- Added native stable API helpers required for the ongoing pure Dart/FFI model
  port work: `tanh`, `variance`, and `addmm`.
- Added regression coverage for the new native math helpers.

## 26.0309.1907

- Added Dart-side vendored model APIs for Qwen3.5 text generation and
  KittenTTS synthesis.
- Added separate `mlx-vlm` and `mlx-audio` Python inference helpers used by
  the new Dart wrappers.

## 26.0309.1900

- Split vendored model-reference integration into independent files for
  `mlx-vlm` and `mlx-audio`, with a thin aggregate hub runner on top.

## 26.0309.1846

- Added vendored `mlx-vlm` and `mlx-audio` reference sources under `vendors/`
  for Qwen3.5 and KittenTTS integration work.
- Added multi-backend benchmark runners for
  `mlx-community/Qwen3.5-9B-MLX-4bit`,
  `mlx-community/Qwen3.5-35B-A3B-4bit`, and
  `mlx-community/kitten-tts-nano-0.8-6bit`.
- Added a Dart wrapper that launches the vendored Python reference runners
  through `uv` so these models can be exercised from the repository's normal
  benchmarking workflow.

## 26.0308.1557

- Added vendored Apple-platform MLX native sources, raw bindings, and native
  build hooks for macOS and iOS.
- Added a broad stable Dart API covering arrays, tensor algebra, random,
  FFT, linalg, quantization, convolutions, scan helpers, misc numeric helpers,
  streams, distributed wrappers, and in-memory IO readers/writers.
- Added regression tests for the stable Dart surface plus Flutter example
  builds for macOS and iOS simulator.
- Added pub.dev-ready documentation, publishing metadata, CI, and release
  workflow support.
