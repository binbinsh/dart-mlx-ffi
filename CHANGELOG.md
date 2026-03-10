# Changelog

### 26.310.2016

- Consolidated root documentation for pub.dev publishing into
  `README.md`, `CHANGELOG.md`, and `AGENTS.md`.
- Added a 13-model Python MLX vs Dart MLX benchmark table using
  `warmup=3` and `iters=10`, with focused notes for the two TTS paths.
- Tightened package metadata to macOS-only support and removed the GitHub
  workflow files from the repository.

### 26.0310.0203

- Moved the ongoing `mlx-vlm` and `mlx-audio` model migration work fully back
  under `benchmark/`, keeping the package core focused on reusable FFI and
  stable API surface area.

### 26.0310.0157

- Reorganized the native migration layout to follow `mlx-vlm` and `mlx-audio`
  more closely, including shared tensor-map loading and shared audio
  preprocessing helpers.

### 26.0309.1956

- Restructured the native VLM and audio migration work into source-style
  modules as a staging step before moving the model-specific code back under
  `benchmark/`.

### 26.0309.1950

- Split the native Qwen3.5 benchmark implementation into multiple
  under-800-line files.
- Added synthetic regression coverage for the dense text path.
- Added offline KittenTTS phoneme-token input helpers.

### 26.0309.1936

- Added stable API helpers required for native model migration:
  `tanh`, `variance`, and `addmm`.
- Added regression tests for the new math helpers.

### 26.0309.1907

- Added Dart-side vendored model APIs for Qwen3.5 text generation and
  KittenTTS synthesis.
- Added separate Python inference helpers for `mlx-vlm` and `mlx-audio`.

### 26.0309.1900

- Split vendored reference-model integration into independent files for
  `mlx-vlm` and `mlx-audio`, with an aggregate runner on top.

### 26.0309.1846

- Added vendored `mlx-vlm` and `mlx-audio` sources under `vendors/` for
  Qwen3.5 and KittenTTS integration work.
- Added benchmark runners for:
  - `mlx-community/Qwen3.5-9B-MLX-4bit`
  - `mlx-community/Qwen3.5-35B-A3B-4bit`
  - `mlx-community/kitten-tts-nano-0.8-6bit`

### 26.0308.1557

- Added vendored Apple-platform MLX native sources, raw bindings, and native
  build hooks for macOS and iOS.
- Added the broad stable Dart API for arrays, tensor algebra, random, FFT,
  linalg, quantization, convolutions, streams, distributed wrappers, and
  in-memory IO.
- Added regression tests, Flutter example builds, CI, and publishing metadata.
