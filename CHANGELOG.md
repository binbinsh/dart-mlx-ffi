# Changelog

### 26.405.15

- Removed the experimental private ANE and Core ML bridge surfaces from the Dart API, native build, tests, local tooling, and vendored `espresso_ane` sources so the package scope is MLX-only again.
- Added a Dart `PaddleOCR-VL` runner under `lib/src/models/paddle_ocr_vl/` and exported `PaddleOcrVlRunner` / `PaddleOcrVlConfig` from `lib/models.dart`.
- Simplified the package build hook and native CMake configuration after the ANE removal, including dropping the `DART_MLX_ENABLE_PRIVATE_ANE` toggle and the now-unused `coremltools` Python dependency.
- Patched the vendored MLX Metal build scripts to honor the active Apple SDK and deployment flags for iOS builds, and to skip `jaccl` on iOS.

### 26.404.11

- Replaced `mlx-community/Kimi-K2-Instruct-4bit` in the publish-time text matrix with the official `unsloth/gemma-4-E2B-it-UD-MLX-4bit` MLX snapshot and verified `Python MLX` vs `Dart MLX` parity at `0` max abs diff.
- Added a dedicated `unsloth_mlx` publish benchmark runner so release reports can use official MLX snapshots that require patched `mlx-lm` model definitions, including `Gemma 4`.
- Regenerated the 10-model publish report and refreshed `README.md` / `models/text_lm/README.md` to document the current benchmark matrix, timings, and `HF_HUB_DISABLE_XET=1` reproduce path for large Hub downloads.

### 26.331.11

- Returned the package scope to MLX-only Apple-platform runtime support and removed the experimental GGUF / `llama.cpp` layer from the public API surface.
- Kept `pubspec.yaml` platform metadata aligned with the actual supported targets so pub.dev shows both `iOS` and `macOS`.
- Kept the MLX-side model runners, export/import workflow, and publish-time benchmark coverage for text, VLM, TTS, and ASR checkpoints.
- Refreshed package metadata, build hooks, and documentation to describe an MLX-only package again.

### 26.325.7

- Moved Dart model implementations out of `benchmark/` and into `lib/src/models/`, including `parakeet_tdt`, `qwen2_5`, `kitten_tts`, shared helpers, and synthetic benchmark code.
- Added `lib/models.dart` as the unified public export surface for Dart model implementations.
- Renamed benchmark sweep scripts away from `recent_*` naming and introduced `publish_model_list.json` plus `parakeet_tdt_sweep.py`.
- Added `TDT v3` to the publish benchmark list and regenerated `benchmark/out/publish_report.json` with `14` rows.
- Fixed multiple `dart-mlx-ffi` native bridge ops to run on `default_device_stream()` where appropriate, including `addmm` and `conv2d`, improving Dart MLX parity and speed.
- Fixed stale benchmark tooling/docs after the model-layout move, including the generic benchmark runner plus ignored local `tmp/` and `output/` artifacts.
- Updated `README.md`, `models/README.md`, and `AGENTS.md` to reflect the new model layout, benchmark layout, and version format.

### 26.310.2051

- Removed the external `documentation:` link so pub.dev can show the generated API reference directly.
- Reordered the 13-model benchmark table in `README.md` to group `text`, `vlm`, and `tts` results separately and simplified model display names.
- Cleaned analyzer and `dart doc --dry-run` warnings to zero, including the raw generated bindings and benchmark helper scripts.
- Added a local `tool/generate_docs.sh` helper and ignored local `vendors/` workspace residue from git status.

### 26.310.2016

- Consolidated root documentation for pub.dev publishing into `README.md`, `CHANGELOG.md`, and `AGENTS.md`.
- Added a 13-model Python MLX vs Dart MLX benchmark table using `warmup=3` and `iters=10`, with focused notes for the two TTS paths.
- Tightened package metadata to macOS-only support and removed the GitHub workflow files from the repository.

### 26.308.1557

- Added vendored Apple-platform MLX native sources, raw bindings, and native build hooks for macOS and iOS.
- Added the broad stable Dart API for arrays, tensor algebra, random, FFT, linalg, quantization, convolutions, streams, distributed wrappers, and in-memory IO.
- Added regression tests, Flutter example builds, CI, and publishing metadata.
