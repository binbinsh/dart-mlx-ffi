# Changelog

### 26.2026.6

- Moved Dart model implementations out of `benchmark/` and into `lib/src/models/`, including `parakeet_tdt`, `qwen2_5`, `kitten_tts`, shared helpers, and synthetic benchmark code.
- Added `lib/models.dart` as the unified public export surface for Dart model implementations.
- Added experimental private ANE bridge surfaces, Core ML helpers, and project-local tooling under `private_ane/`.
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
