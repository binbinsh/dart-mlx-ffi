# Changelog

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
