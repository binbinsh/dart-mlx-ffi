# Changelog

### Unreleased

- Renamed the package to `dart_inference` and the repository identity to `dart-inference`.
- Switched the package version format to `1.yyyy.commit-count`; this release is `1.2026.36`.
- Moved vendored native dependencies from `third_party/` to `vendors/` and updated native build paths plus publish filters.
- Added the first real Zig MLX executor template for `dart_inference_linear`, using Zig-owned safetensors weights with `mlx_matmul`/`mlx_add` before returning runtime ABI tensors.
- Added a Zig-owned MLX C type/output materialization layer and wired the runtime MLX path to copy future `mlx_array` executor outputs into the Dart runtime tensor ABI.
- Added Zig-side MLX quantization metadata parsing for affine/default quantized snapshots so executor selection does not need Dart to inspect config files.
- Added Zig-owned MLX model metadata discovery for `config.json`, `tokenizer.json`, and `generation_config.json`, including `model_type` and architecture diagnostics parsed in Zig.
- Added an Apple-only Zig `mlx-c` safetensors weight loader that keeps loaded parameter and metadata maps inside the Zig-owned MLX session and merges multi-file safetensors layouts without involving Dart.
- Added Zig-owned MLX artifact session discovery for local safetensors layouts, including session diagnostics for artifact kind and weight shard count.
- Moved explicit MLX runtime session creation into Zig and added the first Zig-side managed tensor-to-`mlx_array` conversion skeleton before the executor returns its not-yet-implemented error.
- Made the Zig runtime the only Dart-facing native build output and stopped producing the old Dart-facing MLX code asset from the build hook.
- Routed explicit MLX runtime loads through the Zig runtime boundary so future `mlx-c` execution cannot silently fall back to the private C++ adapter path.
- Added an Apple-only private `dart_inference_mlx_c` build target and linked it from the Zig runtime so MLX migration work can call `mlx-c` from Zig instead of Dart.
- Removed the former raw/shim/stable APIs, legacy model runners, legacy tests, old Dart-facing MLX C++ bridge, and stale local MLX benchmark/probe/example entry points from the package source.
- Renamed native runtime adapter internals to the `DINF/dinf_` prefix, keeping only `dart_inference_runtime_*` as the Dart-facing ABI.
- Added a Dart ONNX Runtime convenience layer (`DartOnnxSession`) on top of the shared model runtime API.
- Added Linux/NVIDIA ONNX model runners for structured UniFrontend and Kokoro TTS, including structured tokenizer, TN/SSML post-processing, Kokoro phonemization, and a composed UniFrontend+Kokoro TTS runtime.
- Added a library-level UniFrontend+Kokoro TTS registry loader so Dart/Flutter apps can run the stack in-process without depending on the HTTP server wrapper.
- Added package-level `dart_inference:onnx_server`, `dart_inference:structured_smoke_infer`, `dart_inference:structured_frontend_infer`, `dart_inference:tts_infer`, `dart_inference:tts_server`, and `dart_inference:tts_backends_status` tools for pure-Dart ONNX serving, benchmarking, structured frontend inference, direct TTS inference, and provider-status reporting.
- Added a TTS backend runtime registry and capability catalog that mark `kokoro` as pure Dart ONNX ready and record the ONNX-export blockers for the remaining local providers.
- Added explicit `close()` lifecycle hooks for structured UniFrontend, Kokoro, the composed UniFrontend+Kokoro runtime, and the TTS backend registry so long-lived Dart/Flutter apps can release ONNX Runtime GPU sessions without waiting for process exit.
- Added Dart-side `.dart_inference_runtime_env.json` discovery for ONNX Runtime preload libraries, including automatic sibling `cuda/lib` and `tensorrt/lib` inference from the staged ORT library path.
- Added ONNX Runtime dependency bundling for provider sidecar libraries, Linux SONAME aliases, optional CUDA/cuDNN/cuBLAS/TensorRT preloading from Dart `backendOptions` or CLI flags, and TensorRT cache/workspace/subgraph provider options so GPU EPs can run without relying on a Python process.
- Fixed structured UniFrontend SSML composition to choose Chinese TN aliases for Chinese text when the multi-head model emits overlapping English and Chinese TN spans.
- Added CUDA EP memory-limit options and defaulted the UniFrontend+Kokoro loader to a 16 GiB CUDA arena cap to avoid runaway ONNX Runtime arena reservation while leaving enough room for Kokoro synthesis.
- Switched Kokoro phonemization to a strict-by-default Dart FFI eSpeak-NG path with LRU caching, SSML `<phoneme>` tag support, and tone3-pinyin conversion for UniFrontend polyphone output, keeping the `espeak-ng` process path only as an explicit fallback.
- Added Kokoro phoneme vocab filtering and chunked ONNX synthesis so long phoneme inputs are split at the 510-token Kokoro budget instead of being silently truncated.
- Moved CUDA/TensorRT ONNX Runtime provider append handling into the native runtime bridge and made the build hook fall back from Ninja to platform CMake generators when Ninja is unavailable.

### 26.414.19

- Switched the iPhone `PaddleOCR-VL-1.5` default KV-cache scheme from `turboquant` to `uniform` after validating the real `photo_render_512` case on-device, restoring the expected token prefix and lowering peak memory.
- Kept `uniform 8-bit` as the stable iPhone default and rejected `uniform 4-bit` as a product path because it only saved `128 KiB` while drifting from the reference output.
- Added a dedicated `PaddleOCR-VL` runtime regression test to lock the default KV scheme and debug override behavior.
- Removed `PaddleOcrVlRunnerDebug` and `PaddleOcrVlDebugOverrides` from the public `lib/models.dart` export surface so the package API is no longer exposing internal profiling hooks.
- Replaced the old Flutter profiling harness example with a minimal demo app and trimmed the publish surface via `.pubignore` so internal iPhone tuning assets no longer ship to pub.dev.

### 26.405.15

- Removed the experimental private ANE and Core ML bridge surfaces from the Dart API, native build, tests, local tooling, and vendored `espresso_ane` sources so the package scope is MLX-only again.
- Added a Dart `PaddleOCR-VL` runner under `lib/src/models/paddle_ocr_vl/` and exported `PaddleOcrVlRunner` / `PaddleOcrVlConfig` from `lib/models.dart`.
- Simplified the package build hook and native CMake configuration after the ANE removal, including dropping the `DART_INFERENCE_ENABLE_PRIVATE_ANE` toggle and the now-unused `coremltools` Python dependency.
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
- Fixed multiple `dart-inference` native bridge ops to run on `default_device_stream()` where appropriate, including `addmm` and `conv2d`, improving Dart MLX parity and speed.
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
