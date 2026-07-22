# Changelog

### Unreleased

- Removed the Dart model aggregate and model re-export entrypoints from
  `dart_inference`; model-specific Dart code now lives under
  `0x-software/models/<model>/dart/` and callers import those model modules
  directly.
- Renamed the package to `dart_inference` and the repository identity to `dart-inference`.
- Switched the package version format to `1.yyyy.commit-count`; this release is `1.2026.124`.
- Fixed clean parallel MLX builds by ordering Metal library generation before
  the embedded metallib target consumes its output.
- Added a generic TTS ONNX component bundle that can inspect, load, run, and metadata-smoke any provider ONNX target declared in the catalog, so newly exported Chatterbox, IndexTTS2, NeuTTS Air, Dia2, VibeVoice, and Qwen3-TTS graphs automatically use the same Dart -> native FFI -> ONNX Runtime path.
- Added Sarashina2 TensorRT EPContext preparation and auto-discovery so the direct Dart/FFI ONNX path can use a cached TensorRT flow-step engine without a TTS server process.
- Fixed ONNX session `requireProvider=false` fallback so CUDA/TensorRT initialization failures retry with sanitized CPU options instead of blocking component bundle loads.
- Added the Sarashina2.2-TTS local text path: raw text now goes through the native `tokenizer.sara2tok` runtime, split Llama ONNX semantic generation, native semantic-token sampling, and the CosyVoice2-style flow/HiFT decoder, with prompt audio tokens wired into semantic generation when prompt text is supplied.
- Moved Sarashina2 prompt token-id composition into the Dart tokenizer ABI and feeds the split LLM from a native-backed token-id buffer instead of expanding tokenizer output through a Dart list.
- Added the NeuTTS Air partial Dart/FFI runtime surface: Qwen2 tokenizer sidecar loading, prompt special-token resolution, native `<|speech_N|>` parsing/formatting, prompt token-id construction, registry opt-in, and a NeuCodec decoder wrapper that will run `neucodec_decoder.onnx` once exported.
- Keeps NeuTTS Air codec-token text parsing in a native-backed buffer through the decoder path, while preserving the public decoded-token list for metadata.
- Keeps caller-provided NeuTTS Air codec-token lists in a native-backed request source before decoder execution, avoiding a second list-to-native conversion in the ONNX path.
- Added native-backed NeuTTS Air prompt-token-id builders and lets the provider path skip prompt-token-id metadata lists while preserving prompt token counts.
- Lets the NeuTTS Air provider path skip codec-token metadata list/text materialization while preserving codec token counts for response metadata.
- Added Qwen2 tokenizer sidecar generation/loading for vocab+merges tokenizers, so CosyVoice2 and NeuTTS Air can start from native-backed tokenizer tables instead of reparsing Hugging Face tokenizer assets in the Dart hot path.
- Moved CosyVoice2 text-tokenization plus LLM prefill embedding construction behind one native FFI ABI call, leaving Dart to pass raw text, prompt speech tokens, and native embedding tables instead of materializing text token lists in the synthesis path.
- Moved CosyVoice2 RAS repetition-window checks onto native-backed decode history buffers, leaving Dart to own RNG and loop control without allocating per-step history sublists.
- Feeds CosyVoice2 flow embedding directly from the native-backed RAS decode buffer, avoiding a Dart token-list round trip between LLM decode and flow encoder.
- Lets the CosyVoice2 provider path skip generated speech-token list materialization, keeping only the token count while the flow encoder consumes the native RAS buffer.
- Moved CosyVoice2 flow-token prompt/generated concatenation into native embedding, so Dart no longer builds an intermediate joined speech-token list before `flow_encoder_fp32`.
- Added a CosyVoice2 flow embedding slice ABI so Sarashina2 can pass full semantic-token sources with an offset/count instead of building a decoded-token list for the flow encoder input.
- Keeps Sarashina2 external semantic-token text parsing in a native-backed buffer through the flow decoder path, while direct/debug calls can still request full token metadata lists.
- Keeps Sarashina2 LLM-generated semantic tokens in a native-backed buffer through flow decoding, with Dart token lists only materialized when metadata is requested.
- Moved Sarashina2 and NeuTTS Air native-backed speech/semantic token range validation into Dart helpers, avoiding Dart-side token iteration on parsed-buffer decode paths.
- Validates Sarashina2 and NeuTTS Air request token-list sources after copying them into the native decode buffer, so provider paths avoid a separate Dart-side validation pass.
- Skips redundant Sarashina2/NeuTTS Air token validation after native token parsers and lets the NeuTTS Air provider pass its already-validated native codec source straight to the decoder.
- Short-circuits empty Dart int-token lists in CosyVoice2, Sarashina2, and NeuTTS Air FFI wrappers instead of allocating empty typed-list copies.
- Moved CosyVoice2 prompt-audio linear resampling and mel-layout transposition into Dart helpers used by zero-shot prompt extraction.
- Moved CosyVoice2 zero-shot prompt 128-mel, matcha 80-mel, and kaldi fbank extraction into a dedicated native prompt module, so prompt ONNX inputs now stay native-backed through speech-tokenizer, CampPlus, and flow conditioning.
- Added a reusable CosyVoice2 native prompt-plan handle with precomputed window/filterbank tables, so repeated zero-shot prompt extraction no longer rebuilds mel constants from Dart or per prompt.
- Precomputed CosyVoice2 prompt DFT rotation step tables in the native prompt plan, reducing per-frame trigonometric work without allocating full DFT bases.
- Moved CosyVoice2 prompt fbank cepstral mean normalization into a native in-place helper and feeds CampPlus from the resulting native buffer.
- Moved CosyVoice2 zero-shot prompt token/feature clipping into native and keeps clipped prompt tensors native-backed for LLM prefill and flow conditioning.
- Keeps CosyVoice2/Sarashina2 prompt speaker embeddings native-backed through flow conditioning and closes prompt-extraction native buffers after prompt-audio synthesis.
- Moved CosyVoice2 diffusion timestep scheduling into a reusable native-filled native tensor, avoiding per-step Dart `Float32List([t, t])` allocation.
- Reused the CosyVoice2 diffusion duplicate-batch input buffer across estimator steps instead of allocating a fresh native tensor every step.
- Reused CosyVoice2 LLM attention-mask buffers across decode steps instead of allocating a new native mask for every generated token.
- Reused one CosyVoice2 LLM speech-token embedding buffer across autoregressive decode steps through the existing native embedding ABI.
- Reused one Sarashina2 LLM token-embedding buffer across semantic-token decode steps through the same native embedding ABI.
- Reused native empty float tensors for CosyVoice2 no-prompt flow conditioning and streaming HiFT cache inputs instead of constructing empty Dart heap tensors.
- Reads CosyVoice2/Sarashina2 prompt speech-token counts from native buffer sizes instead of creating typed-data views just to count tokens.
- Returned CosyVoice2/Sarashina2 provider WAV bytes from the native encoder while HiFT output tensors are still alive, so provider synthesis no longer re-encodes copied Dart audio samples.
- Let CosyVoice2/Sarashina2 provider synthesis skip Dart float mel/audio copies entirely while still reporting sample counts and returning native-encoded WAV bytes.
- Lets the Sarashina2 provider path skip semantic-token metadata list materialization, preserving token counts while direct/debug calls keep full token lists by default.
- Added structured TTS backend ONNX migration manifests and a library-level provider asset audit so blocked local providers report exact Dart/FFI/ONNX target graphs, current source weights, and missing required runtime assets without relying on CLI-local path heuristics.
- Made CosyVoice2 migration status ONNX-target-first: all core components now have explicit Dart/FFI/ONNX target paths with legacy source artifacts recorded only as export sources, loaded CosyVoice2 ONNX components can be retrieved and run by name through the bundle API, and newly exported ONNX components can smoke from runtime input metadata without per-component Dart input code.
- Moved ONNX provider alias/default selection into native open-option normalization, removed duplicate provider/option alias policy from the C++ ORT adapter, added native bridge base fields to adapter diagnostics, and exposed TensorRT EP cache/workspace/subgraph flags plus reusable TensorRT dependency auditing for TTS, structured frontend, smoke, and ONNX server startup paths.
- Moved `VectorStore` cosine search, batched insertion, top-k ranking, and `l2Normalise` math back into Dart, removing the native vector helper ABI.
- Moved Kokoro float32 concatenation and float32-to-PCM16 WAV encoding into shared Dart helpers, including a chunked path that writes multi-chunk synthesis output without pre-concatenating audio.
- Moved UniFrontend structured-logit argmax, sigmoid active-label scanning, binary span detection, BIOES decoding, span-type majority voting, and input/mask filling into Dart helpers backed by reusable native tensor buffers.
- Moved Kokoro ONNX input framing, voice-row selection, speed scalar writes, config vocab loading, phoneme filtering/chunk planning, voice archive loading, and NPY parsing into Dart helpers.
- Moved ONNX convenience tensor helpers onto native-backed native buffers, so helper-created int64/float32/bool inputs no longer need a second native scratch copy during runtime calls.
- Moved `RuntimeTensor` typed factories onto native-backed native buffers, making factory-created runtime inputs native-backed by default.
- Moved Kokoro ONNX audio output chunk handoff to native tensor pointers, so multi-chunk WAV encoding no longer copies each generated audio chunk into a Dart `Float32List` first.
- Moved UniFrontend target matching, candidate-id rows, token/attention rows, char/mask rows, MmBERT BPE tokenization, SSML composition/stripping, emphasis trimming, TN item selection, Chinese detection, and fallback TN verbalization into Dart.
- Moved Kokoro eSpeak-NG dynamic library loading, initialization, voice selection, phoneme calls, language routing, SSML/pinyin handling, text normalization, and cleanup to direct Dart FFI plus Dart text helpers.
- Removed old Dart-facing helper ABI declarations for Kokoro, UniFrontend, BPE, decoder, text, eSpeak, and vector operations.
- Moved Kokoro `config.json` vocab parsing, voice `.npz` loading, ZIP store extraction, and float32 `.npy` parsing into Dart.
- Fixed native-backed Kokoro `.npz` entry names to return null-terminated ABI strings instead of borrowing raw ZIP name slices.
- Fixed UniFrontend emphasis span normalization so decoded spans are normalized before the mutable destination list is cleared.
- Split the Dart-facing vector ABI exports into `vector_api.native` so `runtime.native` stays below the source-size limit during continued migration.
- Split native-backed memory snapshot collection into `mem.native`, keeping `dinf_mem` stable while leaving `runtime.native` room for further migration work under the source-size limit.
- Trimmed Dart-side run-path allocation by writing typed-list inputs directly into the reusable native descriptor arena before `dinf_run`, avoiding transient `RuntimeTensor` and shape-list wrappers.
- Replaced the private adapter diagnostics JSON callback with typed diagnostic entries and added a `vendors/` rule forbidding vendored source edits unless explicitly requested.
- Replaced the private native-to-C++ open-options JSON with typed `DinfOptionEntry` arrays, so Dart options stay typed through native before reaching the native adapter backends.
- Removed the private C++ `dinf_cpp_mem_json` path; the native CLI now reads typed `DinfMemoryInfo` directly before rendering its report JSON.
- Removed stale native-side JSON capability/status helpers now that runtime backend info, accelerator defaults, and engine order use typed fields and masks.
- Removed the obsolete native resolver/fallback JSON compatibility paths, leaving runtime artifact selection on the typed resolver ABI only.
- Moved `nativeRuntimeMode` dispatch from native-backed JSON parsing to typed `dinf_open` entry lookup, avoiding an extra open-path parse before native session creation.
- Replaced Dart-side native open-option JSON encoding with the typed `dinf_open` entry ABI, leaving native to merge options and build the private adapter JSON.
- Replaced Dart session-diagnostics JSON parsing with the typed `dinf_diag` entry ABI, moving adapter diagnostics JSON flattening behind native.
- Replaced native memory-snapshot JSON with the typed `dinf_mem` ABI so Dart receives cross-platform memory fields without decoding native JSON.
- Replaced runtime backend-info JSON with the typed `dinf_info` ABI, removing the Dart-side JSON decode and obsolete private adapter info symbol.
- Replaced Hugging Face artifact reference JSON with the compact `dinf_hf_ref` packed-text ABI, removing native JSON decode from Dart artifact parsing.
- Replaced Core ML layout discovery JSON with the compact `dinf_coreml_layout` packed-text ABI, removing another Dart-side native JSON decode.
- Split the ONNX C++ adapter into small session and pipeline modules so future `Dart -> native FFI -> C/C++` migration work stays under the source-size limit.
- Replaced ONNX preload-library discovery JSON with the compact `dinf_ort_libs` packed-text ABI, removing another Dart-side native JSON decode.
- Replaced Dart resolver/fallback JSON calls with typed `dinf_resolve` and `dinf_fallback` ABIs so runtime artifact selection no longer builds or parses JSON in Dart.
- Moved native runtime open-option assembly into native through `dinf_open_opts` and removed the obsolete `dinf_open` Dart-facing entry point.
- Replaced the capabilities JSON binding with native-backed accelerator bitmasks through `dinf_accel_mask`.
- Moved unresolved remote artifact rejection fully into the native open path, removing the Dart preflight call and obsolete `dinf_artifact_remote` binding.
- Moved Hugging Face auth-token environment lookup into native through `dinf_hf_token`.
- Moved runtime engine/artifact selection into native through the native resolver ABI, leaving Dart to pass compact artifact descriptors and unwrap the selected artifact.
- Moved registered-runtime fallback selection into native through the native fallback ABI, leaving Dart to pass registered engine ids and unwrap the selected fallback artifact.
- Moved Hugging Face default cache root platform/env policy into native through `dinf_hf_cache_root`.
- Split native runtime tensor ABI, validation, and native buffer ownership helpers into `abi.native` to keep `runtime.native` below the source-size limit before further migration.
- Moved Hugging Face artifact reference parsing, directory artifact policy, and cache path construction into native through `dinf_hf_*`.
- Moved unresolved remote artifact detection into native resolver policy.
- Moved the default ONNX Runtime CUDA/TensorRT preload library list out of Dart so `dinf_ort_libs` owns the default names.
- Moved model bundle artifact path resolution into native through `dinf_artifact_path`.
- Moved runtime artifact platform compatibility, MLX preview gating, registered MLX artifact detection, engine order, and default accelerator policy into native resolver policy.
- Moved runtime platform/capability discovery into native through `dinf_platform_id` and `dinf_accel_mask`, so Dart no longer owns native backend accelerator defaults.
- Moved Core ML bundle layout discovery into native through `dinf_coreml_layout`, so Dart no longer owns CoreML-LLM chunk sorting, monolithic bundle detection, pipeline JSON detection, or sidecar scanning.
- Moved ONNX Runtime preload library discovery into native through `dinf_ort_libs`, so Dart no longer owns runtime env-file parsing, CUDA/TensorRT directory inference, or preload library filesystem scanning for session setup.
- Shortened the Dart-facing native ABI from `dart_inference_runtime_*` to `dinf_*`, shortened the private adapter symbols to `dinf_cpp_*`, and renamed the code-asset marker file to `rt_bindings.dart`.
- Routed runtime tensor dtype/shape/byte-length validation through native's shared tensor layout rules before echo, MLX, or private adapter execution.
- Moved `NativeTensorBuffer` dtype/shape byte-length computation into native through `dinf_alloc_tensor`, so Dart no longer duplicates tensor layout rules for zero-copy input buffers.
- Moved Linux memory snapshots into native by reading `/proc/self/status` directly from `dinf_mem`, while leaving non-Linux platform-specific memory probes behind the private adapter.
- Moved `nativeRuntimeMode` dispatch to native-backed open handling instead of substring matching Dart-encoded options.
- Removed the stale unused `dart_inference_bindings_generated.dart` raw-binding placeholder from the public package surface.
- Renamed the private Apple `mlx-c` dependency to `dinf_mlx_c` and reused the Dart-side runtime input tensor descriptor arena across calls so the hot path keeps fewer per-run allocations outside native.
- Moved `.mlxfn` positional input ordering into native by parsing artifact-side `inputs.json` / `input_order`, reporting it in diagnostics, and reordering named runtime tensors before `mlx_imported_function_apply`.
- Restored the vendored MLX, `mlx-c`, `fmt`, and `metal-cpp` source trees to the publish package so Apple builds have the private `mlx-c` dependency sources locally.
- Split native-backed MLX artifact discovery and config/quantization parsing into `mlx_artifact.native`, leaving `mlx_backend.native` focused on session state, weight loading, and executor dispatch.
- Moved vendored native dependencies from `third_party/` to `vendors/` and updated native build paths plus publish filters.
- Updated native MLX backend metadata so Apple builds report `enabled: true` and expose the registered `.mlxfn` plus `dart_inference_linear` executor surface.
- Removed the second native-side MLX output copy by moving materialized C-allocator buffers directly into the Dart-facing runtime tensor batch.
- Added native-backed `.mlxfn` imported-function execution so exported MLX bundles now run through `dinf_* -> native -> mlx-c`, and moved the benchmark helper off the removed Dart MLX import runner.
- Registered MLX in the bundled runtime registry while keeping default resolver selection limited to implemented `.mlxfn`/`mlx-function` artifacts.
- Split native runtime input tensor conversion into `mlx_input.native` to keep the MLX backend extensible under the source-file size limit.
- Added the first real native MLX executor template for `dart_inference_linear`, using native-backed safetensors weights with `mlx_matmul`/`mlx_add` before returning runtime ABI tensors.
- Added a native-backed MLX C type/output materialization layer and wired the runtime MLX path to copy future `mlx_array` executor outputs into the Dart runtime tensor ABI.
- Added native-side MLX quantization metadata parsing for affine/default quantized snapshots so executor selection does not need Dart to inspect config files.
- Added native-backed MLX model metadata discovery for `config.json`, `tokenizer.json`, and `generation_config.json`, including `model_type` and architecture diagnostics parsed in native.
- Added an Apple-only native `mlx-c` safetensors weight loader that keeps loaded parameter and metadata maps inside the native-backed MLX session and merges multi-file safetensors layouts without involving Dart.
- Added native-backed MLX artifact session discovery for local safetensors layouts, including session diagnostics for artifact kind and weight shard count.
- Moved explicit MLX runtime session creation into native and added the first native-side managed tensor-to-`mlx_array` conversion skeleton before the executor returns its not-yet-implemented error.
- Made the native runtime the only Dart-facing native build output and stopped producing the old Dart-facing MLX code asset from the build hook.
- Routed explicit MLX runtime loads through the native runtime boundary so future `mlx-c` execution cannot silently fall back to the private C++ adapter path.
- Added an Apple-only private `mlx-c` build target and linked it from the native runtime so MLX migration work can call `mlx-c` from native instead of Dart.
- Removed the former raw/shim/stable APIs, legacy model runners, legacy tests, old Dart-facing MLX C++ bridge, and stale local MLX benchmark/probe/example entry points from the package source.
- Renamed native runtime adapter internals to the `DINF/dinf_` prefix.
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
- Added a Dart `PaddleOCR-VL` runner now owned under `models/paddle_ocr_vl/dart/` and exported through the model Module aggregate.
- Simplified the package build hook and native CMake configuration after the ANE removal, including dropping the `DART_INFERENCE_ENABLE_PRIVATE_ANE` toggle and the now-unused `coremltools` Python dependency.
- Patched the vendored MLX Metal build scripts to honor the active Apple SDK and deployment flags for iOS builds, and to skip `jaccl` on iOS.

### 26.404.11

- Replaced `mlx-community/Kimi-K2-Instruct-4bit` in the publish-time text matrix with the official `unsloth/gemma-4-E2B-it-UD-MLX-4bit` MLX snapshot and verified `Python MLX` vs `Dart MLX` parity at `0` max abs diff.
- Added a dedicated `unsloth_mlx` publish benchmark runner so release reports can use official MLX snapshots that require patched `mlx-lm` model definitions, including `Gemma 4`.
- Regenerated the 10-model publish report and refreshed `README.md` / `models/dart/tool/text_lm/README.md` to document the current benchmark matrix, timings, and `HF_HUB_DISABLE_XET=1` reproduce path for large Hub downloads.

### 26.331.11

- Returned the package scope to MLX-only Apple-platform runtime support and removed the experimental GGUF / `llama.cpp` layer from the public API surface.
- Kept `pubspec.yaml` platform metadata aligned with the actual supported targets so pub.dev shows both `iOS` and `macOS`.
- Kept the MLX-side model runners, export/import workflow, and publish-time benchmark coverage for text, VLM, TTS, and ASR checkpoints.
- Refreshed package metadata, build hooks, and documentation to describe an MLX-only package again.

### 26.325.7

- Moved Dart model implementations out of `benchmark/` and, in the current repository layout, into `models/<model>/dart/`, including `parakeet_tdt`, `qwen2_5`, `kitten_tts`, shared helpers, and synthetic benchmark code.
- Added `models/dart/lib/` as the unified public export surface for Dart model implementations.
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
