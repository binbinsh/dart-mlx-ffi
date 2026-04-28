# Changelog

### Unreleased

- Renamed the package to `dart_inference` and the repository identity to `dart-inference`.
- Switched the package version format to `1.yyyy.commit-count`; this release is `1.2026.85`.
- Added a generic TTS ONNX component bundle that can inspect, load, run, and metadata-smoke any provider ONNX target declared in the catalog, so newly exported Chatterbox, IndexTTS2, NeuTTS Air, Dia2, VibeVoice, and Qwen3-TTS graphs automatically use the same Dart -> Zig -> ONNX Runtime path.
- Added structured TTS backend ONNX migration manifests and a library-level provider asset audit so blocked local providers report exact Dart/Zig/ONNX target graphs, current source weights, and missing required runtime assets without relying on CLI-local path heuristics.
- Made CosyVoice2 migration status ONNX-target-first: all core components now have explicit Dart/Zig/ONNX target paths with legacy source artifacts recorded only as export sources, loaded CosyVoice2 ONNX components can be retrieved and run by name through the bundle API, and newly exported ONNX components can smoke from runtime input metadata without per-component Dart input code.
- Moved ONNX provider alias/default selection into Zig open-option normalization, removed duplicate provider/option alias policy from the C++ ORT adapter, added Zig bridge base fields to adapter diagnostics, and exposed TensorRT EP cache/workspace/subgraph flags plus reusable TensorRT dependency auditing for TTS, structured frontend, smoke, and ONNX server startup paths.
- Moved `VectorStore` cosine search, batched index insertion, top-k ranking, and `l2Normalise` math into a Zig-owned native vector module exposed through `dinf_vec_*`, leaving Dart to hold only document metadata and result wrapping.
- Moved Kokoro float32 concatenation and float32-to-PCM16 WAV encoding into Zig through `dinf_audio_*`, including a chunked path that writes multi-chunk synthesis output without Dart pre-concatenating audio.
- Moved UniFrontend structured-logit argmax and sigmoid-mask scanning into Zig through `dinf_dec_*`, reusing native-backed runtime output tensors without copying logits back through Dart loops.
- Removed the legacy Dart heap-tensor fallback paths from UniFrontend decoding, so structured logits must stay native-backed and all argmax/sigmoid/BIOES/span-type work goes through Zig.
- Removed the obsolete Dart-facing `dinf_dec_sigmoid` mask ABI after structured decoding moved to native active-label and span helpers.
- Moved UniFrontend binary span boundary scanning into Zig through `dinf_dec_spans`, avoiding Dart-side per-character scanning after sigmoid thresholding.
- Moved UniFrontend BIOES argmax/span decoding into Zig through `dinf_dec_bioes`, so emphasis and TN span heads no longer materialize Dart id lists first.
- Moved UniFrontend TN span type majority voting into Zig through `dinf_dec_span_types`, avoiding Dart-side per-character type tallying.
- Moved UniFrontend emotion active-label scanning and best-label fallback into Zig through `dinf_dec_active`.
- Moved UniFrontend encoded input tensors onto Zig-owned native buffers with Zig-side bulk fill, so structured frontend inference no longer copies fixed input/mask tensors from Dart heap buffers before each native run.
- Consolidated UniFrontend native input reset into `dinf_struct_reset`, reducing the per-batch buffer clear/fill path from eight Dart FFI calls to one Zig call.
- Moved Kokoro ONNX input tensors onto reusable Zig-owned native buffers with dynamic tensor views, avoiding the runtime scratch-copy step for token ids, style vectors, and speed scalars.
- Consolidated Kokoro per-chunk ONNX input framing, voice-row selection, and speed scalar writes into `dinf_kok_inputs`, replacing the Dart-side multi-call prep path.
- Moved the remaining Kokoro voice-row copy helper to `dinf_kok_row` and removed the unused Dart-facing generic fill/copy ABI bindings.
- Moved ONNX convenience tensor helpers onto Zig-owned native buffers, so helper-created int64/float32/bool inputs no longer need a second native scratch copy during runtime calls.
- Moved `RuntimeTensor` typed factories onto Zig-owned native buffers, making factory-created runtime inputs native-backed by default.
- Moved Kokoro ONNX audio output chunk handoff to native tensor pointers, so multi-chunk WAV encoding no longer copies each generated audio chunk into a Dart `Float32List` first.
- Moved UniFrontend token-span, target, and candidate mask writes into Zig fill helpers, keeping structured frontend mask mutation on native-backed buffers.
- Consolidated UniFrontend target/candidate mask row writes behind `dinf_struct_matches`, letting Zig consume matcher results directly instead of round-tripping candidate ids through Dart lists.
- Moved UniFrontend pronunciation candidate-id rows into the Zig target matcher handle, so matched targets return precomputed native candidate ids instead of rebuilding them from Dart maps during input encoding.
- Moved UniFrontend token/attention and char/mask row writes into Zig helpers backed by reusable native scratch buffers.
- Moved UniFrontend char-vocab lookup and char/mask row filling into `dinf_fill_chars_i64`, removing the Dart-side per-character id loop in structured input encoding.
- Moved UniFrontend MmBERT BPE tokenization into a Zig-owned tokenizer handle through `dinf_bpe_*`, so structured input encoding fills native token ids and token offsets without Dart-side BPE merge loops or token-list copies.
- Changed UniFrontend token input encoding to use `dinf_bpe_fill`, letting Zig write `input_ids`, `attention_mask`, and token offsets in one pass without a Dart token scratch buffer or a second fill call.
- Moved UniFrontend SSML stripping for TTS text into Zig through `dinf_text_strip_ssml`, replacing Dart-side RegExp tag removal on the frontend output path.
- Moved UniFrontend SSML composition, XML escaping, and Dart UTF-16 span slicing into Zig through `dinf_text_ssml`.
- Moved UniFrontend emphasis span trimming and whitespace-merge normalization into Zig through `dinf_text_norm_spans`.
- Moved UniFrontend English/Chinese TN item overlap selection and ordering into Zig through `dinf_text_select_tn`.
- Moved UniFrontend Chinese-script detection into Zig through `dinf_text_has_zh`, removing the TTS routing RegExp from Dart.
- Moved UniFrontend fallback TN text verbalization and Chinese/English TN preference scoring into Zig through `dinf_text_tn_*`.
- Moved Kokoro eSpeak-NG dynamic library loading, initialization, voice selection, and phoneme calls behind Zig through `dinf_esp_*`, removing Dart-side direct eSpeak C FFI.
- Removed the Dart-side `espeak-ng` process fallback and moved eSpeak data-path environment/default directory resolution into Zig, keeping Kokoro phonemization on the `Dart -> Zig -> C library` path.
- Consolidated the default Kokoro phonemizer pipeline behind `dinf_esp_kok_*`, so normalization, language routing, mixed-script splitting, SSML/pinyin handling, eSpeak calls, post-processing, and cleanup run inside Zig with a single Dart call per uncached request.
- Moved Kokoro `config.json` vocab parsing into Zig through `dinf_kok_vocab`, so runtime load no longer JSON-decodes the token vocab or fills vocab buffers in Dart.
- Moved Kokoro voice `.npz` loading, ZIP store/deflate extraction, and float32 `.npy` parsing into Zig through `dinf_kok_npz`, so Dart no longer depends on `archive` or materializes decompressed voice arrays before native buffer adoption.
- Fixed Zig-owned Kokoro `.npz` entry names to return null-terminated ABI strings instead of borrowing raw ZIP name slices.
- Moved Kokoro phoneme filtering, chunk planning, and token id generation into Zig through `dinf_kok_*`, so TTS synthesis consumes native token chunks without Dart-side chunk strings or token-list copies.
- Moved Kokoro English text normalization and eSpeak phoneme post-processing into Zig through `dinf_kok_norm` and `dinf_kok_post`.
- Moved Kokoro SSML plain-text extraction, explicit phoneme cleanup, and pinyin phoneme-tag detection/normalization into Zig through compact `dinf_kok_*` text helpers.
- Moved Kokoro phonemizer language routing and mixed CJK/Latin run splitting into Zig through `dinf_kok_lang` and `dinf_kok_runs`.
- Moved Kokoro `<phoneme ph="...">` SSML parsing into Zig through `dinf_kok_ssml`, removing the Dart-side tag scanning RegExp.
- Moved Kokoro eSpeak phoneme cleanup into Zig through `dinf_kok_clean`, replacing per-call Dart regex cleanup on the phonemizer output path.
- Fixed UniFrontend emphasis span normalization so decoded spans are normalized before the mutable destination list is cleared.
- Split the Dart-facing vector ABI exports into `vector_api.zig` so `runtime.zig` stays below the source-size limit during continued migration.
- Split Zig-owned memory snapshot collection into `mem.zig`, keeping `dinf_mem` stable while leaving `runtime.zig` room for further migration work under the source-size limit.
- Trimmed Dart-side run-path allocation by writing typed-list inputs directly into the reusable native descriptor arena before `dinf_run`, avoiding transient `RuntimeTensor` and shape-list wrappers.
- Replaced the private adapter diagnostics JSON callback with typed diagnostic entries and added a `vendors/` rule forbidding vendored source edits unless explicitly requested.
- Replaced the private Zig-to-C++ open-options JSON with typed `DinfOptionEntry` arrays, so Dart options stay typed through Zig before reaching the native adapter backends.
- Removed the private C++ `dinf_cpp_mem_json` path; the native CLI now reads typed `DinfMemoryInfo` directly before rendering its report JSON.
- Removed stale Zig-side JSON capability/status helpers now that runtime backend info, accelerator defaults, and engine order use typed fields and masks.
- Removed the obsolete Zig resolver/fallback JSON compatibility paths, leaving runtime artifact selection on the typed resolver ABI only.
- Moved `zigRuntimeMode` dispatch from Zig-owned JSON parsing to typed `dinf_open` entry lookup, avoiding an extra open-path parse before native session creation.
- Replaced Dart-side native open-option JSON encoding with the typed `dinf_open` entry ABI, leaving Zig to merge options and build the private adapter JSON.
- Replaced Dart session-diagnostics JSON parsing with the typed `dinf_diag` entry ABI, moving adapter diagnostics JSON flattening behind Zig.
- Replaced native memory-snapshot JSON with the typed `dinf_mem` ABI so Dart receives cross-platform memory fields without decoding native JSON.
- Replaced runtime backend-info JSON with the typed `dinf_info` ABI, removing the Dart-side JSON decode and obsolete private adapter info symbol.
- Replaced Hugging Face artifact reference JSON with the compact `dinf_hf_ref` packed-text ABI, removing native JSON decode from Dart artifact parsing.
- Replaced Core ML layout discovery JSON with the compact `dinf_coreml_layout` packed-text ABI, removing another Dart-side native JSON decode.
- Split the ONNX C++ adapter into small session and pipeline modules so future `Dart -> Zig -> C/C++` migration work stays under the source-size limit.
- Replaced ONNX preload-library discovery JSON with the compact `dinf_ort_libs` packed-text ABI, removing another Dart-side native JSON decode.
- Replaced Dart resolver/fallback JSON calls with typed `dinf_resolve` and `dinf_fallback` ABIs so runtime artifact selection no longer builds or parses JSON in Dart.
- Moved native runtime open-option assembly into Zig through `dinf_open_opts` and removed the obsolete `dinf_open` Dart-facing entry point.
- Replaced the capabilities JSON binding with Zig-owned accelerator bitmasks through `dinf_accel_mask`.
- Moved unresolved remote artifact rejection fully into the Zig open path, removing the Dart preflight call and obsolete `dinf_artifact_remote` binding.
- Moved Hugging Face auth-token environment lookup into Zig through `dinf_hf_token`.
- Moved runtime engine/artifact selection into Zig through the native resolver ABI, leaving Dart to pass compact artifact descriptors and unwrap the selected artifact.
- Moved registered-runtime fallback selection into Zig through the native fallback ABI, leaving Dart to pass registered engine ids and unwrap the selected fallback artifact.
- Moved Hugging Face default cache root platform/env policy into Zig through `dinf_hf_cache_root`.
- Split Zig runtime tensor ABI, validation, and native buffer ownership helpers into `abi.zig` to keep `runtime.zig` below the source-size limit before further migration.
- Moved Hugging Face artifact reference parsing, directory artifact policy, and cache path construction into Zig through `dinf_hf_*`.
- Moved unresolved remote artifact detection into Zig resolver policy.
- Moved the default ONNX Runtime CUDA/TensorRT preload library list out of Dart so `dinf_ort_libs` owns the default names.
- Moved model bundle artifact path resolution into Zig through `dinf_artifact_path`.
- Moved runtime artifact platform compatibility, MLX preview gating, registered MLX artifact detection, engine order, and default accelerator policy into Zig resolver policy.
- Moved runtime platform/capability discovery into Zig through `dinf_platform_id` and `dinf_accel_mask`, so Dart no longer owns native backend accelerator defaults.
- Moved Core ML bundle layout discovery into Zig through `dinf_coreml_layout`, so Dart no longer owns CoreML-LLM chunk sorting, monolithic bundle detection, pipeline JSON detection, or sidecar scanning.
- Moved ONNX Runtime preload library discovery into Zig through `dinf_ort_libs`, so Dart no longer owns runtime env-file parsing, CUDA/TensorRT directory inference, or preload library filesystem scanning for session setup.
- Shortened the Dart-facing native ABI from `dart_inference_runtime_*` to `dinf_*`, shortened the private adapter symbols to `dinf_cpp_*`, and renamed the code-asset marker file to `rt_bindings.dart`.
- Routed runtime tensor dtype/shape/byte-length validation through Zig's shared tensor layout rules before echo, MLX, or private adapter execution.
- Moved `NativeTensorBuffer` dtype/shape byte-length computation into Zig through `dinf_alloc_tensor`, so Dart no longer duplicates tensor layout rules for zero-copy input buffers.
- Moved Linux memory snapshots into Zig by reading `/proc/self/status` directly from `dinf_mem`, while leaving non-Linux platform-specific memory probes behind the private adapter.
- Moved `zigRuntimeMode` dispatch to Zig-owned open handling instead of substring matching Dart-encoded options.
- Removed the stale unused `dart_inference_bindings_generated.dart` raw-binding placeholder from the public package surface.
- Renamed the private Apple `mlx-c` dependency to `dinf_zig_mlx_c` and reused the Dart-side runtime input tensor descriptor arena across calls so the hot path keeps fewer per-run allocations outside Zig.
- Moved `.mlxfn` positional input ordering into Zig by parsing artifact-side `inputs.json` / `input_order`, reporting it in diagnostics, and reordering named runtime tensors before `mlx_imported_function_apply`.
- Restored the vendored MLX, `mlx-c`, `fmt`, and `metal-cpp` source trees to the publish package so Apple builds have the private `mlx-c` dependency sources locally.
- Split Zig-owned MLX artifact discovery and config/quantization parsing into `mlx_artifact.zig`, leaving `mlx_backend.zig` focused on session state, weight loading, and executor dispatch.
- Moved vendored native dependencies from `third_party/` to `vendors/` and updated native build paths plus publish filters.
- Updated Zig MLX backend metadata so Apple builds report `enabled: true` and expose the registered `.mlxfn` plus `dart_inference_linear` executor surface.
- Removed the second Zig-side MLX output copy by moving materialized C-allocator buffers directly into the Dart-facing runtime tensor batch.
- Added Zig-owned `.mlxfn` imported-function execution so exported MLX bundles now run through `dinf_* -> Zig -> mlx-c`, and moved the benchmark helper off the removed Dart MLX import runner.
- Registered MLX in the bundled runtime registry while keeping default resolver selection limited to implemented `.mlxfn`/`mlx-function` artifacts.
- Split Zig runtime input tensor conversion into `mlx_input.zig` to keep the MLX backend extensible under the source-file size limit.
- Added the first real Zig MLX executor template for `dart_inference_linear`, using Zig-owned safetensors weights with `mlx_matmul`/`mlx_add` before returning runtime ABI tensors.
- Added a Zig-owned MLX C type/output materialization layer and wired the runtime MLX path to copy future `mlx_array` executor outputs into the Dart runtime tensor ABI.
- Added Zig-side MLX quantization metadata parsing for affine/default quantized snapshots so executor selection does not need Dart to inspect config files.
- Added Zig-owned MLX model metadata discovery for `config.json`, `tokenizer.json`, and `generation_config.json`, including `model_type` and architecture diagnostics parsed in Zig.
- Added an Apple-only Zig `mlx-c` safetensors weight loader that keeps loaded parameter and metadata maps inside the Zig-owned MLX session and merges multi-file safetensors layouts without involving Dart.
- Added Zig-owned MLX artifact session discovery for local safetensors layouts, including session diagnostics for artifact kind and weight shard count.
- Moved explicit MLX runtime session creation into Zig and added the first Zig-side managed tensor-to-`mlx_array` conversion skeleton before the executor returns its not-yet-implemented error.
- Made the Zig runtime the only Dart-facing native build output and stopped producing the old Dart-facing MLX code asset from the build hook.
- Routed explicit MLX runtime loads through the Zig runtime boundary so future `mlx-c` execution cannot silently fall back to the private C++ adapter path.
- Added an Apple-only private `mlx-c` build target and linked it from the Zig runtime so MLX migration work can call `mlx-c` from Zig instead of Dart.
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
