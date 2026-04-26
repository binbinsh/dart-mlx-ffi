# Runtime Benchmarks

This directory contains the promotion harness for cross-platform model runtime
support. A model can be marked `production` only after the matrix report passes
correctness, speed, peak-memory, and device-profile gates for iOS, macOS,
Windows, Linux, and Android.

`fixtures/` contains the canonical tiny text, image, audio, tool-schema, and
embedding-query inputs used for smoke and parity runs. Models declare
`required_fixtures` in `models.yaml` / `hf_artifacts.yaml`; `run_all.py` blocks
cells that are missing those required inputs before they reach promotion.

Runner output must be JSON with this shape:

```json
{
  "model_id": "qwen3_5",
  "platform": "ios",
  "engine": "coreml",
  "artifact": "path/to/artifact",
  "correctness": {
    "token_ids": [1, 2, 3],
    "top1_match_ratio": 1.0,
    "top3_coverage_ratio": 1.0
  },
  "metrics": {
    "prefill_tokens_per_second": 100.0,
    "decode_tokens_per_second": 20.0,
    "ttft_ms": 250.0,
    "end_to_end_ms": 1200.0,
    "peak_memory_bytes": 123456789
  },
  "device_profile": {
    "provider": "ANE",
    "raw_peak_memory_field": "phys_footprint",
    "runtime_diagnostics": {
      "engine": "coreml",
      "compute_units": "cpuAndNeuralEngine",
      "compute_plan": {
        "available": true,
        "preferred_device_counts": {"ane": 42}
      }
    }
  }
}
```

`compare.py` accepts a combined report containing `baseline` and `candidate`
objects in that runner format and emits the promotion verdict.

`combine_reports.py` builds that combined input from two runner outputs:

```sh
uv run python benchmark/runtime/combine_reports.py \
  --baseline benchmark/out/runtime/qwen3_5/ios/coreml_llm.json \
  --candidate benchmark/out/runtime/qwen3_5/ios/coreml.json \
  --out benchmark/out/runtime/qwen3_5/ios/report.json

uv run python benchmark/runtime/compare.py \
  benchmark/out/runtime/qwen3_5/ios/report.json \
  --out benchmark/out/runtime/qwen3_5/ios/verdict.json
```

`audit.py` checks the promotion evidence expected by `models.yaml`:

```sh
uv run python benchmark/runtime/audit.py \
  --out-root benchmark/out/runtime \
  --out benchmark/out/runtime/audit.json
```

`run_matrix.py` runs one model/platform matrix cell end to end:

```sh
uv run python benchmark/runtime/resolve_hf_artifacts.py \
  --model-id qwen3_5 \
  --platform macos \
  --out benchmark/runtime/artifacts.local.yaml

uv run python benchmark/runtime/run_all.py \
  --artifacts benchmark/runtime/artifacts.local.yaml \
  --model-id qwen3_5 \
  --platform macos \
  --engine coreml \
  --run
```

CoreML-LLM baseline tasks use `--task text|function|embedding|vlm`. Function
tasks can pass `--tools-file` as either one function schema object or an array
of CoreML-LLM tool declarations, embedding tasks can pass
`--embedding-query-file` and `--embedding-dim`, and VLM tasks must pass
`--image-file`. These options are also supported in `artifacts.local.yaml` and
are forwarded by `run_all.py`.

`run_all.py` plans or runs all required platform cells from a local artifact
map. Generate that map from Hugging Face first:

```sh
uv run python benchmark/runtime/resolve_hf_artifacts.py \
  --out benchmark/runtime/artifacts.local.yaml
```

When a model is still blocked, search Hugging Face for newly published runtime
artifacts before editing the catalog:

```sh
uv run python benchmark/runtime/search_hf_artifacts.py
```

The search report separates directly loadable runtime candidates from component
sidecars. Component-only hits, such as MiniCPM token2wav/CosyVoice ONNX files
or a vision-only Core ML bundle, are evidence for the blocked reason but do not
count as platform runtime artifacts.

GGUF files are useful for llama.cpp/Metal or CPU execution, but they are not
direct Core ML conversion inputs. Core ML conversion should prefer the original
Hugging Face safetensors/PyTorch source model, because GGUF stores llama.cpp
metadata and quantized tensor layouts rather than a Core ML/PyTorch graph. When
a GGUF model card is the only obvious lead, use the acquisition helper. It first
checks for existing Hugging Face Core ML artifacts, then falls back to a
CoreML-LLM conversion plan from the original source model:

```sh
uv run --group gguf-tools python benchmark/runtime/coreml_acquire.py \
  --gguf path/to/model.gguf \
  --dry-run
```

For lower-level GGUF metadata inspection, use:

```sh
uv run --group gguf-tools python benchmark/runtime/gguf_coreml_bridge.py \
  --gguf path/to/model.gguf \
  --source-model Qwen/Qwen3.5-0.8B \
  --dry-run
```

If the GGUF metadata contains a Hugging Face source URL/repository, the bridge
uses it automatically. Otherwise pass `--source-model`; without a source model
the bridge reports `state: blocked` rather than pretending a direct GGUF to
Core ML conversion is production-safe.

Some models are platform-ready only through a fallback engine. The gap report
shows those cases explicitly:

```sh
uv run python benchmark/runtime/engine_gap_report.py \
  --model-id paddle_ocr_vl
```

When no ready runtime artifact exists, use the conversion recipes instead of
editing the catalog by hand. The converter runs the real exporter command,
validates that an artifact was produced, then loads the artifact through
`artifact_health.py` before it is allowed to unblock any platform. Conversion
and health reports are written next to the artifact, and passing conversions
emit an artifact-map overlay:

```sh
uv run python benchmark/runtime/convert_artifacts.py \
  --model-id glm4_7_flash \
  --engine onnx \
  --dry-run

uv run python benchmark/runtime/convert_artifacts.py \
  --model-id glm4_7_flash \
  --engine onnx \
  --base-artifacts benchmark/runtime/artifacts.local.yaml \
  --out benchmark/runtime/artifacts.converted.yaml
```

Recipes may also define `timeout_seconds` and `env` to keep long exporters
from hanging indefinitely and to inject model-specific runtime toggles.
Timeouts are recorded as structured `conversion_failed` reports so matrix
planning can continue.

The default recipes use real native exporters:

- `coreml`: clones CoreML-LLM into `benchmark/artifacts/tools/coreml-llm` when
  needed, then runs its `conversion/convert.py` through `uv run
  --with-requirements`.
- `onnx`: runs `optimum-cli export onnx` through the `onnx-convert`
  dependency group with transient `transformers<5` and `torch>=2.11.0`
  overrides, because the current Optimum CLI is not yet compatible with this
  repo's main Transformers 5 runtime dependency.
- `litert`: runs `litert-torch`'s Hugging Face generative exporter via
  `benchmark/runtime/converters/litert_hf_export.py` through the
  `litert-convert` dependency group.
- `litert` (ONNX fallback path): runs `onnx2tf` via
  `benchmark/runtime/converters/onnx_to_litert.py` through the
  `onnx2tf-convert` dependency group for models where only ONNX artifacts are
  available. The converter now applies a targeted `onnx2tf` SequenceEmpty
  hotfix, retries with auto-generated `-prf` replacement JSON when available,
  and can try multiple ONNX source candidates in one run.

Large models should normally be converted on a workstation with enough disk and
RAM, then validated with the same runtime matrix. Converted artifact-map cells
set `artifact_source: converted`, `artifact_health_passed: true`, and
`unblock_platform: true`; this only removes the "no HF artifact" blocker. The
model still remains `staging` until every required platform has passing
correctness, speed, peak-memory, and device-profile reports. If an upstream
exporter does not support a given custom architecture, or if the converted
artifact cannot be loaded by the native runtime, the model stays
blocked/staging until a model-specific converter or runtime fix is added.

PaddleOCR-VL is a useful edge case: Hugging Face currently provides MLX and
ONNX artifacts, but no directly loadable Core ML or LiteRT artifact. The
resolver therefore records Apple fallback from `coreml` to `mlx` and Android
fallback from `litert` to `onnx`. `audit.py` and `promote.py` treat those
fallbacks as staging-only blockers, even if the fallback engine produces
passing reports. Its componentized ONNX artifact should also pass
`artifact_health.py` before matrix execution; this keeps upstream ORT load
crashes, such as a decoder session crash, recorded as a failed health report
instead of aborting the full run. Generate the missing native artifacts with:

```sh
uv run python benchmark/runtime/convert_artifacts.py \
  --model-id paddle_ocr_vl \
  --engine coreml \
  --out benchmark/runtime/artifacts.converted.yaml

uv run python benchmark/runtime/convert_artifacts.py \
  --model-id paddle_ocr_vl \
  --engine litert \
  --out benchmark/runtime/artifacts.converted.yaml
```

For `silero_vad` and `kitten_tts`, Android currently resolves through ONNX
fallback while `onnx2tf` LiteRT conversion attempts are tracked as
engine blockers (currently `onnx2tf_if_subgraph_binding_bug` for `silero_vad`
and `onnx2tf_unsupported_operator_loop` for `kitten_tts`) with per-model
`conversion_record.json`, `onnx_to_litert_report.json`, and `conversion.log`
evidence under
`benchmark/artifacts/converted/<model>/litert/`.

```sh
uv run python benchmark/runtime/run_all.py \
  --artifacts benchmark/runtime/artifacts.local.yaml \
  --plan-out benchmark/out/runtime/plan.json \
  --allow-fail
```

Add `--run` to execute every ready cell in that plan.
`models.yaml` includes every model family already exposed by the package's MLX
model manifest, plus staged expansion targets. Apple Core ML cells use
CoreML-LLM when a same-model baseline exists; other native cells can run
`coreml`, `onnx`, or `litert` as both baseline and candidate from the same
artifact map, or consume pre-collected reports when a platform is remote.
By default, local path existence is checked only for real `--run` executions.
Use `--path-check always` while editing an artifact map to catch unresolved or
missing paths during planning.
Real executions also check that a cell can run on the current host. Run each
platform cell on its target host with `--platform`, or mark non-local cells with
`executor: remote` / `executor: device` when the artifact map points at
pre-collected reports. Android cells may also use `executor: adb`, which pushes
local fixtures to a connected adb device, runs a device-side command or runner
for both baseline and candidate when needed, pulls the reports, then feeds them
through the same `run_matrix.py` compare/promote flow.
For iOS Core ML cells, set `ios_device_smoke: true` in the artifact map to run
`ios_flutter_smoke.py` as an artifact-health stage (real wireless/iOS-device
load + peak-memory snapshot) before local compare.
For Android ONNX/LiteRT cells, set `android_device_smoke: true` to run
`android_flutter_smoke.py` as artifact-health before local compare.
`run_all.py` now surfaces device-smoke/runtime-health failures as structured
`failure_class` + `failure_reason` fields in execution results (for example
`runtime_version_mismatch`, `interpreter_create_failed`,
`section_index_required`, `network_unavailable`).

The public Dart runtime can also resolve built-in `hf://` artifacts directly:

```dart
final registry = RuntimeRegistry.native();
final session = await registry.loadAsync(
  ModelManifest.builtIn()['silero_vad']!,
  options: const RuntimeOptions(engine: RuntimeEngine.onnx),
);
```

`HuggingFaceArtifactCache` stores files under `DART_INFERENCE_HF_CACHE`, or the
platform cache directory when that environment variable is unset. Synchronous
`RuntimeRegistry.load()` remains local-only; use `loadAsync()` whenever remote
artifacts may need to be downloaded. `run_matrix.py`, `run_all.py`, and the
native runners also accept `--artifact hf://...` plus optional
`--hf-cache-root`.

Connected devices can be inspected with:

```sh
uv run python benchmark/runtime/device_probe.py --platform all
```

For iOS wireless devices, run a direct native runtime smoke on the Flutter
example app:

```sh
uv run python benchmark/runtime/ios_flutter_smoke.py \
  --model-id silero_vad \
  --engine coreml

# Optional: force an artifact (for hf:// overrides)
uv run python benchmark/runtime/ios_flutter_smoke.py \
  --model-id silero_vad \
  --engine coreml \
  --artifact hf://FluidInference/silero-vad-coreml/silero-vad-unified-v6.0.0.mlmodelc
```

The app emits chunked runtime-smoke markers
(`DINF_RUNTIME_SMOKE_RESULT_BEGIN/CHUNK/END`) containing load diagnostics and
memory snapshots from `RuntimeRegistry.loadAsync()`. Reports default to
`benchmark/out/runtime/<model>/ios/device_smoke.json` unless `--out` is set.

For Android devices, run the same smoke path through Flutter:

```sh
uv run python benchmark/runtime/android_flutter_smoke.py \
  --model-id qwen2_5 \
  --engine litert \
  --artifact hf://litert-community/Qwen2.5-0.5B-Instruct/Qwen2.5-0.5B-Instruct_seq128_q8_ekv1280.tflite
```

Android reports default to
`benchmark/out/runtime/<model>/android/device_smoke.json`.
The Android smoke runner defaults to `--build-mode release`; pass
`--build-mode debug` when you need full Flutter debug logs.

Android adb execution can collect both baseline and candidate reports on the
device. Pass `remote_baseline_report` / `remote_candidate_report` when reports
already exist, or pass a `device_runner` / `device_command` that writes them:

```sh
uv run python benchmark/runtime/adb_runner.py \
  --model-id qwen3_5 \
  --engine litert \
  --baseline-engine litert \
  --artifact /data/local/tmp/qwen3_5/model.tflite \
  --remote-baseline-report /data/local/tmp/qwen3_5/android_baseline.json \
  --remote-candidate-report /data/local/tmp/qwen3_5/android_candidate.json \
  --device-runner /data/local/tmp/dinf_runner \
  --delegate xnnpack \
  --require-delegate
```

The bundled native CLI can be built as that device runner:

```sh
cmake -S native/runtime -B /tmp/dinf_runtime_android \
  -DDINF_BUILD_CLI=ON \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-26

cmake --build /tmp/dinf_runtime_android --target dinf_runner
adb push /tmp/dinf_runtime_android/libdart_inference_runtime.so \
  /data/local/tmp/libdart_inference_runtime.so
adb push /tmp/dinf_runtime_android/dinf_runner \
  /data/local/tmp/dinf_runner
adb shell chmod +x /data/local/tmp/dinf_runner
```

Use `LD_LIBRARY_PATH=/data/local/tmp` in `device_command` when the runner and
shared library are deployed outside the app sandbox.

When `input_json` contains tensor sidecars via `file` or `path`, `adb_runner.py`
pushes those local files to the device and rewrites the device-side input JSON.
It also pushes optional fixtures such as `prompt_file`, `tools_file`,
`embedding_query_file`, `image_file`, and `audio_file`. Custom
`device_command` templates can reference both local names, for example
`{image_file}`, and device paths, for example `{remote_image_file}`.

Android native runner reports use process PSS as `peak_memory_bytes` and also
include `android_pss`, `android_peak_pss`, `android_native_heap_pss`, and
`android_java_heap_pss` in the raw memory snapshot. The native and Java heap
values are aggregated from `/proc/self/smaps` mappings, so promotion reports
keep the platform raw fields alongside the normalized peak-memory gate.

After a matrix run, `promote.py` converts verdicts into manifest-compatible
validation metadata and marks only fully passing models as `production`:

```sh
uv run python benchmark/runtime/promote.py \
  --out-root benchmark/out/runtime \
  --out benchmark/out/runtime/promotion_patch.json
```

When using converted artifacts, pass the same artifact map to `audit.py` and
`promote.py` so converted cells unblock the catalog-level HF artifact gaps:

```sh
uv run python benchmark/runtime/audit.py \
  --artifacts benchmark/runtime/artifacts.converted.yaml \
  --out-root benchmark/out/runtime

uv run python benchmark/runtime/promote.py \
  --artifacts benchmark/runtime/artifacts.converted.yaml \
  --out-root benchmark/out/runtime \
  --out benchmark/out/runtime/promotion_patch.json
```

The CoreML-LLM baseline path can normalize a pre-existing report recorded for
the same artifact map cell:

```sh
uv run python benchmark/runtime/runners/coreml_llm_runner.py \
  --model-id qwen3_5 \
  --platform ios \
  --artifact "$COREML_LLM_ARTIFACT" \
  --raw-report "$COREML_LLM_RAW_REPORT"
```

or invoke the SwiftPM baseline runner in
`benchmark/coreml-llm/swift_baseline` directly:

```sh
uv run python benchmark/runtime/runners/coreml_llm_runner.py \
  --model-id qwen3_5 \
  --platform ios \
  --artifact "$COREML_LLM_ARTIFACT" \
  --prompt-file benchmark/runtime/fixtures/text_prompt.txt \
  --out benchmark/out/runtime/qwen3_5/ios/coreml_llm.json
```

Task-specific examples:

```sh
uv run python benchmark/runtime/runners/coreml_llm_runner.py \
  --model-id function_gemma \
  --platform ios \
  --artifact "$FUNCTION_GEMMA_COREML_ARTIFACT" \
  --task function \
  --prompt-file benchmark/runtime/fixtures/text_prompt.txt \
  --tools-file benchmark/runtime/fixtures/tool_schema.json \
  --out benchmark/out/runtime/function_gemma/ios/coreml_llm.json

uv run python benchmark/runtime/runners/coreml_llm_runner.py \
  --model-id embedding_gemma \
  --platform ios \
  --artifact "$EMBEDDING_GEMMA_COREML_ARTIFACT" \
  --task embedding \
  --embedding-query-file benchmark/runtime/fixtures/embedding_query.txt \
  --embedding-dim 768 \
  --out benchmark/out/runtime/embedding_gemma/ios/coreml_llm.json
```

`dart_runtime_runner.dart` is the generic Dart entrypoint for native Core ML,
ONNX Runtime, and LiteRT artifacts. It accepts an input JSON file:

Native backend notes:

- Core ML loads `.mlmodelc` / `.mlpackage` artifacts on Apple platforms and
  defaults to `.cpuAndNeuralEngine`. Directory artifacts may contain
  `chunk*.mlmodelc` and `prefill_chunk*.mlmodelc`; use
  `--coreml-mode decode|prefill` to select the chunk group. When diagnostics
  are enabled, the backend records `MLComputePlan` preferred/supported device
  placement counts in the report. Componentized Core ML artifacts can also be
  wrapped in a pipeline JSON:

```json
{
  "format": "dart_inference.coreml_pipeline.v1",
  "stages": [
    {
      "name": "vision",
      "model": "vision_encoder.mlmodelc",
      "outputs": {"last_hidden_state": "vision_tokens"}
    },
    {
      "name": "decoder",
      "model": "decoder.mlmodelc",
      "inputs": {"encoder_hidden_states": "vision_tokens"}
    }
  ],
  "outputs": ["logits"]
}
```

  This is the Core ML counterpart to the ONNX pipeline runner: every stage is a
  real `MLModel` or a small native pipeline op. `inputs` maps stage input names
  to tensors in the pipeline tensor map, and `outputs` renames stage outputs
  for later stages or final report selection. Compute-plan diagnostics audit
  every model stage, so ANE/GPU/CPU placement gates still work for
  componentized models.
- ONNX Runtime is compiled in when the build hook sees
  `DART_INFERENCE_ENABLE_ORT=1`, `DART_INFERENCE_ORT_INCLUDE_DIR`, and
  `DART_INFERENCE_ORT_LIBRARY`.
  Set `DART_INFERENCE_ORT_RUNTIME_LIBRARY` when the link library is an import
  library and the runtime `.dll` / `.so` / `.dylib` must be bundled separately.
  Use `--provider cuda|dml|openvino|qnn|xnnpack|cpu` to request a provider,
  and `--require-provider` when fallback to CPU should fail the run.
  To use the ONNX Runtime Python wheel's bundled dynamic library for native
  smoke tests, run:

```sh
eval "$(uv run python benchmark/runtime/ort_env.py --fetch-headers --shell)"
```

  The helper discovers the wheel library, downloads the matching official C API
  header under `benchmark/artifacts/tools/onnxruntime`, and prints the build
  environment variables consumed by the hook/CMake backend.
  Componentized ONNX artifacts can be wrapped in a pipeline JSON instead of a
  single `.onnx` path:

```json
{
  "format": "dart_inference.onnx_pipeline.v1",
  "stages": [
    {
      "name": "vision",
      "model": "onnx/vision_encoder.onnx",
      "outputs": {"last_hidden_state": "vision_tokens"}
    },
    {
      "name": "decoder",
      "model": "onnx/decoder_model_merged.onnx",
      "inputs": {"encoder_hidden_states": "vision_tokens"}
    }
  ],
  "outputs": ["logits"]
}
```

  Each stage is a real ONNX Runtime session. `inputs` maps stage input names to
  tensors already present in the pipeline tensor map, while `outputs` renames
  stage outputs for later stages or final report selection. Diagnostics include
  per-stage provider/input/output names, and provider gates check every stage.
  Pipeline stages may also use the native op
  `{"op": "scatter_embeddings"}` to replace text embedding rows with visual
  embeddings before a decoder stage. The op expects `base` / `updates` /
  `indices` inputs, defaults to `inputs_embeds` / `image_embeds` /
  `image_token_indices`, and currently supports a single leading batch. This
  is used by componentized VLM artifacts such as PaddleOCR-VL where
  `embed_tokens`, `vision_encoder`, and `decoder` are exported separately.
  To verify the native C API path end to end, run:

```sh
uv run python benchmark/runtime/ort_smoke.py
```

  This builds the native runner with ORT enabled, generates a tiny ONNX graph,
  executes both single-model and two-stage pipeline paths through the C++
  backend, and checks the numeric outputs.
- LiteRT uses the TensorFlow Lite C API and loads
  `DART_INFERENCE_LITERT_LIBRARY` / `DART_INFERENCE_TFLITE_LIBRARY` when provided, falling
  back to a bundled adjacent `libtensorflowlite_c` / `tensorflowlite_c.dll`
  next to the runtime bridge, then the platform library name. The build hook
  bundles that dependency when either environment variable points at a local
  LiteRT/TensorFlow Lite C library. Optional side libraries can be passed with
  `DART_INFERENCE_LITERT_EXTRA_LIBRARIES` (or backend option
  `litertExtraLibraries`), and are preloaded before model creation. This is
  useful for Select TF Ops/Flex (`libtensorflowlite_flex_jni.so`) models.
  XNNPACK is enabled when available unless
  disabled by metadata. `--delegate gpu|nnapi|xnnpack` requests a specific
  delegate, and `--require-delegate` makes missing delegate support fail the
  run instead of falling back.
  `benchmark/runtime/litert_env.py` mirrors `ort_env.py` for runtime setup and,
  on Android, defaults to the Google LiteRT 1.4.x track
  (`com.google.ai.edge.litert:litert`) and can auto-download fallback
  `org.tensorflow:tensorflow-lite` AARs, then extract
  `libtensorflowlite_jni.so` and `libtensorflowlite_flex_jni.so` into
  `benchmark/artifacts/tools/litert/`. Select TF Ops/Flex side libraries are
  optional and no longer block runtime setup when a matching AAR is unavailable.
  Raw `.tflite` files execute directly. `.task` and `.litertlm` containers are
  scanned for embedded `TFL3` flatbuffers; a single embedded model is extracted
  automatically, while multi-section containers require
  `--litert-section-index N` / `backendOptions.litertSectionIndex` or a
  higher-level LiteRT-LM/MediaPipe runner.

All native backend diagnostics include the discovered `input_names` and
`output_names` where the runtime exposes them. These names are kept in matrix
reports to make fixture generation and signature mismatches auditable.

Core ML native execution can be smoke-tested with:

```sh
uv run --with coremltools python benchmark/runtime/coreml_smoke.py
```

The smoke generates a tiny identity Core ML model, compiles it with Xcode's
`coremlcompiler`, runs it through the C++ Core ML backend, and validates the
numeric output plus diagnostics.

```json
{
  "input": {
    "dtype": "float32",
    "shape": [1, 4],
    "values": [1, 2, 3, 4]
  }
}
```

For real model fixtures, the runner also accepts an explicit `inputs` object,
nested numeric arrays, raw base64 tensor bytes, and binary tensor files relative
to the JSON file:

```json
{
  "metadata": {"prompt": "translate: hello"},
  "inputs": {
    "input_ids": {
      "dtype": "int64",
      "shape": [1, 4],
      "values": [[151644, 872, 198, 9906]]
    },
    "pixel_values": {
      "dtype": "float16",
      "shape": [1, 3, 224, 224],
      "file": "pixel_values.f16"
    },
    "attention_mask": {
      "dtype": "int32",
      "shape": [1, 4],
      "base64": "AQAAAAEAAAABAAAAAQAAAA=="
    }
  }
}
```

Supported dtypes are `bool`, `float16`, `float32`, `float64`, `int32`,
`int64`, and `uint8`. File and base64 payloads are raw tensor bytes in host
byte order.

Generate model-shaped inputs from Hugging Face tokenizers/processors with:

```sh
uv run python benchmark/runtime/prepare_inputs.py \
  --model-id qwen3_5 \
  --prompt-file benchmark/runtime/fixtures/text_prompt.txt \
  --out benchmark/out/runtime_inputs/qwen3_5.json
```

VLM processors that depend on `torchvision`, including PaddleOCR-VL-1.5, should
be run with the `vlm-prepare` dependency group:

```sh
uv run --group vlm-prepare python benchmark/runtime/prepare_inputs.py \
  --model-id paddle_ocr_vl \
  --task vlm \
  --trust-remote-code \
  --prompt-file benchmark/runtime/fixtures/text_prompt.txt \
  --image-file benchmark/runtime/fixtures/image.png \
  --out benchmark/out/runtime_inputs/paddle_ocr_vl.json
```

For ONNX smoke runs, pass `--onnx-artifact <model.onnx>` to align the generated
inputs to the graph signature and synthesize zero tensors for required cache or
side inputs that the tokenizer does not produce. `--onnx-artifact` also accepts
`dart_inference.onnx_pipeline.v1` JSON specs; `prepare_inputs.py` walks each
component graph, skips tensors produced by earlier stages, derives
`image_token_indices` for `scatter_embeddings`, preserves ONNX `bool` inputs
such as `use_cache_branch`, and writes large tensors as binary sidecars next to
the JSON so the same fixture works with Dart, native CLI, and adb execution.

`run_all.py --prepare-inputs --run` wires this into the matrix: it generates a
per-cell input JSON under `benchmark/out/runtime/<model>/<platform>/input.json`
before launching the backend runner. Without `--prepare-inputs`, the matrix
uses the explicit `input_json` from the artifact map unchanged.

Native artifact load health checks run before prepared inputs and backend
execution when `--artifact-health-check run` is active. ONNX checks run in the
`onnx-convert` dependency group and load each pipeline model stage in a
subprocess. Core ML and LiteRT checks call the Dart runtime runner in
`--health-check` mode, which creates the native session and records diagnostics
without running inference. In both paths, a native `SIGSEGV` is captured in
`benchmark/out/runtime/_artifact_health/*.json` instead of terminating the
matrix driver. Op-only pipeline stages, such as `scatter_embeddings`, are
validated without ORT.

Example:

```sh
dart run benchmark/runtime/dart_runtime_runner.dart \
  --model-id tiny \
  --engine coreml \
  --artifact "$COREML_ARTIFACT" \
  --input-json benchmark/runtime/fixtures/tiny_input.json \
  --iters 10 \
  --num-threads 4 \
  --out benchmark/out/runtime/tiny/macos/coreml.json
```
