# dart_inference

`dart_inference` is a Dart and Flutter FFI package for cross-platform model
inference through a pinned native runtime.

The native call stack is intentionally narrow:

```text
Dart API -> dinf_* ABI -> native -> private C/C++/ObjC++ libs
```

Dart exposes the migrated MLX stable/raw tensor APIs through package-owned FFI.
Model sessions, tensor packing, backend dispatch, and native memory ownership
belong behind the native runtime boundary. MLX tensor APIs are exposed through
the package-owned Dart FFI bridge for compatibility with `dart_mlx_ffi`.

## Status

- Package name: `dart_inference`
- Version format: `1.yyyy.commit-count`
- Dart-facing native symbols: `dinf_*`
- Private adapter symbols: `dinf_cpp_*`
- Current native backends: Core ML, ONNX Runtime, LiteRT, MLX on Apple targets
- MLX: stable tensor API, raw bindings, shim bridge, and migrated MLX model
  helpers are available through Dart FFI

## Public Entry Points

- `package:dart_inference/dart_inference.dart`
- `package:dart_inference/runtime.dart`
- `package:dart_inference/models.dart`
- `package:dart_inference/mlx.dart`
- `package:dart_inference/raw.dart`
- `package:dart_inference/dart_mlx_ffi.dart`

## Runtime Example

```dart
import 'dart:typed_data';

import 'package:dart_inference/runtime.dart';

void main() {
  final backend = NativeRuntimeBackend.info();
  print(backend);

  final input = NativeTensorBuffer.float32([1, 4]);
  input.asFloat32List().setAll(0, <double>[1, 2, 3, 4]);

  // Pass input.tensor to a ModelSession.run(...) call. NativeTensorBuffer keeps
  // the input outside the Dart heap and avoids the per-run scratch copy.
  input.close();
}
```

For ordinary Dart typed-data inputs, the runtime copies into reusable native
scratch buffers owned by the session. Outputs are exposed as typed-data views
over native memory and released by `ModelOutputs.close()`.

## Native Build

The package build hook builds the Dart-facing native runtime library:

- `dart_inference_runtime`: native FFI ABI library loaded by Dart FFI
- `dart_inference_mlx`: Apple-target MLX bridge code asset when the platform
  and Metal toolchain are available

No external native compiler toolchain is downloaded by the hook. The runtime
is built from the C/C++/ObjC++ sources under `native/runtime` and `native/mlx_c`
through CMake.

Useful runtime build environment variables:

- `DART_INFERENCE_ENABLE_ORT=1`
- `DART_INFERENCE_ORT_INCLUDE_DIR=/path/to/onnxruntime/include`
- `DART_INFERENCE_ORT_LIBRARY=/path/to/libonnxruntime.so`
- `DART_INFERENCE_ORT_RUNTIME_LIBRARY=/path/to/libonnxruntime.so`
- `DART_INFERENCE_LITERT_LIBRARY=/path/to/libtensorflowlite.so`
- `DART_INFERENCE_LITERT_EXTRA_LIBRARIES=/path/a.so:/path/b.so`

The Dart helpers also read `.dart_inference_runtime_env.json`, or the file named
by `DART_INFERENCE_RUNTIME_ENV_FILE`, for staged native runtime paths.

## Model Helpers

`models.dart` exports runtime-first helpers:

- shared model specs, manifests, tuning, metadata, and small utilities
- Kokoro ONNX TTS helpers
- UniFrontend structured text helpers
- MLX model helpers migrated from `dart_mlx_ffi`
- TTS backend catalog and registry helpers

### UniFrontend TTS Smoke

`dart_inference:tts_backends_status` can audit UniFrontend provider assets and
smoke local ONNX TTS components through the Dart -> native FFI -> ONNX Runtime path:

```sh
dart run dart_inference:tts_backends_status \
  --root /path/to/unifrontend \
  --provider cuda \
  --smoke-onnx
```

The status output keeps graph-level readiness separate from provider-level
synthesis readiness. `graphReadyForSynthesis` only means the declared ONNX
targets exist/load/smoke; `readyForSynthesis` also requires a provider-level
Dart/FFI orchestrator in the catalog. For example, Chatterbox can report all
three declared graphs present while remaining blocked until tokenizer, CFG
sampling, chunk orchestration, and WAV assembly are implemented.

CosyVoice2 status is ONNX-target-first: existing `.pt`, `.zip`, and
`.safetensors` files are reported as export sources, while runtime readiness is
based on the required split ONNX targets (`llm_prefill.onnx`,
`llm_decode.onnx`, `llm_decoder_head.onnx`, `flow.encoder.fp32.onnx`,
`flow.decoder.estimator.fp32.onnx`, `hift.onnx`, and `hift_streaming.onnx`) plus
the runtime sidecars. Generate the flow sidecar once from the PyTorch checkpoint:

```sh
uv run --group cosyvoice2-tools python tool/generate_flow_support.py \
  --model-dir /path/to/unifrontend/src/ttsbackends/providers/cosyvoice2/models/CosyVoice2-0.5B

uv run python tool/generate_qwen2_tokenizer_sidecar.py \
  --tokenizer-dir /path/to/unifrontend/src/ttsbackends/providers/cosyvoice2/models/CosyVoice2-0.5B/CosyVoice-BlankEN \
  --specials cosyvoice2
```

CosyVoice2 can be loaded by `tts_infer` with `--tts-provider cosyvoice2`; the
HTTP and local Flutter clients keep it opt-in through
`DART_TTS_ENABLE_COSYVOICE2=1` until Python-reference parity, longer prompt/text
coverage, and warm performance numbers are recorded. With the sidecars present,
the short no-prompt, prompt, and `hift_streaming` smoke paths run through CUDA.
The Qwen2 tokenizer sidecar is loaded directly by native when present, so runtime
startup does not need to parse `vocab.json` and `merges.txt`.
The current CosyVoice2 provider is still a staging orchestrator: ONNX session
execution stays behind the Dart -> native FFI -> native ONNX Runtime boundary, and
flow-side tensor prep uses Dart helpers, but the remaining autoregressive and
diffusion scheduler loops should move behind that boundary before marking the
provider production ready.

Sarashina2.2-TTS is available through the direct Dart/FFI runtime. The direct
path can decode externally generated `<|semantic_N|>` tokens once the
CosyVoice2-style flow, HiFT, campplus, speech-tokenizer, and support npz assets
have been exported into the model directory. Token parsing, prompt formatting,
semantic sampling, and the base tokenizer use the Dart -> native runtime ABI; raw
text generation enters the split Llama ONNX prefill/decode graphs through the
native `tokenizer.sara2tok` runtime. The split Llama driver expects a token
embedding sidecar generated from `model.safetensors`:

```sh
uv run --group cosyvoice2-tools python tool/generate_flow_support.py \
  --model-dir /path/to/unifrontend/src/ttsbackends/providers/sarashina2-tts/models/sarashina2.2-tts

uv run python tool/generate_sarashina2_llm_embeddings.py \
  --model-dir /path/to/unifrontend/src/ttsbackends/providers/sarashina2-tts/models/sarashina2.2-tts

uv run python tool/generate_sarashina2_tokenizer_sidecar.py \
  --model-dir /path/to/unifrontend/src/ttsbackends/providers/sarashina2-tts/models/sarashina2.2-tts
```

Use the model-specific CLI for direct synthesis without starting a server or
loading the generic TTS registry:

```sh
dart run bin/sarashina2_synthesize.dart \
  --model-dir /path/to/unifrontend/src/ttsbackends/providers/sarashina2-tts/models/sarashina2.2-tts \
  --text "こんにちは" \
  --output-wav out.wav \
  --json
```

For the fastest local NVIDIA path, build the Sarashina2 TensorRT EPContext once.
This keeps synthesis as a direct Dart -> native FFI -> ONNX Runtime call; it does not
start a TTS server. The context is auto-discovered from the UniFrontend artifact
directory by later `--provider tensorrt` runs:

```sh
dart run tool/prepare_sarashina2_tensorrt_context.dart \
  --root /path/to/unifrontend \
  --force \
  --json

dart run bin/sarashina2_synthesize.dart \
  --model-dir /path/to/unifrontend/src/ttsbackends/providers/sarashina2-tts/models/sarashina2.2-tts \
  --provider tensorrt \
  --text "こんにちは" \
  --output-wav out.wav \
  --json
```

Use `--provider cuda` for the standard ONNX Runtime CUDA path, `--provider cpu`
for CPU validation, or add `--allow-cpu-fallback` when a CUDA/TensorRT request
should retry CPU instead of failing if the requested GPU provider cannot
initialize.

The same CLI also accepts pre-tokenized prompt ids with `--prompt-token-ids` or
decoder-only semantic tokens with `--semantic-token-text`,
`--semantic-token-file`, `--semantic-tokens`, or `--semantic-tokens-file`.
Prompt audio can condition both the semantic generator and flow decoder when
`--prompt-wav` is paired with the prompt transcription in `--prompt-text`;
without `--prompt-text`, prompt audio is used for flow decoding only. Semantic
generation defaults to the upstream-style sampler (`temperature=0.9`,
`topP=0.95`, `frequencyPenalty=1.0`), with candidate filtering and sampling
handled by native; pass `--temperature`, `--top-p`, `--frequency-penalty`, and
`--seed` for reproducible local runs. A single CLI process reuses the loaded
runtime across `--repeat` runs and extracts `--prompt-wav` once before the loop,
while each generated sequence keeps its own LLM KV-cache. In Dart code, build a
`Sarashina2SynthesisRequest` and pass it to the direct `runtime.synthesize(...)`
API; it accepts raw text, `promptTokenIds`, or external semantic tokens and
returns WAV bytes plus semantic/timing metadata. For repeated calls with one
speaker, call `runtime.extractPrompt(promptAudio)` once and include the returned
`Sarashina2Prompt` in each request.
The upstream Python generator embeds a SilentCipher watermark by default; the
current Dart/FFI runtime does not yet implement that watermark step and reports
it as not embedded.

For LLM/tokenizer-only validation, use the smoke runner before loading the
flow/HiFT decoder:

```sh
dart run bin/sarashina2_llm_smoke.dart \
  --model-dir /path/to/unifrontend/src/ttsbackends/providers/sarashina2-tts/models/sarashina2.2-tts \
  --text "hello" \
  --steps 8 \
  --json
```

NeuTTS Air is registered as `neutts-air`. The Dart/FFI layer can now load its
HuggingFace `tokenizer.json`, resolve the NeuTTS prompt special-token ids, build
the upstream prompt token layout, and parse/format `<|speech_N|>` codec tokens
through native. When `neutts_air_lm.onnx` and `neucodec_decoder.onnx` are present,
provider-level synthesis uses Qwen2 native tokenization, no-cache LM ONNX logits,
native int64 decode-token buffers, native top-k/top-p speech-token sampling,
NeuCodec ONNX, and shared WAV output.
The HTTP server can be opted in with
`DART_TTS_ENABLE_NEUTTS_AIR=1`; `/synthesize` accepts `codecTokenText`,
`codecTokens`, `referencePhones`, `inputPhones`, and `referenceCodes` for this
path. Split prefill/decode KV graphs remain the performance target.

For lower startup overhead, generate the Qwen2 tokenizer sidecar once. Use the
NeuTTS preset so the Dart tokenizer registers only the prompt control tokens, not
all 65k `<|speech_N|>` codec tokens:

```sh
uv run python tool/generate_qwen2_tokenizer_sidecar.py \
  --model-dir /path/to/unifrontend/src/ttsbackends/providers/neutts-air/models \
  --specials neutts-air
```

The provider asset audit is also ONNX-target-first for the remaining local TTS
backends. It reports each target graph path, the source weight file it should be
exported from, missing required ONNX assets, and whether that provider can run
through the pure `Dart -> native FFI -> ONNX Runtime` path today.

Catalog-declared TTS ONNX targets also get a generic component bundle in the
status output. When an exported target graph appears on disk, the same bundle
can load it, run it by component name, and build synthetic smoke inputs from
ONNX Runtime metadata without adding provider-specific Dart glue.

Use `--provider tensorrt --trt-cache-dir <dir> --trt-workspace-mb <mb>` to
benchmark TensorRT EP where ONNX Runtime exposes it. Add `--trt-fp16` only when
the graph has been checked for acceptable FP16 parity. The status output also
includes a TensorRT dependency audit, so missing `libnvinfer*` libraries are
reported separately from model-load failures and TensorRT loads are skipped
until the runtime dependencies are visible. `tts_infer`, `tts_server`,
`structured_frontend_infer`, `structured_smoke_infer`, and `onnx_server` use
the same preflight check for strict TensorRT requests.

## Local Validation

```sh
dart analyze
dart test
dart pub publish --dry-run
```

The exact native library path varies by host architecture and platform.
