/// Core ML runtime helpers (Phase 2 of the PaddleOCR-VL-1.5
/// re-architecture).
///
/// This file is the Dart-side surface for the iOS17+/macOS14+ stateful
/// Core ML pipeline. The native plumbing lives in
/// `native/runtime/coreml_stateful.{h,mm}`.
///
/// ## `pipeline.json` schema v2
///
/// Schema v2 extends the existing pipeline manifest (`format:
/// "dart_inference.coreml_pipeline.v1"` is still accepted unchanged) with
/// two optional per-stage fields:
///
/// * `compute_units` — one of `"cpu_and_ne"` (alias
///   `"cpu_and_neural_engine"`), `"cpu_and_gpu"`, `"cpu_only"`, or
///   `"all"`. When omitted, the stage inherits the session-wide default
///   passed via the runtime options (`coremlComputeUnits`).
/// * `stateful` — `true` opts the stage in to the iOS17+/macOS14+
///   `MLState` path. When omitted or `false`, the stage runs through the
///   stateless `predictionFromFeatures:error:` API exactly as before.
///
/// Example for the PaddleOCR-VL-1.5 pipeline (ADR §2 / §6):
///
/// ```jsonc
/// {
///   "schema_version": 2,
///   "format": "dart_inference.coreml_pipeline.v1",
///   "stages": [
///     {
///       "name": "vision_embed",
///       "model": "vision_embed.mlpackage",
///       "compute_units": "cpu_and_gpu",
///       "stateful": false,
///       "inputs":  { "pixel_values": "pixel_values",
///                    "input_ids": "input_ids",
///                    "image_grid_thw": "image_grid_thw" },
///       "outputs": { "inputs_embeds": "inputs_embeds" }
///     },
///     {
///       "name": "prefill_decoder",
///       "model": "prefill_decoder.mlpackage",
///       "compute_units": "cpu_and_gpu",
///       "stateful": true,
///       "inputs":  { "inputs_embeds": "inputs_embeds",
///                    "position_ids": "position_ids",
///                    "causal_mask":  "causal_mask" },
///       "outputs": { "last_logits": "last_logits" }
///     },
///     {
///       "name": "decode_decoder",
///       "model": "decode_decoder.mlpackage",
///       "compute_units": "cpu_and_ne",
///       "stateful": true,
///       "inputs":  { "token_embed":  "token_embed",
///                    "position_ids": "position_ids",
///                    "past_kv_len":  "past_kv_len" },
///       "outputs": { "logits": "logits" }
///     }
///   ]
/// }
/// ```
///
/// ## State lifecycle
///
/// `MLState` is allocated lazily on the first `run()` call into a
/// stateful stage and persists across every subsequent `run()` on the
/// same session. To clear the cache between OCR requests call
/// [resetCoremlState]; the next `run()` into a stateful stage will
/// allocate a fresh state. State is also released when the session is
/// closed.
library;

import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart' as pkg_ffi;

import 'native_bindings.dart' as native;

/// Schema version emitted and consumed by Phase 2.
const int coremlPipelineSchemaVersion = 2;

/// Recognised values for the per-stage `compute_units` field in
/// pipeline.json schema v2. Documented here to keep the converter
/// (Phase 1) and the runtime (Phase 2) in sync.
const Set<String> coremlComputeUnitsTokens = <String>{
  'cpu_only',
  'cpu_and_ne',
  'cpu_and_neural_engine',
  'cpu_and_gpu',
  'all',
};

/// Thrown when [resetCoremlState] fails. Wraps the native error string.
class CoremlResetStateException implements Exception {
  CoremlResetStateException(this.code, this.message);

  final int code;
  final String message;

  @override
  String toString() => 'CoremlResetStateException(code=$code, $message)';
}

/// Implemented by native Core ML sessions that can clear their `MLState`.
abstract interface class CoremlStateResettable {
  void resetCoremlState();
}

/// Clear the `MLState` held by a Core ML pipeline session.
///
/// `session` must be the opaque pointer returned by `dinf_open` for a
/// Core ML engine. Throws [CoremlResetStateException] on any non-zero
/// status from the native entry point.
void resetCoremlState(ffi.Pointer<ffi.Void> session) {
  final errorOut = pkg_ffi.calloc<ffi.Pointer<ffi.Char>>();
  try {
    final rc = native.coremlResetState(session, errorOut);
    if (rc != 0) {
      final ptr = errorOut.value;
      final message = ptr == ffi.nullptr
          ? 'unknown error'
          : ptr.cast<pkg_ffi.Utf8>().toDartString();
      if (ptr != ffi.nullptr) {
        native.freeStr(ptr);
      }
      throw CoremlResetStateException(rc, message);
    }
  } finally {
    pkg_ffi.calloc.free(errorOut);
  }
}
