#pragma once

// Stateful Core ML session helpers (iOS18+/macOS15+ MLState).
//
// Phase 2 of the PaddleOCR-VL-1.5 re-architecture. See
// `mise-kitchen-prep/docs/adr/0001-paddleocr-vl-coreml-rearchitecture.md`
// §6 and `mise-kitchen-prep/docs/plans/paddleocr-vl-coreml-rebuild.md`
// Phase 2 for the contract this header implements.
//
// The non-stateful Core ML path in `coreml_backend.mm` is unchanged.
// The hooks in this header are only consumed when a stage in
// `pipeline.json` (schema v2) opts in via `"stateful": true`.

#include <cstdint>
#include <string>

#include "nlohmann/json.hpp"
#include "runtime_bridge.h"

#if defined(__APPLE__)
#import <CoreML/CoreML.h>
#endif

// ---------------------------------------------------------------------------
// Schema v2 parsing helpers (pipeline.json).
// ---------------------------------------------------------------------------

// Parse a per-stage `compute_units` string from pipeline.json schema v2.
// Accepts: "cpu_and_ne" / "cpu_and_neural_engine", "cpu_and_gpu",
// "cpu_only", "all". Falls back to `fallback` when the value is missing
// or unrecognised (mirrors the legacy session-wide default).
//
// Returned as the platform's `MLComputeUnits` enum on Apple builds, and
// as an opaque int (the underlying enum value) on stub builds so the
// header stays cross-platform.
#if defined(__APPLE__)
MLComputeUnits dinf_coreml_parse_compute_units(
    const nlohmann::json& stage,
    MLComputeUnits fallback);
#endif

// Returns true when the JSON stage object opts in to stateful prediction
// (`"stateful": true`). Defaults to false when the field is absent.
bool dinf_coreml_stage_is_stateful(const nlohmann::json& stage);

// ---------------------------------------------------------------------------
// Reset hook (cross-platform). The Core ML session implements this; other
// backends keep the default no-op which surfaces as an error from the C
// entry point so callers cannot accidentally reset a non-Core ML session.
// ---------------------------------------------------------------------------

class DinfCoreMlResettable {
 public:
  virtual ~DinfCoreMlResettable() = default;

  // Returns 0 on success. Non-zero with `*error` populated on failure.
  // Implementations should be a no-op (returning 0) when no stateful
  // stages were configured.
  virtual int ResetCoremlState(std::string* error) = 0;
};

// ---------------------------------------------------------------------------
// State store, owned by a stateful Core ML session.
// ---------------------------------------------------------------------------
//
// Only declared as an opaque forward on non-Apple builds; the stub TU
// never instantiates it.
#if defined(__APPLE__)

API_AVAILABLE(macos(15.0), ios(18.0))
@interface DinfCoreMlStatefulStore : NSObject

// Returns the existing state for `stageName`, lazily allocating a fresh
// state from `model` on first access. Subsequent calls return the same
// state pointer until `reset` is invoked.
- (MLState *)stateForStage:(NSString*)stageName fromModel:(MLModel*)model;

// Drops every retained state. The next `stateForStage:fromModel:` call
// for any stage will allocate a new one.
- (void)reset;

@end

// Run a single stateful prediction. Encapsulates the
// `predictionFromFeatures:usingState:options:error:` invocation along
// with the iOS18/macOS15 availability guard. Returns nil and writes a
// human-readable error string into `error` on failure (including when
// the deployment target lacks MLState support).
id<MLFeatureProvider> dinf_coreml_predict_with_state(
    MLModel* model,
    id<MLFeatureProvider> inputs,
    DinfCoreMlStatefulStore* store,
    NSString* stageName,
    std::string* error);

#endif  // __APPLE__

// ---------------------------------------------------------------------------
// C entry point (also wired into runtime_bridge.h).
// ---------------------------------------------------------------------------

extern "C" {

// Clear the Core ML stateful KV cache associated with `session`.
//
// `session` must be a session previously returned by `dinf_cpp_open`
// with `engine == DINF_ENGINE_COREML`. After this call returns 0, the
// next `dinf_cpp_run` invocation that hits a stateful stage will
// allocate a fresh `MLState` instead of reusing the previous one.
//
// Returns 0 on success. Returns a non-zero error code and writes a
// newly-allocated error string into `*error` (caller releases via
// `dinf_cpp_free_str`) on failure: passing a null session, a session
// that is not a Core ML session, or a session whose pipeline declared
// no stateful stages.
//
// Thread-safety: callers must serialise this against `dinf_cpp_run` on
// the same session. The runtime does not lock for them.
DINF_RUNTIME_EXPORT int32_t dinf_coreml_reset_state(
    DinfRuntimeSession* session,
    char** error);

}  // extern "C"
