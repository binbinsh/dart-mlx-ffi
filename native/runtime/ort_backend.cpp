#include "runtime_bridge.h"

#if DINF_ENABLE_ORT
#include "ort_pipeline.h"
#include "ort_session.h"
#endif

#if !DINF_ENABLE_ORT
namespace {

std::string ort_disabled_error() {
  return "ONNX Runtime backend was built without ORT headers/library. "
         "Set DART_INFERENCE_ENABLE_ORT=1 and provide DART_INFERENCE_ORT_INCLUDE_DIR "
         "and DART_INFERENCE_ORT_LIBRARY when building.";
}

}  // namespace
#endif

DinfRuntimeSession* dinf_create_onnx_session(
    const char* model_path,
    const char* options_json,
    std::string* error) {
#if DINF_ENABLE_ORT
  nlohmann::json spec;
  if (dinf_ort::IsPipelineSpec(model_path, &spec)) {
    auto pipeline =
        dinf_ort::CreatePipeline(model_path, spec, options_json, error);
    return pipeline.release();
  }
  auto session = dinf_ort::CreateSession(model_path, options_json, error);
  return session.release();
#else
  (void)model_path;
  (void)options_json;
  *error = ort_disabled_error();
  return nullptr;
#endif
}
