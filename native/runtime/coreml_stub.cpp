#include "runtime_bridge.h"

DinfRuntimeSession* dinf_create_coreml_session(
    const char*,
    const DinfOptions*,
    std::string* error) {
  *error = "Core ML backend is only available on Apple platforms.";
  return nullptr;
}

extern "C" int32_t dinf_coreml_reset_state(
    DinfRuntimeSession*,
    char** error) {
  if (error != nullptr) {
    *error = dinf_copy_string(
        "Core ML backend is only available on Apple platforms.");
  }
  return 1;
}
