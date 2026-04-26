#include "runtime_bridge.h"

DinfRuntimeSession* dinf_create_coreml_session(
    const char*,
    const DinfOptions*,
    std::string* error) {
  *error = "Core ML backend is only available on Apple platforms.";
  return nullptr;
}
