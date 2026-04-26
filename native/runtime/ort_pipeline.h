#pragma once

#include "runtime_bridge.h"

#include <filesystem>
#include <memory>
#include <string>

#if DINF_ENABLE_ORT
#include "nlohmann/json.hpp"

namespace dinf_ort {

bool IsPipelineSpec(
    const std::filesystem::path& path,
    nlohmann::json* spec);

std::unique_ptr<DinfRuntimeSession> CreatePipeline(
    const std::filesystem::path& spec_path,
    const nlohmann::json& spec,
    const char* options_json,
    std::string* error);

}  // namespace dinf_ort

#endif
