#pragma once

#include "runtime_bridge.h"

#include <string>

using DinfPipelineStageFactory = DinfRuntimeSession* (*)(
    const char* model_path,
    const DinfOptions* options,
    std::string* error);

DinfRuntimeSession* dinf_try_create_pipeline_session(
    const char* model_path,
    const DinfOptions* options,
    const char* engine_name,
    const char* pipeline_format,
    DinfPipelineStageFactory stage_factory,
    std::string* error);
