std::string ProviderBoolValue(bool value) {
  return value ? "1" : "0";
}

bool AddSessionConfigBool(
    const OrtApi* api,
    OrtSessionOptions* options,
    const DinfOptions* runtime_options,
    const char* option_key,
    const char* config_key,
    std::string* error) {
  if (!dinf_option_exists(runtime_options, option_key)) {
    return true;
  }
  const std::string value =
      ProviderBoolValue(dinf_option_bool(runtime_options, option_key, false));
  return Ok(api, api->AddSessionConfigEntry(options, config_key, value.c_str()), error);
}

bool ApplySessionConfigEntries(
    const OrtApi* api,
    OrtSessionOptions* options,
    const DinfOptions* runtime_options,
    std::string* error) {
  return AddSessionConfigBool(
             api,
             options,
             runtime_options,
             "useEnvAllocators",
             "session.use_env_allocators",
             error) &&
         AddSessionConfigBool(
             api,
             options,
             runtime_options,
             "ortUseEnvAllocators",
             "session.use_env_allocators",
             error) &&
         AddSessionConfigBool(
             api,
             options,
             runtime_options,
             "disableCpuEpFallback",
             "session.disable_cpu_ep_fallback",
             error) &&
         AddSessionConfigBool(
             api,
             options,
             runtime_options,
             "ortDisableCpuEpFallback",
             "session.disable_cpu_ep_fallback",
             error);
}

void AddProviderOption(
    std::vector<std::string>* keys,
    std::vector<std::string>* values,
    const char* key,
    const std::string& value) {
  keys->emplace_back(key);
  values->push_back(value);
}

bool AddBoolProviderOption(
    std::vector<std::string>* keys,
    std::vector<std::string>* values,
    const DinfOptions* runtime_options,
    const char* option_key,
    const char* provider_key) {
  if (!dinf_option_exists(runtime_options, option_key)) {
    return false;
  }
  AddProviderOption(
      keys,
      values,
      provider_key,
      ProviderBoolValue(dinf_option_bool(runtime_options, option_key, false)));
  return true;
}

bool AddIntProviderOption(
    std::vector<std::string>* keys,
    std::vector<std::string>* values,
    const DinfOptions* runtime_options,
    const char* option_key,
    const char* provider_key) {
  if (!dinf_option_exists(runtime_options, option_key)) {
    return false;
  }
  AddProviderOption(
      keys,
      values,
      provider_key,
      std::to_string(dinf_option_int(runtime_options, option_key, 0)));
  return true;
}

bool AddStringProviderOption(
    std::vector<std::string>* keys,
    std::vector<std::string>* values,
    const DinfOptions* runtime_options,
    const char* option_key,
    const char* provider_key) {
  const std::string value = dinf_option_string(runtime_options, option_key);
  if (value.empty()) {
    return false;
  }
  AddProviderOption(keys, values, provider_key, value);
  return true;
}

void AppendProviderOptionPtrs(
    const std::vector<std::string>& keys,
    const std::vector<std::string>& values,
    std::vector<const char*>* key_ptrs,
    std::vector<const char*>* value_ptrs) {
  key_ptrs->reserve(keys.size());
  value_ptrs->reserve(values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    key_ptrs->push_back(keys[i].c_str());
    value_ptrs->push_back(values[i].c_str());
  }
}

bool RequiresCudaV2Options(const DinfOptions* runtime_options) {
  return dinf_option_exists(runtime_options, "cudaEnableGraph") ||
         dinf_option_exists(runtime_options, "enableCudaGraph") ||
         dinf_option_exists(runtime_options, "cudaUseTf32") ||
         dinf_option_exists(runtime_options, "cudaTunableOpEnable") ||
         dinf_option_exists(runtime_options, "cudaTunableOpTuningEnable") ||
         dinf_option_exists(runtime_options, "cudaTunableOpMaxTuningMs");
}

bool AppendCudaProviderV2(
    const OrtApi* api,
    OrtSessionOptions* options,
    const DinfOptions* runtime_options,
    std::string* error) {
  if (api->CreateCUDAProviderOptions == nullptr ||
      api->UpdateCUDAProviderOptions == nullptr ||
      api->SessionOptionsAppendExecutionProvider_CUDA_V2 == nullptr ||
      api->ReleaseCUDAProviderOptions == nullptr) {
    *error = "ONNX Runtime CUDA V2 provider options are unavailable";
    return false;
  }
  OrtCUDAProviderOptionsV2* cuda_options = nullptr;
  if (!Ok(api, api->CreateCUDAProviderOptions(&cuda_options), error)) {
    return false;
  }
  std::vector<std::string> keys;
  std::vector<std::string> values;
  AddProviderOption(
      &keys,
      &values,
      "device_id",
      std::to_string(std::max(0, dinf_option_int(runtime_options, "deviceId", 0))));
  const int cuda_mem_limit_mb =
      dinf_option_int(runtime_options, "cudaMemoryLimitMb", 0);
  if (cuda_mem_limit_mb > 0) {
    AddProviderOption(
        &keys,
        &values,
        "gpu_mem_limit",
        std::to_string(static_cast<size_t>(cuda_mem_limit_mb) * 1024ULL * 1024ULL));
  }
  const int arena_extend_strategy =
      dinf_option_int(runtime_options, "cudaArenaExtendStrategy", -1);
  if (arena_extend_strategy >= 0) {
    AddProviderOption(
        &keys,
        &values,
        "arena_extend_strategy",
        std::to_string(arena_extend_strategy));
  }
  if (!AddBoolProviderOption(
          &keys,
          &values,
          runtime_options,
          "cudaDoCopyInDefaultStream",
          "do_copy_in_default_stream")) {
    AddBoolProviderOption(
        &keys,
        &values,
        runtime_options,
        "doCopyInDefaultStream",
        "do_copy_in_default_stream");
  }
  if (!AddBoolProviderOption(
          &keys,
          &values,
          runtime_options,
          "cudaEnableGraph",
          "enable_cuda_graph")) {
    AddBoolProviderOption(
        &keys,
        &values,
        runtime_options,
        "enableCudaGraph",
        "enable_cuda_graph");
  }
  AddBoolProviderOption(
      &keys, &values, runtime_options, "cudaUseTf32", "use_tf32");
  AddBoolProviderOption(
      &keys,
      &values,
      runtime_options,
      "cudaTunableOpEnable",
      "tunable_op_enable");
  AddBoolProviderOption(
      &keys,
      &values,
      runtime_options,
      "cudaTunableOpTuningEnable",
      "tunable_op_tuning_enable");
  const int tunable_max_ms =
      dinf_option_int(runtime_options, "cudaTunableOpMaxTuningMs", -1);
  if (tunable_max_ms >= 0) {
    AddProviderOption(
        &keys,
        &values,
        "tunable_op_max_tuning_duration_ms",
        std::to_string(tunable_max_ms));
  }

  std::vector<const char*> key_ptrs;
  std::vector<const char*> value_ptrs;
  AppendProviderOptionPtrs(keys, values, &key_ptrs, &value_ptrs);
  const bool updated = Ok(
      api,
      api->UpdateCUDAProviderOptions(
          cuda_options, key_ptrs.data(), value_ptrs.data(), key_ptrs.size()),
      error);
  if (!updated) {
    api->ReleaseCUDAProviderOptions(cuda_options);
    return false;
  }
  const bool appended = Ok(
      api,
      api->SessionOptionsAppendExecutionProvider_CUDA_V2(options, cuda_options),
      error);
  api->ReleaseCUDAProviderOptions(cuda_options);
  return appended;
}

bool RequiresTensorRtV2Options(const DinfOptions* runtime_options) {
  return dinf_option_exists(runtime_options, "trtEngineCachePrefix") ||
         dinf_option_exists(runtime_options, "trtTimingCacheEnable") ||
         dinf_option_exists(runtime_options, "trtTimingCachePath") ||
         dinf_option_exists(runtime_options, "trtForceTimingCache") ||
         dinf_option_exists(runtime_options, "trtContextMemorySharingEnable") ||
         dinf_option_exists(runtime_options, "trtLayerNormFp32Fallback") ||
         dinf_option_exists(runtime_options, "trtCudaGraphEnable") ||
         dinf_option_exists(runtime_options, "trtBuilderOptimizationLevel") ||
         dinf_option_exists(runtime_options, "trtAuxiliaryStreams") ||
         dinf_option_exists(runtime_options, "trtTacticSources") ||
         dinf_option_exists(runtime_options, "trtExtraPluginLibPaths") ||
         dinf_option_exists(runtime_options, "trtDetailedBuildLog") ||
         dinf_option_exists(runtime_options, "trtBuildHeuristicsEnable") ||
         dinf_option_exists(runtime_options, "trtSparsityEnable") ||
         dinf_option_exists(runtime_options, "trtProfileMinShapes") ||
         dinf_option_exists(runtime_options, "trtProfileMaxShapes") ||
         dinf_option_exists(runtime_options, "trtProfileOptShapes") ||
         dinf_option_exists(runtime_options, "trtEngineHwCompatible") ||
         dinf_option_exists(runtime_options, "trtOpTypesToExclude") ||
         dinf_option_exists(runtime_options, "trtDumpEpContextModel") ||
         dinf_option_exists(runtime_options, "trtEpContextFilePath") ||
         dinf_option_exists(runtime_options, "trtEpContextEmbedMode");
}

bool AppendTensorRtProviderV2(
    const OrtApi* api,
    OrtSessionOptions* options,
    const DinfOptions* runtime_options,
    std::string* error) {
  if (api->CreateTensorRTProviderOptions == nullptr ||
      api->UpdateTensorRTProviderOptions == nullptr ||
      api->SessionOptionsAppendExecutionProvider_TensorRT_V2 == nullptr ||
      api->ReleaseTensorRTProviderOptions == nullptr) {
    *error = "ONNX Runtime TensorRT V2 provider options are unavailable";
    return false;
  }
  OrtTensorRTProviderOptionsV2* trt_options = nullptr;
  if (!Ok(api, api->CreateTensorRTProviderOptions(&trt_options), error)) {
    return false;
  }

  std::vector<std::string> keys;
  std::vector<std::string> values;
  AddProviderOption(
      &keys,
      &values,
      "device_id",
      std::to_string(std::max(0, dinf_option_int(runtime_options, "deviceId", 0))));
  AddIntProviderOption(
      &keys,
      &values,
      runtime_options,
      "trtMaxPartitionIterations",
      "trt_max_partition_iterations");
  AddIntProviderOption(
      &keys,
      &values,
      runtime_options,
      "trtMinSubgraphSize",
      "trt_min_subgraph_size");
  const int trt_workspace_mb =
      dinf_option_int(runtime_options, "trtWorkspaceMemoryLimitMb", 0);
  if (trt_workspace_mb > 0) {
    AddProviderOption(
        &keys,
        &values,
        "trt_max_workspace_size",
        std::to_string(static_cast<size_t>(trt_workspace_mb) * 1024ULL * 1024ULL));
  }

  AddBoolProviderOption(&keys, &values, runtime_options, "trtFp16", "trt_fp16_enable");
  AddBoolProviderOption(&keys, &values, runtime_options, "trtInt8", "trt_int8_enable");
  AddStringProviderOption(
      &keys,
      &values,
      runtime_options,
      "trtInt8CalibrationTableName",
      "trt_int8_calibration_table_name");
  AddBoolProviderOption(
      &keys,
      &values,
      runtime_options,
      "trtInt8UseNativeCalibrationTable",
      "trt_int8_use_native_calibration_table");
  AddBoolProviderOption(&keys, &values, runtime_options, "trtDlaEnable", "trt_dla_enable");
  AddIntProviderOption(&keys, &values, runtime_options, "trtDlaCore", "trt_dla_core");
  AddBoolProviderOption(
      &keys, &values, runtime_options, "trtDumpSubgraphs", "trt_dump_subgraphs");

  const std::string trt_cache_path =
      dinf_option_string(runtime_options, "trtEngineCachePath");
  if (!trt_cache_path.empty()) {
    AddProviderOption(&keys, &values, "trt_engine_cache_enable", "1");
    AddProviderOption(&keys, &values, "trt_engine_cache_path", trt_cache_path);
  } else {
    AddBoolProviderOption(
        &keys,
        &values,
        runtime_options,
        "trtEngineCacheEnable",
        "trt_engine_cache_enable");
  }
  AddStringProviderOption(
      &keys,
      &values,
      runtime_options,
      "trtEngineCachePrefix",
      "trt_engine_cache_prefix");
  AddBoolProviderOption(
      &keys,
      &values,
      runtime_options,
      "trtForceSequentialEngineBuild",
      "trt_force_sequential_engine_build");

  const std::string trt_timing_cache_path =
      dinf_option_string(runtime_options, "trtTimingCachePath");
  if (!trt_timing_cache_path.empty()) {
    AddProviderOption(&keys, &values, "trt_timing_cache_enable", "1");
    AddProviderOption(&keys, &values, "trt_timing_cache_path", trt_timing_cache_path);
  } else {
    AddBoolProviderOption(
        &keys,
        &values,
        runtime_options,
        "trtTimingCacheEnable",
        "trt_timing_cache_enable");
  }
  AddBoolProviderOption(
      &keys, &values, runtime_options, "trtForceTimingCache", "trt_force_timing_cache");
  AddBoolProviderOption(
      &keys,
      &values,
      runtime_options,
      "trtContextMemorySharingEnable",
      "trt_context_memory_sharing_enable");
  AddBoolProviderOption(
      &keys,
      &values,
      runtime_options,
      "trtLayerNormFp32Fallback",
      "trt_layer_norm_fp32_fallback");
  AddBoolProviderOption(
      &keys, &values, runtime_options, "trtCudaGraphEnable", "trt_cuda_graph_enable");
  AddIntProviderOption(
      &keys,
      &values,
      runtime_options,
      "trtBuilderOptimizationLevel",
      "trt_builder_optimization_level");
  AddIntProviderOption(
      &keys, &values, runtime_options, "trtAuxiliaryStreams", "trt_auxiliary_streams");
  AddStringProviderOption(
      &keys, &values, runtime_options, "trtTacticSources", "trt_tactic_sources");
  AddStringProviderOption(
      &keys,
      &values,
      runtime_options,
      "trtExtraPluginLibPaths",
      "trt_extra_plugin_lib_paths");
  AddBoolProviderOption(
      &keys, &values, runtime_options, "trtDetailedBuildLog", "trt_detailed_build_log");
  AddBoolProviderOption(
      &keys,
      &values,
      runtime_options,
      "trtBuildHeuristicsEnable",
      "trt_build_heuristics_enable");
  AddBoolProviderOption(
      &keys, &values, runtime_options, "trtSparsityEnable", "trt_sparsity_enable");
  AddStringProviderOption(
      &keys, &values, runtime_options, "trtProfileMinShapes", "trt_profile_min_shapes");
  AddStringProviderOption(
      &keys, &values, runtime_options, "trtProfileMaxShapes", "trt_profile_max_shapes");
  AddStringProviderOption(
      &keys, &values, runtime_options, "trtProfileOptShapes", "trt_profile_opt_shapes");
  AddBoolProviderOption(
      &keys,
      &values,
      runtime_options,
      "trtEngineHwCompatible",
      "trt_engine_hw_compatible");
  AddStringProviderOption(
      &keys, &values, runtime_options, "trtOpTypesToExclude", "trt_op_types_to_exclude");
  AddBoolProviderOption(
      &keys,
      &values,
      runtime_options,
      "trtDumpEpContextModel",
      "trt_dump_ep_context_model");
  AddStringProviderOption(
      &keys, &values, runtime_options, "trtEpContextFilePath", "trt_ep_context_file_path");
  AddBoolProviderOption(
      &keys, &values, runtime_options, "trtEpContextEmbedMode", "trt_ep_context_embed_mode");

  std::vector<const char*> key_ptrs;
  std::vector<const char*> value_ptrs;
  AppendProviderOptionPtrs(keys, values, &key_ptrs, &value_ptrs);
  const bool updated = Ok(
      api,
      api->UpdateTensorRTProviderOptions(
          trt_options, key_ptrs.data(), value_ptrs.data(), key_ptrs.size()),
      error);
  if (!updated) {
    api->ReleaseTensorRTProviderOptions(trt_options);
    return false;
  }
  const bool appended = Ok(
      api,
      api->SessionOptionsAppendExecutionProvider_TensorRT_V2(options, trt_options),
      error);
  api->ReleaseTensorRTProviderOptions(trt_options);
  return appended;
}

bool AppendProvider(
    const OrtApi* api,
    OrtSessionOptions* options,
    const std::string& provider,
    const std::vector<std::string>& providers,
    const DinfOptions* runtime_options,
    bool* appended,
    std::string* error) {
  if (appended != nullptr) {
    *appended = false;
  }
  if (provider.empty() || provider == "CPUExecutionProvider") {
    return true;
  }
  if (!ContainsProvider(providers, provider)) {
    if (dinf_option_bool(runtime_options, "requireProvider", false)) {
      *error = "Requested ONNX Runtime provider is unavailable: " + provider;
      return false;
    }
    return true;
  }

  if (provider == "CUDAExecutionProvider" &&
      api->SessionOptionsAppendExecutionProvider_CUDA != nullptr) {
    std::string cuda_v2_error;
    if (AppendCudaProviderV2(api, options, runtime_options, &cuda_v2_error)) {
      if (appended != nullptr) {
        *appended = true;
      }
      return true;
    }
    if (RequiresCudaV2Options(runtime_options)) {
      if (dinf_option_bool(runtime_options, "requireProvider", false)) {
        *error = cuda_v2_error;
        return false;
      }
      return true;
    }
    OrtCUDAProviderOptions cuda_options{};
    cuda_options.device_id =
        std::max(0, dinf_option_int(runtime_options, "deviceId", 0));
    const int cuda_mem_limit_mb =
        dinf_option_int(runtime_options, "cudaMemoryLimitMb", 0);
    if (cuda_mem_limit_mb > 0) {
      cuda_options.gpu_mem_limit =
          static_cast<size_t>(cuda_mem_limit_mb) * 1024ULL * 1024ULL;
    }
    const int arena_extend_strategy =
        dinf_option_int(runtime_options, "cudaArenaExtendStrategy", -1);
    if (arena_extend_strategy >= 0) {
      cuda_options.arena_extend_strategy = arena_extend_strategy;
    }
    if (dinf_option_exists(runtime_options, "cudaDoCopyInDefaultStream")) {
      cuda_options.do_copy_in_default_stream =
          dinf_option_bool(runtime_options, "cudaDoCopyInDefaultStream", true)
          ? 1
          : 0;
    }
    std::string append_error;
    if (Ok(api,
           api->SessionOptionsAppendExecutionProvider_CUDA(
               options,
               &cuda_options),
           &append_error)) {
      if (appended != nullptr) {
        *appended = true;
      }
      return true;
    }
    if (dinf_option_bool(runtime_options, "requireProvider", false)) {
      *error = append_error;
      return false;
    }
  }

  if (provider == "TensorrtExecutionProvider" &&
      (api->SessionOptionsAppendExecutionProvider_TensorRT != nullptr ||
       api->SessionOptionsAppendExecutionProvider_TensorRT_V2 != nullptr)) {
    std::string trt_v2_error;
    if (AppendTensorRtProviderV2(api, options, runtime_options, &trt_v2_error)) {
      if (appended != nullptr) {
        *appended = true;
      }
      return true;
    }
    if (RequiresTensorRtV2Options(runtime_options)) {
      if (dinf_option_bool(runtime_options, "requireProvider", false)) {
        *error = trt_v2_error;
        return false;
      }
      return true;
    }
    if (api->SessionOptionsAppendExecutionProvider_TensorRT == nullptr) {
      if (dinf_option_bool(runtime_options, "requireProvider", false)) {
        *error = trt_v2_error;
        return false;
      }
      return true;
    }
    OrtTensorRTProviderOptions trt_options{};
    trt_options.device_id =
        std::max(0, dinf_option_int(runtime_options, "deviceId", 0));
    trt_options.trt_max_partition_iterations = std::max(
        0,
        dinf_option_int(runtime_options, "trtMaxPartitionIterations", 0));
    trt_options.trt_min_subgraph_size =
        std::max(0, dinf_option_int(runtime_options, "trtMinSubgraphSize", 0));
    const int trt_workspace_mb =
        dinf_option_int(runtime_options, "trtWorkspaceMemoryLimitMb", 0);
    if (trt_workspace_mb > 0) {
      trt_options.trt_max_workspace_size =
          static_cast<size_t>(trt_workspace_mb) * 1024ULL * 1024ULL;
    }
    trt_options.trt_fp16_enable =
        dinf_option_bool(runtime_options, "trtFp16", false) ? 1 : 0;
    trt_options.trt_int8_enable =
        dinf_option_bool(runtime_options, "trtInt8", false) ? 1 : 0;
    trt_options.trt_dump_subgraphs =
        dinf_option_bool(runtime_options, "trtDumpSubgraphs", false) ? 1 : 0;
    const std::string trt_cache_path =
        dinf_option_string(runtime_options, "trtEngineCachePath");
    if (!trt_cache_path.empty()) {
      trt_options.trt_engine_cache_enable = 1;
      trt_options.trt_engine_cache_path = trt_cache_path.c_str();
    } else {
      trt_options.trt_engine_cache_enable =
          dinf_option_bool(runtime_options, "trtEngineCacheEnable", false)
          ? 1
          : 0;
    }
    trt_options.trt_force_sequential_engine_build =
        dinf_option_bool(runtime_options, "trtForceSequentialEngineBuild", false)
        ? 1
        : 0;
    std::string append_error;
    if (Ok(api,
           api->SessionOptionsAppendExecutionProvider_TensorRT(
               options,
               &trt_options),
           &append_error)) {
      if (appended != nullptr) {
        *appended = true;
      }
      return true;
    }
    if (dinf_option_bool(runtime_options, "requireProvider", false)) {
      *error = append_error;
      return false;
    }
  }

  std::string append_error;
  if (!Ok(api,
          api->SessionOptionsAppendExecutionProvider(
              options, provider.c_str(), nullptr, nullptr, 0),
          &append_error)) {
    if (dinf_option_bool(runtime_options, "requireProvider", false)) {
      *error = append_error;
      return false;
    }
    return true;
  }
  if (appended != nullptr) {
    *appended = true;
  }
  return true;
}
