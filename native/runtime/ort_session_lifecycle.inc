Session::Session(
    const OrtApi* api,
    std::shared_ptr<OrtEnv> env,
    OrtSessionOptions* options,
    OrtSession* session,
    OrtAllocator* allocator,
    OrtMemoryInfo* memory_info,
    OrtMemoryInfo* device_memory_info,
    std::string provider,
    std::vector<std::string> available_providers,
    std::vector<std::string> input_names,
    std::vector<std::string> output_names,
    std::vector<std::string> cpu_output_names,
    std::vector<std::string> device_output_names,
    std::vector<TensorMetadata> input_metadata,
    std::vector<TensorMetadata> output_metadata,
    int num_threads,
    bool provider_appended,
    bool use_io_binding,
    bool use_device_outputs,
    bool use_output_views,
    bool sync_bound_inputs,
    bool sync_bound_outputs,
    bool cache_bound_outputs,
    std::string prepacked_weights_key,
    std::shared_ptr<void> prepacked_weights,
    OrtIoBinding* io_binding,
    OrtRunOptions* run_options,
    int cuda_graph_id)
    : api_(api),
      env_(env),
      options_(options),
      session_(session),
      io_binding_(io_binding),
      run_options_(run_options),
      allocator_(allocator),
      memory_info_(memory_info),
      device_memory_info_(device_memory_info),
      provider_(std::move(provider)),
      prepacked_weights_key_(std::move(prepacked_weights_key)),
      available_providers_(std::move(available_providers)),
      input_names_(std::move(input_names)),
      output_names_(std::move(output_names)),
      cpu_output_names_(std::move(cpu_output_names)),
      device_output_names_(std::move(device_output_names)),
      input_metadata_(std::move(input_metadata)),
      output_metadata_(std::move(output_metadata)),
      prepacked_weights_(std::move(prepacked_weights)),
      num_threads_(num_threads),
      cuda_graph_id_(cuda_graph_id),
      provider_appended_(provider_appended),
      use_io_binding_(use_io_binding),
      use_device_outputs_(use_device_outputs),
      use_output_views_(use_output_views),
      sync_bound_inputs_(sync_bound_inputs),
      sync_bound_outputs_(sync_bound_outputs),
      cache_bound_outputs_(cache_bound_outputs),
      bound_outputs_cached_(false) {}

Session::~Session() {
  if (io_binding_ != nullptr) {
    api_->ReleaseIoBinding(io_binding_);
  }
  if (run_options_ != nullptr) {
    api_->ReleaseRunOptions(run_options_);
  }
  if (device_memory_info_ != nullptr) {
    api_->ReleaseMemoryInfo(device_memory_info_);
  }
  api_->ReleaseMemoryInfo(memory_info_);
  api_->ReleaseSession(session_);
  api_->ReleaseSessionOptions(options_);
}
