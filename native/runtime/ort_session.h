#pragma once

#include "runtime_bridge.h"

#include <memory>
#include <string>
#include <vector>

#if DINF_ENABLE_ORT
#include <onnxruntime_c_api.h>

namespace dinf_ort {

struct TensorMetadata {
  std::string name;
  std::string onnx_type;
  std::string dtype;
  int32_t dtype_id = 0;
  std::vector<int64_t> shape;
  std::vector<std::string> symbolic_shape;
};

class Session final : public DinfRuntimeSession {
 public:
  Session(
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
      int cuda_graph_id);

  ~Session() override;

  const std::vector<std::string>& InputNames() const;

  const std::vector<std::string>& OutputNames() const;

  int Run(
      const DinfNamedTensor* inputs,
      size_t input_count,
      DinfNamedTensor** outputs,
      size_t* output_count,
      std::string* error) override;

  void Diagnostics(
      DinfDiagBuilder* out,
      const std::string& prefix) const override;

 private:
  bool OutputUsesDeviceHandle(const std::string& name) const;

  void ReleaseValues(
      std::vector<OrtValue*>& values,
      const std::vector<uint8_t>* owned = nullptr);

  const OrtApi* api_;
  std::shared_ptr<OrtEnv> env_;
  OrtSessionOptions* options_;
  OrtSession* session_;
  OrtIoBinding* io_binding_;
  OrtRunOptions* run_options_;
  OrtAllocator* allocator_;
  OrtMemoryInfo* memory_info_;
  OrtMemoryInfo* device_memory_info_;
  std::string provider_;
  std::string prepacked_weights_key_;
  std::vector<std::string> available_providers_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<std::string> cpu_output_names_;
  std::vector<std::string> device_output_names_;
  std::vector<TensorMetadata> input_metadata_;
  std::vector<TensorMetadata> output_metadata_;
  std::shared_ptr<void> prepacked_weights_;
  int num_threads_;
  int cuda_graph_id_;
  bool provider_appended_;
  bool use_io_binding_;
  bool use_device_outputs_;
  bool use_output_views_;
  bool sync_bound_inputs_;
  bool sync_bound_outputs_;
  bool cache_bound_outputs_;
  bool bound_outputs_cached_;
};

std::unique_ptr<Session> CreateSession(
    const char* model_path,
    const DinfOptions* runtime_options,
    std::string* error);

std::vector<int64_t> TensorShape(const DinfTensor& tensor);

}  // namespace dinf_ort

#endif
