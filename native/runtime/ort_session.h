#pragma once

#include "runtime_bridge.h"

#include <memory>
#include <string>
#include <vector>

#if DINF_ENABLE_ORT
#include <onnxruntime_c_api.h>

namespace dinf_ort {

class Session final : public DinfRuntimeSession {
 public:
  Session(
      const OrtApi* api,
      OrtEnv* env,
      OrtSessionOptions* options,
      OrtSession* session,
      OrtAllocator* allocator,
      OrtMemoryInfo* memory_info,
      std::string provider,
      std::vector<std::string> available_providers,
      std::vector<std::string> input_names,
      std::vector<std::string> output_names,
      int num_threads,
      bool provider_appended);

  ~Session() override;

  const std::vector<std::string>& InputNames() const;

  const std::vector<std::string>& OutputNames() const;

  int Run(
      const DinfNamedTensor* inputs,
      size_t input_count,
      DinfNamedTensor** outputs,
      size_t* output_count,
      std::string* error) override;

  std::string DiagnosticsJson() const override;

 private:
  void ReleaseValues(std::vector<OrtValue*>& values);

  void ReleaseNames(std::vector<char*>& names);

  const OrtApi* api_;
  OrtEnv* env_;
  OrtSessionOptions* options_;
  OrtSession* session_;
  OrtAllocator* allocator_;
  OrtMemoryInfo* memory_info_;
  std::string provider_;
  std::vector<std::string> available_providers_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  int num_threads_;
  bool provider_appended_;
};

std::unique_ptr<Session> CreateSession(
    const char* model_path,
    const char* options_json,
    std::string* error);

std::vector<int64_t> TensorShape(const DinfTensor& tensor);

}  // namespace dinf_ort

#endif
