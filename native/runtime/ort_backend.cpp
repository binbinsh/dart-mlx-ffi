#include "runtime_bridge.h"
#include "options.h"
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if DMF_ENABLE_ORT
#include <onnxruntime_c_api.h>
#include "nlohmann/json.hpp"
#if defined(_WIN32)
#include <windows.h>
#endif
#endif

#if DMF_ENABLE_ORT && defined(__ANDROID__)
extern "C" OrtStatus* OrtSessionOptionsAppendExecutionProvider_Nnapi(
    OrtSessionOptions* options, uint32_t nnapi_flags);
#endif
#if DMF_ENABLE_ORT
namespace {
using json = nlohmann::json;

ONNXTensorElementDataType ort_dtype(int32_t dtype) {
  switch (dtype) {
    case DMF_DTYPE_FLOAT32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case DMF_DTYPE_INT32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case DMF_DTYPE_INT64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case DMF_DTYPE_UINT8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    case DMF_DTYPE_FLOAT64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    case DMF_DTYPE_FLOAT16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    case DMF_DTYPE_BOOL:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    default:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
}

int32_t dmf_dtype(ONNXTensorElementDataType dtype) {
  switch (dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return DMF_DTYPE_FLOAT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return DMF_DTYPE_INT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return DMF_DTYPE_INT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return DMF_DTYPE_UINT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return DMF_DTYPE_FLOAT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return DMF_DTYPE_FLOAT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return DMF_DTYPE_BOOL;
    default:
      return 0;
  }
}

std::string status_message(const OrtApi* api, OrtStatus* status) {
  if (status == nullptr) {
    return "";
  }
  std::string message = api->GetErrorMessage(status);
  api->ReleaseStatus(status);
  return message;
}

bool ok(const OrtApi* api, OrtStatus* status, std::string* error) {
  if (status == nullptr) {
    return true;
  }
  *error = status_message(api, status);
  return false;
}

std::string lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

std::vector<std::string> available_providers(
    const OrtApi* api,
    std::string* error) {
  char** raw = nullptr;
  int count = 0;
  if (!ok(api, api->GetAvailableProviders(&raw, &count), error)) {
    return {};
  }
  std::vector<std::string> providers;
  for (int i = 0; i < count; ++i) {
    providers.emplace_back(raw[i]);
  }
  std::string release_error;
  ok(api, api->ReleaseAvailableProviders(raw, count), &release_error);
  return providers;
}

bool contains_provider(
    const std::vector<std::string>& providers,
    const std::string& provider) {
  const std::string expected = lower(provider);
  for (const auto& item : providers) {
    if (lower(item) == expected) {
      return true;
    }
  }
  return false;
}

std::string matching_provider(
    const std::vector<std::string>& providers,
    const std::string& provider) {
  const std::string expected = lower(provider);
  for (const auto& item : providers) {
    if (lower(item) == expected) {
      return item;
    }
  }
  return "";
}

std::vector<std::string> session_names(
    const OrtApi* api,
    OrtSession* session,
    OrtAllocator* allocator,
    bool inputs,
    std::string* error) {
  size_t count = 0;
  if (inputs) {
    if (!ok(api, api->SessionGetInputCount(session, &count), error)) {
      return {};
    }
  } else if (!ok(api, api->SessionGetOutputCount(session, &count), error)) {
    return {};
  }
  std::vector<std::string> names;
  names.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    char* name = nullptr;
    OrtStatus* status = inputs
        ? api->SessionGetInputName(session, i, allocator, &name)
        : api->SessionGetOutputName(session, i, allocator, &name);
    if (!ok(api, status, error)) {
      return {};
    }
    names.emplace_back(name == nullptr ? "" : name);
    allocator->Free(allocator, name);
  }
  return names;
}

std::string canonical_provider(std::string provider) {
  const std::string value = lower(provider);
  if (value.empty() || value == "cpu") {
    return "CPUExecutionProvider";
  }
  if (value == "cuda") {
    return "CUDAExecutionProvider";
  }
  if (value == "tensorrt" || value == "trt") {
    return "TensorrtExecutionProvider";
  }
  if (value == "directml" || value == "dml") {
    return "DmlExecutionProvider";
  }
  if (value == "openvino") {
    return "OpenVINOExecutionProvider";
  }
  if (value == "rocm") {
    return "ROCMExecutionProvider";
  }
  if (value == "qnn" || value == "npu") {
    return "QNNExecutionProvider";
  }
  if (value == "nnapi" || value == "androidnnapi") {
    return "NNAPIExecutionProvider";
  }
  if (value == "xnnpack") {
    return "XnnpackExecutionProvider";
  }
  return provider;
}

std::string requested_provider(const char* options_json) {
  std::string requested =
      dmf_option_string(options_json, "provider",
          dmf_option_string(options_json, "executionProvider",
              dmf_option_string(options_json, "ortProvider")));
  if (!requested.empty()) {
    return requested;
  }
  if (dmf_options_contains_token(options_json, "npu")) {
    return "npu";
  }
  if (dmf_options_contains_token(options_json, "gpu")) {
    return "gpu";
  }
  return "cpu";
}

std::string choose_provider(
    const char* options_json,
    const std::vector<std::string>& providers) {
  const std::vector<std::string> gpu_order = {
      "CUDAExecutionProvider",
      "TensorrtExecutionProvider",
      "DmlExecutionProvider",
      "ROCMExecutionProvider",
      "OpenVINOExecutionProvider",
      "CoreMLExecutionProvider",
      "XnnpackExecutionProvider",
  };
  const std::vector<std::string> npu_order = {
      "QNNExecutionProvider",
      "NNAPIExecutionProvider",
      "OpenVINOExecutionProvider",
      "XnnpackExecutionProvider",
  };
  std::string requested =
      dmf_option_string(options_json, "provider",
          dmf_option_string(options_json, "executionProvider",
              dmf_option_string(options_json, "ortProvider")));
  if (!requested.empty()) {
    const std::string requested_value = lower(requested);
    if (requested_value == "gpu" || requested_value == "npu") {
      const auto& requested_order =
          requested_value == "npu" ? npu_order : gpu_order;
      for (const auto& provider : requested_order) {
        const std::string matched = matching_provider(providers, provider);
        if (!matched.empty()) {
          return matched;
        }
      }
    }
    const std::string canonical = canonical_provider(requested);
    const std::string matched = matching_provider(providers, canonical);
    return matched.empty() ? canonical : matched;
  }
  if (!dmf_options_contains_token(options_json, "gpu") &&
      !dmf_options_contains_token(options_json, "npu")) {
    return "CPUExecutionProvider";
  }
  const auto& order =
      dmf_options_contains_token(options_json, "npu") ? npu_order : gpu_order;
  for (const auto& provider : order) {
    const std::string matched = matching_provider(providers, provider);
    if (!matched.empty()) {
      return matched;
    }
  }
  return "CPUExecutionProvider";
}

std::string provider_fallback_reason(
    const std::string& requested,
    const std::string& selected,
    const std::string& effective) {
  const std::string value = lower(requested);
  if (selected != effective) {
    return "selected_provider_unavailable_or_append_failed";
  }
  const std::string selected_value = lower(selected);
  if (value == "npu" && selected_value != "qnnexecutionprovider" &&
      selected_value != "nnapiexecutionprovider") {
    return "generic_npu_fallback";
  }
  if (value == "gpu" && selected == "XnnpackExecutionProvider") {
    return "generic_gpu_fallback";
  }
  if (value != "cpu" && value != "gpu" && value != "npu" &&
      lower(canonical_provider(requested)) != selected_value) {
    return "requested_provider_unavailable";
  }
  return "";
}

bool append_provider(
    const OrtApi* api,
    OrtSessionOptions* options,
    const std::string& provider,
    const std::vector<std::string>& providers,
    const char* options_json,
    bool* appended,
    std::string* append_error_out,
    std::string* error) {
  if (appended != nullptr) {
    *appended = false;
  }
  if (append_error_out != nullptr) {
    append_error_out->clear();
  }
  if (provider.empty() || provider == "CPUExecutionProvider") {
    return true;
  }
  if (!contains_provider(providers, provider)) {
    if (dmf_option_bool(options_json, "requireProvider", false)) {
      *error = "Requested ONNX Runtime provider is unavailable: " + provider;
      return false;
    }
    return true;
  }
  std::string append_error;
#if defined(__ANDROID__)
  if (lower(provider) == "nnapiexecutionprovider") {
    const uint32_t nnapi_flags = static_cast<uint32_t>(
        std::max(0, dmf_option_int(options_json, "nnapiFlags", 0)));
    if (!ok(api,
            OrtSessionOptionsAppendExecutionProvider_Nnapi(options, nnapi_flags),
            &append_error)) {
      if (append_error_out != nullptr) {
        *append_error_out = append_error;
      }
      if (dmf_option_bool(options_json, "requireProvider", false)) {
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
#endif
  if (!ok(api,
          api->SessionOptionsAppendExecutionProvider(
              options, provider.c_str(), nullptr, nullptr, 0),
          &append_error)) {
    if (append_error_out != nullptr) {
      *append_error_out = append_error;
    }
    if (dmf_option_bool(options_json, "requireProvider", false)) {
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

#if defined(_WIN32)
std::wstring utf8_to_wide(const char* value) {
  if (value == nullptr) {
    return {};
  }
  const int size = MultiByteToWideChar(CP_UTF8, 0, value, -1, nullptr, 0);
  if (size <= 0) {
    return {};
  }
  std::wstring out(size - 1, L'\0');
  MultiByteToWideChar(CP_UTF8, 0, value, -1, out.data(), size);
  return out;
}
#endif

class OrtSessionWrapper final : public DmfRuntimeSession {
 public:
  OrtSessionWrapper(
      const OrtApi* api,
      OrtEnv* env,
      OrtSessionOptions* options,
      OrtSession* session,
      OrtAllocator* allocator,
      OrtMemoryInfo* memory_info,
      std::string provider,
      std::string requested_provider,
      std::string selected_provider,
      std::string provider_fallback_reason,
      std::string provider_append_error,
      std::vector<std::string> available_providers,
      std::vector<std::string> input_names,
      std::vector<std::string> output_names,
      int num_threads,
      bool provider_appended)
      : api_(api),
        env_(env),
        options_(options),
        session_(session),
        allocator_(allocator),
        memory_info_(memory_info),
        provider_(std::move(provider)),
        requested_provider_(std::move(requested_provider)),
        selected_provider_(std::move(selected_provider)),
        provider_fallback_reason_(std::move(provider_fallback_reason)),
        provider_append_error_(std::move(provider_append_error)),
        available_providers_(std::move(available_providers)),
        input_names_(std::move(input_names)),
        output_names_(std::move(output_names)),
        num_threads_(num_threads),
        provider_appended_(provider_appended) {}

  ~OrtSessionWrapper() override {
    api_->ReleaseMemoryInfo(memory_info_);
    api_->ReleaseSession(session_);
    api_->ReleaseSessionOptions(options_);
    api_->ReleaseEnv(env_);
  }

  const std::vector<std::string>& InputNames() const { return input_names_; }

  const std::vector<std::string>& OutputNames() const { return output_names_; }

  int Run(
      const DmfNamedTensor* inputs,
      size_t input_count,
      DmfNamedTensor** outputs,
      size_t* output_count,
      std::string* error) override {
    std::vector<OrtValue*> input_values(input_count);
    std::vector<const char*> input_names(input_count);
    for (size_t i = 0; i < input_count; ++i) {
      input_names[i] = inputs[i].name;
      const auto dtype = ort_dtype(inputs[i].tensor.dtype);
      if (dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
        *error = "Unsupported ONNX Runtime input dtype";
        release_values(input_values);
        return 1;
      }
      if (!ok(api_, api_->CreateTensorWithDataAsOrtValue(
                        memory_info_,
                        inputs[i].tensor.data,
                        inputs[i].tensor.byte_length,
                        inputs[i].tensor.shape,
                        inputs[i].tensor.rank,
                        dtype,
                        &input_values[i]),
              error)) {
        release_values(input_values);
        return 1;
      }
    }

    size_t output_name_count = 0;
    if (!ok(api_, api_->SessionGetOutputCount(session_, &output_name_count), error)) {
      release_values(input_values);
      return 1;
    }
    std::vector<char*> owned_names;
    std::vector<const char*> output_names;
    owned_names.reserve(output_name_count);
    output_names.reserve(output_name_count);
    for (size_t i = 0; i < output_name_count; ++i) {
      char* name = nullptr;
      if (!ok(api_, api_->SessionGetOutputName(session_, i, allocator_, &name), error)) {
        release_values(input_values);
        release_names(owned_names);
        return 1;
      }
      output_names.push_back(name);
      owned_names.push_back(name);
    }

    std::vector<OrtValue*> output_values(output_name_count);
    if (!ok(api_,
            api_->Run(
                session_,
                nullptr,
                input_names.data(),
                input_values.data(),
                input_values.size(),
                output_names.data(),
                output_names.size(),
                output_values.data()),
            error)) {
      release_values(input_values);
      release_values(output_values);
      release_names(owned_names);
      return 1;
    }

    std::vector<DmfNamedTensor> produced;
    for (size_t i = 0; i < output_values.size(); ++i) {
      OrtTensorTypeAndShapeInfo* info = nullptr;
      if (!ok(api_, api_->GetTensorTypeAndShape(output_values[i], &info), error)) {
        release_values(input_values);
        release_values(output_values);
        release_names(owned_names);
        return 1;
      }
      ONNXTensorElementDataType ort_type;
      if (!ok(api_, api_->GetTensorElementType(info, &ort_type), error)) {
        api_->ReleaseTensorTypeAndShapeInfo(info);
        release_values(input_values);
        release_values(output_values);
        release_names(owned_names);
        return 1;
      }
      const int32_t dtype = dmf_dtype(ort_type);
      if (dtype == 0) {
        api_->ReleaseTensorTypeAndShapeInfo(info);
        continue;
      }
      size_t rank = 0;
      if (!ok(api_, api_->GetDimensionsCount(info, &rank), error)) {
        api_->ReleaseTensorTypeAndShapeInfo(info);
        release_values(input_values);
        release_values(output_values);
        release_names(owned_names);
        return 1;
      }
      std::vector<int64_t> shape(rank);
      if (!ok(api_, api_->GetDimensions(info, shape.data(), rank), error)) {
        api_->ReleaseTensorTypeAndShapeInfo(info);
        release_values(input_values);
        release_values(output_values);
        release_names(owned_names);
        return 1;
      }
      size_t count = 0;
      if (!ok(api_, api_->GetTensorShapeElementCount(info, &count), error)) {
        api_->ReleaseTensorTypeAndShapeInfo(info);
        release_values(input_values);
        release_values(output_values);
        release_names(owned_names);
        return 1;
      }
      api_->ReleaseTensorTypeAndShapeInfo(info);
      void* data = nullptr;
      if (!ok(api_, api_->GetTensorMutableData(output_values[i], &data), error)) {
        release_values(input_values);
        release_values(output_values);
        release_names(owned_names);
        return 1;
      }
      produced.push_back(dmf_make_tensor(
          output_names[i],
          dtype,
          shape,
          data,
          count * dmf_dtype_size(dtype)));
    }

    release_values(input_values);
    release_values(output_values);
    release_names(owned_names);
    *output_count = produced.size();
    *outputs = static_cast<DmfNamedTensor*>(
        std::malloc(sizeof(DmfNamedTensor) * produced.size()));
    if (!produced.empty()) {
      std::memcpy(*outputs, produced.data(), sizeof(DmfNamedTensor) * produced.size());
    }
    return 0;
  }

  std::string DiagnosticsJson() const override {
    std::string out = std::string("{\"engine\":\"onnx\",\"provider\":\"") +
           dmf_json_escape(provider_) + "\",\"effective_provider\":\"" +
           dmf_json_escape(provider_) + "\",\"requested_provider\":\"" +
           dmf_json_escape(requested_provider_) + "\",\"selected_provider\":\"" +
           dmf_json_escape(selected_provider_) + "\",\"provider_appended\":" +
           (provider_appended_ ? "true" : "false") + ",\"num_threads\":" +
           std::to_string(num_threads_) + ",\"available_providers\":" +
           dmf_json_string_array(available_providers_) +
           ",\"input_names\":" + dmf_json_string_array(input_names_) +
           ",\"output_names\":" + dmf_json_string_array(output_names_);
    if (!provider_append_error_.empty()) {
      out += ",\"provider_append_error\":\"" +
             dmf_json_escape(provider_append_error_) + "\"";
    }
    if (!provider_fallback_reason_.empty()) {
      out += ",\"provider_fallback\":{\"requested\":\"" +
             dmf_json_escape(requested_provider_) + "\",\"selected\":\"" +
             dmf_json_escape(selected_provider_) + "\",\"effective\":\"" +
             dmf_json_escape(provider_) + "\",\"reason\":\"" +
             dmf_json_escape(provider_fallback_reason_) + "\"";
      if (!provider_append_error_.empty()) {
        out += ",\"append_error\":\"" +
               dmf_json_escape(provider_append_error_) + "\"";
      }
      out += "}";
    }
    out += "}";
    return out;
  }

 private:
  void release_values(std::vector<OrtValue*>& values) {
    for (auto* value : values) {
      if (value != nullptr) {
        api_->ReleaseValue(value);
      }
    }
  }

  void release_names(std::vector<char*>& names) {
    for (auto* name : names) {
      if (name != nullptr) {
        allocator_->Free(allocator_, name);
      }
    }
    names.clear();
  }

  const OrtApi* api_;
  OrtEnv* env_;
  OrtSessionOptions* options_;
  OrtSession* session_;
  OrtAllocator* allocator_;
  OrtMemoryInfo* memory_info_;
  std::string provider_;
  std::string requested_provider_;
  std::string selected_provider_;
  std::string provider_fallback_reason_;
  std::string provider_append_error_;
  std::vector<std::string> available_providers_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  int num_threads_;
  bool provider_appended_;
};

std::unique_ptr<OrtSessionWrapper> create_ort_session(
    const char* model_path,
    const char* options_json,
    std::string* error) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  OrtEnv* env = nullptr;
  OrtSessionOptions* options = nullptr;
  OrtSession* session = nullptr;
  OrtAllocator* allocator = nullptr;
  OrtMemoryInfo* memory_info = nullptr;
  if (!ok(api, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "dart_mlx_ffi", &env), error)) {
    return nullptr;
  }
  if (!ok(api, api->CreateSessionOptions(&options), error)) {
    api->ReleaseEnv(env);
    return nullptr;
  }
  const int num_threads = std::max(1, dmf_option_int(options_json, "numThreads", 1));
  if (!ok(api, api->SetIntraOpNumThreads(options, num_threads), error)) {
    api->ReleaseSessionOptions(options);
    api->ReleaseEnv(env);
    return nullptr;
  }
  if (!ok(api, api->SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL), error)) {
    api->ReleaseSessionOptions(options);
    api->ReleaseEnv(env);
    return nullptr;
  }
  std::string provider_error;
  const std::vector<std::string> providers = available_providers(api, &provider_error);
  if (!provider_error.empty()) {
    api->ReleaseSessionOptions(options);
    api->ReleaseEnv(env);
    *error = provider_error;
    return nullptr;
  }
  const std::string requested = requested_provider(options_json);
  const std::string provider = choose_provider(options_json, providers);
  bool provider_appended = false;
  std::string provider_append_error;
  if (!append_provider(
          api,
          options,
          provider,
          providers,
          options_json,
          &provider_appended,
          &provider_append_error,
          error)) {
    api->ReleaseSessionOptions(options);
    api->ReleaseEnv(env);
    return nullptr;
  }
  const std::string effective_provider =
      provider == "CPUExecutionProvider" || provider_appended
      ? provider
      : "CPUExecutionProvider";
  const std::string fallback_reason =
      provider_fallback_reason(requested, provider, effective_provider);
#if defined(_WIN32)
  const std::wstring wide_path = utf8_to_wide(model_path);
  const auto* ort_path = wide_path.c_str();
#else
  const auto* ort_path = model_path;
#endif
  if (!ok(api, api->CreateSession(env, ort_path, options, &session), error)) {
    api->ReleaseSessionOptions(options);
    api->ReleaseEnv(env);
    return nullptr;
  }
  if (!ok(api, api->GetAllocatorWithDefaultOptions(&allocator), error)) {
    api->ReleaseSession(session);
    api->ReleaseSessionOptions(options);
    api->ReleaseEnv(env);
    return nullptr;
  }
  if (!ok(api,
          api->CreateCpuMemoryInfo(
              OrtArenaAllocator, OrtMemTypeDefault, &memory_info),
          error)) {
    api->ReleaseSession(session);
    api->ReleaseSessionOptions(options);
    api->ReleaseEnv(env);
    return nullptr;
  }
  std::vector<std::string> input_names =
      session_names(api, session, allocator, true, error);
  if (!error->empty()) {
    api->ReleaseMemoryInfo(memory_info);
    api->ReleaseSession(session);
    api->ReleaseSessionOptions(options);
    api->ReleaseEnv(env);
    return nullptr;
  }
  std::vector<std::string> output_names =
      session_names(api, session, allocator, false, error);
  if (!error->empty()) {
    api->ReleaseMemoryInfo(memory_info);
    api->ReleaseSession(session);
    api->ReleaseSessionOptions(options);
    api->ReleaseEnv(env);
    return nullptr;
  }
  return std::unique_ptr<OrtSessionWrapper>(new OrtSessionWrapper(
      api,
      env,
      options,
      session,
      allocator,
      memory_info,
      effective_provider,
      requested,
      provider,
      fallback_reason,
      provider_append_error,
      providers,
      std::move(input_names),
      std::move(output_names),
      num_threads,
      provider_appended));
}

std::vector<int64_t> tensor_shape(const DmfNativeTensor& tensor) {
  std::vector<int64_t> shape;
  shape.reserve(tensor.rank);
  for (int i = 0; i < tensor.rank; ++i) {
    shape.push_back(tensor.shape[i]);
  }
  return shape;
}

struct OwnedTensorArray {
  DmfNamedTensor* values = nullptr;
  size_t count = 0;

  ~OwnedTensorArray() {
    dmf_runtime_free_tensors(values, static_cast<intptr_t>(count));
  }
};

using TensorMap = std::map<std::string, const DmfNamedTensor*>;
using StringMap = std::map<std::string, std::string>;

StringMap string_map_from_json(const json& value) {
  StringMap result;
  if (!value.is_object()) {
    return result;
  }
  for (const auto& item : value.items()) {
    if (item.value().is_string()) {
      result[item.key()] = item.value().get<std::string>();
    }
  }
  return result;
}

std::filesystem::path resolve_pipeline_path(
    const std::filesystem::path& spec_path,
    const std::string& value) {
  std::filesystem::path path(value);
  if (path.is_absolute()) {
    return path;
  }
  return spec_path.parent_path() / path;
}

struct PipelineStage {
  std::string name;
  std::string op;
  std::string model_path;
  StringMap inputs;
  StringMap outputs;
  std::unique_ptr<OrtSessionWrapper> session;
};

std::string mapped_name(
    const StringMap& values,
    const std::string& name,
    const std::string& fallback) {
  const auto found = values.find(name);
  return found == values.end() ? fallback : found->second;
}

const DmfNamedTensor* find_tensor(
    const TensorMap& tensors,
    const std::string& name,
    std::string* error) {
  const auto found = tensors.find(name);
  if (found == tensors.end()) {
    *error = "ONNX pipeline missing tensor '" + name + "'";
    return nullptr;
  }
  return found->second;
}

int64_t index_value(const DmfNativeTensor& tensor, size_t index) {
  if (tensor.dtype == DMF_DTYPE_INT64) {
    return static_cast<const int64_t*>(tensor.data)[index];
  }
  if (tensor.dtype == DMF_DTYPE_INT32) {
    return static_cast<const int32_t*>(tensor.data)[index];
  }
  return -1;
}

OwnedTensorArray* scatter_embeddings(
    const PipelineStage& stage,
    const TensorMap& tensors,
    std::string* error) {
  const std::string base_key =
      mapped_name(stage.inputs, "base", mapped_name(stage.inputs, "inputs_embeds", "inputs_embeds"));
  const std::string updates_key =
      mapped_name(stage.inputs, "updates", mapped_name(stage.inputs, "image_embeds", "image_embeds"));
  const std::string indices_key =
      mapped_name(stage.inputs, "indices", "image_token_indices");
  const auto* base = find_tensor(tensors, base_key, error);
  const auto* updates = find_tensor(tensors, updates_key, error);
  const auto* indices = find_tensor(tensors, indices_key, error);
  if (base == nullptr || updates == nullptr || indices == nullptr) {
    return nullptr;
  }
  if (base->tensor.dtype != updates->tensor.dtype) {
    *error = "scatter_embeddings requires base and updates to share dtype";
    return nullptr;
  }
  if (base->tensor.rank < 2 || updates->tensor.rank != 2) {
    *error = "scatter_embeddings expects base rank >= 2 and updates rank 2";
    return nullptr;
  }
  if (indices->tensor.dtype != DMF_DTYPE_INT64 &&
      indices->tensor.dtype != DMF_DTYPE_INT32) {
    *error = "scatter_embeddings indices must be int32 or int64";
    return nullptr;
  }
  const int64_t sequence = base->tensor.shape[base->tensor.rank - 2];
  const int64_t hidden = base->tensor.shape[base->tensor.rank - 1];
  int64_t leading = 1;
  for (int32_t i = 0; i < base->tensor.rank - 2; ++i) {
    leading *= base->tensor.shape[i];
  }
  if (leading != 1) {
    *error = "scatter_embeddings currently supports a single leading batch";
    return nullptr;
  }
  if (sequence <= 0 || hidden <= 0) {
    *error = "scatter_embeddings base shape must have positive sequence and hidden dims";
    return nullptr;
  }
  if (updates->tensor.shape[1] != hidden) {
    *error = "scatter_embeddings update hidden size does not match base";
    return nullptr;
  }
  const size_t index_count =
      static_cast<size_t>(indices->tensor.byte_length / dmf_dtype_size(indices->tensor.dtype));
  if (index_count != static_cast<size_t>(updates->tensor.shape[0])) {
    *error = "scatter_embeddings update count does not match index count";
    return nullptr;
  }
  const size_t row_bytes = static_cast<size_t>(hidden) * dmf_dtype_size(base->tensor.dtype);
  const auto* update_bytes = static_cast<const uint8_t*>(updates->tensor.data);
  std::vector<uint8_t> merged(static_cast<size_t>(base->tensor.byte_length));
  std::memcpy(merged.data(), base->tensor.data, merged.size());
  for (size_t i = 0; i < index_count; ++i) {
    const int64_t raw_index = index_value(indices->tensor, i);
    if (raw_index < 0 || raw_index >= sequence) {
      *error = "scatter_embeddings index is out of sequence bounds";
      return nullptr;
    }
    const size_t offset = static_cast<size_t>(raw_index) * row_bytes;
    if (offset + row_bytes > merged.size()) {
      *error = "scatter_embeddings computed offset is out of bounds";
      return nullptr;
    }
    std::memcpy(merged.data() + offset, update_bytes + i * row_bytes, row_bytes);
  }
  const std::string output_name =
      mapped_name(stage.outputs, "output", "inputs_embeds");
  auto* holder = new OwnedTensorArray();
  holder->count = 1;
  holder->values = static_cast<DmfNamedTensor*>(std::malloc(sizeof(DmfNamedTensor)));
  holder->values[0] = dmf_make_tensor(
      output_name.c_str(),
      base->tensor.dtype,
      tensor_shape(base->tensor),
      merged.data(),
      merged.size());
  return holder;
}

class OrtPipelineSession final : public DmfRuntimeSession {
 public:
  OrtPipelineSession(
      std::string spec_path,
      std::vector<PipelineStage> stages,
      StringMap requested_outputs)
      : spec_path_(std::move(spec_path)),
        stages_(std::move(stages)),
        requested_outputs_(std::move(requested_outputs)) {}

  int Run(
      const DmfNamedTensor* inputs,
      size_t input_count,
      DmfNamedTensor** outputs,
      size_t* output_count,
      std::string* error) override {
    TensorMap tensors;
    for (size_t i = 0; i < input_count; ++i) {
      tensors[inputs[i].name == nullptr ? "" : inputs[i].name] = &inputs[i];
    }
    std::vector<std::unique_ptr<OwnedTensorArray>> owned;
    std::vector<std::string> last_keys;
    for (auto& stage : stages_) {
      if (!stage.op.empty()) {
        std::unique_ptr<OwnedTensorArray> holder;
        if (stage.op == "scatter_embeddings") {
          holder.reset(scatter_embeddings(stage, tensors, error));
        } else {
          *error = "Unsupported ONNX pipeline op: " + stage.op;
          return 1;
        }
        if (!holder) {
          return 1;
        }
        last_keys.clear();
        for (size_t i = 0; i < holder->count; ++i) {
          const std::string key =
              holder->values[i].name == nullptr ? "" : holder->values[i].name;
          tensors[key] = &holder->values[i];
          last_keys.push_back(key);
        }
        owned.push_back(std::move(holder));
        continue;
      }

      std::vector<DmfNamedTensor> selected;
      std::vector<std::string> selected_names;
      const auto& required = stage.session->InputNames();
      selected.reserve(required.size());
      selected_names.reserve(required.size());
      for (const auto& input_name : required) {
        const auto mapped = stage.inputs.find(input_name);
        const std::string source =
            mapped == stage.inputs.end() ? input_name : mapped->second;
        const auto found = tensors.find(source);
        if (found == tensors.end()) {
          *error = "ONNX pipeline stage '" + stage.name +
                   "' is missing input '" + source + "' for '" + input_name + "'";
          return 1;
        }
        selected.push_back(*found->second);
        selected_names.push_back(input_name);
      }
      for (size_t i = 0; i < selected.size(); ++i) {
        selected[i].name = const_cast<char*>(selected_names[i].c_str());
      }

      DmfNamedTensor* stage_outputs = nullptr;
      size_t stage_output_count = 0;
      const int status = stage.session->Run(
          selected.data(),
          selected.size(),
          &stage_outputs,
          &stage_output_count,
          error);
      if (status != 0) {
        dmf_runtime_free_tensors(
            stage_outputs,
            static_cast<intptr_t>(stage_output_count));
        return status;
      }

      auto holder = std::make_unique<OwnedTensorArray>();
      holder->values = stage_outputs;
      holder->count = stage_output_count;
      last_keys.clear();
      for (size_t i = 0; i < stage_output_count; ++i) {
        const std::string output_name =
            stage_outputs[i].name == nullptr ? "" : stage_outputs[i].name;
        const auto mapped = stage.outputs.find(output_name);
        const std::string key =
            mapped == stage.outputs.end() ? output_name : mapped->second;
        tensors[key] = &stage_outputs[i];
        last_keys.push_back(key);
      }
      owned.push_back(std::move(holder));
    }

    StringMap requested = requested_outputs_;
    if (requested.empty()) {
      for (const auto& key : last_keys) {
        requested[key] = key;
      }
    }
    std::vector<DmfNamedTensor> produced;
    for (const auto& item : requested) {
      const auto found = tensors.find(item.second);
      if (found == tensors.end()) {
        *error = "ONNX pipeline requested missing output '" + item.second + "'";
        return 1;
      }
      const auto* source = found->second;
      produced.push_back(dmf_make_tensor(
          item.first.c_str(),
          source->tensor.dtype,
          tensor_shape(source->tensor),
          source->tensor.data,
          source->tensor.byte_length));
    }
    *output_count = produced.size();
    *outputs = static_cast<DmfNamedTensor*>(
        std::malloc(sizeof(DmfNamedTensor) * produced.size()));
    if (!produced.empty()) {
      std::memcpy(*outputs, produced.data(), sizeof(DmfNamedTensor) * produced.size());
    }
    return 0;
  }

  std::string DiagnosticsJson() const override {
    std::string stages = "[";
    for (size_t i = 0; i < stages_.size(); ++i) {
      if (i > 0) {
        stages += ",";
      }
      stages += "{\"name\":\"" + dmf_json_escape(stages_[i].name) + "\"";
      if (!stages_[i].op.empty()) {
        stages += ",\"op\":\"" + dmf_json_escape(stages_[i].op) + "\"";
      } else {
        stages += ",\"model\":\"" + dmf_json_escape(stages_[i].model_path) +
                  "\",\"diagnostics\":" + stages_[i].session->DiagnosticsJson();
      }
      stages += "}";
    }
    stages += "]";
    return "{\"engine\":\"onnx\",\"pipeline\":true,\"spec\":\"" +
           dmf_json_escape(spec_path_) + "\",\"stage_count\":" +
           std::to_string(stages_.size()) + ",\"stages\":" + stages + "}";
  }

 private:
  std::string spec_path_;
  std::vector<PipelineStage> stages_;
  StringMap requested_outputs_;
};

bool is_pipeline_spec(const std::filesystem::path& path, json* spec) {
  if (path.extension() != ".json") {
    return false;
  }
  std::ifstream file(path);
  if (!file) {
    return false;
  }
  json parsed;
  file >> parsed;
  if (!parsed.is_object() || !parsed.contains("stages")) {
    return false;
  }
  const std::string format = parsed.value("format", "");
  if (!format.empty() && format != "dart_mlx_ffi.onnx_pipeline.v1") {
    return false;
  }
  *spec = std::move(parsed);
  return true;
}

StringMap requested_outputs_from_json(const json& spec) {
  if (!spec.contains("outputs")) {
    return {};
  }
  const auto& outputs = spec.at("outputs");
  if (outputs.is_object()) {
    return string_map_from_json(outputs);
  }
  StringMap result;
  if (outputs.is_array()) {
    for (const auto& item : outputs) {
      if (item.is_string()) {
        const std::string name = item.get<std::string>();
        result[name] = name;
      }
    }
  }
  return result;
}

std::unique_ptr<OrtPipelineSession> create_pipeline_session(
    const std::filesystem::path& spec_path,
    const json& spec,
    const char* options_json,
    std::string* error) {
  if (!spec.at("stages").is_array()) {
    *error = "ONNX pipeline stages must be an array.";
    return nullptr;
  }
  std::vector<PipelineStage> stages;
  for (const auto& item : spec.at("stages")) {
    if (!item.is_object()) {
      *error = "ONNX pipeline stage must be an object.";
      return nullptr;
    }
    PipelineStage stage;
    stage.name = item.value("name", std::string("stage_") + std::to_string(stages.size()));
    stage.inputs = string_map_from_json(item.value("inputs", json::object()));
    stage.outputs = string_map_from_json(item.value("outputs", json::object()));
    if (item.contains("op")) {
      if (item.contains("model")) {
        *error = "ONNX pipeline stage must contain either op or model, not both.";
        return nullptr;
      }
      if (!item.at("op").is_string()) {
        *error = "ONNX pipeline op stage must contain a string op.";
        return nullptr;
      }
      stage.op = item.at("op").get<std::string>();
    } else {
      if (!item.contains("model")) {
        *error = "ONNX pipeline model stage must contain a model path.";
        return nullptr;
      }
      if (!item.at("model").is_string()) {
        *error = "ONNX pipeline model stage must contain a string model path.";
        return nullptr;
      }
      const auto path = resolve_pipeline_path(spec_path, item.at("model").get<std::string>());
      stage.model_path = path.string();
      stage.session = create_ort_session(stage.model_path.c_str(), options_json, error);
      if (!stage.session) {
        return nullptr;
      }
    }
    stages.push_back(std::move(stage));
  }
  return std::make_unique<OrtPipelineSession>(
      spec_path.string(),
      std::move(stages),
      requested_outputs_from_json(spec));
}

}  // namespace
#else
namespace {
std::string ort_disabled_error() {
  return "ONNX Runtime backend was built without ORT headers/library. "
         "Set DART_MLX_ENABLE_ORT=1 and provide DART_MLX_ORT_INCLUDE_DIR "
         "and DART_MLX_ORT_LIBRARY when building.";
}
}  // namespace
#endif

DmfRuntimeSession* dmf_create_onnx_session(
    const char* model_path,
    const char* options_json,
    std::string* error) {
#if DMF_ENABLE_ORT
  json spec;
  if (is_pipeline_spec(model_path, &spec)) {
    auto pipeline = create_pipeline_session(model_path, spec, options_json, error);
    return pipeline.release();
  }
  auto session = create_ort_session(model_path, options_json, error);
  return session.release();
#else
  (void)model_path;
  (void)options_json;
  *error = ort_disabled_error();
  return nullptr;
#endif
}
