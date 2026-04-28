#include "ort_session.h"

#if DINF_ENABLE_ORT
#include "options.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <utility>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace dinf_ort {
namespace {

ONNXTensorElementDataType OrtDtype(int32_t dtype) {
  switch (dtype) {
    case DINF_DTYPE_FLOAT32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case DINF_DTYPE_INT32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case DINF_DTYPE_INT64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case DINF_DTYPE_UINT8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    case DINF_DTYPE_FLOAT64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    case DINF_DTYPE_FLOAT16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    case DINF_DTYPE_BOOL:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    default:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
}

int32_t RuntimeDtype(ONNXTensorElementDataType dtype) {
  switch (dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return DINF_DTYPE_FLOAT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return DINF_DTYPE_INT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return DINF_DTYPE_INT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return DINF_DTYPE_UINT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return DINF_DTYPE_FLOAT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return DINF_DTYPE_FLOAT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return DINF_DTYPE_BOOL;
    default:
      return 0;
  }
}

std::string OrtDtypeName(ONNXTensorElementDataType dtype) {
  switch (dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return "float32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return "uint8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return "int8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return "uint16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return "int16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return "int32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return "int64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      return "string";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return "bool";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return "float16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return "float64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return "uint32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return "uint64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
      return "complex64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
      return "complex128";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return "bfloat16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
      return "float8e4m3fn";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ:
      return "float8e4m3fnuz";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:
      return "float8e5m2";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ:
      return "float8e5m2fnuz";
    default:
      return "undefined";
  }
}

std::string OnnxTypeName(ONNXType type) {
  switch (type) {
    case ONNX_TYPE_TENSOR:
      return "tensor";
    case ONNX_TYPE_SEQUENCE:
      return "sequence";
    case ONNX_TYPE_MAP:
      return "map";
    case ONNX_TYPE_OPAQUE:
      return "opaque";
    case ONNX_TYPE_SPARSETENSOR:
      return "sparse_tensor";
    case ONNX_TYPE_OPTIONAL:
      return "optional";
    default:
      return "unknown";
  }
}

std::string StatusMessage(const OrtApi* api, OrtStatus* status) {
  if (status == nullptr) {
    return "";
  }
  std::string message = api->GetErrorMessage(status);
  api->ReleaseStatus(status);
  return message;
}

bool Ok(const OrtApi* api, OrtStatus* status, std::string* error) {
  if (status == nullptr) {
    return true;
  }
  *error = StatusMessage(api, status);
  return false;
}

#if defined(_WIN32)
std::wstring Utf8ToWide(const char* value) {
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

std::vector<std::string> SplitPaths(const std::string& raw) {
  std::vector<std::string> out;
  std::string current;
  for (const char ch : raw) {
    if (ch == ':' || ch == ';' || ch == ',' || ch == '\n' || ch == '\r') {
      if (!current.empty()) {
        out.push_back(current);
        current.clear();
      }
      continue;
    }
    current.push_back(ch);
  }
  if (!current.empty()) {
    out.push_back(current);
  }
  return out;
}

bool PreloadLibraries(const DinfOptions* runtime_options, std::string* error) {
  const std::string raw =
      dinf_option_string(runtime_options, "preloadLibraries",
          dinf_option_string(runtime_options, "preloadRuntimeLibraries"));
  if (raw.empty()) {
    return true;
  }
  const bool required =
      dinf_option_bool(runtime_options, "requirePreloadLibraries",
          dinf_option_bool(runtime_options, "requireProvider", false));
  for (const auto& path : SplitPaths(raw)) {
    if (path.empty()) {
      continue;
    }
#if defined(_WIN32)
    const std::wstring wide = Utf8ToWide(path.c_str());
    HMODULE handle = LoadLibraryW(wide.c_str());
    if (handle == nullptr && required) {
      *error = "Failed to preload runtime library: " + path;
      return false;
    }
#else
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr && required) {
      const char* dl_error = dlerror();
      *error = std::string("Failed to preload runtime library ") + path +
               ": " + (dl_error == nullptr ? "" : dl_error);
      return false;
    }
#endif
  }
  return true;
}

std::vector<std::string> AvailableProviders(
    const OrtApi* api,
    std::string* error) {
  char** raw = nullptr;
  int count = 0;
  if (!Ok(api, api->GetAvailableProviders(&raw, &count), error)) {
    return {};
  }
  std::vector<std::string> providers;
  for (int i = 0; i < count; ++i) {
    providers.emplace_back(raw[i]);
  }
  std::string release_error;
  Ok(api, api->ReleaseAvailableProviders(raw, count), &release_error);
  return providers;
}

bool ContainsProvider(
    const std::vector<std::string>& providers,
    const std::string& provider) {
  return std::find(providers.begin(), providers.end(), provider) !=
         providers.end();
}

std::vector<std::string> SessionNames(
    const OrtApi* api,
    OrtSession* session,
    OrtAllocator* allocator,
    bool inputs,
    std::string* error) {
  size_t count = 0;
  if (inputs) {
    if (!Ok(api, api->SessionGetInputCount(session, &count), error)) {
      return {};
    }
  } else if (!Ok(api, api->SessionGetOutputCount(session, &count), error)) {
    return {};
  }
  std::vector<std::string> names;
  names.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    char* name = nullptr;
    OrtStatus* status = inputs
        ? api->SessionGetInputName(session, i, allocator, &name)
        : api->SessionGetOutputName(session, i, allocator, &name);
    if (!Ok(api, status, error)) {
      return {};
    }
    names.emplace_back(name == nullptr ? "" : name);
    allocator->Free(allocator, name);
  }
  return names;
}

std::vector<TensorMetadata> SessionMetadata(
    const OrtApi* api,
    OrtSession* session,
    OrtAllocator* allocator,
    bool inputs,
    std::string* error) {
  size_t count = 0;
  if (inputs) {
    if (!Ok(api, api->SessionGetInputCount(session, &count), error)) {
      return {};
    }
  } else if (!Ok(api, api->SessionGetOutputCount(session, &count), error)) {
    return {};
  }

  std::vector<TensorMetadata> metadata;
  metadata.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    char* name = nullptr;
    OrtStatus* name_status = inputs
        ? api->SessionGetInputName(session, i, allocator, &name)
        : api->SessionGetOutputName(session, i, allocator, &name);
    if (!Ok(api, name_status, error)) {
      return {};
    }

    TensorMetadata item;
    item.name = name == nullptr ? "" : name;
    allocator->Free(allocator, name);

    OrtTypeInfo* type_info = nullptr;
    OrtStatus* type_status = inputs
        ? api->SessionGetInputTypeInfo(session, i, &type_info)
        : api->SessionGetOutputTypeInfo(session, i, &type_info);
    if (!Ok(api, type_status, error)) {
      return {};
    }

    ONNXType onnx_type = ONNX_TYPE_UNKNOWN;
    if (!Ok(api, api->GetOnnxTypeFromTypeInfo(type_info, &onnx_type), error)) {
      api->ReleaseTypeInfo(type_info);
      return {};
    }
    item.onnx_type = OnnxTypeName(onnx_type);

    if (onnx_type == ONNX_TYPE_TENSOR) {
      const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
      if (!Ok(api, api->CastTypeInfoToTensorInfo(type_info, &tensor_info), error)) {
        api->ReleaseTypeInfo(type_info);
        return {};
      }
      if (tensor_info != nullptr) {
        ONNXTensorElementDataType dtype =
            ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        if (!Ok(api, api->GetTensorElementType(tensor_info, &dtype), error)) {
          api->ReleaseTypeInfo(type_info);
          return {};
        }
        item.dtype = OrtDtypeName(dtype);
        item.dtype_id = RuntimeDtype(dtype);

        size_t rank = 0;
        if (!Ok(api, api->GetDimensionsCount(tensor_info, &rank), error)) {
          api->ReleaseTypeInfo(type_info);
          return {};
        }
        item.shape.resize(rank);
        if (rank > 0) {
          if (!Ok(api, api->GetDimensions(tensor_info, item.shape.data(), rank), error)) {
            api->ReleaseTypeInfo(type_info);
            return {};
          }
          std::vector<const char*> symbols(rank);
          std::string symbol_error;
          if (Ok(
                  api,
                  api->GetSymbolicDimensions(
                      tensor_info,
                      symbols.data(),
                      rank),
                  &symbol_error)) {
            item.symbolic_shape.reserve(rank);
            for (const auto* symbol : symbols) {
              item.symbolic_shape.emplace_back(symbol == nullptr ? "" : symbol);
            }
          }
        }
      }
    }

    api->ReleaseTypeInfo(type_info);
    metadata.push_back(std::move(item));
  }
  return metadata;
}

std::vector<std::string> MetadataNames(
    const std::vector<TensorMetadata>& metadata) {
  std::vector<std::string> names;
  names.reserve(metadata.size());
  for (const auto& item : metadata) {
    names.push_back(item.name);
  }
  return names;
}

std::vector<std::string> ShapeSignature(const TensorMetadata& metadata) {
  std::vector<std::string> signature;
  signature.reserve(metadata.shape.size());
  for (size_t i = 0; i < metadata.shape.size(); ++i) {
    if (i < metadata.symbolic_shape.size() &&
        !metadata.symbolic_shape[i].empty()) {
      signature.push_back(metadata.symbolic_shape[i]);
    } else {
      signature.push_back(std::to_string(metadata.shape[i]));
    }
  }
  return signature;
}

void AddIntList(
    DinfDiagBuilder* out,
    const std::string& path,
    const std::vector<int64_t>& values) {
  out->AddList(path);
  for (size_t i = 0; i < values.size(); ++i) {
    out->AddInt(dinf_diag_path(path, std::to_string(i)), values[i]);
  }
}

void AddTensorMetadataList(
    DinfDiagBuilder* out,
    const std::string& path,
    const std::vector<TensorMetadata>& metadata) {
  out->AddList(path);
  for (size_t i = 0; i < metadata.size(); ++i) {
    const auto& item = metadata[i];
    const std::string item_path = dinf_diag_path(path, std::to_string(i));
    out->AddMap(item_path);
    out->AddString(dinf_diag_path(item_path, "name"), item.name);
    out->AddString(dinf_diag_path(item_path, "onnx_type"), item.onnx_type);
    if (!item.dtype.empty()) {
      out->AddString(dinf_diag_path(item_path, "dtype"), item.dtype);
    }
    out->AddInt(dinf_diag_path(item_path, "dtype_id"), item.dtype_id);
    out->AddInt(dinf_diag_path(item_path, "rank"), item.shape.size());
    AddIntList(out, dinf_diag_path(item_path, "shape"), item.shape);
    out->AddStringList(
        dinf_diag_path(item_path, "symbolic_shape"),
        item.symbolic_shape);
    out->AddStringList(
        dinf_diag_path(item_path, "shape_signature"),
        ShapeSignature(item));
  }
}

std::string RequestedProvider(const DinfOptions* runtime_options) {
  const std::string provider =
      dinf_option_string(runtime_options, "provider");
  if (!provider.empty()) {
    return provider;
  }
  return "CPUExecutionProvider";
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
      api->SessionOptionsAppendExecutionProvider_TensorRT != nullptr) {
    OrtTensorRTProviderOptions trt_options{};
    trt_options.device_id =
        std::max(0, dinf_option_int(runtime_options, "deviceId", 0));
    trt_options.trt_max_partition_iterations =
        std::max(0, dinf_option_int(runtime_options, "trtMaxPartitionIterations", 0));
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
          dinf_option_bool(runtime_options, "trtEngineCacheEnable", false) ? 1 : 0;
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

}  // namespace

Session::Session(
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
    std::vector<TensorMetadata> input_metadata,
    std::vector<TensorMetadata> output_metadata,
    int num_threads,
    bool provider_appended)
    : api_(api),
      env_(env),
      options_(options),
      session_(session),
      allocator_(allocator),
      memory_info_(memory_info),
      provider_(std::move(provider)),
      available_providers_(std::move(available_providers)),
      input_names_(std::move(input_names)),
      output_names_(std::move(output_names)),
      input_metadata_(std::move(input_metadata)),
      output_metadata_(std::move(output_metadata)),
      num_threads_(num_threads),
      provider_appended_(provider_appended) {}

Session::~Session() {
  api_->ReleaseMemoryInfo(memory_info_);
  api_->ReleaseSession(session_);
  api_->ReleaseSessionOptions(options_);
  api_->ReleaseEnv(env_);
}

const std::vector<std::string>& Session::InputNames() const {
  return input_names_;
}

const std::vector<std::string>& Session::OutputNames() const {
  return output_names_;
}

int Session::Run(
    const DinfNamedTensor* inputs,
    size_t input_count,
    DinfNamedTensor** outputs,
    size_t* output_count,
    std::string* error) {
  std::vector<OrtValue*> input_values(input_count);
  std::vector<const char*> input_names(input_count);
  for (size_t i = 0; i < input_count; ++i) {
    input_names[i] = inputs[i].name;
    const auto dtype = OrtDtype(inputs[i].tensor.dtype);
    if (dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
      *error = "Unsupported ONNX Runtime input dtype";
      ReleaseValues(input_values);
      return 1;
    }
    if (!Ok(api_, api_->CreateTensorWithDataAsOrtValue(
                      memory_info_,
                      inputs[i].tensor.data,
                      inputs[i].tensor.byte_length,
                      inputs[i].tensor.shape,
                      inputs[i].tensor.rank,
                      dtype,
                      &input_values[i]),
            error)) {
      ReleaseValues(input_values);
      return 1;
    }
  }

  std::vector<const char*> output_names;
  output_names.reserve(output_names_.size());
  for (const auto& name : output_names_) {
    output_names.push_back(name.c_str());
  }

  std::vector<OrtValue*> output_values(output_names.size());
  if (!Ok(api_,
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
    ReleaseValues(input_values);
    ReleaseValues(output_values);
    return 1;
  }

  std::vector<DinfNamedTensor> produced;
  for (size_t i = 0; i < output_values.size(); ++i) {
    OrtTensorTypeAndShapeInfo* info = nullptr;
    if (!Ok(api_, api_->GetTensorTypeAndShape(output_values[i], &info), error)) {
      ReleaseValues(input_values);
      ReleaseValues(output_values);
      return 1;
    }
    ONNXTensorElementDataType ort_type;
    if (!Ok(api_, api_->GetTensorElementType(info, &ort_type), error)) {
      api_->ReleaseTensorTypeAndShapeInfo(info);
      ReleaseValues(input_values);
      ReleaseValues(output_values);
      return 1;
    }
    const int32_t dtype = RuntimeDtype(ort_type);
    if (dtype == 0) {
      api_->ReleaseTensorTypeAndShapeInfo(info);
      continue;
    }
    size_t rank = 0;
    if (!Ok(api_, api_->GetDimensionsCount(info, &rank), error)) {
      api_->ReleaseTensorTypeAndShapeInfo(info);
      ReleaseValues(input_values);
      ReleaseValues(output_values);
      return 1;
    }
    std::vector<int64_t> shape(rank);
    if (!Ok(api_, api_->GetDimensions(info, shape.data(), rank), error)) {
      api_->ReleaseTensorTypeAndShapeInfo(info);
      ReleaseValues(input_values);
      ReleaseValues(output_values);
      return 1;
    }
    size_t count = 0;
    if (!Ok(api_, api_->GetTensorShapeElementCount(info, &count), error)) {
      api_->ReleaseTensorTypeAndShapeInfo(info);
      ReleaseValues(input_values);
      ReleaseValues(output_values);
      return 1;
    }
    api_->ReleaseTensorTypeAndShapeInfo(info);
    void* data = nullptr;
    if (!Ok(api_, api_->GetTensorMutableData(output_values[i], &data), error)) {
      ReleaseValues(input_values);
      ReleaseValues(output_values);
      return 1;
    }
    produced.push_back(dinf_make_tensor(
        output_names[i],
        dtype,
        shape,
        data,
        count * dinf_dtype_size(dtype)));
  }

  ReleaseValues(input_values);
  ReleaseValues(output_values);
  *output_count = produced.size();
  *outputs = static_cast<DinfNamedTensor*>(
      std::malloc(sizeof(DinfNamedTensor) * produced.size()));
  if (!produced.empty()) {
    std::memcpy(*outputs, produced.data(), sizeof(DinfNamedTensor) * produced.size());
  }
  return 0;
}

void Session::Diagnostics(
    DinfDiagBuilder* out,
    const std::string& prefix) const {
  out->AddString(dinf_diag_path(prefix, "engine"), "onnx");
  out->AddString(dinf_diag_path(prefix, "provider"), provider_);
  out->AddBool(dinf_diag_path(prefix, "provider_appended"), provider_appended_);
  out->AddInt(dinf_diag_path(prefix, "num_threads"), num_threads_);
  out->AddStringList(
      dinf_diag_path(prefix, "available_providers"),
      available_providers_);
  out->AddStringList(dinf_diag_path(prefix, "input_names"), input_names_);
  out->AddStringList(dinf_diag_path(prefix, "output_names"), output_names_);
  AddTensorMetadataList(
      out,
      dinf_diag_path(prefix, "input_metadata"),
      input_metadata_);
  AddTensorMetadataList(
      out,
      dinf_diag_path(prefix, "output_metadata"),
      output_metadata_);
}

void Session::ReleaseValues(std::vector<OrtValue*>& values) {
  for (auto* value : values) {
    if (value != nullptr) {
      api_->ReleaseValue(value);
    }
  }
}

void Session::ReleaseNames(std::vector<char*>& names) {
  for (auto* name : names) {
    if (name != nullptr) {
      allocator_->Free(allocator_, name);
    }
  }
  names.clear();
}

std::unique_ptr<Session> CreateSession(
    const char* model_path,
    const DinfOptions* runtime_options,
    std::string* error) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  OrtEnv* env = nullptr;
  OrtSessionOptions* options = nullptr;
  OrtSession* session = nullptr;
  OrtAllocator* allocator = nullptr;
  OrtMemoryInfo* memory_info = nullptr;
  if (!Ok(api, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "dart_inference", &env), error)) {
    return nullptr;
  }
  if (!Ok(api, api->CreateSessionOptions(&options), error)) {
    api->ReleaseEnv(env);
    return nullptr;
  }
  const int num_threads = std::max(1, dinf_option_int(runtime_options, "numThreads", 1));
  if (!Ok(api, api->SetIntraOpNumThreads(options, num_threads), error)) {
    api->ReleaseSessionOptions(options);
    api->ReleaseEnv(env);
    return nullptr;
  }
  if (!Ok(api, api->SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL), error)) {
    api->ReleaseSessionOptions(options);
    api->ReleaseEnv(env);
    return nullptr;
  }
  if (!PreloadLibraries(runtime_options, error)) {
    api->ReleaseSessionOptions(options);
    api->ReleaseEnv(env);
    return nullptr;
  }
  std::string provider_error;
  const std::vector<std::string> providers = AvailableProviders(api, &provider_error);
  if (!provider_error.empty()) {
    api->ReleaseSessionOptions(options);
    api->ReleaseEnv(env);
    *error = provider_error;
    return nullptr;
  }
  const std::string provider = RequestedProvider(runtime_options);
  bool provider_appended = false;
  if (!AppendProvider(
          api,
          options,
          provider,
          providers,
          runtime_options,
          &provider_appended,
          error)) {
    api->ReleaseSessionOptions(options);
    api->ReleaseEnv(env);
    return nullptr;
  }
  const std::string effective_provider =
      provider == "CPUExecutionProvider" || provider_appended
      ? provider
      : "CPUExecutionProvider";
#if defined(_WIN32)
  const std::wstring wide_path = Utf8ToWide(model_path);
  const auto* ort_path = wide_path.c_str();
#else
  const auto* ort_path = model_path;
#endif
  if (!Ok(api, api->CreateSession(env, ort_path, options, &session), error)) {
    api->ReleaseSessionOptions(options);
    api->ReleaseEnv(env);
    return nullptr;
  }
  if (!Ok(api, api->GetAllocatorWithDefaultOptions(&allocator), error)) {
    api->ReleaseSession(session);
    api->ReleaseSessionOptions(options);
    api->ReleaseEnv(env);
    return nullptr;
  }
  if (!Ok(api,
          api->CreateCpuMemoryInfo(
              OrtArenaAllocator, OrtMemTypeDefault, &memory_info),
          error)) {
    api->ReleaseSession(session);
    api->ReleaseSessionOptions(options);
    api->ReleaseEnv(env);
    return nullptr;
  }
  std::vector<TensorMetadata> input_metadata =
      SessionMetadata(api, session, allocator, true, error);
  if (!error->empty()) {
    api->ReleaseMemoryInfo(memory_info);
    api->ReleaseSession(session);
    api->ReleaseSessionOptions(options);
    api->ReleaseEnv(env);
    return nullptr;
  }
  std::vector<TensorMetadata> output_metadata =
      SessionMetadata(api, session, allocator, false, error);
  if (!error->empty()) {
    api->ReleaseMemoryInfo(memory_info);
    api->ReleaseSession(session);
    api->ReleaseSessionOptions(options);
    api->ReleaseEnv(env);
    return nullptr;
  }
  std::vector<std::string> input_names = MetadataNames(input_metadata);
  std::vector<std::string> output_names = MetadataNames(output_metadata);
  return std::unique_ptr<Session>(new Session(
      api,
      env,
      options,
      session,
      allocator,
      memory_info,
      effective_provider,
      providers,
      std::move(input_names),
      std::move(output_names),
      std::move(input_metadata),
      std::move(output_metadata),
      num_threads,
      provider_appended));
}

std::vector<int64_t> TensorShape(const DinfTensor& tensor) {
  std::vector<int64_t> shape;
  shape.reserve(tensor.rank);
  for (int i = 0; i < tensor.rank; ++i) {
    shape.push_back(tensor.shape[i]);
  }
  return shape;
}

}  // namespace dinf_ort

#endif
