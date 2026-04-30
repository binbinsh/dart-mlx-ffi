#include "ort_session.h"

#if DINF_ENABLE_ORT
#include "options.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
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

struct SharedPrepackedWeights {
  SharedPrepackedWeights(
      const OrtApi* api,
      OrtPrepackedWeightsContainer* container)
      : api(api), container(container) {}

  ~SharedPrepackedWeights() {
    if (container != nullptr) {
      api->ReleasePrepackedWeightsContainer(container);
    }
  }

  const OrtApi* api;
  OrtPrepackedWeightsContainer* container;
};

std::mutex& PrepackedWeightsMutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<std::string, std::weak_ptr<SharedPrepackedWeights>>&
PrepackedWeightsRegistry() {
  static std::unordered_map<std::string, std::weak_ptr<SharedPrepackedWeights>>
      registry;
  return registry;
}

std::shared_ptr<SharedPrepackedWeights> SharedPrepackedWeightsFor(
    const OrtApi* api,
    const std::string& key,
    std::string* error) {
  if (key.empty()) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(PrepackedWeightsMutex());
  auto& registry = PrepackedWeightsRegistry();
  const auto found = registry.find(key);
  if (found != registry.end()) {
    if (auto existing = found->second.lock()) {
      return existing;
    }
    registry.erase(found);
  }
  OrtPrepackedWeightsContainer* container = nullptr;
  if (!Ok(api, api->CreatePrepackedWeightsContainer(&container), error)) {
    return nullptr;
  }
  auto shared = std::make_shared<SharedPrepackedWeights>(api, container);
  registry[key] = shared;
  return shared;
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

std::string TrimAsciiSpace(const std::string& value) {
  size_t begin = 0;
  while (begin < value.size() &&
         (value[begin] == ' ' || value[begin] == '\t')) {
    begin += 1;
  }
  size_t end = value.size();
  while (end > begin && (value[end - 1] == ' ' || value[end - 1] == '\t')) {
    end -= 1;
  }
  return value.substr(begin, end - begin);
}

std::vector<std::string> SplitNames(const std::string& raw) {
  std::vector<std::string> out;
  std::string current;
  for (const char ch : raw) {
    if (ch == ',' || ch == ';' || ch == '\n' || ch == '\r') {
      const std::string name = TrimAsciiSpace(current);
      if (!name.empty()) {
        out.push_back(name);
      }
      current.clear();
      continue;
    }
    current.push_back(ch);
  }
  const std::string name = TrimAsciiSpace(current);
  if (!name.empty()) {
    out.push_back(name);
  }
  return out;
}

bool ContainsName(
    const std::vector<std::string>& names,
    const std::string& name) {
  return std::find(names.begin(), names.end(), name) != names.end();
}

#include "ort_session_tensor.inc"

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

bool StaticTensorShapes(const std::vector<TensorMetadata>& metadata) {
  for (const auto& item : metadata) {
    if (item.onnx_type != "tensor") {
      return false;
    }
    for (const auto dim : item.shape) {
      if (dim < 0) {
        return false;
      }
    }
  }
  return true;
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

std::shared_ptr<OrtEnv> SharedEnv(const OrtApi* api, std::string* error) {
  static std::mutex mutex;
  static std::weak_ptr<OrtEnv> env_ref;
  std::lock_guard<std::mutex> lock(mutex);
  if (auto existing = env_ref.lock()) {
    return existing;
  }
  OrtEnv* raw = nullptr;
  if (!Ok(
          api,
          api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "dart_inference", &raw),
          error)) {
    return {};
  }
  auto env = std::shared_ptr<OrtEnv>(
      raw,
      [api](OrtEnv* value) {
        if (value != nullptr) {
          api->ReleaseEnv(value);
        }
      });
  env_ref = env;
  return env;
}

#include "ort_session_provider.inc"

}  // namespace

#include "ort_session_lifecycle.inc"

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
  std::vector<uint8_t> input_owned(input_count, 1);
  std::vector<const char*> input_names(input_count);
  for (size_t i = 0; i < input_count; ++i) {
    input_names[i] = inputs[i].name;
    if (inputs[i].tensor.memory_kind == DINF_TENSOR_MEMORY_HANDLE) {
      const auto* handle = inputs[i].tensor.handle;
      if (handle == nullptr || handle->value == nullptr) {
        *error = "ONNX Runtime input tensor handle is null";
        ReleaseValues(input_values, &input_owned);
        return 1;
      }
      input_values[i] = static_cast<OrtValue*>(handle->value);
      input_owned[i] = 0;
      continue;
    }
    if (inputs[i].tensor.memory_kind != DINF_TENSOR_MEMORY_CPU) {
      *error = "Unsupported ONNX Runtime input memory kind";
      ReleaseValues(input_values, &input_owned);
      return 1;
    }
    const auto dtype = OrtDtype(inputs[i].tensor.dtype);
    if (dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
      *error = "Unsupported ONNX Runtime input dtype";
      ReleaseValues(input_values, &input_owned);
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
      ReleaseValues(input_values, &input_owned);
      return 1;
    }
  }

  std::vector<const char*> output_names;
  output_names.reserve(output_names_.size());
  for (const auto& name : output_names_) {
    output_names.push_back(name.c_str());
  }

  std::vector<OrtValue*> output_values;
  if (use_io_binding_ && io_binding_ != nullptr) {
    api_->ClearBoundInputs(io_binding_);
    for (size_t i = 0; i < input_values.size(); ++i) {
      if (!Ok(api_,
              api_->BindInput(io_binding_, input_names[i], input_values[i]),
              error)) {
        ReleaseValues(input_values, &input_owned);
        return 1;
      }
    }
    if (!cache_bound_outputs_ || !bound_outputs_cached_) {
      api_->ClearBoundOutputs(io_binding_);
      for (size_t i = 0; i < output_names.size(); ++i) {
        const auto* name = output_names[i];
        const auto& output_name = output_names_[i];
        OrtMemoryInfo* target_memory_info =
            OutputUsesDeviceHandle(output_name) ? device_memory_info_ : memory_info_;
        if (!Ok(api_,
                api_->BindOutputToDevice(io_binding_, name, target_memory_info),
                error)) {
          ReleaseValues(input_values, &input_owned);
          return 1;
        }
      }
      bound_outputs_cached_ = cache_bound_outputs_;
    }
    if (sync_bound_inputs_ &&
        !Ok(api_, api_->SynchronizeBoundInputs(io_binding_), error)) {
      ReleaseValues(input_values, &input_owned);
      return 1;
    }
    if (!Ok(api_, api_->RunWithBinding(session_, run_options_, io_binding_), error)) {
      ReleaseValues(input_values, &input_owned);
      return 1;
    }
    if (sync_bound_outputs_ &&
        !Ok(api_, api_->SynchronizeBoundOutputs(io_binding_), error)) {
      ReleaseValues(input_values, &input_owned);
      return 1;
    }
    OrtValue** bound_outputs = nullptr;
    size_t bound_output_count = 0;
    if (!Ok(api_,
            api_->GetBoundOutputValues(
                io_binding_,
                allocator_,
                &bound_outputs,
                &bound_output_count),
            error)) {
      ReleaseValues(input_values, &input_owned);
      return 1;
    }
    output_values.assign(bound_outputs, bound_outputs + bound_output_count);
    if (bound_outputs != nullptr) {
      allocator_->Free(allocator_, bound_outputs);
    }
  } else {
    output_values.resize(output_names.size());
    if (!Ok(api_,
            api_->Run(
                session_,
                run_options_,
                input_names.data(),
                input_values.data(),
                input_values.size(),
                output_names.data(),
                output_names.size(),
                output_values.data()),
            error)) {
      ReleaseValues(input_values, &input_owned);
      ReleaseValues(output_values);
      return 1;
    }
  }

  std::vector<DinfNamedTensor> produced;
  for (size_t i = 0; i < output_values.size(); ++i) {
    OrtTensorTypeAndShapeInfo* info = nullptr;
    if (!Ok(api_, api_->GetTensorTypeAndShape(output_values[i], &info), error)) {
      ReleaseValues(input_values, &input_owned);
      ReleaseValues(output_values);
      return 1;
    }
    ONNXTensorElementDataType ort_type;
    if (!Ok(api_, api_->GetTensorElementType(info, &ort_type), error)) {
      api_->ReleaseTensorTypeAndShapeInfo(info);
      ReleaseValues(input_values, &input_owned);
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
      FreeProduced(&produced);
      ReleaseValues(input_values, &input_owned);
      ReleaseValues(output_values);
      return 1;
    }
    std::vector<int64_t> shape(rank);
    if (!Ok(api_, api_->GetDimensions(info, shape.data(), rank), error)) {
      api_->ReleaseTensorTypeAndShapeInfo(info);
      FreeProduced(&produced);
      ReleaseValues(input_values, &input_owned);
      ReleaseValues(output_values);
      return 1;
    }
    size_t count = 0;
    if (!Ok(api_, api_->GetTensorShapeElementCount(info, &count), error)) {
      api_->ReleaseTensorTypeAndShapeInfo(info);
      FreeProduced(&produced);
      ReleaseValues(input_values, &input_owned);
      ReleaseValues(output_values);
      return 1;
    }
    api_->ReleaseTensorTypeAndShapeInfo(info);
    const size_t byte_length = count * dinf_dtype_size(dtype);
    const std::string output_name =
        i < output_names_.size() ? output_names_[i] : std::string();
    if (OutputUsesDeviceHandle(output_name)) {
      DinfNamedTensor tensor = MakeHandleTensor(
          api_,
          output_name.c_str(),
          dtype,
          shape,
          byte_length,
          output_values[i]);
      if (tensor.name == nullptr ||
          (rank > 0 && tensor.tensor.shape == nullptr) ||
          tensor.tensor.handle == nullptr) {
        std::free(tensor.name);
        std::free(tensor.tensor.shape);
        if (tensor.tensor.handle != nullptr) {
          dinf_release_tensor_handle(tensor.tensor.handle);
          output_values[i] = nullptr;
        }
        FreeProduced(&produced);
        ReleaseValues(input_values, &input_owned);
        ReleaseValues(output_values);
        *error = "Failed to allocate ONNX Runtime output tensor handle";
        return 1;
      }
      output_values[i] = nullptr;
      produced.push_back(tensor);
      continue;
    }
    void* data = nullptr;
    if (!Ok(api_, api_->GetTensorMutableData(output_values[i], &data), error)) {
      FreeProduced(&produced);
      ReleaseValues(input_values, &input_owned);
      ReleaseValues(output_values);
      return 1;
    }
    if (use_output_views_) {
      DinfNamedTensor tensor = MakeCpuViewTensor(
          api_,
          output_name.c_str(),
          dtype,
          shape,
          byte_length,
          data,
          output_values[i]);
      if (tensor.name == nullptr ||
          (rank > 0 && tensor.tensor.shape == nullptr) ||
          tensor.tensor.handle == nullptr) {
        std::free(tensor.name);
        std::free(tensor.tensor.shape);
        if (tensor.tensor.handle != nullptr) {
          dinf_release_tensor_handle(tensor.tensor.handle);
          output_values[i] = nullptr;
        }
        FreeProduced(&produced);
        ReleaseValues(input_values, &input_owned);
        ReleaseValues(output_values);
        *error = "Failed to allocate ONNX Runtime CPU output tensor view";
        return 1;
      }
      output_values[i] = nullptr;
      produced.push_back(tensor);
      continue;
    }
    produced.push_back(dinf_make_tensor(
        output_name.c_str(),
        dtype,
        shape,
        data,
        byte_length));
  }

  ReleaseValues(input_values, &input_owned);
  ReleaseValues(output_values);
  *output_count = 0;
  *outputs = nullptr;
  if (produced.empty()) {
    return 0;
  }
  auto* output_array = static_cast<DinfNamedTensor*>(
      std::malloc(sizeof(DinfNamedTensor) * produced.size()));
  if (output_array == nullptr) {
    FreeProduced(&produced);
    *error = "Failed to allocate ONNX Runtime output tensor array";
    return 1;
  }
  std::memcpy(
      output_array,
      produced.data(),
      sizeof(DinfNamedTensor) * produced.size());
  *outputs = output_array;
  *output_count = produced.size();
  return 0;
}

#include "ort_session_diag.inc"

bool Session::OutputUsesDeviceHandle(const std::string& name) const {
  if (!use_io_binding_ || !use_device_outputs_ ||
      device_memory_info_ == nullptr) {
    return false;
  }
  if (!device_output_names_.empty()) {
    return ContainsName(device_output_names_, name);
  }
  return !ContainsName(cpu_output_names_, name);
}

void Session::ReleaseValues(
    std::vector<OrtValue*>& values,
    const std::vector<uint8_t>* owned) {
  for (size_t i = 0; i < values.size(); ++i) {
    if (owned != nullptr && (i >= owned->size() || (*owned)[i] == 0)) {
      continue;
    }
    auto* value = values[i];
    if (value != nullptr) {
      api_->ReleaseValue(value);
      values[i] = nullptr;
    }
  }
}

std::unique_ptr<Session> CreateSession(
    const char* model_path,
    const DinfOptions* runtime_options,
    std::string* error) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  std::shared_ptr<OrtEnv> env = SharedEnv(api, error);
  OrtSessionOptions* options = nullptr;
  OrtSession* session = nullptr;
  OrtAllocator* allocator = nullptr;
  OrtMemoryInfo* memory_info = nullptr;
  OrtMemoryInfo* device_memory_info = nullptr;
  OrtRunOptions* run_options = nullptr;
  if (!env) {
    return nullptr;
  }
  if (!Ok(api, api->CreateSessionOptions(&options), error)) {
    return nullptr;
  }
  const int num_threads = std::max(1, dinf_option_int(runtime_options, "numThreads", 1));
  if (!Ok(api, api->SetIntraOpNumThreads(options, num_threads), error)) {
    api->ReleaseSessionOptions(options);
    return nullptr;
  }
  if (!Ok(api, api->SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL), error)) {
    api->ReleaseSessionOptions(options);
    return nullptr;
  }
  if (!ApplySessionConfigEntries(api, options, runtime_options, error)) {
    api->ReleaseSessionOptions(options);
    return nullptr;
  }
  if (dinf_option_bool(runtime_options, "disableMemPattern", false) &&
      !Ok(api, api->DisableMemPattern(options), error)) {
    api->ReleaseSessionOptions(options);
    return nullptr;
  }
  if (dinf_option_bool(runtime_options, "disableCpuMemArena", false) &&
      !Ok(api, api->DisableCpuMemArena(options), error)) {
    api->ReleaseSessionOptions(options);
    return nullptr;
  }
  const std::string execution_mode =
      dinf_option_string(runtime_options, "executionMode");
  if (!execution_mode.empty()) {
    const ExecutionMode mode =
        execution_mode == "parallel" || execution_mode == "ORT_PARALLEL"
        ? ORT_PARALLEL
        : ORT_SEQUENTIAL;
    if (!Ok(api, api->SetSessionExecutionMode(options, mode), error)) {
      api->ReleaseSessionOptions(options);
      return nullptr;
    }
  }
  const std::string optimized_model_path =
      dinf_option_string(runtime_options, "optimizedModelFilePath");
  if (!optimized_model_path.empty()) {
#if defined(_WIN32)
    const std::wstring wide_optimized_path =
        Utf8ToWide(optimized_model_path.c_str());
    const auto* optimized_path = wide_optimized_path.c_str();
#else
    const auto* optimized_path = optimized_model_path.c_str();
#endif
    if (!Ok(api, api->SetOptimizedModelFilePath(options, optimized_path), error)) {
      api->ReleaseSessionOptions(options);
      return nullptr;
    }
  }
  if (!PreloadLibraries(runtime_options, error)) {
    api->ReleaseSessionOptions(options);
    return nullptr;
  }
  std::string provider_error;
  const std::vector<std::string> providers = AvailableProviders(api, &provider_error);
  if (!provider_error.empty()) {
    api->ReleaseSessionOptions(options);
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
  const std::string prepacked_weights_key =
      dinf_option_string(runtime_options, "prepackedWeightsKey");
  std::shared_ptr<SharedPrepackedWeights> prepacked_weights;
  if (!prepacked_weights_key.empty()) {
    prepacked_weights =
        SharedPrepackedWeightsFor(api, prepacked_weights_key, error);
    if (!prepacked_weights) {
      api->ReleaseSessionOptions(options);
      return nullptr;
    }
  }
  OrtStatus* create_status = nullptr;
  if (prepacked_weights) {
    create_status = api->CreateSessionWithPrepackedWeightsContainer(
        env.get(),
        ort_path,
        options,
        prepacked_weights->container,
        &session);
  } else {
    create_status = api->CreateSession(env.get(), ort_path, options, &session);
  }
  if (!Ok(api, create_status, error)) {
    api->ReleaseSessionOptions(options);
    return nullptr;
  }
  if (!Ok(api, api->GetAllocatorWithDefaultOptions(&allocator), error)) {
    api->ReleaseSession(session);
    api->ReleaseSessionOptions(options);
    return nullptr;
  }
  if (!Ok(api,
          api->CreateCpuMemoryInfo(
              OrtArenaAllocator, OrtMemTypeDefault, &memory_info),
          error)) {
    api->ReleaseSession(session);
    api->ReleaseSessionOptions(options);
    return nullptr;
  }
  OrtIoBinding* io_binding = nullptr;
  const bool use_io_binding = dinf_option_bool(
      runtime_options,
      "useIoBinding",
      dinf_option_bool(runtime_options, "ioBinding", false));
  const bool requested_device_outputs = dinf_option_bool(
      runtime_options,
      "useDeviceOutputs",
      dinf_option_bool(runtime_options, "deviceOutputs", false));
  const bool cuda_backed_provider =
      (effective_provider == "CUDAExecutionProvider" ||
       effective_provider == "TensorrtExecutionProvider") &&
      provider_appended;
  const bool use_device_outputs =
      use_io_binding && requested_device_outputs && cuda_backed_provider;
  const bool use_output_views = dinf_option_bool(
      runtime_options,
      "useOutputViews",
      dinf_option_bool(runtime_options, "zeroCopyOutputs", false));
  const bool sync_bound_inputs = dinf_option_bool(
      runtime_options,
      "syncBoundInputs",
      dinf_option_bool(runtime_options, "syncIoBindingInputs", true));
  const bool sync_bound_outputs = dinf_option_bool(
      runtime_options,
      "syncBoundOutputs",
      dinf_option_bool(runtime_options, "syncIoBindingOutputs", true));
  const bool requested_cache_bound_outputs = dinf_option_bool(
      runtime_options,
      "cacheBoundOutputs",
      dinf_option_bool(runtime_options, "cacheIoBindingOutputs", false));
  if (use_device_outputs &&
      !Ok(api,
          api->CreateMemoryInfo(
              "Cuda",
              OrtDeviceAllocator,
              std::max(0, dinf_option_int(runtime_options, "deviceId", 0)),
              OrtMemTypeDefault,
              &device_memory_info),
          error)) {
    api->ReleaseMemoryInfo(memory_info);
    api->ReleaseSession(session);
    api->ReleaseSessionOptions(options);
    return nullptr;
  }
  if (use_io_binding &&
      !Ok(api, api->CreateIoBinding(session, &io_binding), error)) {
    if (device_memory_info != nullptr) {
      api->ReleaseMemoryInfo(device_memory_info);
    }
    api->ReleaseMemoryInfo(memory_info);
    api->ReleaseSession(session);
    api->ReleaseSessionOptions(options);
    return nullptr;
  }
  const int cuda_graph_id = dinf_option_int(runtime_options, "cudaGraphId", -1);
  if (cuda_graph_id >= 0) {
    if (!Ok(api, api->CreateRunOptions(&run_options), error)) {
      if (io_binding != nullptr) {
        api->ReleaseIoBinding(io_binding);
      }
      if (device_memory_info != nullptr) {
        api->ReleaseMemoryInfo(device_memory_info);
      }
      api->ReleaseMemoryInfo(memory_info);
      api->ReleaseSession(session);
      api->ReleaseSessionOptions(options);
      return nullptr;
    }
    const std::string graph_id = std::to_string(cuda_graph_id);
    if (!Ok(
            api,
            api->AddRunConfigEntry(
                run_options, "gpu_graph_id", graph_id.c_str()),
            error)) {
      api->ReleaseRunOptions(run_options);
      if (io_binding != nullptr) {
        api->ReleaseIoBinding(io_binding);
      }
      if (device_memory_info != nullptr) {
        api->ReleaseMemoryInfo(device_memory_info);
      }
      api->ReleaseMemoryInfo(memory_info);
      api->ReleaseSession(session);
      api->ReleaseSessionOptions(options);
      return nullptr;
    }
  }
  std::vector<TensorMetadata> input_metadata =
      SessionMetadata(api, session, allocator, true, error);
  if (!error->empty()) {
    if (run_options != nullptr) {
      api->ReleaseRunOptions(run_options);
    }
    if (io_binding != nullptr) {
      api->ReleaseIoBinding(io_binding);
    }
    if (device_memory_info != nullptr) {
      api->ReleaseMemoryInfo(device_memory_info);
    }
    api->ReleaseMemoryInfo(memory_info);
    api->ReleaseSession(session);
    api->ReleaseSessionOptions(options);
    return nullptr;
  }
  std::vector<TensorMetadata> output_metadata =
      SessionMetadata(api, session, allocator, false, error);
  if (!error->empty()) {
    if (run_options != nullptr) {
      api->ReleaseRunOptions(run_options);
    }
    if (io_binding != nullptr) {
      api->ReleaseIoBinding(io_binding);
    }
    if (device_memory_info != nullptr) {
      api->ReleaseMemoryInfo(device_memory_info);
    }
    api->ReleaseMemoryInfo(memory_info);
    api->ReleaseSession(session);
    api->ReleaseSessionOptions(options);
    return nullptr;
  }
  const bool cache_bound_outputs =
      requested_cache_bound_outputs && StaticTensorShapes(output_metadata);
  std::vector<std::string> input_names = MetadataNames(input_metadata);
  std::vector<std::string> output_names = MetadataNames(output_metadata);
  std::vector<std::string> cpu_output_names =
      SplitNames(dinf_option_string(runtime_options, "cpuOutputNames"));
  std::vector<std::string> device_output_names =
      SplitNames(dinf_option_string(runtime_options, "deviceOutputNames"));
  return std::unique_ptr<Session>(new Session(
      api,
      std::move(env),
      options,
      session,
      allocator,
      memory_info,
      device_memory_info,
      effective_provider,
      providers,
      std::move(input_names),
      std::move(output_names),
      std::move(cpu_output_names),
      std::move(device_output_names),
      std::move(input_metadata),
      std::move(output_metadata),
      num_threads,
      provider_appended,
      use_io_binding,
      use_device_outputs,
      use_output_views,
      sync_bound_inputs,
      sync_bound_outputs,
      cache_bound_outputs,
      prepacked_weights_key,
      std::static_pointer_cast<void>(prepacked_weights),
      io_binding,
      run_options,
      cuda_graph_id));
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
