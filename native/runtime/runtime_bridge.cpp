#include "runtime_bridge.h"

#include <cstdlib>
#include <cstring>

namespace {

void set_error(char** out, const std::string& message) {
  if (out != nullptr) {
    *out = dinf_copy_string(message);
  }
}

}  // namespace

extern "C" DinfRuntimeSession* dinf_cpp_runtime_create(
    int32_t engine,
    const char* model_path,
    const char* options_json,
    char** error) {
  if (model_path == nullptr) {
    set_error(error, "model_path is null");
    return nullptr;
  }
  std::string message;
  DinfRuntimeSession* session = nullptr;
  switch (engine) {
    case DINF_ENGINE_COREML:
      session = dinf_create_coreml_session(model_path, options_json, &message);
      break;
    case DINF_ENGINE_ONNX:
      session = dinf_create_onnx_session(model_path, options_json, &message);
      break;
    case DINF_ENGINE_LITERT:
      session = dinf_create_litert_session(model_path, options_json, &message);
      break;
    default:
      message = "Unsupported native runtime engine";
      break;
  }
  if (session == nullptr) {
    set_error(error, message.empty() ? "Failed to create runtime session"
                                     : message);
  }
  return session;
}

extern "C" void dinf_cpp_runtime_free(DinfRuntimeSession* session) {
  delete session;
}

extern "C" int32_t dinf_cpp_runtime_run(
    DinfRuntimeSession* session,
    const DartInferenceNamedTensor* inputs,
    intptr_t input_count,
    DartInferenceNamedTensor** outputs,
    intptr_t* output_count,
    char** error) {
  if (session == nullptr) {
    set_error(error, "runtime session is null");
    return 1;
  }
  if (outputs == nullptr || output_count == nullptr) {
    set_error(error, "output pointers are null");
    return 1;
  }
  *outputs = nullptr;
  *output_count = 0;
  size_t produced = 0;
  std::string message;
  const int status = session->Run(
      inputs,
      input_count < 0 ? 0 : static_cast<size_t>(input_count),
      outputs,
      &produced,
      &message);
  *output_count = static_cast<intptr_t>(produced);
  if (status != 0) {
    set_error(error, message.empty() ? "runtime run failed" : message);
  }
  return status;
}

extern "C" void dinf_cpp_runtime_free_tensors(
    DartInferenceNamedTensor* tensors,
    intptr_t count) {
  if (tensors == nullptr) {
    return;
  }
  for (intptr_t i = 0; i < count; ++i) {
    std::free(tensors[i].name);
    std::free(tensors[i].tensor.shape);
    std::free(tensors[i].tensor.data);
  }
  std::free(tensors);
}

extern "C" void dinf_cpp_runtime_free_string(char* value) {
  std::free(value);
}

extern "C" char* dinf_cpp_runtime_backend_json() {
  return dinf_copy_string(
      "{\"native_backend\":\"cpp-adapter\",\"abi\":\"dinf_cpp_runtime_adapter_v1\"}");
}

extern "C" char* dinf_cpp_runtime_diagnostics_json(DinfRuntimeSession* session) {
  if (session == nullptr) {
    return dinf_copy_string("{}");
  }
  return dinf_copy_string(session->DiagnosticsJson());
}

DartInferenceNamedTensor dinf_make_tensor(
    const char* name,
    int32_t dtype,
    const std::vector<int64_t>& shape,
    const void* data,
    size_t byte_length) {
  DartInferenceNamedTensor tensor{};
  tensor.name = dinf_copy_string(name == nullptr ? "" : name);
  tensor.tensor.dtype = dtype;
  tensor.tensor.rank = static_cast<int32_t>(shape.size());
  tensor.tensor.byte_length = static_cast<intptr_t>(byte_length);
  tensor.tensor.shape = static_cast<int64_t*>(
      std::malloc(sizeof(int64_t) * shape.size()));
  if (!shape.empty()) {
    std::memcpy(
        tensor.tensor.shape,
        shape.data(),
        sizeof(int64_t) * shape.size());
  }
  tensor.tensor.data = std::malloc(byte_length);
  if (byte_length > 0 && data != nullptr) {
    std::memcpy(tensor.tensor.data, data, byte_length);
  }
  return tensor;
}

size_t dinf_dtype_size(int32_t dtype) {
  switch (dtype) {
    case DINF_DTYPE_FLOAT32:
    case DINF_DTYPE_INT32:
      return 4;
    case DINF_DTYPE_INT64:
    case DINF_DTYPE_FLOAT64:
      return 8;
    case DINF_DTYPE_UINT8:
    case DINF_DTYPE_BOOL:
      return 1;
    case DINF_DTYPE_FLOAT16:
      return 2;
    default:
      return 0;
  }
}

char* dinf_copy_string(const std::string& value) {
  auto* out = static_cast<char*>(std::malloc(value.size() + 1));
  if (out == nullptr) {
    return nullptr;
  }
  std::memcpy(out, value.c_str(), value.size() + 1);
  return out;
}

std::string dinf_json_escape(const std::string& value) {
  std::string out;
  out.reserve(value.size() + 2);
  for (const char c : value) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out.push_back(c);
        break;
    }
  }
  return out;
}

std::string dinf_json_string_array(const std::vector<std::string>& values) {
  std::string out = "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      out += ",";
    }
    out += "\"";
    out += dinf_json_escape(values[i]);
    out += "\"";
  }
  out += "]";
  return out;
}
