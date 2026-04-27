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

DinfDiagBuilder::~DinfDiagBuilder() {
  Free();
}

void DinfDiagBuilder::AddString(
    const std::string& path,
    const std::string& value) {
  char* owned_path = Copy(path);
  char* owned_text = Copy(value);
  if (owned_path == nullptr || owned_text == nullptr) {
    std::free(owned_path);
    std::free(owned_text);
    return;
  }
  Add({owned_path, DINF_OPTION_STRING, owned_text, 0, 0, 0});
}

void DinfDiagBuilder::AddInt(const std::string& path, int64_t value) {
  char* owned_path = Copy(path);
  if (owned_path == nullptr) {
    return;
  }
  Add({owned_path, DINF_OPTION_INT, nullptr, value, 0, 0});
}

void DinfDiagBuilder::AddBool(const std::string& path, bool value) {
  char* owned_path = Copy(path);
  if (owned_path == nullptr) {
    return;
  }
  Add({owned_path, DINF_OPTION_BOOL, nullptr, 0, 0, value ? 1 : 0});
}

void DinfDiagBuilder::AddDouble(const std::string& path, double value) {
  char* owned_path = Copy(path);
  if (owned_path == nullptr) {
    return;
  }
  Add({owned_path, DINF_OPTION_DOUBLE, nullptr, 0, value, 0});
}

void DinfDiagBuilder::AddMap(const std::string& path) {
  char* owned_path = Copy(path);
  if (owned_path == nullptr) {
    return;
  }
  Add({owned_path, DINF_OPTION_MAP, nullptr, 0, 0, 0});
}

void DinfDiagBuilder::AddList(const std::string& path) {
  char* owned_path = Copy(path);
  if (owned_path == nullptr) {
    return;
  }
  Add({owned_path, DINF_OPTION_LIST, nullptr, 0, 0, 0});
}

void DinfDiagBuilder::AddNull(const std::string& path) {
  char* owned_path = Copy(path);
  if (owned_path == nullptr) {
    return;
  }
  Add({owned_path, DINF_OPTION_NULL, nullptr, 0, 0, 0});
}

void DinfDiagBuilder::AddStringList(
    const std::string& path,
    const std::vector<std::string>& values) {
  AddList(path);
  for (size_t i = 0; i < values.size(); ++i) {
    AddString(dinf_diag_path(path, std::to_string(i)), values[i]);
  }
}

DinfOptionEntry* DinfDiagBuilder::Release(intptr_t* count) {
  if (count != nullptr) {
    *count = 0;
  }
  if (entries_.empty()) {
    return nullptr;
  }
  auto* out = static_cast<DinfOptionEntry*>(
      std::malloc(sizeof(DinfOptionEntry) * entries_.size()));
  if (out == nullptr) {
    return nullptr;
  }
  std::memcpy(out, entries_.data(), sizeof(DinfOptionEntry) * entries_.size());
  if (count != nullptr) {
    *count = static_cast<intptr_t>(entries_.size());
  }
  entries_.clear();
  return out;
}

char* DinfDiagBuilder::Copy(const std::string& value) {
  return dinf_copy_string(value);
}

void DinfDiagBuilder::Add(DinfOptionEntry entry) {
  entries_.push_back(entry);
}

void DinfDiagBuilder::Free() {
  for (const auto& entry : entries_) {
    std::free(const_cast<char*>(entry.path));
    std::free(const_cast<char*>(entry.text));
  }
  entries_.clear();
}

void DinfRuntimeSession::Diagnostics(
    DinfDiagBuilder* out,
    const std::string& prefix) const {
  (void)out;
  (void)prefix;
}

extern "C" DinfRuntimeSession* dinf_cpp_open(
    int32_t engine,
    const char* model_path,
    const DinfOptionEntry* options,
    intptr_t option_count,
    char** error) {
  if (model_path == nullptr) {
    set_error(error, "model_path is null");
    return nullptr;
  }
  const DinfOptions runtime_options{
      options,
      option_count < 0 ? 0 : option_count,
  };
  std::string message;
  DinfRuntimeSession* session = nullptr;
  switch (engine) {
    case DINF_ENGINE_COREML:
      session = dinf_create_coreml_session(
          model_path,
          &runtime_options,
          &message);
      break;
    case DINF_ENGINE_ONNX:
      session = dinf_create_onnx_session(
          model_path,
          &runtime_options,
          &message);
      break;
    case DINF_ENGINE_LITERT:
      session = dinf_create_litert_session(
          model_path,
          &runtime_options,
          &message);
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

extern "C" void dinf_cpp_close(DinfRuntimeSession* session) {
  delete session;
}

extern "C" int32_t dinf_cpp_run(
    DinfRuntimeSession* session,
    const DinfNamedTensor* inputs,
    intptr_t input_count,
    DinfNamedTensor** outputs,
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

extern "C" void dinf_cpp_free_tensors(
    DinfNamedTensor* tensors,
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

extern "C" void dinf_cpp_free_str(char* value) {
  std::free(value);
}

extern "C" DinfOptionEntry* dinf_cpp_diag(
    DinfRuntimeSession* session,
    intptr_t* count) {
  if (count != nullptr) {
    *count = 0;
  }
  if (session == nullptr) {
    return nullptr;
  }
  DinfDiagBuilder builder;
  session->Diagnostics(&builder, "");
  return builder.Release(count);
}

extern "C" void dinf_cpp_free_options(
    DinfOptionEntry* entries,
    intptr_t count) {
  if (entries == nullptr) {
    return;
  }
  for (intptr_t i = 0; i < count; ++i) {
    std::free(const_cast<char*>(entries[i].path));
    std::free(const_cast<char*>(entries[i].text));
  }
  std::free(entries);
}

DinfNamedTensor dinf_make_tensor(
    const char* name,
    int32_t dtype,
    const std::vector<int64_t>& shape,
    const void* data,
    size_t byte_length) {
  DinfNamedTensor tensor{};
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

std::string dinf_diag_path(
    const std::string& parent,
    const std::string& key) {
  if (parent.empty()) {
    return key;
  }
  return parent + '\x1f' + key;
}
