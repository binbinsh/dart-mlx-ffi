#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#if defined(_WIN32)
#define DINF_RUNTIME_EXPORT __declspec(dllexport)
#else
#define DINF_RUNTIME_EXPORT __attribute__((visibility("default")))
#endif

enum DinfRuntimeEngine {
  DINF_ENGINE_MLX = 0,
  DINF_ENGINE_COREML = 1,
  DINF_ENGINE_ONNX = 2,
  DINF_ENGINE_LITERT = 3,
};

enum DinfTensorDType {
  DINF_DTYPE_FLOAT32 = 1,
  DINF_DTYPE_INT32 = 2,
  DINF_DTYPE_INT64 = 3,
  DINF_DTYPE_UINT8 = 4,
  DINF_DTYPE_FLOAT64 = 5,
  DINF_DTYPE_FLOAT16 = 6,
  DINF_DTYPE_BOOL = 7,
};

struct DartInferenceNativeTensor {
  int32_t dtype;
  int32_t rank;
  int64_t* shape;
  intptr_t byte_length;
  void* data;
};

struct DartInferenceNamedTensor {
  char* name;
  DartInferenceNativeTensor tensor;
};

class DinfRuntimeSession {
 public:
  virtual ~DinfRuntimeSession() = default;

  virtual int Run(
      const DartInferenceNamedTensor* inputs,
      size_t input_count,
      DartInferenceNamedTensor** outputs,
      size_t* output_count,
      std::string* error) = 0;

  virtual std::string DiagnosticsJson() const { return "{}"; }
};

extern "C" {

DINF_RUNTIME_EXPORT DinfRuntimeSession* dinf_cpp_runtime_create(
    int32_t engine,
    const char* model_path,
    const char* options_json,
    char** error);

DINF_RUNTIME_EXPORT void dinf_cpp_runtime_free(DinfRuntimeSession* session);

DINF_RUNTIME_EXPORT int32_t dinf_cpp_runtime_run(
    DinfRuntimeSession* session,
    const DartInferenceNamedTensor* inputs,
    intptr_t input_count,
    DartInferenceNamedTensor** outputs,
    intptr_t* output_count,
    char** error);

DINF_RUNTIME_EXPORT void dinf_cpp_runtime_free_tensors(
    DartInferenceNamedTensor* tensors,
    intptr_t count);

DINF_RUNTIME_EXPORT void dinf_cpp_runtime_free_string(char* value);

DINF_RUNTIME_EXPORT char* dinf_cpp_runtime_backend_json();

DINF_RUNTIME_EXPORT char* dinf_cpp_runtime_memory_info_json();

DINF_RUNTIME_EXPORT char* dinf_cpp_runtime_diagnostics_json(
    DinfRuntimeSession* session);
}

DartInferenceNamedTensor dinf_make_tensor(
    const char* name,
    int32_t dtype,
    const std::vector<int64_t>& shape,
    const void* data,
    size_t byte_length);

size_t dinf_dtype_size(int32_t dtype);

char* dinf_copy_string(const std::string& value);

std::string dinf_json_escape(const std::string& value);

std::string dinf_json_string_array(const std::vector<std::string>& values);

DinfRuntimeSession* dinf_create_coreml_session(
    const char* model_path,
    const char* options_json,
    std::string* error);

DinfRuntimeSession* dinf_create_onnx_session(
    const char* model_path,
    const char* options_json,
    std::string* error);

DinfRuntimeSession* dinf_create_litert_session(
    const char* model_path,
    const char* options_json,
    std::string* error);
