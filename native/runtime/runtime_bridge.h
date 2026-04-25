#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum DmfRuntimeEngine {
  DMF_ENGINE_MLX = 0,
  DMF_ENGINE_COREML = 1,
  DMF_ENGINE_ONNX = 2,
  DMF_ENGINE_LITERT = 3,
};

enum DmfTensorDType {
  DMF_DTYPE_FLOAT32 = 1,
  DMF_DTYPE_INT32 = 2,
  DMF_DTYPE_INT64 = 3,
  DMF_DTYPE_UINT8 = 4,
  DMF_DTYPE_FLOAT64 = 5,
  DMF_DTYPE_FLOAT16 = 6,
  DMF_DTYPE_BOOL = 7,
};

struct DmfNativeTensor {
  int32_t dtype;
  int32_t rank;
  int64_t* shape;
  intptr_t byte_length;
  void* data;
};

struct DmfNamedTensor {
  char* name;
  DmfNativeTensor tensor;
};

class DmfRuntimeSession {
 public:
  virtual ~DmfRuntimeSession() = default;

  virtual int Run(
      const DmfNamedTensor* inputs,
      size_t input_count,
      DmfNamedTensor** outputs,
      size_t* output_count,
      std::string* error) = 0;

  virtual std::string DiagnosticsJson() const { return "{}"; }
};

extern "C" {

DmfRuntimeSession* dmf_runtime_create(
    int32_t engine,
    const char* model_path,
    const char* options_json,
    char** error);

void dmf_runtime_free(DmfRuntimeSession* session);

int32_t dmf_runtime_run(
    DmfRuntimeSession* session,
    const DmfNamedTensor* inputs,
    intptr_t input_count,
    DmfNamedTensor** outputs,
    intptr_t* output_count,
    char** error);

void dmf_runtime_free_tensors(DmfNamedTensor* tensors, intptr_t count);

void dmf_runtime_free_string(char* value);

char* dmf_runtime_memory_info_json();

char* dmf_runtime_diagnostics_json(DmfRuntimeSession* session);
}

DmfNamedTensor dmf_make_tensor(
    const char* name,
    int32_t dtype,
    const std::vector<int64_t>& shape,
    const void* data,
    size_t byte_length);

size_t dmf_dtype_size(int32_t dtype);

char* dmf_copy_string(const std::string& value);

std::string dmf_json_escape(const std::string& value);

std::string dmf_json_string_array(const std::vector<std::string>& values);

DmfRuntimeSession* dmf_create_coreml_session(
    const char* model_path,
    const char* options_json,
    std::string* error);

DmfRuntimeSession* dmf_create_onnx_session(
    const char* model_path,
    const char* options_json,
    std::string* error);

DmfRuntimeSession* dmf_create_litert_session(
    const char* model_path,
    const char* options_json,
    std::string* error);
