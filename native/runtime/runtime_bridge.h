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

struct DinfTensor {
  int32_t dtype;
  int32_t rank;
  int64_t* shape;
  intptr_t byte_length;
  void* data;
};

struct DinfNamedTensor {
  char* name;
  DinfTensor tensor;
};

struct DinfMemoryInfo {
  const char* native_backend;
  uint64_t peak_memory_bytes;
  uint64_t vm_hwm;
  uint64_t vm_rss;
  uint64_t phys_footprint;
  uint64_t resident_size;
  uint64_t virtual_size;
  uint64_t peak_working_set;
  uint64_t working_set;
  uint64_t android_peak_pss;
  uint64_t android_pss;
  uint64_t android_rss;
  uint64_t android_native_heap_pss;
  uint64_t android_java_heap_pss;
  uint64_t android_native_heap_private_dirty;
  uint64_t android_java_heap_private_dirty;
};

enum DinfOptionKind {
  DINF_OPTION_STRING = 1,
  DINF_OPTION_INT = 2,
  DINF_OPTION_BOOL = 3,
  DINF_OPTION_MAP = 4,
  DINF_OPTION_LIST = 5,
  DINF_OPTION_DOUBLE = 6,
  DINF_OPTION_NULL = 7,
};

struct DinfOptionEntry {
  const char* path;
  int32_t kind;
  const char* text;
  int64_t int_value;
  double double_value;
  int32_t bool_value;
};

struct DinfOptions {
  const DinfOptionEntry* entries;
  intptr_t count;
};

class DinfRuntimeSession {
 public:
  virtual ~DinfRuntimeSession() = default;

  virtual int Run(
      const DinfNamedTensor* inputs,
      size_t input_count,
      DinfNamedTensor** outputs,
      size_t* output_count,
      std::string* error) = 0;

  virtual std::string DiagnosticsJson() const { return "{}"; }
};

extern "C" {

DINF_RUNTIME_EXPORT DinfRuntimeSession* dinf_cpp_open(
    int32_t engine,
    const char* model_path,
    const DinfOptionEntry* options,
    intptr_t option_count,
    char** error);

DINF_RUNTIME_EXPORT void dinf_cpp_close(DinfRuntimeSession* session);

DINF_RUNTIME_EXPORT int32_t dinf_cpp_run(
    DinfRuntimeSession* session,
    const DinfNamedTensor* inputs,
    intptr_t input_count,
    DinfNamedTensor** outputs,
    intptr_t* output_count,
    char** error);

DINF_RUNTIME_EXPORT void dinf_cpp_free_tensors(
    DinfNamedTensor* tensors,
    intptr_t count);

DINF_RUNTIME_EXPORT void dinf_cpp_free_str(char* value);

DINF_RUNTIME_EXPORT void dinf_cpp_mem(DinfMemoryInfo* out);

DINF_RUNTIME_EXPORT char* dinf_cpp_diag_json(
    DinfRuntimeSession* session);
}

DinfNamedTensor dinf_make_tensor(
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
    const DinfOptions* options,
    std::string* error);

DinfRuntimeSession* dinf_create_onnx_session(
    const char* model_path,
    const DinfOptions* options,
    std::string* error);

DinfRuntimeSession* dinf_create_litert_session(
    const char* model_path,
    const DinfOptions* options,
    std::string* error);
