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

enum DinfTensorMemoryKind {
  DINF_TENSOR_MEMORY_CPU = 0,
  DINF_TENSOR_MEMORY_HANDLE = 1,
  DINF_TENSOR_MEMORY_CPU_VIEW = 2,
};

using DinfTensorHandleRelease = void (*)(void* value, void* context);

struct DinfTensorHandle {
  void* value;
  void* context;
  DinfTensorHandleRelease release;
};

struct DinfTensor {
  int32_t dtype;
  int32_t rank;
  int64_t* shape;
  intptr_t byte_length;
  void* data;
  DinfTensorHandle* handle;
  int32_t memory_kind;
  int32_t reserved;
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

struct DinfInfo {
  const char* native_backend;
  const char* runtime_version;
  const char* async_model;
  const char* abi;
  const char* mlx_owner;
  const char* mlx_api;
  int32_t mlx_linked;
  int32_t mlx_enabled;
  const char* mlx_artifacts;
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

struct DinfResolveArtifact {
  int32_t engine;
  const char* path;
  const char* format;
  const char* target_platforms;
};

struct DinfResolveResult {
  int32_t engine;
  int32_t accel_mask;
  int32_t fallback_engine;
};

class DinfDiagBuilder {
 public:
  ~DinfDiagBuilder();

  void AddString(const std::string& path, const std::string& value);
  void AddInt(const std::string& path, int64_t value);
  void AddBool(const std::string& path, bool value);
  void AddDouble(const std::string& path, double value);
  void AddMap(const std::string& path);
  void AddList(const std::string& path);
  void AddNull(const std::string& path);
  void AddStringList(
      const std::string& path,
      const std::vector<std::string>& values);

  DinfOptionEntry* Release(intptr_t* count);

 private:
  char* Copy(const std::string& value);
  void Add(DinfOptionEntry entry);
  void Free();

  std::vector<DinfOptionEntry> entries_;
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

  virtual void Diagnostics(
      DinfDiagBuilder* out,
      const std::string& prefix) const;
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

DINF_RUNTIME_EXPORT DinfOptionEntry* dinf_cpp_diag(
    DinfRuntimeSession* session,
    intptr_t* count);

DINF_RUNTIME_EXPORT void dinf_cpp_free_options(
    DinfOptionEntry* entries,
    intptr_t count);

// Reset the iOS17+/macOS14+ MLState held by a Core ML pipeline session.
//
// `session` must have been created via `dinf_cpp_open` with
// `engine == DINF_ENGINE_COREML`. After this call the next stateful
// stage executed by `dinf_cpp_run` will allocate a fresh MLState. On
// success returns 0; on failure returns non-zero and writes an error
// string into `*error` (release with `dinf_cpp_free_str`). Implemented
// in `coreml_stateful.mm` (Apple) and `coreml_stub.cpp` (others).
DINF_RUNTIME_EXPORT int32_t dinf_coreml_reset_state(
    DinfRuntimeSession* session,
    char** error);
}

DinfNamedTensor dinf_make_tensor(
    const char* name,
    int32_t dtype,
    const std::vector<int64_t>& shape,
    const void* data,
    size_t byte_length);

DinfTensorHandle* dinf_make_tensor_handle(
    void* value,
    void* context,
    DinfTensorHandleRelease release);

void dinf_release_tensor_handle(DinfTensorHandle* handle);

size_t dinf_dtype_size(int32_t dtype);

char* dinf_copy_string(const std::string& value);

std::string dinf_json_escape(const std::string& value);

std::string dinf_json_string_array(const std::vector<std::string>& values);

std::string dinf_diag_path(
    const std::string& parent,
    const std::string& key);

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
