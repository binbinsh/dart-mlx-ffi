#include "runtime_bridge.h"
#include "options.h"
#include "runtime_pipeline.h"

#include <algorithm>
#include <cstdarg>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#include <limits.h>
#endif

namespace {

struct TfLiteModel;
struct TfLiteInterpreterOptions;
struct TfLiteInterpreter;
struct TfLiteTensor;
struct TfLiteDelegate;

using TfLiteStatus = int;

struct TfLiteNnapiDelegateOptions {
  int execution_preference;
  const char* accelerator_name;
  const char* cache_dir;
  const char* model_token;
  int disallow_nnapi_cpu;
  int allow_fp16;
  int max_number_delegated_partitions;
  void* nnapi_support_library_handle;
};

class SharedLibrary {
 public:
  explicit SharedLibrary(const char* path) : path_(path == nullptr ? "" : path) {
#if defined(_WIN32)
    handle_ = LoadLibraryA(path);
#else
    handle_ = dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
  }

  ~SharedLibrary() {
#if defined(_WIN32)
    if (handle_ != nullptr) {
      FreeLibrary(static_cast<HMODULE>(handle_));
    }
#else
    if (handle_ != nullptr) {
      dlclose(handle_);
    }
#endif
  }

  bool ok() const { return handle_ != nullptr; }

  const std::string& path() const { return path_; }

  template <typename T>
  bool load(const char* name, T* out) {
#if defined(_WIN32)
    *out = reinterpret_cast<T>(GetProcAddress(static_cast<HMODULE>(handle_), name));
#else
    *out = reinterpret_cast<T>(dlsym(handle_, name));
#endif
    return *out != nullptr;
  }

 private:
  std::string path_;
  void* handle_ = nullptr;
};

struct LiteRtApi {
  TfLiteModel* (*ModelCreateFromFile)(const char*);
  void (*ModelDelete)(TfLiteModel*);
  TfLiteInterpreterOptions* (*OptionsCreate)();
  void (*OptionsDelete)(TfLiteInterpreterOptions*);
  void (*OptionsSetNumThreads)(TfLiteInterpreterOptions*, int32_t);
  void (*OptionsSetErrorReporter)(
      TfLiteInterpreterOptions*,
      void (*)(void*, const char*, va_list),
      void*);
  void (*OptionsAddDelegate)(TfLiteInterpreterOptions*, TfLiteDelegate*);
  TfLiteInterpreter* (*InterpreterCreate)(
      const TfLiteModel*,
      const TfLiteInterpreterOptions*);
  void (*InterpreterDelete)(TfLiteInterpreter*);
  TfLiteStatus (*AllocateTensors)(TfLiteInterpreter*);
  TfLiteStatus (*Invoke)(TfLiteInterpreter*);
  int32_t (*InputTensorCount)(const TfLiteInterpreter*);
  int32_t (*OutputTensorCount)(const TfLiteInterpreter*);
  TfLiteTensor* (*InputTensor)(TfLiteInterpreter*, int32_t);
  const TfLiteTensor* (*OutputTensor)(const TfLiteInterpreter*, int32_t);
  TfLiteStatus (*ResizeInputTensor)(
      TfLiteInterpreter*,
      int32_t,
      const int*,
      int32_t);
  TfLiteStatus (*TensorCopyFromBuffer)(TfLiteTensor*, const void*, size_t);
  TfLiteStatus (*TensorCopyToBuffer)(const TfLiteTensor*, void*, size_t);
  size_t (*TensorByteSize)(const TfLiteTensor*);
  int32_t (*TensorType)(const TfLiteTensor*);
  int32_t (*TensorNumDims)(const TfLiteTensor*);
  int32_t (*TensorDim)(const TfLiteTensor*, int32_t);
  const char* (*TensorName)(const TfLiteTensor*);
  TfLiteDelegate* (*XNNPackDelegateCreate)(const void*);
  void (*XNNPackDelegateDelete)(TfLiteDelegate*);
  TfLiteDelegate* (*GpuDelegateCreate)(const void*);
  void (*GpuDelegateDelete)(TfLiteDelegate*);
  TfLiteDelegate* (*NnApiDelegateCreate)(const TfLiteNnapiDelegateOptions*);
  TfLiteNnapiDelegateOptions (*NnApiDelegateOptionsDefault)();
  void (*NnApiDelegateDelete)(TfLiteDelegate*);
};

const char* env(const char* name) {
  const char* value = std::getenv(name);
  return value == nullptr || value[0] == '\0' ? nullptr : value;
}

void append_unique(std::vector<std::string>* values, const std::string& value) {
  if (value.empty()) {
    return;
  }
  if (std::find(values->begin(), values->end(), value) != values->end()) {
    return;
  }
  values->push_back(value);
}

std::vector<std::string> parse_library_list(const std::string& raw) {
  std::vector<std::string> values;
  std::string token;
  const auto flush = [&]() {
    if (token.empty()) {
      return;
    }
    const auto start = token.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
      token.clear();
      return;
    }
    const auto end = token.find_last_not_of(" \t\r\n");
    append_unique(&values, token.substr(start, end - start + 1));
    token.clear();
  };
  for (const char c : raw) {
    const bool separator =
#if defined(_WIN32)
        c == ';' || c == ',' || c == '\n' || c == '\r';
#else
        c == ':' || c == ';' || c == ',' || c == '\n' || c == '\r';
#endif
    if (separator) {
      flush();
      continue;
    }
    token.push_back(c);
  }
  flush();
  return values;
}

std::filesystem::path runtime_library_dir() {
#if defined(_WIN32)
  HMODULE module = nullptr;
  char path[MAX_PATH] = {};
  if (GetModuleHandleExA(
          GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
          reinterpret_cast<LPCSTR>(&runtime_library_dir),
          &module) &&
      GetModuleFileNameA(module, path, MAX_PATH) > 0) {
    return std::filesystem::path(path).parent_path();
  }
#else
  Dl_info info;
  if (dladdr(reinterpret_cast<void*>(&runtime_library_dir), &info) != 0 &&
      info.dli_fname != nullptr) {
    return std::filesystem::path(info.dli_fname).parent_path();
  }
#endif
  return {};
}

std::vector<std::string> library_candidates() {
  std::vector<std::string> candidates;
  if (const char* value = env("DART_INFERENCE_LITERT_LIBRARY")) {
    append_unique(&candidates, value);
  }
  if (const char* value = env("DART_INFERENCE_TFLITE_LIBRARY")) {
    append_unique(&candidates, value);
  }
  const auto adjacent = runtime_library_dir();
  if (!adjacent.empty()) {
#if defined(_WIN32)
    append_unique(&candidates, (adjacent / "tensorflowlite_c.dll").string());
    append_unique(&candidates, (adjacent / "tensorflowlite.dll").string());
#elif defined(__APPLE__)
    append_unique(&candidates, (adjacent / "libtensorflowlite_c.dylib").string());
    append_unique(&candidates, (adjacent / "libtensorflowlite.dylib").string());
#else
    append_unique(&candidates, (adjacent / "libtensorflowlite_c.so").string());
    append_unique(&candidates, (adjacent / "libtensorflowlite.so").string());
    append_unique(&candidates, (adjacent / "libtensorflowlite_jni.so").string());
#endif
  }
#if defined(_WIN32)
  append_unique(&candidates, "tensorflowlite_c.dll");
#elif defined(__APPLE__)
  append_unique(&candidates, "libtensorflowlite_c.dylib");
#else
  append_unique(&candidates, "libtensorflowlite_c.so");
  append_unique(&candidates, "libtensorflowlite_jni.so");
#endif
  return candidates;
}

std::vector<std::string> support_library_candidates(const DinfOptions* runtime_options) {
  std::vector<std::string> candidates;
  if (const char* raw = env("DART_INFERENCE_LITERT_EXTRA_LIBRARIES")) {
    for (const auto& value : parse_library_list(raw)) {
      append_unique(&candidates, value);
    }
  }
  if (const char* value = env("DART_INFERENCE_TFLITE_FLEX_LIBRARY")) {
    append_unique(&candidates, value);
  }
  for (const auto key : {"litertExtraLibraries", "extraLibraries"}) {
    const std::string values = dinf_option_string(runtime_options, key);
    for (const auto& value : parse_library_list(values)) {
      append_unique(&candidates, value);
    }
  }
  const auto adjacent = runtime_library_dir();
  if (!adjacent.empty()) {
#if defined(_WIN32)
    append_unique(&candidates, (adjacent / "tensorflowlite_flex.dll").string());
#elif defined(__APPLE__)
    append_unique(&candidates, (adjacent / "libtensorflowlite_flex.dylib").string());
#else
    append_unique(&candidates, (adjacent / "libtensorflowlite_flex_jni.so").string());
    append_unique(&candidates, (adjacent / "libtensorflowlite_flex.so").string());
#endif
  }
#if defined(_WIN32)
  append_unique(&candidates, "tensorflowlite_flex.dll");
#elif defined(__APPLE__)
  append_unique(&candidates, "libtensorflowlite_flex.dylib");
#else
  append_unique(&candidates, "libtensorflowlite_flex_jni.so");
  append_unique(&candidates, "libtensorflowlite_flex.so");
#endif
  return candidates;
}

struct SupportLibraryLoadResult {
  std::vector<std::unique_ptr<SharedLibrary>> libraries;
  std::vector<std::string> loaded;
  std::vector<std::string> attempted;
};

SupportLibraryLoadResult preload_support_libraries(const DinfOptions* runtime_options) {
  SupportLibraryLoadResult result;
  for (const auto& candidate : support_library_candidates(runtime_options)) {
    append_unique(&result.attempted, candidate);
    auto library = std::make_unique<SharedLibrary>(candidate.c_str());
    if (!library->ok()) {
      continue;
    }
    append_unique(&result.loaded, candidate);
    result.libraries.push_back(std::move(library));
  }
  return result;
}

bool load_api(SharedLibrary& library, LiteRtApi* api, std::string* error) {
#define LOAD(name, field) \
  if (!library.load(name, &api->field)) { \
    *error = std::string("Missing LiteRT/TFLite symbol: ") + name; \
    return false; \
  }
  LOAD("TfLiteModelCreateFromFile", ModelCreateFromFile)
  LOAD("TfLiteModelDelete", ModelDelete)
  LOAD("TfLiteInterpreterOptionsCreate", OptionsCreate)
  LOAD("TfLiteInterpreterOptionsDelete", OptionsDelete)
  LOAD("TfLiteInterpreterOptionsSetNumThreads", OptionsSetNumThreads)
  LOAD("TfLiteInterpreterOptionsAddDelegate", OptionsAddDelegate)
  LOAD("TfLiteInterpreterCreate", InterpreterCreate)
  LOAD("TfLiteInterpreterDelete", InterpreterDelete)
  LOAD("TfLiteInterpreterAllocateTensors", AllocateTensors)
  LOAD("TfLiteInterpreterInvoke", Invoke)
  LOAD("TfLiteInterpreterGetInputTensorCount", InputTensorCount)
  LOAD("TfLiteInterpreterGetOutputTensorCount", OutputTensorCount)
  LOAD("TfLiteInterpreterGetInputTensor", InputTensor)
  LOAD("TfLiteInterpreterGetOutputTensor", OutputTensor)
  LOAD("TfLiteInterpreterResizeInputTensor", ResizeInputTensor)
  LOAD("TfLiteTensorCopyFromBuffer", TensorCopyFromBuffer)
  LOAD("TfLiteTensorCopyToBuffer", TensorCopyToBuffer)
  LOAD("TfLiteTensorByteSize", TensorByteSize)
  LOAD("TfLiteTensorType", TensorType)
  LOAD("TfLiteTensorNumDims", TensorNumDims)
  LOAD("TfLiteTensorDim", TensorDim)
  LOAD("TfLiteTensorName", TensorName)
#undef LOAD
  library.load("TfLiteXNNPackDelegateCreate", &api->XNNPackDelegateCreate);
  library.load("TfLiteXNNPackDelegateDelete", &api->XNNPackDelegateDelete);
  library.load("TfLiteGpuDelegateV2Create", &api->GpuDelegateCreate);
  library.load("TfLiteGpuDelegateV2Delete", &api->GpuDelegateDelete);
  if (api->GpuDelegateCreate == nullptr) {
    library.load("TfLiteGpuDelegateCreate", &api->GpuDelegateCreate);
  }
  if (api->GpuDelegateDelete == nullptr) {
    library.load("TfLiteGpuDelegateDelete", &api->GpuDelegateDelete);
  }
  library.load("TfLiteNnApiDelegateCreate", &api->NnApiDelegateCreate);
  library.load("TfLiteNnApiDelegateDelete", &api->NnApiDelegateDelete);
  if (api->NnApiDelegateCreate == nullptr) {
    library.load("TfLiteNnapiDelegateCreate", &api->NnApiDelegateCreate);
  }
  if (api->NnApiDelegateDelete == nullptr) {
    library.load("TfLiteNnapiDelegateDelete", &api->NnApiDelegateDelete);
  }
  library.load(
      "TfLiteNnapiDelegateOptionsDefault",
      &api->NnApiDelegateOptionsDefault);
  library.load(
      "TfLiteInterpreterOptionsSetErrorReporter",
      &api->OptionsSetErrorReporter);
  return true;
}

std::string lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

std::string requested_delegate(const DinfOptions* runtime_options) {
  return lower(dinf_option_string(runtime_options, "delegate"));
}

bool xnnpack_enabled(const DinfOptions* runtime_options) {
  if (!dinf_option_bool(runtime_options, "enableXNNPack", true) ||
      !dinf_option_bool(runtime_options, "enableXnnpack", true) ||
      !dinf_option_bool(runtime_options, "useXNNPACK", true) ||
      !dinf_option_bool(runtime_options, "useXnnpack", true)) {
    return false;
  }
  return requested_delegate(runtime_options) != "none";
}

bool wants_xnnpack(const DinfOptions* runtime_options) {
  if (!xnnpack_enabled(runtime_options)) {
    return false;
  }
  const std::string delegate = requested_delegate(runtime_options);
  return delegate.empty() || delegate == "xnnpack" || delegate == "gpu" ||
         delegate == "nnapi" || delegate == "npu" || delegate == "qnn" ||
         dinf_options_contains_token(runtime_options, "cpu");
}

struct DelegateHandle {
  TfLiteDelegate* delegate = nullptr;
  void (*deleter)(TfLiteDelegate*) = nullptr;
};

void release_delegate_handles(std::vector<DelegateHandle>* delegates) {
  for (auto& handle : *delegates) {
    if (handle.delegate != nullptr && handle.deleter != nullptr) {
      handle.deleter(handle.delegate);
    }
  }
  delegates->clear();
}

struct LiteRtErrorCollector {
  std::string message;
};

void litert_error_reporter(void* user_data, const char* format, va_list args) {
  if (user_data == nullptr || format == nullptr) {
    return;
  }
  char buffer[1024];
  const int written = std::vsnprintf(buffer, sizeof(buffer), format, args);
  if (written <= 0) {
    return;
  }
  auto* collector = static_cast<LiteRtErrorCollector*>(user_data);
  if (!collector->message.empty()) {
    collector->message.push_back(' ');
  }
  const size_t length = static_cast<size_t>(written);
  collector->message.append(buffer, std::min(length, sizeof(buffer) - 1));
}

void configure_error_reporter(
    const LiteRtApi& api,
    TfLiteInterpreterOptions* options,
    LiteRtErrorCollector* collector) {
  if (api.OptionsSetErrorReporter == nullptr ||
      options == nullptr ||
      collector == nullptr) {
    return;
  }
  api.OptionsSetErrorReporter(options, litert_error_reporter, collector);
}

std::string compact_error_text(std::string value) {
  if (value.empty()) {
    return value;
  }
  for (char& c : value) {
    if (c == '\n' || c == '\r' || c == '\t') {
      c = ' ';
    }
  }
  std::string compact;
  compact.reserve(value.size());
  bool previous_space = false;
  for (const char c : value) {
    const bool is_space = std::isspace(static_cast<unsigned char>(c));
    if (is_space) {
      if (!previous_space) {
        compact.push_back(' ');
      }
      previous_space = true;
      continue;
    }
    compact.push_back(c);
    previous_space = false;
  }
  if (compact.size() > 600) {
    compact.resize(600);
    compact += "...";
  }
  return compact;
}

struct TfliteSection {
  size_t offset = 0;
  size_t size = 0;
};

struct LiteRtArtifact {
  std::string source_path;
  std::string model_path;
  std::string temp_path;
  std::string source_format = "tflite";
  int section_count = 0;
  int selected_section = -1;
};

std::vector<uint8_t> read_file(const char* path, std::string* error) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    *error = std::string("Unable to open LiteRT artifact: ") + path;
    return {};
  }
  input.seekg(0, std::ios::end);
  const std::streamoff size = input.tellg();
  input.seekg(0, std::ios::beg);
  if (size < 0) {
    *error = std::string("Unable to stat LiteRT artifact: ") + path;
    return {};
  }
  std::vector<uint8_t> data(static_cast<size_t>(size));
  if (!data.empty()) {
    input.read(reinterpret_cast<char*>(data.data()), size);
  }
  if (!input && !data.empty()) {
    *error = std::string("Unable to read LiteRT artifact: ") + path;
    return {};
  }
  return data;
}

bool is_tflite_buffer(const std::vector<uint8_t>& data, size_t offset) {
  return offset + 8 <= data.size() &&
         data[offset + 4] == 'T' &&
         data[offset + 5] == 'F' &&
         data[offset + 6] == 'L' &&
         data[offset + 7] == '3';
}

std::vector<TfliteSection> find_tflite_sections(
    const std::vector<uint8_t>& data) {
  std::vector<TfliteSection> sections;
  for (size_t offset = 0; offset + 8 <= data.size(); offset += 4) {
    if (!is_tflite_buffer(data, offset)) {
      continue;
    }
    size_t end = data.size();
    for (size_t next = offset + 4; next + 8 <= data.size(); next += 4) {
      if (is_tflite_buffer(data, next)) {
        end = next;
        break;
      }
    }
    sections.push_back({offset, end - offset});
    offset = end > 4 ? end - 4 : end;
  }
  return sections;
}

std::string extension_of(const char* path) {
  return lower(std::filesystem::path(path).extension().string());
}

bool write_section_file(
    const std::vector<uint8_t>& data,
    TfliteSection section,
    const char* source_path,
    int index,
    std::string* out_path,
    std::string* error) {
  const auto stamp = std::chrono::steady_clock::now()
                         .time_since_epoch()
                         .count();
  const auto name = std::string("dart_inference_litert_") +
                    std::to_string(std::hash<std::string>{}(source_path)) +
                    "_" + std::to_string(index) + "_" +
                    std::to_string(stamp) + ".tflite";
  const auto path = std::filesystem::temp_directory_path() / name;
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    *error = "Unable to create temporary extracted TFLite file";
    return false;
  }
  output.write(
      reinterpret_cast<const char*>(data.data() + section.offset),
      static_cast<std::streamsize>(section.size));
  if (!output) {
    *error = "Unable to write temporary extracted TFLite file";
    return false;
  }
  *out_path = path.string();
  return true;
}

bool prepare_litert_artifact(
    const char* model_path,
    const DinfOptions* runtime_options,
    LiteRtArtifact* artifact,
    std::string* error) {
  artifact->source_path = model_path;
  artifact->model_path = model_path;
  const std::string extension = extension_of(model_path);
  if (extension != ".task" && extension != ".litertlm") {
    artifact->source_format = "tflite";
    return true;
  }

  artifact->source_format = extension == ".task" ? "task" : "litertlm";
  const std::vector<uint8_t> data = read_file(model_path, error);
  if (!error->empty()) {
    return false;
  }
  const std::vector<TfliteSection> sections = find_tflite_sections(data);
  artifact->section_count = static_cast<int>(sections.size());
  if (sections.empty()) {
    *error = "LiteRT container does not contain raw TFLite FlatBuffer "
             "sections. Use a LiteRT-LM or MediaPipe task runtime for this "
             "artifact, or provide a directly loadable .tflite file.";
    return false;
  }

  const int requested =
      dinf_option_int(runtime_options, "litertSectionIndex", -1);
  int selected = requested;
  if (selected < 0 && sections.size() == 1) {
    selected = 0;
  }
  if (selected < 0) {
    *error = "LiteRT container has multiple TFLite sections (" +
             std::to_string(sections.size()) +
             "). Set backendOptions.litertSectionIndex to select one, or "
             "use a full LiteRT-LM runtime for multi-section LLM execution.";
    return false;
  }
  if (selected >= static_cast<int>(sections.size())) {
    *error = "litertSectionIndex is out of range for LiteRT container";
    return false;
  }
  artifact->selected_section = selected;
  if (!write_section_file(
          data,
          sections[selected],
          model_path,
          selected,
          &artifact->temp_path,
          error)) {
    return false;
  }
  artifact->model_path = artifact->temp_path;
  return true;
}

bool add_xnnpack_delegate(
    const LiteRtApi& api,
    TfLiteInterpreterOptions* options,
    std::vector<DelegateHandle>* delegates,
    std::string* error,
    bool required) {
  if (api.XNNPackDelegateCreate == nullptr ||
      api.XNNPackDelegateDelete == nullptr) {
    if (required) {
      *error = "LiteRT XNNPACK delegate symbols are not available";
      return false;
    }
    return true;
  }
  TfLiteDelegate* delegate = api.XNNPackDelegateCreate(nullptr);
  if (delegate == nullptr) {
    if (required) {
      *error = "TfLiteXNNPackDelegateCreate failed";
      return false;
    }
    return true;
  }
  api.OptionsAddDelegate(options, delegate);
  delegates->push_back({delegate, api.XNNPackDelegateDelete});
  return true;
}

bool add_symbol_delegate(
    TfLiteDelegate* (*create)(const void*),
    void (*destroy)(TfLiteDelegate*),
    const char* name,
    TfLiteInterpreterOptions* options,
    const LiteRtApi& api,
    std::vector<DelegateHandle>* delegates,
    std::string* error,
    bool required) {
  if (create == nullptr || destroy == nullptr) {
    if (required) {
      *error = std::string("LiteRT delegate symbols are not available: ") +
               name;
      return false;
    }
    return true;
  }
  TfLiteDelegate* delegate = create(nullptr);
  if (delegate == nullptr) {
    if (required) {
      *error = std::string("LiteRT delegate create failed: ") + name;
      return false;
    }
    return true;
  }
  api.OptionsAddDelegate(options, delegate);
  delegates->push_back({delegate, destroy});
  return true;
}

bool add_nnapi_delegate(
    const LiteRtApi& api,
    TfLiteInterpreterOptions* options,
    const DinfOptions* runtime_options,
    std::vector<DelegateHandle>* delegates,
    std::string* error,
    bool required) {
  if (api.NnApiDelegateCreate == nullptr ||
      api.NnApiDelegateOptionsDefault == nullptr ||
      api.NnApiDelegateDelete == nullptr) {
    if (required) {
      *error = "LiteRT delegate symbols are not available: nnapi";
      return false;
    }
    return true;
  }
  TfLiteNnapiDelegateOptions nnapi_options =
      api.NnApiDelegateOptionsDefault();
  nnapi_options.disallow_nnapi_cpu =
      dinf_option_bool(runtime_options, "nnapiDisallowCpu", false) ? 1 : 0;
  nnapi_options.allow_fp16 =
      dinf_option_bool(runtime_options, "nnapiAllowFp16", false) ? 1 : 0;
  const int max_partitions =
      dinf_option_int(runtime_options, "nnapiMaxDelegatedPartitions", -1);
  if (max_partitions > 0) {
    nnapi_options.max_number_delegated_partitions = max_partitions;
  }
  const int preference = dinf_option_int(runtime_options, "nnapiPreference", -2);
  if (preference >= -1 && preference <= 2) {
    nnapi_options.execution_preference = preference;
  }
  TfLiteDelegate* delegate = api.NnApiDelegateCreate(&nnapi_options);
  if (delegate == nullptr) {
    if (required) {
      *error = "LiteRT delegate create failed: nnapi";
      return false;
    }
    return true;
  }
  api.OptionsAddDelegate(options, delegate);
  delegates->push_back({delegate, api.NnApiDelegateDelete});
  return true;
}

bool resolve_input_indices(
    const LiteRtApi& api,
    TfLiteInterpreter* interpreter,
    const DinfNamedTensor* inputs,
    size_t input_count,
    std::vector<int32_t>* indices,
    std::string* error) {
  const int32_t expected_inputs = api.InputTensorCount(interpreter);
  if (input_count != static_cast<size_t>(expected_inputs)) {
    *error = "LiteRT input count does not match model signature";
    return false;
  }
  indices->assign(input_count, -1);
  std::vector<bool> used(expected_inputs, false);
  for (size_t input_index = 0; input_index < input_count; ++input_index) {
    if (inputs[input_index].name == nullptr || inputs[input_index].name[0] == '\0') {
      continue;
    }
    for (int32_t model_index = 0; model_index < expected_inputs; ++model_index) {
      if (used[model_index]) {
        continue;
      }
      TfLiteTensor* tensor = api.InputTensor(interpreter, model_index);
      const char* tensor_name = tensor == nullptr ? nullptr : api.TensorName(tensor);
      if (tensor_name != nullptr &&
          std::strcmp(tensor_name, inputs[input_index].name) == 0) {
        (*indices)[input_index] = model_index;
        used[model_index] = true;
        break;
      }
    }
  }
  int32_t next = 0;
  for (size_t input_index = 0; input_index < input_count; ++input_index) {
    if ((*indices)[input_index] >= 0) {
      continue;
    }
    while (next < expected_inputs && used[next]) {
      ++next;
    }
    if (next >= expected_inputs) {
      *error = "LiteRT input index resolution failed";
      return false;
    }
    (*indices)[input_index] = next;
    used[next] = true;
  }
  return true;
}

int32_t litert_dtype_to_dmf(int32_t type) {
  switch (type) {
    case 1:
      return DINF_DTYPE_FLOAT32;
    case 2:
      return DINF_DTYPE_INT32;
    case 3:
      return DINF_DTYPE_UINT8;
    case 4:
      return DINF_DTYPE_INT64;
    case 6:
      return DINF_DTYPE_BOOL;
    case 10:
      return DINF_DTYPE_FLOAT16;
    case 11:
      return DINF_DTYPE_FLOAT64;
    default:
      return 0;
  }
}

int32_t dinf_dtype_to_litert(int32_t dtype) {
  switch (dtype) {
    case DINF_DTYPE_FLOAT32:
      return 1;
    case DINF_DTYPE_INT32:
      return 2;
    case DINF_DTYPE_UINT8:
      return 3;
    case DINF_DTYPE_INT64:
      return 4;
    case DINF_DTYPE_BOOL:
      return 6;
    case DINF_DTYPE_FLOAT16:
      return 10;
    case DINF_DTYPE_FLOAT64:
      return 11;
    default:
      return -1;
  }
}

class LiteRtSession final : public DinfRuntimeSession {
 public:
  LiteRtSession(
      std::unique_ptr<SharedLibrary> library,
      LiteRtApi api,
      TfLiteModel* model,
      TfLiteInterpreterOptions* options,
      TfLiteInterpreter* interpreter,
      std::vector<DelegateHandle> delegates,
      LiteRtArtifact artifact,
      int num_threads,
      std::string requested_delegate,
      std::vector<std::string> delegate_names,
      std::vector<std::unique_ptr<SharedLibrary>> support_libraries,
      std::vector<std::string> support_library_attempts,
      std::vector<std::string> support_library_loaded,
      bool delegate_fallback_attempted)
      : library_(std::move(library)),
        api_(api),
        model_(model),
        options_(options),
        interpreter_(interpreter),
        delegates_(std::move(delegates)),
        artifact_(std::move(artifact)),
        num_threads_(num_threads),
        requested_delegate_(std::move(requested_delegate)),
        delegate_names_(std::move(delegate_names)),
        support_libraries_(std::move(support_libraries)),
        support_library_attempts_(std::move(support_library_attempts)),
        support_library_loaded_(std::move(support_library_loaded)),
        delegate_fallback_attempted_(delegate_fallback_attempted) {}

  ~LiteRtSession() override {
    api_.InterpreterDelete(interpreter_);
    for (auto& handle : delegates_) {
      if (handle.delegate != nullptr && handle.deleter != nullptr) {
        handle.deleter(handle.delegate);
      }
    }
    api_.OptionsDelete(options_);
    api_.ModelDelete(model_);
    if (!artifact_.temp_path.empty()) {
      std::remove(artifact_.temp_path.c_str());
    }
  }

  int Run(
      const DinfNamedTensor* inputs,
      size_t input_count,
      DinfNamedTensor** outputs,
      size_t* output_count,
      std::string* error) override {
    std::vector<int32_t> input_indices;
    if (!resolve_input_indices(
            api_, interpreter_, inputs, input_count, &input_indices, error)) {
      return 1;
    }
    for (size_t i = 0; i < input_count; ++i) {
      std::vector<int> dims;
      for (int32_t axis = 0; axis < inputs[i].tensor.rank; ++axis) {
        dims.push_back(static_cast<int>(inputs[i].tensor.shape[axis]));
      }
      if (api_.ResizeInputTensor(
              interpreter_, input_indices[i], dims.data(), dims.size()) != 0) {
        *error = "TfLiteInterpreterResizeInputTensor failed";
        return 1;
      }
    }
    if (api_.AllocateTensors(interpreter_) != 0) {
      *error = "TfLiteInterpreterAllocateTensors failed";
      return 1;
    }
    for (size_t i = 0; i < input_count; ++i) {
      TfLiteTensor* tensor = api_.InputTensor(interpreter_, input_indices[i]);
      if (tensor == nullptr) {
        *error = "LiteRT input tensor is null";
        return 1;
      }
      if (api_.TensorType(tensor) != dinf_dtype_to_litert(inputs[i].tensor.dtype)) {
        *error = "LiteRT input dtype does not match model signature";
        return 1;
      }
      if (api_.TensorByteSize(tensor) !=
          static_cast<size_t>(inputs[i].tensor.byte_length)) {
        *error = "LiteRT input byte length does not match model signature";
        return 1;
      }
      if (api_.TensorCopyFromBuffer(
              tensor, inputs[i].tensor.data, inputs[i].tensor.byte_length) != 0) {
        *error = "TfLiteTensorCopyFromBuffer failed";
        return 1;
      }
    }
    if (api_.Invoke(interpreter_) != 0) {
      *error = "TfLiteInterpreterInvoke failed";
      return 1;
    }
    const int32_t count = api_.OutputTensorCount(interpreter_);
    std::vector<DinfNamedTensor> produced;
    for (int32_t i = 0; i < count; ++i) {
      const TfLiteTensor* tensor = api_.OutputTensor(interpreter_, i);
      if (tensor == nullptr) {
        continue;
      }
      const int32_t dtype = litert_dtype_to_dmf(api_.TensorType(tensor));
      if (dtype == 0) {
        continue;
      }
      const size_t byte_length = api_.TensorByteSize(tensor);
      std::vector<uint8_t> data(byte_length);
      if (api_.TensorCopyToBuffer(tensor, data.data(), data.size()) != 0) {
        *error = "TfLiteTensorCopyToBuffer failed";
        return 1;
      }
      std::vector<int64_t> shape;
      const int32_t rank = api_.TensorNumDims(tensor);
      for (int32_t axis = 0; axis < rank; ++axis) {
        shape.push_back(api_.TensorDim(tensor, axis));
      }
      const char* name = api_.TensorName(tensor);
      produced.push_back(dinf_make_tensor(
          name == nullptr ? "" : name,
          dtype,
          shape,
          data.data(),
          data.size()));
    }
    *output_count = produced.size();
    *outputs = static_cast<DinfNamedTensor*>(
        std::malloc(sizeof(DinfNamedTensor) * produced.size()));
    if (!produced.empty()) {
      std::memcpy(*outputs, produced.data(), sizeof(DinfNamedTensor) * produced.size());
    }
    return 0;
  }

  void Diagnostics(
      DinfDiagBuilder* out,
      const std::string& prefix) const override {
    out->AddString(dinf_diag_path(prefix, "engine"), "litert");
    out->AddString(dinf_diag_path(prefix, "library"), library_->path());
    out->AddInt(dinf_diag_path(prefix, "num_threads"), num_threads_);
    out->AddString(
        dinf_diag_path(prefix, "requested_delegate"),
        requested_delegate_);
    out->AddStringList(dinf_diag_path(prefix, "delegates"), delegate_names_);
    out->AddStringList(
        dinf_diag_path(prefix, "support_libraries_loaded"),
        support_library_loaded_);
    out->AddStringList(
        dinf_diag_path(prefix, "support_libraries_attempted"),
        support_library_attempts_);
    out->AddStringList(
        dinf_diag_path(prefix, "input_names"),
        tensor_name_values(/*inputs=*/true));
    out->AddStringList(
        dinf_diag_path(prefix, "output_names"),
        tensor_name_values(/*inputs=*/false));
    out->AddString(
        dinf_diag_path(prefix, "artifact_path"),
        artifact_.source_path);
    out->AddString(
        dinf_diag_path(prefix, "resolved_artifact_path"),
        artifact_.model_path);
    out->AddString(
        dinf_diag_path(prefix, "artifact_format"),
        artifact_.source_format);
    out->AddInt(
        dinf_diag_path(prefix, "tflite_section_count"),
        artifact_.section_count);
    out->AddInt(
        dinf_diag_path(prefix, "selected_tflite_section"),
        artifact_.selected_section);
    out->AddBool(
        dinf_diag_path(prefix, "delegate_fallback_attempted"),
        delegate_fallback_attempted_);
  }

 private:
  std::vector<std::string> tensor_name_values(bool inputs) const {
    const int32_t count = inputs
        ? api_.InputTensorCount(interpreter_)
        : api_.OutputTensorCount(interpreter_);
    std::vector<std::string> names;
    names.reserve(count < 0 ? 0 : static_cast<size_t>(count));
    for (int32_t i = 0; i < count; ++i) {
      const TfLiteTensor* tensor = inputs
          ? api_.InputTensor(interpreter_, i)
          : api_.OutputTensor(interpreter_, i);
      const char* name = tensor == nullptr ? nullptr : api_.TensorName(tensor);
      names.emplace_back(name == nullptr ? "" : name);
    }
    return names;
  }

  std::unique_ptr<SharedLibrary> library_;
  LiteRtApi api_{};
  TfLiteModel* model_ = nullptr;
  TfLiteInterpreterOptions* options_ = nullptr;
  TfLiteInterpreter* interpreter_ = nullptr;
  std::vector<DelegateHandle> delegates_;
  LiteRtArtifact artifact_;
  int num_threads_;
  std::string requested_delegate_;
  std::vector<std::string> delegate_names_;
  std::vector<std::unique_ptr<SharedLibrary>> support_libraries_;
  std::vector<std::string> support_library_attempts_;
  std::vector<std::string> support_library_loaded_;
  bool delegate_fallback_attempted_;
};

}  // namespace

DinfRuntimeSession* dinf_create_litert_session(
    const char* model_path,
    const DinfOptions* runtime_options,
    std::string* error) {
  if (auto* pipeline = dinf_try_create_pipeline_session(
          model_path,
          runtime_options,
          "litert",
          "dart_mlx_ffi.litert_pipeline.v1",
          dinf_create_litert_session,
          error)) {
    return pipeline;
  }
  if (!error->empty()) {
    return nullptr;
  }
  std::vector<std::string> attempted;
  for (const auto& candidate : library_candidates()) {
    attempted.push_back(candidate);
    auto library = std::make_unique<SharedLibrary>(candidate.c_str());
    if (!library->ok()) {
      continue;
    }
    LiteRtApi api{};
    if (!load_api(*library, &api, error)) {
      return nullptr;
    }
    SupportLibraryLoadResult support_libraries =
        preload_support_libraries(runtime_options);
    LiteRtArtifact artifact;
    if (!prepare_litert_artifact(model_path, runtime_options, &artifact, error)) {
      return nullptr;
    }
    TfLiteModel* model = api.ModelCreateFromFile(artifact.model_path.c_str());
    if (model == nullptr) {
      *error = "TfLiteModelCreateFromFile failed for " + artifact.model_path;
      if (!artifact.temp_path.empty()) {
        std::remove(artifact.temp_path.c_str());
      }
      return nullptr;
    }
    TfLiteInterpreterOptions* options = api.OptionsCreate();
    LiteRtErrorCollector error_collector;
    configure_error_reporter(api, options, &error_collector);
    const int num_threads = std::max(1, dinf_option_int(runtime_options, "numThreads", 1));
    api.OptionsSetNumThreads(options, num_threads);
    std::vector<DelegateHandle> delegates;
    std::vector<std::string> delegate_names;
    const bool require_delegate =
        dinf_option_bool(runtime_options, "requireDelegate", false);
    const std::string delegate = requested_delegate(runtime_options);
    if (delegate == "gpu") {
      const auto before = delegates.size();
      if (!add_symbol_delegate(
              api.GpuDelegateCreate,
              api.GpuDelegateDelete,
              "gpu",
              options,
              api,
              &delegates,
              error,
              require_delegate)) {
        api.OptionsDelete(options);
        api.ModelDelete(model);
        return nullptr;
      }
      if (delegates.size() > before) {
        delegate_names.push_back("gpu");
      }
    }
    if (delegate == "nnapi" || delegate == "npu") {
      const auto before = delegates.size();
      if (!add_nnapi_delegate(
              api, options, runtime_options, &delegates, error, require_delegate)) {
        api.OptionsDelete(options);
        api.ModelDelete(model);
        return nullptr;
      }
      if (delegates.size() > before) {
        delegate_names.push_back("nnapi");
      }
    }
    if (delegate == "qnn" && require_delegate) {
      api.OptionsDelete(options);
      api.ModelDelete(model);
      *error = "LiteRT QNN delegate is not available through the bundled C API";
      return nullptr;
    }
    if (delegate_names.empty() && wants_xnnpack(runtime_options)) {
      const auto before = delegates.size();
      if (!add_xnnpack_delegate(
              api, options, &delegates, error, require_delegate)) {
        api.OptionsDelete(options);
        api.ModelDelete(model);
        return nullptr;
      }
      if (delegates.size() > before) {
        delegate_names.push_back("xnnpack");
      }
    }
    bool attempted_delegate_fallback = false;
    TfLiteInterpreter* interpreter = api.InterpreterCreate(model, options);
    if (interpreter == nullptr && !delegates.empty() && !require_delegate) {
      attempted_delegate_fallback = true;
      release_delegate_handles(&delegates);
      api.OptionsDelete(options);
      options = api.OptionsCreate();
      configure_error_reporter(api, options, &error_collector);
      api.OptionsSetNumThreads(options, num_threads);
      interpreter = api.InterpreterCreate(model, options);
      if (interpreter != nullptr) {
        delegate_names.clear();
      }
    }
    if (interpreter == nullptr) {
      release_delegate_handles(&delegates);
      api.OptionsDelete(options);
      api.ModelDelete(model);
      *error = "TfLiteInterpreterCreate failed for " + artifact.model_path;
      if (!delegate_names.empty()) {
        *error += " with delegates " + dinf_json_string_array(delegate_names);
      } else {
        *error += " with no delegates";
      }
      if (attempted_delegate_fallback) {
        *error += " (fallback without delegates was also attempted)";
      }
      const std::string runtime_error = compact_error_text(error_collector.message);
      if (!runtime_error.empty()) {
        *error += " [tflite_error: " + runtime_error + "]";
      }
      if (!support_libraries.loaded.empty()) {
        *error += " [support libraries loaded: " +
                  dinf_json_string_array(support_libraries.loaded) + "]";
      } else if (!support_libraries.attempted.empty()) {
        *error += " [no optional support libraries loaded; attempted " +
                  dinf_json_string_array(support_libraries.attempted) + "]";
      }
      return nullptr;
    }
    return new LiteRtSession(
        std::move(library),
        api,
        model,
        options,
        interpreter,
        std::move(delegates),
        std::move(artifact),
        num_threads,
        delegate,
        std::move(delegate_names),
        std::move(support_libraries.libraries),
        std::move(support_libraries.attempted),
        std::move(support_libraries.loaded),
        attempted_delegate_fallback);
  }
  *error = "Unable to load LiteRT/TFLite C library. Set "
           "DART_INFERENCE_LITERT_LIBRARY or DART_INFERENCE_TFLITE_LIBRARY. Tried: " +
           dinf_json_string_array(attempted);
  return nullptr;
}
