#include "runtime_bridge.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <sstream>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

namespace {

void set_error(char** out, const std::string& message) {
  if (out != nullptr) {
    *out = dinf_copy_string(message);
  }
}

bool artifact_remote(const char* value) {
  if (value == nullptr) {
    return false;
  }
  const std::string text(value);
  return text.find("://") != std::string::npos;
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
    if (tensors[i].tensor.memory_kind == DINF_TENSOR_MEMORY_CPU) {
      std::free(tensors[i].tensor.data);
    }
    dinf_release_tensor_handle(tensors[i].tensor.handle);
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

extern "C" DINF_RUNTIME_EXPORT DinfRuntimeSession* dinf_open(
    int32_t engine,
    const char* model_path,
    int32_t prefer_mask,
    int32_t diagnostics,
    int32_t num_threads,
    const DinfOptionEntry* metadata_entries,
    intptr_t metadata_count,
    const DinfOptionEntry* backend_entries,
    intptr_t backend_count,
    char** error) {
  (void)prefer_mask;
  (void)diagnostics;
  if (artifact_remote(model_path)) {
    set_error(
        error,
        "Runtime artifact must be resolved to a local path before native execution.");
    return nullptr;
  }
  std::vector<DinfOptionEntry> options;
  if (metadata_entries != nullptr && metadata_count > 0) {
    options.insert(
        options.end(),
        metadata_entries,
        metadata_entries + metadata_count);
  }
  if (backend_entries != nullptr && backend_count > 0) {
    options.insert(
        options.end(),
        backend_entries,
        backend_entries + backend_count);
  }
  const std::string num_threads_key = "numThreads";
  if (num_threads > 0) {
    options.push_back(
        {num_threads_key.c_str(), DINF_OPTION_INT, nullptr, num_threads, 0, 0});
  }
  return dinf_cpp_open(
      engine,
      model_path,
      options.empty() ? nullptr : options.data(),
      static_cast<intptr_t>(options.size()),
      error);
}

extern "C" DINF_RUNTIME_EXPORT void dinf_close(DinfRuntimeSession* session) {
  dinf_cpp_close(session);
}

extern "C" DINF_RUNTIME_EXPORT int32_t dinf_run(
    DinfRuntimeSession* session,
    const DinfNamedTensor* inputs,
    intptr_t input_count,
    DinfNamedTensor** outputs,
    intptr_t* output_count,
    char** error) {
  return dinf_cpp_run(
      session,
      inputs,
      input_count,
      outputs,
      output_count,
      error);
}

extern "C" DINF_RUNTIME_EXPORT void dinf_free_tensors(
    DinfNamedTensor* tensors,
    intptr_t count) {
  dinf_cpp_free_tensors(tensors, count);
}

extern "C" DINF_RUNTIME_EXPORT void dinf_free_str(char* value) {
  dinf_cpp_free_str(value);
}

extern "C" DINF_RUNTIME_EXPORT DinfOptionEntry* dinf_diag(
    DinfRuntimeSession* session,
    intptr_t* count) {
  return dinf_cpp_diag(session, count);
}

extern "C" DINF_RUNTIME_EXPORT void dinf_free_diag(
    DinfOptionEntry* entries,
    intptr_t count) {
  dinf_cpp_free_options(entries, count);
}

extern "C" DINF_RUNTIME_EXPORT int32_t dinf_info(DinfInfo* out) {
  if (out == nullptr) {
    return 1;
  }
  out->native_backend = "cpp";
  out->runtime_version = "";
  out->async_model = "sync";
  out->abi = "dart_ffi_cpp_v1";
  out->mlx_owner = "none";
  out->mlx_api = "";
  out->mlx_linked = 0;
  out->mlx_enabled = 0;
  out->mlx_artifacts = "";
  return 0;
}

extern "C" DINF_RUNTIME_EXPORT int32_t dinf_platform_id() {
#if defined(__APPLE__)
#if TARGET_OS_IPHONE
  return 0;
#else
  return 1;
#endif
#elif defined(_WIN32)
  return 2;
#elif defined(__ANDROID__)
  return 4;
#elif defined(__linux__)
  return 3;
#else
  return 5;
#endif
}

extern "C" DINF_RUNTIME_EXPORT int32_t dinf_accel_mask(int32_t engine) {
  switch (engine) {
    case DINF_ENGINE_COREML:
      return 1 | 2 | 4;
    case DINF_ENGINE_ONNX:
      return 1 | 2;
    case DINF_ENGINE_LITERT:
      return 1 | 2 | 8;
    case DINF_ENGINE_MLX:
      return 1 | 2;
    default:
      return 1;
  }
}

extern "C" DINF_RUNTIME_EXPORT int32_t dinf_mem(DinfMemoryInfo* out) {
  dinf_cpp_mem(out);
  return 0;
}

extern "C" DINF_RUNTIME_EXPORT void* dinf_alloc(intptr_t byte_length) {
  if (byte_length <= 0) {
    return nullptr;
  }
  return std::malloc(static_cast<size_t>(byte_length));
}

extern "C" DINF_RUNTIME_EXPORT void dinf_free_buf(void* value) {
  std::free(value);
}

extern "C" DINF_RUNTIME_EXPORT void* dinf_alloc_tensor(
    int32_t dtype,
    const int64_t* shape,
    int32_t rank,
    intptr_t* byte_length,
    char** error) {
  if (byte_length != nullptr) {
    *byte_length = 0;
  }
  if (rank < 0 || (rank > 0 && shape == nullptr)) {
    set_error(error, "invalid tensor shape");
    return nullptr;
  }
  size_t count = 1;
  for (int32_t i = 0; i < rank; ++i) {
    if (shape[i] < 0) {
      set_error(error, "negative tensor dimension");
      return nullptr;
    }
    count *= static_cast<size_t>(shape[i]);
  }
  const size_t bytes = count * dinf_dtype_size(dtype);
  if (byte_length != nullptr) {
    *byte_length = static_cast<intptr_t>(bytes);
  }
  if (bytes == 0) {
    return nullptr;
  }
  void* out = std::malloc(bytes);
  if (out == nullptr) {
    set_error(error, "failed to allocate tensor buffer");
  }
  return out;
}

namespace {

std::vector<std::string> split_text(const char* text, char sep) {
  std::vector<std::string> values;
  if (text == nullptr || text[0] == '\0') {
    return values;
  }
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, sep)) {
    if (!item.empty()) {
      values.push_back(item);
    }
  }
  return values;
}

bool path_exists(const std::filesystem::path& path) {
  std::error_code ec;
  return std::filesystem::exists(path, ec);
}

bool ends_with(const std::string& value, const std::string& suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) ==
             0;
}

std::string platform_name(int32_t platform) {
  switch (platform) {
    case 0:
      return "ios";
    case 1:
      return "macos";
    case 2:
      return "windows";
    case 3:
      return "linux";
    case 4:
      return "android";
    default:
      return "unknown";
  }
}

bool artifact_matches_platform(
    const DinfResolveArtifact& artifact,
    int32_t platform) {
  if (artifact.target_platforms == nullptr ||
      artifact.target_platforms[0] == '\0') {
    return true;
  }
  const std::string wanted = platform_name(platform);
  std::string current;
  for (const char ch : std::string(artifact.target_platforms)) {
    if (ch == ':' || ch == ';' || ch == ',' || ch == '\n' || ch == '\r' ||
        ch == ' ' || ch == '\t') {
      if (current == wanted) {
        return true;
      }
      current.clear();
      continue;
    }
    current.push_back(ch);
  }
  if (current == wanted) {
    return true;
  }
  return false;
}

bool is_mlx_registered_artifact(const DinfResolveArtifact& artifact) {
  const std::string format = artifact.format == nullptr ? "" : artifact.format;
  if (format == "mlx-function" || format == "mlxfn") {
    return true;
  }
  const std::string path = artifact.path == nullptr ? "" : artifact.path;
  return ends_with(path, ".mlxfn") || ends_with(path, "/function.mlxfn");
}

bool artifact_matches_policy(
    const DinfResolveArtifact& artifact,
    int32_t platform,
    bool allow_preview_mlx) {
  if (!artifact_matches_platform(artifact, platform)) {
    return false;
  }
  if (artifact.engine != DINF_ENGINE_MLX) {
    return true;
  }
  return allow_preview_mlx || is_mlx_registered_artifact(artifact);
}

std::vector<int32_t> engine_order(int32_t platform) {
  switch (platform) {
    case 0:
    case 1:
      return {DINF_ENGINE_COREML, DINF_ENGINE_MLX, DINF_ENGINE_ONNX};
    case 2:
    case 3:
      return {DINF_ENGINE_ONNX};
    case 4:
      return {DINF_ENGINE_LITERT, DINF_ENGINE_ONNX};
    default:
      return {DINF_ENGINE_COREML, DINF_ENGINE_ONNX, DINF_ENGINE_LITERT};
  }
}

const DinfResolveArtifact* find_artifact(
    const DinfResolveArtifact* artifacts,
    intptr_t count,
    int32_t engine,
    int32_t platform,
    bool allow_preview_mlx) {
  if (artifacts == nullptr || count <= 0) {
    return nullptr;
  }
  for (intptr_t i = 0; i < count; ++i) {
    if (artifacts[i].engine == engine &&
        artifact_matches_policy(
            artifacts[i],
            platform,
            allow_preview_mlx)) {
      return &artifacts[i];
    }
  }
  return nullptr;
}

int32_t fallback_artifact_engine(
    int32_t platform,
    const int32_t* registered_engines,
    intptr_t registered_count,
    const DinfResolveArtifact* artifacts,
    intptr_t artifact_count) {
  for (intptr_t i = 0; i < artifact_count; ++i) {
    const int32_t engine = artifacts[i].engine;
    bool registered = false;
    for (intptr_t j = 0; j < registered_count; ++j) {
      if (registered_engines[j] == engine) {
        registered = true;
        break;
      }
    }
    if (!registered) {
      continue;
    }
    if (artifact_matches_policy(artifacts[i], platform, true)) {
      return engine;
    }
  }
  return -1;
}

std::string hf_repo_cache_key(const char* repo) {
  std::string key = repo == nullptr ? "" : repo;
  std::string out;
  for (const char ch : key) {
    if (ch == '/') {
      out += "--";
    } else {
      out.push_back(ch);
    }
  }
  return out;
}

std::string parse_hf_ref_text(const char* value) {
  if (value == nullptr) {
    return "";
  }
  const std::string uri(value);
  constexpr const char* prefix = "hf://";
  if (uri.rfind(prefix, 0) != 0) {
    return "";
  }
  return uri.substr(std::strlen(prefix));
}

bool parse_hf_body(
    const std::string& body,
    std::string* repo,
    std::string* artifact) {
  const size_t first = body.find('/');
  const size_t second = first == std::string::npos
      ? std::string::npos
      : body.find('/', first + 1);
  if (first == std::string::npos || second == std::string::npos) {
    return false;
  }
  *repo = body.substr(0, second);
  *artifact = body.substr(second + 1);
  return !repo->empty() && !artifact->empty();
}

bool is_bundle_name(const std::string& name) {
  return ends_with(name, ".mlmodelc") || ends_with(name, ".mlpackage");
}

std::string strip_bundle_suffix(const std::string& name) {
  if (ends_with(name, ".mlmodelc")) {
    return name.substr(0, name.size() - std::strlen(".mlmodelc"));
  }
  if (ends_with(name, ".mlpackage")) {
    return name.substr(0, name.size() - std::strlen(".mlpackage"));
  }
  return "";
}

bool parse_chunk_index(
    const std::string& name,
    const std::string& prefix,
    int* index) {
  const std::string stem = strip_bundle_suffix(name);
  if (stem.empty()) {
    return false;
  }
  if (prefix == "chunk" && stem == "chunk_head") {
    *index = 1000000;
    return true;
  }
  if (stem.rfind(prefix, 0) != 0) {
    return false;
  }
  std::string rest = stem.substr(prefix.size());
  if (!rest.empty() && rest.front() == '_') {
    rest.erase(rest.begin());
  }
  if (rest.empty()) {
    return false;
  }
  for (const char ch : rest) {
    if (ch < '0' || ch > '9') {
      return false;
    }
  }
  *index = std::stoi(rest);
  return true;
}

std::string join_list(const std::vector<std::string>& values, char sep) {
  std::string out;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      out.push_back(sep);
    }
    out += values[i];
  }
  return out;
}

std::string home_dir() {
  const char* home = std::getenv("HOME");
  if (home != nullptr && home[0] != '\0') {
    return home;
  }
#if defined(_WIN32)
  const char* user_profile = std::getenv("USERPROFILE");
  if (user_profile != nullptr && user_profile[0] != '\0') {
    return user_profile;
  }
#endif
  return ".";
}

}  // namespace

extern "C" DINF_RUNTIME_EXPORT int32_t dinf_resolve(
    const char* model_id,
    int32_t platform,
    int32_t requested_engine,
    int32_t allow_fallback,
    int32_t prefer_mask,
    const DinfResolveArtifact* artifacts,
    intptr_t artifact_count,
    DinfResolveResult* result,
    char** error) {
  (void)model_id;
  if (result == nullptr) {
    set_error(error, "resolve result is null");
    return 1;
  }
  result->engine = -1;
  result->accel_mask = prefer_mask != 0 ? prefer_mask : 1;
  result->fallback_engine = -1;
  if (requested_engine >= 0 &&
      find_artifact(
          artifacts,
          artifact_count,
          requested_engine,
          platform,
          true) != nullptr) {
    result->engine = requested_engine;
    result->accel_mask = prefer_mask != 0
        ? prefer_mask
        : dinf_accel_mask(requested_engine);
    return 0;
  }
  if (!allow_fallback) {
    set_error(error, "Requested runtime engine has no compatible artifact.");
    return 1;
  }
  for (const int32_t engine : engine_order(platform)) {
    const auto* artifact =
        find_artifact(artifacts, artifact_count, engine, platform, false);
    if (artifact == nullptr) {
      continue;
    }
    result->engine = engine;
    result->accel_mask =
        prefer_mask != 0 ? prefer_mask : dinf_accel_mask(engine);
    result->fallback_engine = requested_engine >= 0 ? requested_engine : -1;
    return 0;
  }
  set_error(error, "No compatible runtime artifact found.");
  return 1;
}

extern "C" DINF_RUNTIME_EXPORT int32_t dinf_fallback(
    int32_t platform,
    const int32_t* registered_engines,
    intptr_t registered_count,
    const DinfResolveArtifact* artifacts,
    intptr_t artifact_count) {
  if (registered_engines == nullptr || registered_count <= 0) {
    return -1;
  }
  return fallback_artifact_engine(
      platform,
      registered_engines,
      registered_count,
      artifacts,
      artifact_count);
}

extern "C" DINF_RUNTIME_EXPORT char* dinf_artifact_path(
    const char* root_path,
    const char* path) {
  if (path == nullptr) {
    return dinf_copy_string("");
  }
  const std::filesystem::path artifact(path);
  if (artifact_remote(path) || artifact.is_absolute() || root_path == nullptr ||
      root_path[0] == '\0') {
    return dinf_copy_string(path);
  }
  return dinf_copy_string((std::filesystem::path(root_path) / artifact).string());
}

extern "C" DINF_RUNTIME_EXPORT char* dinf_hf_cache_root() {
  const char* hf_home = std::getenv("HF_HOME");
  if (hf_home != nullptr && hf_home[0] != '\0') {
    return dinf_copy_string(hf_home);
  }
  const char* xdg = std::getenv("XDG_CACHE_HOME");
  if (xdg != nullptr && xdg[0] != '\0') {
    return dinf_copy_string((std::filesystem::path(xdg) / "huggingface").string());
  }
  return dinf_copy_string(
      (std::filesystem::path(home_dir()) / ".cache" / "huggingface").string());
}

extern "C" DINF_RUNTIME_EXPORT char* dinf_hf_token() {
  const char* token = std::getenv("HF_TOKEN");
  if (token == nullptr || token[0] == '\0') {
    token = std::getenv("HUGGING_FACE_HUB_TOKEN");
  }
  return token == nullptr ? nullptr : dinf_copy_string(token);
}

extern "C" DINF_RUNTIME_EXPORT char* dinf_hf_cache_path(
    const char* cache_root,
    const char* repo,
    const char* revision,
    const char* artifact_path) {
  const std::string repo_key = hf_repo_cache_key(repo);
  const std::filesystem::path path =
      std::filesystem::path(cache_root == nullptr ? "" : cache_root) /
      "hub" /
      ("models--" + repo_key) /
      "snapshots" /
      (revision == nullptr || revision[0] == '\0' ? "main" : revision) /
      (artifact_path == nullptr ? "" : artifact_path);
  return dinf_copy_string(path.string());
}

extern "C" DINF_RUNTIME_EXPORT char* dinf_hf_ref(
    const char* source_uri,
    const char* artifact_path,
    const char* repo,
    const char* artifact,
    const char* revision) {
  std::string repo_value = repo == nullptr ? "" : repo;
  std::string artifact_value = artifact == nullptr ? "" : artifact;
  std::string revision_value =
      revision == nullptr || revision[0] == '\0' ? "main" : revision;
  if ((repo_value.empty() || artifact_value.empty()) && source_uri != nullptr) {
    parse_hf_body(parse_hf_ref_text(source_uri), &repo_value, &artifact_value);
  }
  if ((repo_value.empty() || artifact_value.empty()) &&
      artifact_path != nullptr) {
    parse_hf_body(
        parse_hf_ref_text(artifact_path),
        &repo_value,
        &artifact_value);
  }
  if (repo_value.empty() || artifact_value.empty()) {
    return nullptr;
  }
  return dinf_copy_string(repo_value + "\x1f" + artifact_value + "\x1f" + revision_value);
}

extern "C" DINF_RUNTIME_EXPORT int32_t dinf_hf_dir_artifact(
    const char* artifact_path) {
  if (artifact_path == nullptr) {
    return 0;
  }
  const std::string value(artifact_path);
  return ends_with(value, ".mlmodelc") || ends_with(value, ".mlpackage") ? 1 : 0;
}

extern "C" DINF_RUNTIME_EXPORT char* dinf_coreml_layout(
    const char* root_path) {
  const std::string root = root_path == nullptr ? "" : root_path;
  const char field_sep = '\x1f';
  const char list_sep = '\x1e';
  const auto root_fs = std::filesystem::path(root);
  std::string pipeline;
  std::string config;
  std::string monolithic;
  std::vector<std::pair<int, std::string>> decode_chunks;
  std::vector<std::pair<int, std::string>> prefill_chunks;
  std::vector<std::string> sidecars;

  std::error_code ec;
  if (std::filesystem::is_regular_file(root_fs, ec) && ends_with(root, ".json")) {
    pipeline = root;
  } else if (ends_with(root, ".mlmodelc") || ends_with(root, ".mlpackage")) {
    monolithic = root;
  } else if (std::filesystem::is_directory(root_fs, ec)) {
    if (path_exists(root_fs / "model_config.json")) {
      config = (root_fs / "model_config.json").string();
    }
    for (const auto& entry :
         std::filesystem::directory_iterator(root_fs, ec)) {
      if (ec) {
        break;
      }
      const std::string name = entry.path().filename().string();
      if (entry.is_directory(ec) && is_bundle_name(name)) {
        const std::string full = entry.path().string();
        if (name == "model.mlmodelc" || name == "model.mlpackage") {
          monolithic = full;
          continue;
        }
        int index = 0;
        if (parse_chunk_index(name, "prefill_chunk", &index)) {
          prefill_chunks.push_back({index, full});
          continue;
        }
        if (parse_chunk_index(name, "chunk", &index)) {
          decode_chunks.push_back({index, full});
          continue;
        }
      }
      if (entry.is_regular_file(ec) && name != "model_config.json") {
        sidecars.push_back(entry.path().string());
      }
    }
  }
  const auto by_index = [](const auto& left, const auto& right) {
    return left.first < right.first;
  };
  std::sort(decode_chunks.begin(), decode_chunks.end(), by_index);
  std::sort(prefill_chunks.begin(), prefill_chunks.end(), by_index);
  std::sort(sidecars.begin(), sidecars.end());

  std::vector<std::string> decode;
  std::vector<std::string> prefill;
  for (const auto& item : decode_chunks) {
    decode.push_back(item.second);
  }
  for (const auto& item : prefill_chunks) {
    prefill.push_back(item.second);
  }

  std::string out = root;
  out.push_back(field_sep);
  out += pipeline;
  out.push_back(field_sep);
  out += config;
  out.push_back(field_sep);
  out += monolithic;
  out.push_back(field_sep);
  out += join_list(decode, list_sep);
  out.push_back(field_sep);
  out += join_list(prefill, list_sep);
  out.push_back(field_sep);
  out += join_list(sidecars, list_sep);
  return dinf_copy_string(out);
}

extern "C" DINF_RUNTIME_EXPORT char* dinf_ort_libs(
    const char* runtime_env_file,
    const char* search_roots,
    const char* explicit_libraries,
    const char* library_dirs,
    const char* library_names) {
  (void)runtime_env_file;
  (void)search_roots;
  std::vector<std::string> out;
  for (const auto& value : split_text(explicit_libraries, '\x1e')) {
    if (path_exists(value)) {
      out.push_back(value);
    }
  }
  const auto dirs = split_text(library_dirs, '\x1e');
  for (const auto& name : split_text(library_names, '\x1e')) {
    for (const auto& dir : dirs) {
      const auto path = std::filesystem::path(dir) / name;
      if (path_exists(path)) {
        out.push_back(path.string());
        break;
      }
    }
  }
  std::string joined;
  for (size_t i = 0; i < out.size(); ++i) {
    if (i > 0) {
      joined.push_back('\x1e');
    }
    joined += out[i];
  }
  return dinf_copy_string(joined);
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
  tensor.tensor.memory_kind = DINF_TENSOR_MEMORY_CPU;
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

DinfTensorHandle* dinf_make_tensor_handle(
    void* value,
    void* context,
    DinfTensorHandleRelease release) {
  auto* handle = static_cast<DinfTensorHandle*>(
      std::malloc(sizeof(DinfTensorHandle)));
  if (handle == nullptr) {
    return nullptr;
  }
  handle->value = value;
  handle->context = context;
  handle->release = release;
  return handle;
}

void dinf_release_tensor_handle(DinfTensorHandle* handle) {
  if (handle == nullptr) {
    return;
  }
  if (handle->release != nullptr) {
    handle->release(handle->value, handle->context);
  }
  std::free(handle);
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
