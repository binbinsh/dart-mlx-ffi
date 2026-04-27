#include "runtime_bridge.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <deque>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace {

struct Args {
  std::map<std::string, std::string> values;
  std::vector<std::string> flags;

  bool has_flag(const std::string& name) const {
    return std::find(flags.begin(), flags.end(), name) != flags.end();
  }

  std::string option(
      const std::string& name,
      const std::string& fallback = "") const {
    const auto it = values.find(name);
    return it == values.end() ? fallback : it->second;
  }

  std::string required(const std::string& name) const {
    const auto value = option(name);
    if (value.empty()) {
      throw std::runtime_error("Missing --" + name);
    }
    return value;
  }
};

struct RuntimeOptionEntries {
  std::vector<DinfOptionEntry> entries;
  std::deque<std::string> strings;

  const char* own(std::string value) {
    strings.push_back(std::move(value));
    return strings.back().c_str();
  }

  void add_string(const char* path, const std::string& value) {
    if (value.empty()) {
      return;
    }
    entries.push_back({
        path,
        DINF_OPTION_STRING,
        own(value),
        0,
        0,
        0,
    });
  }

  void add_int(const char* path, int64_t value) {
    entries.push_back({
        path,
        DINF_OPTION_INT,
        nullptr,
        value,
        0,
        0,
    });
  }

  void add_bool(const char* path, bool value) {
    entries.push_back({
        path,
        DINF_OPTION_BOOL,
        nullptr,
        0,
        0,
        value ? 1 : 0,
    });
  }
};

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string raw = argv[i];
    if (raw.rfind("--", 0) != 0) {
      throw std::runtime_error("Unexpected positional argument: " + raw);
    }
    const std::string name = raw.substr(2);
    if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
      args.values[name] = argv[++i];
    } else {
      args.flags.push_back(name);
    }
  }
  return args;
}

json read_json_file(const std::string& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("Unable to open " + path);
  }
  json value;
  file >> value;
  return value;
}

void write_json_file(const std::string& path, const json& value) {
  std::ofstream file(path);
  if (!file) {
    throw std::runtime_error("Unable to write " + path);
  }
  file << value.dump(2) << "\n";
}

int engine_id(const std::string& engine) {
  if (engine == "coreml") {
    return DINF_ENGINE_COREML;
  }
  if (engine == "onnx") {
    return DINF_ENGINE_ONNX;
  }
  if (engine == "litert") {
    return DINF_ENGINE_LITERT;
  }
  throw std::runtime_error("Unsupported engine: " + engine);
}

int dtype_id(const std::string& dtype) {
  if (dtype == "float32") {
    return DINF_DTYPE_FLOAT32;
  }
  if (dtype == "int32") {
    return DINF_DTYPE_INT32;
  }
  if (dtype == "int64") {
    return DINF_DTYPE_INT64;
  }
  if (dtype == "uint8") {
    return DINF_DTYPE_UINT8;
  }
  if (dtype == "float64") {
    return DINF_DTYPE_FLOAT64;
  }
  if (dtype == "float16") {
    return DINF_DTYPE_FLOAT16;
  }
  if (dtype == "bool") {
    return DINF_DTYPE_BOOL;
  }
  throw std::runtime_error("Unsupported dtype: " + dtype);
}

std::string dtype_name(int dtype) {
  switch (dtype) {
    case DINF_DTYPE_FLOAT32:
      return "float32";
    case DINF_DTYPE_INT32:
      return "int32";
    case DINF_DTYPE_INT64:
      return "int64";
    case DINF_DTYPE_UINT8:
      return "uint8";
    case DINF_DTYPE_FLOAT64:
      return "float64";
    case DINF_DTYPE_FLOAT16:
      return "float16";
    case DINF_DTYPE_BOOL:
      return "bool";
    default:
      return "unknown";
  }
}

size_t dtype_size(int dtype) {
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

char* copy_string(const std::string& value) {
  char* copy = static_cast<char*>(std::malloc(value.size() + 1));
  if (copy == nullptr) {
    return nullptr;
  }
  std::memcpy(copy, value.data(), value.size());
  copy[value.size()] = '\0';
  return copy;
}

DinfNamedTensor make_tensor(
    const char* name,
    int32_t dtype,
    const std::vector<int64_t>& shape,
    const void* data,
    size_t byte_length) {
  DinfNamedTensor tensor{};
  tensor.name = copy_string(name == nullptr ? "" : name);
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

int base64_value(char c) {
  if (c >= 'A' && c <= 'Z') {
    return c - 'A';
  }
  if (c >= 'a' && c <= 'z') {
    return c - 'a' + 26;
  }
  if (c >= '0' && c <= '9') {
    return c - '0' + 52;
  }
  if (c == '+') {
    return 62;
  }
  if (c == '/') {
    return 63;
  }
  return -1;
}

std::vector<uint8_t> base64_decode(const std::string& text) {
  std::vector<uint8_t> bytes;
  int value = 0;
  int value_bits = -8;
  for (unsigned char raw : text) {
    const char c = static_cast<char>(raw);
    if (std::isspace(raw)) {
      continue;
    }
    if (c == '=') {
      break;
    }
    const int digit = base64_value(c);
    if (digit < 0) {
      throw std::runtime_error("Invalid base64 tensor bytes.");
    }
    value = (value << 6) | digit;
    value_bits += 6;
    if (value_bits >= 0) {
      bytes.push_back(static_cast<uint8_t>((value >> value_bits) & 0xff));
      value_bits -= 8;
    }
  }
  return bytes;
}

std::vector<uint8_t> read_binary_file(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Unable to open tensor data file: " + path.string());
  }
  return std::vector<uint8_t>(
      std::istreambuf_iterator<char>(file),
      std::istreambuf_iterator<char>());
}

std::string read_text_file(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("Unable to open tensor data file: " + path.string());
  }
  return std::string(
      std::istreambuf_iterator<char>(file),
      std::istreambuf_iterator<char>());
}

std::filesystem::path resolve_data_path(
    const std::string& value,
    const std::filesystem::path& base_dir) {
  std::filesystem::path path(value);
  if (path.is_absolute() || base_dir.empty()) {
    return path;
  }
  return base_dir / path;
}

template <typename T>
void collect_numeric_values(const json& value, std::vector<T>* typed) {
  if (value.is_array()) {
    for (const auto& item : value) {
      collect_numeric_values(item, typed);
    }
    return;
  }
  typed->push_back(value.get<T>());
}

template <typename T>
std::vector<uint8_t> numeric_bytes(const json& values) {
  if (!values.is_array()) {
    throw std::runtime_error("Tensor values must be an array.");
  }
  std::vector<T> typed;
  collect_numeric_values(values, &typed);
  std::vector<uint8_t> bytes(sizeof(T) * typed.size());
  std::memcpy(bytes.data(), typed.data(), bytes.size());
  return bytes;
}

uint16_t float_to_half(double value) {
  const float narrowed = static_cast<float>(value);
  uint32_t bits = 0;
  std::memcpy(&bits, &narrowed, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000;
  int exponent = static_cast<int>((bits >> 23) & 0xff) - 127 + 15;
  const uint32_t mantissa = bits & 0x7fffff;
  if (exponent <= 0) {
    if (exponent < -10) {
      return static_cast<uint16_t>(sign);
    }
    const uint32_t shifted = (mantissa | 0x800000) >> (1 - exponent);
    return static_cast<uint16_t>(sign | ((shifted + 0x1000) >> 13));
  }
  if (exponent >= 0x1f) {
    if (mantissa == 0) {
      return static_cast<uint16_t>(sign | 0x7c00);
    }
    return static_cast<uint16_t>(sign | 0x7c00 | (mantissa >> 13) | 1);
  }
  return static_cast<uint16_t>(
      sign | (static_cast<uint32_t>(exponent) << 10) |
      ((mantissa + 0x1000) >> 13));
}

void collect_float16_values(const json& value, std::vector<uint16_t>* typed) {
  if (value.is_array()) {
    for (const auto& item : value) {
      collect_float16_values(item, typed);
    }
    return;
  }
  typed->push_back(float_to_half(value.get<double>()));
}

std::vector<uint8_t> float16_numeric_bytes(const json& values) {
  if (!values.is_array()) {
    throw std::runtime_error("Tensor values must be an array.");
  }
  std::vector<uint16_t> typed;
  collect_float16_values(values, &typed);
  std::vector<uint8_t> bytes(sizeof(uint16_t) * typed.size());
  std::memcpy(bytes.data(), typed.data(), bytes.size());
  return bytes;
}

void collect_bool_values(const json& value, std::vector<uint8_t>* typed) {
  if (value.is_array()) {
    for (const auto& item : value) {
      collect_bool_values(item, typed);
    }
    return;
  }
  if (value.is_boolean()) {
    typed->push_back(value.get<bool>() ? 1 : 0);
    return;
  }
  typed->push_back(value.get<double>() == 0.0 ? 0 : 1);
}

std::vector<uint8_t> bool_numeric_bytes(const json& values) {
  if (!values.is_array()) {
    throw std::runtime_error("Tensor values must be an array.");
  }
  std::vector<uint8_t> typed;
  collect_bool_values(values, &typed);
  return typed;
}

std::vector<uint8_t> tensor_bytes_from_json(
    const json& spec,
    const std::string& dtype,
    const std::filesystem::path& base_dir,
    const std::string& input_name) {
  if (spec.contains("values")) {
    const json& values = spec.at("values");
    if (dtype == "float32") {
      return numeric_bytes<float>(values);
    }
    if (dtype == "int32") {
      return numeric_bytes<int32_t>(values);
    }
    if (dtype == "int64") {
      return numeric_bytes<int64_t>(values);
    }
    if (dtype == "uint8") {
      return numeric_bytes<uint8_t>(values);
    }
    if (dtype == "float64") {
      return numeric_bytes<double>(values);
    }
    if (dtype == "float16") {
      return float16_numeric_bytes(values);
    }
    if (dtype == "bool") {
      return bool_numeric_bytes(values);
    }
    throw std::runtime_error("Unsupported dtype: " + dtype);
  }
  if (spec.contains("base64")) {
    if (!spec.at("base64").is_string()) {
      throw std::runtime_error("Input " + input_name + " base64 must be a string.");
    }
    return base64_decode(spec.at("base64").get<std::string>());
  }
  const char* path_key =
      spec.contains("file") ? "file" : (spec.contains("path") ? "path" : nullptr);
  if (path_key != nullptr) {
    if (!spec.at(path_key).is_string()) {
      throw std::runtime_error(
          "Input " + input_name + " " + path_key + " must be a string.");
    }
    const auto path = resolve_data_path(
        spec.at(path_key).get<std::string>(),
        base_dir);
    if (spec.value("encoding", "") == "base64") {
      return base64_decode(read_text_file(path));
    }
    return read_binary_file(path);
  }
  throw std::runtime_error(
      "Input " + input_name + " must define values, base64, file, or path.");
}

std::vector<int64_t> shape_from_json(
    const json& spec,
    int dtype,
    size_t byte_length,
    const std::string& input_name) {
  std::vector<int64_t> shape;
  if (!spec.contains("shape")) {
    const size_t width = dtype_size(dtype);
    if (width == 0 || byte_length % width != 0) {
      throw std::runtime_error(
          "Input " + input_name + " byte length is not divisible by dtype width.");
    }
    shape.push_back(static_cast<int64_t>(byte_length / width));
    return shape;
  }
  if (!spec.at("shape").is_array()) {
    throw std::runtime_error("Input " + input_name + " shape must be an array.");
  }
  for (const auto& dim : spec.at("shape")) {
    shape.push_back(dim.get<int64_t>());
  }
  const size_t width = dtype_size(dtype);
  size_t expected = width;
  for (const auto dim : shape) {
    expected *= static_cast<size_t>(dim);
  }
  if (expected != byte_length) {
    throw std::runtime_error(
        "Input " + input_name + " byte length does not match shape and dtype.");
  }
  return shape;
}

DinfNamedTensor tensor_from_json(
    const std::string& name,
    const json& spec,
    const std::filesystem::path& base_dir) {
  if (!spec.is_object()) {
    throw std::runtime_error("Input " + name + " must be an object.");
  }
  const std::string dtype = spec.at("dtype").get<std::string>();
  const int id = dtype_id(dtype);
  std::vector<uint8_t> bytes =
      tensor_bytes_from_json(spec, dtype, base_dir, name);
  return make_tensor(
      name.c_str(),
      id,
      shape_from_json(spec, id, bytes.size(), name),
      bytes.data(),
      bytes.size());
}

std::vector<DinfNamedTensor> inputs_from_json(
    const json& document,
    const std::filesystem::path& base_dir) {
  const json* inputs = &document;
  if (document.contains("inputs")) {
    inputs = &document.at("inputs");
  }
  if (!inputs->is_object()) {
    throw std::runtime_error("Input JSON must contain an object of tensors.");
  }
  std::vector<DinfNamedTensor> tensors;
  for (const auto& item : inputs->items()) {
    tensors.push_back(tensor_from_json(item.key(), item.value(), base_dir));
  }
  return tensors;
}

void free_inputs(std::vector<DinfNamedTensor>& tensors) {
  for (auto& tensor : tensors) {
    std::free(tensor.name);
    std::free(tensor.tensor.shape);
    std::free(tensor.tensor.data);
  }
  tensors.clear();
}

std::vector<std::string> split_diag_path(const char* path) {
  std::vector<std::string> segments;
  if (path == nullptr || *path == '\0') {
    return segments;
  }
  const std::string raw(path);
  size_t start = 0;
  while (start <= raw.size()) {
    const size_t next = raw.find('\x1f', start);
    segments.push_back(raw.substr(start, next - start));
    if (next == std::string::npos) {
      break;
    }
    start = next + 1;
  }
  return segments;
}

size_t diag_index(const std::string& segment) {
  if (segment.empty() ||
      !std::all_of(segment.begin(), segment.end(), [](unsigned char c) {
        return std::isdigit(c) != 0;
      })) {
    return 0;
  }
  return static_cast<size_t>(std::stoull(segment));
}

json& diag_child(json& node, const std::string& segment) {
  if (node.is_array()) {
    const size_t index = diag_index(segment);
    while (node.size() <= index) {
      node.push_back(nullptr);
    }
    return node[index];
  }
  if (!node.is_object()) {
    node = json::object();
  }
  return node[segment];
}

json diag_value(const DinfOptionEntry& entry) {
  switch (entry.kind) {
    case DINF_OPTION_STRING:
      return entry.text == nullptr ? "" : entry.text;
    case DINF_OPTION_INT:
      return entry.int_value;
    case DINF_OPTION_BOOL:
      return entry.bool_value != 0;
    case DINF_OPTION_MAP:
      return json::object();
    case DINF_OPTION_LIST:
      return json::array();
    case DINF_OPTION_DOUBLE:
      return entry.double_value;
    case DINF_OPTION_NULL:
    default:
      return nullptr;
  }
}

void assign_diag(json& root, const DinfOptionEntry& entry) {
  const auto segments = split_diag_path(entry.path);
  if (segments.empty()) {
    return;
  }
  json* node = &root;
  for (const auto& segment : segments) {
    node = &diag_child(*node, segment);
  }
  *node = diag_value(entry);
}

json diagnostics_json(DinfRuntimeSession* session) {
  intptr_t count = 0;
  DinfOptionEntry* entries = dinf_cpp_diag(session, &count);
  if (entries == nullptr || count <= 0) {
    return json::object();
  }
  json out = json::object();
  for (intptr_t i = 0; i < count; ++i) {
    assign_diag(out, entries[i]);
  }
  dinf_cpp_free_options(entries, count);
  return out;
}

json tensor_values(const DinfTensor& tensor) {
  const auto count = static_cast<size_t>(
      tensor.byte_length / static_cast<intptr_t>(dtype_size(tensor.dtype)));
  json values = json::array();
  switch (tensor.dtype) {
    case DINF_DTYPE_FLOAT32: {
      const auto* data = static_cast<const float*>(tensor.data);
      for (size_t i = 0; i < count; ++i) {
        values.push_back(data[i]);
      }
      break;
    }
    case DINF_DTYPE_INT32: {
      const auto* data = static_cast<const int32_t*>(tensor.data);
      for (size_t i = 0; i < count; ++i) {
        values.push_back(data[i]);
      }
      break;
    }
    case DINF_DTYPE_INT64: {
      const auto* data = static_cast<const int64_t*>(tensor.data);
      for (size_t i = 0; i < count; ++i) {
        values.push_back(data[i]);
      }
      break;
    }
    case DINF_DTYPE_UINT8: {
      const auto* data = static_cast<const uint8_t*>(tensor.data);
      for (size_t i = 0; i < count; ++i) {
        values.push_back(data[i]);
      }
      break;
    }
    case DINF_DTYPE_FLOAT64: {
      const auto* data = static_cast<const double*>(tensor.data);
      for (size_t i = 0; i < count; ++i) {
        values.push_back(data[i]);
      }
      break;
    }
    case DINF_DTYPE_FLOAT16: {
      const auto* data = static_cast<const uint16_t*>(tensor.data);
      for (size_t i = 0; i < count; ++i) {
        values.push_back(data[i]);
      }
      break;
    }
    case DINF_DTYPE_BOOL: {
      const auto* data = static_cast<const uint8_t*>(tensor.data);
      for (size_t i = 0; i < count; ++i) {
        values.push_back(data[i] != 0);
      }
      break;
    }
  }
  return values;
}

json tensor_shape(const DinfTensor& tensor) {
  json shape = json::array();
  for (int i = 0; i < tensor.rank; ++i) {
    shape.push_back(tensor.shape[i]);
  }
  return shape;
}

json correctness(DinfNamedTensor* outputs, intptr_t output_count) {
  json output_values = json::object();
  json output_summaries = json::object();
  for (intptr_t i = 0; i < output_count; ++i) {
    const auto& output = outputs[i];
    const json values = tensor_values(output.tensor);
    json preview = json::array();
    const auto preview_count = std::min<size_t>(values.size(), 16);
    for (size_t j = 0; j < preview_count; ++j) {
      preview.push_back(values[j]);
    }
    const std::string name = output.name == nullptr ? "" : output.name;
    output_summaries[name] = {
        {"dtype", dtype_name(output.tensor.dtype)},
        {"shape", tensor_shape(output.tensor)},
        {"preview", preview},
        {"num_values", values.size()},
    };
    if (values.size() <= 4096) {
      output_values[name] = {
          {"dtype", dtype_name(output.tensor.dtype)},
          {"shape", tensor_shape(output.tensor)},
          {"values", values},
      };
    }
  }
  json result = {{"output_summaries", output_summaries}};
  if (!output_values.empty()) {
    result["output_values"] = output_values;
  }
  return result;
}

RuntimeOptionEntries runtime_options(const Args& args) {
  RuntimeOptionEntries options;
  options.add_bool("diagnostics", true);
  options.add_string("provider", args.option("provider"));
  options.add_string("delegate", args.option("delegate"));
  options.add_string("coremlMode", args.option("coreml-mode"));
  if (!args.option("litert-section-index").empty()) {
    options.add_int(
        "litertSectionIndex",
        std::stoi(args.option("litert-section-index")));
  }
  if (args.has_flag("require-provider")) {
    options.add_bool("requireProvider", true);
  }
  if (args.has_flag("require-delegate")) {
    options.add_bool("requireDelegate", true);
  }
  if (!args.option("num-threads").empty()) {
    options.add_int("numThreads", std::stoi(args.option("num-threads")));
  }
  return options;
}

double millis_since(
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

uint64_t peak_memory_value(const DinfMemoryInfo& memory) {
  if (memory.peak_memory_bytes != 0) {
    return memory.peak_memory_bytes;
  }
  if (memory.vm_hwm != 0) {
    return memory.vm_hwm;
  }
  if (memory.resident_size != 0) {
    return memory.resident_size;
  }
  return 0;
}

DinfMemoryInfo memory_info() {
  DinfMemoryInfo info{};
  dinf_cpp_mem(&info);
  return info;
}

json memory_json(const DinfMemoryInfo& memory) {
  json out = {{"peak_memory_bytes", memory.peak_memory_bytes}};
#if defined(__APPLE__)
  out["phys_footprint"] = memory.phys_footprint;
  out["resident_size"] = memory.resident_size;
  out["virtual_size"] = memory.virtual_size;
#elif defined(_WIN32)
  out["peak_working_set"] = memory.peak_working_set;
  out["working_set"] = memory.working_set;
#elif defined(__ANDROID__)
  out["android_peak_pss"] = memory.android_peak_pss;
  out["android_pss"] = memory.android_pss;
  out["android_rss"] = memory.android_rss;
  out["android_native_heap_pss"] = memory.android_native_heap_pss;
  out["android_java_heap_pss"] = memory.android_java_heap_pss;
  out["android_native_heap_private_dirty"] =
      memory.android_native_heap_private_dirty;
  out["android_java_heap_private_dirty"] =
      memory.android_java_heap_private_dirty;
#elif defined(__linux__)
  out["vm_hwm"] = memory.vm_hwm;
  out["vm_rss"] = memory.vm_rss;
#endif
  return out;
}

void update_peak_memory(uint64_t* peak, const DinfMemoryInfo& memory) {
  *peak = std::max(*peak, peak_memory_value(memory));
}

void usage() {
  std::cout
      << "Usage:\n"
      << "  dinf_runner --model-id <id> "
      << "--engine <coreml|onnx|litert> --artifact <path> "
      << "--input-json <inputs.json> [--platform android] [--out report.json]\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    if (args.has_flag("help")) {
      usage();
      return 0;
    }
    const std::string model_id = args.required("model-id");
    const std::string engine = args.required("engine");
    const std::string artifact = args.required("artifact");
    const std::string input_path = args.required("input-json");
    const std::string platform = args.option("platform", "android");
    const int warmup = std::stoi(args.option("warmup", "1"));
    const int iters = std::stoi(args.option("iters", "5"));

    char* error = nullptr;
    auto options = runtime_options(args);
    DinfRuntimeSession* session = dinf_cpp_open(
        engine_id(engine),
        artifact.c_str(),
        options.entries.empty() ? nullptr : options.entries.data(),
        static_cast<intptr_t>(options.entries.size()),
        &error);
    if (session == nullptr) {
      const std::string message = error == nullptr ? "runtime create failed" : error;
      dinf_cpp_free_str(error);
      throw std::runtime_error(message);
    }

    auto inputs = inputs_from_json(
        read_json_file(input_path),
        std::filesystem::path(input_path).parent_path());
    DinfNamedTensor* outputs = nullptr;
    intptr_t output_count = 0;
    const DinfMemoryInfo memory_before = memory_info();
    uint64_t peak_memory = peak_memory_value(memory_before);

    for (int i = 0; i < warmup; ++i) {
      if (outputs != nullptr) {
        dinf_cpp_free_tensors(outputs, output_count);
        outputs = nullptr;
        output_count = 0;
      }
      if (dinf_cpp_run(
              session,
              inputs.data(),
              static_cast<intptr_t>(inputs.size()),
              &outputs,
              &output_count,
              &error) != 0) {
        const std::string message = error == nullptr ? "runtime run failed" : error;
        dinf_cpp_free_str(error);
        throw std::runtime_error(message);
      }
      update_peak_memory(&peak_memory, memory_info());
    }

    double timed_ms = 0.0;
    for (int i = 0; i < iters; ++i) {
      if (outputs != nullptr) {
        dinf_cpp_free_tensors(outputs, output_count);
        outputs = nullptr;
        output_count = 0;
      }
      const auto start = std::chrono::steady_clock::now();
      if (dinf_cpp_run(
              session,
              inputs.data(),
              static_cast<intptr_t>(inputs.size()),
              &outputs,
              &output_count,
              &error) != 0) {
        const std::string message = error == nullptr ? "runtime run failed" : error;
        dinf_cpp_free_str(error);
        throw std::runtime_error(message);
      }
      const auto end = std::chrono::steady_clock::now();
      timed_ms += millis_since(start, end);
      update_peak_memory(&peak_memory, memory_info());
    }

    const DinfMemoryInfo memory_after = memory_info();
    update_peak_memory(&peak_memory, memory_after);
    const json diagnostics = diagnostics_json(session);
    const double per_iter_ms = iters > 0 ? timed_ms / iters : 0.0;
    const json memory_before_json = memory_json(memory_before);
    const json memory_after_json = memory_json(memory_after);

    json report = {
        {"model_id", model_id},
        {"platform", platform},
        {"engine", engine},
        {"artifact", artifact},
        {"correctness", correctness(outputs, output_count)},
        {"metrics", {{"end_to_end_ms", per_iter_ms}, {"peak_memory_bytes", peak_memory}}},
        {
            "device_profile",
            {
                {"runtime", "dart_inference_native_cli"},
                {"runtime_diagnostics", diagnostics},
                {"memory_before", memory_before_json},
                {"memory_after", memory_after_json},
                {"raw_peak_memory_field",
                 memory_after_json.contains("android_peak_pss")
                     ? "android_peak_pss"
                     : (memory_after_json.contains("vm_hwm")
                            ? "VmHWM"
                            : "peak_memory_bytes")},
            },
        },
    };

    if (!args.option("out").empty()) {
      write_json_file(args.option("out"), report);
    }
    std::cout << report.dump(2) << "\n";

    if (outputs != nullptr) {
      dinf_cpp_free_tensors(outputs, output_count);
    }
    free_inputs(inputs);
    dinf_cpp_close(session);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "dinf_runner failed: " << error.what() << "\n";
    return 2;
  }
}
