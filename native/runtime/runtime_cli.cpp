#include "runtime_bridge.h"

#include <algorithm>
#include <chrono>
#include <cctype>
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
    return DMF_ENGINE_COREML;
  }
  if (engine == "onnx") {
    return DMF_ENGINE_ONNX;
  }
  if (engine == "litert") {
    return DMF_ENGINE_LITERT;
  }
  throw std::runtime_error("Unsupported engine: " + engine);
}

int dtype_id(const std::string& dtype) {
  if (dtype == "float32") {
    return DMF_DTYPE_FLOAT32;
  }
  if (dtype == "int32") {
    return DMF_DTYPE_INT32;
  }
  if (dtype == "int64") {
    return DMF_DTYPE_INT64;
  }
  if (dtype == "uint8") {
    return DMF_DTYPE_UINT8;
  }
  if (dtype == "float64") {
    return DMF_DTYPE_FLOAT64;
  }
  if (dtype == "float16") {
    return DMF_DTYPE_FLOAT16;
  }
  if (dtype == "bool") {
    return DMF_DTYPE_BOOL;
  }
  throw std::runtime_error("Unsupported dtype: " + dtype);
}

std::string dtype_name(int dtype) {
  switch (dtype) {
    case DMF_DTYPE_FLOAT32:
      return "float32";
    case DMF_DTYPE_INT32:
      return "int32";
    case DMF_DTYPE_INT64:
      return "int64";
    case DMF_DTYPE_UINT8:
      return "uint8";
    case DMF_DTYPE_FLOAT64:
      return "float64";
    case DMF_DTYPE_FLOAT16:
      return "float16";
    case DMF_DTYPE_BOOL:
      return "bool";
    default:
      return "unknown";
  }
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
    const size_t width = dmf_dtype_size(dtype);
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
  const size_t width = dmf_dtype_size(dtype);
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

DmfNamedTensor tensor_from_json(
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
  return dmf_make_tensor(
      name.c_str(),
      id,
      shape_from_json(spec, id, bytes.size(), name),
      bytes.data(),
      bytes.size());
}

std::vector<DmfNamedTensor> inputs_from_json(
    const json& document,
    const std::filesystem::path& base_dir) {
  const json* inputs = &document;
  if (document.contains("inputs")) {
    inputs = &document.at("inputs");
  }
  if (!inputs->is_object()) {
    throw std::runtime_error("Input JSON must contain an object of tensors.");
  }
  std::vector<DmfNamedTensor> tensors;
  for (const auto& item : inputs->items()) {
    tensors.push_back(tensor_from_json(item.key(), item.value(), base_dir));
  }
  return tensors;
}

void free_inputs(std::vector<DmfNamedTensor>& tensors) {
  for (auto& tensor : tensors) {
    std::free(tensor.name);
    std::free(tensor.tensor.shape);
    std::free(tensor.tensor.data);
  }
  tensors.clear();
}

json parse_json_string(char* raw) {
  if (raw == nullptr) {
    return json::object();
  }
  const std::string text(raw);
  dmf_runtime_free_string(raw);
  if (text.empty()) {
    return json::object();
  }
  return json::parse(text);
}

json tensor_values(const DmfNativeTensor& tensor) {
  const auto count = static_cast<size_t>(
      tensor.byte_length / static_cast<intptr_t>(dmf_dtype_size(tensor.dtype)));
  json values = json::array();
  switch (tensor.dtype) {
    case DMF_DTYPE_FLOAT32: {
      const auto* data = static_cast<const float*>(tensor.data);
      for (size_t i = 0; i < count; ++i) {
        values.push_back(data[i]);
      }
      break;
    }
    case DMF_DTYPE_INT32: {
      const auto* data = static_cast<const int32_t*>(tensor.data);
      for (size_t i = 0; i < count; ++i) {
        values.push_back(data[i]);
      }
      break;
    }
    case DMF_DTYPE_INT64: {
      const auto* data = static_cast<const int64_t*>(tensor.data);
      for (size_t i = 0; i < count; ++i) {
        values.push_back(data[i]);
      }
      break;
    }
    case DMF_DTYPE_UINT8: {
      const auto* data = static_cast<const uint8_t*>(tensor.data);
      for (size_t i = 0; i < count; ++i) {
        values.push_back(data[i]);
      }
      break;
    }
    case DMF_DTYPE_FLOAT64: {
      const auto* data = static_cast<const double*>(tensor.data);
      for (size_t i = 0; i < count; ++i) {
        values.push_back(data[i]);
      }
      break;
    }
    case DMF_DTYPE_FLOAT16: {
      const auto* data = static_cast<const uint16_t*>(tensor.data);
      for (size_t i = 0; i < count; ++i) {
        values.push_back(data[i]);
      }
      break;
    }
    case DMF_DTYPE_BOOL: {
      const auto* data = static_cast<const uint8_t*>(tensor.data);
      for (size_t i = 0; i < count; ++i) {
        values.push_back(data[i] != 0);
      }
      break;
    }
  }
  return values;
}

json tensor_shape(const DmfNativeTensor& tensor) {
  json shape = json::array();
  for (int i = 0; i < tensor.rank; ++i) {
    shape.push_back(tensor.shape[i]);
  }
  return shape;
}

json correctness(DmfNamedTensor* outputs, intptr_t output_count) {
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

json runtime_options(const Args& args) {
  json backend_options = json::object();
  if (!args.option("provider").empty()) {
    backend_options["provider"] = args.option("provider");
  }
  if (!args.option("delegate").empty()) {
    backend_options["delegate"] = args.option("delegate");
  }
  if (!args.option("coreml-mode").empty()) {
    backend_options["coremlMode"] = args.option("coreml-mode");
  }
  if (!args.option("litert-section-index").empty()) {
    backend_options["litertSectionIndex"] =
        std::stoi(args.option("litert-section-index"));
  }
  if (args.has_flag("require-provider")) {
    backend_options["requireProvider"] = true;
  }
  if (args.has_flag("require-delegate")) {
    backend_options["requireDelegate"] = true;
  }
  json options = {
      {"diagnostics", true},
      {"backendOptions", backend_options},
  };
  if (!args.option("num-threads").empty()) {
    options["numThreads"] = std::stoi(args.option("num-threads"));
  }
  return options;
}

double millis_since(
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

uint64_t peak_memory_value(const json& memory) {
  if (memory.contains("peak_memory_bytes")) {
    return memory.value("peak_memory_bytes", 0ULL);
  }
  if (memory.contains("vm_hwm")) {
    return memory.value("vm_hwm", 0ULL);
  }
  if (memory.contains("resident_size")) {
    return memory.value("resident_size", 0ULL);
  }
  return 0;
}

void update_peak_memory(uint64_t* peak, const json& memory) {
  *peak = std::max(*peak, peak_memory_value(memory));
}

void usage() {
  std::cout
      << "Usage:\n"
      << "  dart_mlx_ffi_runtime_runner --model-id <id> "
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
    DmfRuntimeSession* session = dmf_runtime_create(
        engine_id(engine),
        artifact.c_str(),
        runtime_options(args).dump().c_str(),
        &error);
    if (session == nullptr) {
      const std::string message = error == nullptr ? "runtime create failed" : error;
      dmf_runtime_free_string(error);
      throw std::runtime_error(message);
    }

    auto inputs = inputs_from_json(
        read_json_file(input_path),
        std::filesystem::path(input_path).parent_path());
    DmfNamedTensor* outputs = nullptr;
    intptr_t output_count = 0;
    const json memory_before = parse_json_string(dmf_runtime_memory_info_json());
    uint64_t peak_memory = peak_memory_value(memory_before);

    for (int i = 0; i < warmup; ++i) {
      if (outputs != nullptr) {
        dmf_runtime_free_tensors(outputs, output_count);
        outputs = nullptr;
        output_count = 0;
      }
      if (dmf_runtime_run(
              session,
              inputs.data(),
              static_cast<intptr_t>(inputs.size()),
              &outputs,
              &output_count,
              &error) != 0) {
        const std::string message = error == nullptr ? "runtime run failed" : error;
        dmf_runtime_free_string(error);
        throw std::runtime_error(message);
      }
      update_peak_memory(
          &peak_memory,
          parse_json_string(dmf_runtime_memory_info_json()));
    }

    double timed_ms = 0.0;
    for (int i = 0; i < iters; ++i) {
      if (outputs != nullptr) {
        dmf_runtime_free_tensors(outputs, output_count);
        outputs = nullptr;
        output_count = 0;
      }
      const auto start = std::chrono::steady_clock::now();
      if (dmf_runtime_run(
              session,
              inputs.data(),
              static_cast<intptr_t>(inputs.size()),
              &outputs,
              &output_count,
              &error) != 0) {
        const std::string message = error == nullptr ? "runtime run failed" : error;
        dmf_runtime_free_string(error);
        throw std::runtime_error(message);
      }
      const auto end = std::chrono::steady_clock::now();
      timed_ms += millis_since(start, end);
      update_peak_memory(
          &peak_memory,
          parse_json_string(dmf_runtime_memory_info_json()));
    }

    const json memory_after = parse_json_string(dmf_runtime_memory_info_json());
    update_peak_memory(&peak_memory, memory_after);
    const json diagnostics = parse_json_string(dmf_runtime_diagnostics_json(session));
    const double per_iter_ms = iters > 0 ? timed_ms / iters : 0.0;

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
                {"runtime", "dart_mlx_ffi_native_cli"},
                {"runtime_diagnostics", diagnostics},
                {"memory_before", memory_before},
                {"memory_after", memory_after},
                {"raw_peak_memory_field",
                 memory_after.contains("android_peak_pss")
                     ? "android_peak_pss"
                     : (memory_after.contains("vm_hwm")
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
      dmf_runtime_free_tensors(outputs, output_count);
    }
    free_inputs(inputs);
    dmf_runtime_free(session);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "dart_mlx_ffi_runtime_runner failed: " << error.what() << "\n";
    return 2;
  }
}
