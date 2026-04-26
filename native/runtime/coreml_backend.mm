#include "runtime_bridge.h"
#include "options.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace {

NSString* ns_string(const char* value) {
  return [NSString stringWithUTF8String:value == nullptr ? "" : value];
}

std::string basename(const std::string& path) {
  return std::filesystem::path(path).filename().string();
}

MLComputeUnits compute_units_from_options(const DinfOptions* runtime_options) {
  std::string requested =
      dinf_option_string(runtime_options, "computeUnits",
          dinf_option_string(runtime_options, "coremlComputeUnits"));
  std::transform(requested.begin(), requested.end(), requested.begin(),
                 [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  if (requested.empty()) {
    if (dinf_options_contains_token(runtime_options, "ane")) {
      return MLComputeUnitsCPUAndNeuralEngine;
    }
    if (dinf_options_contains_token(runtime_options, "gpu")) {
      return MLComputeUnitsCPUAndGPU;
    }
    if (dinf_options_contains_token(runtime_options, "cpu")) {
      return MLComputeUnitsCPUOnly;
    }
    return MLComputeUnitsCPUAndNeuralEngine;
  }
  if (requested == "cpuonly" || requested == "cpu") {
    return MLComputeUnitsCPUOnly;
  }
  if (requested == "cpuandgpu" || requested == "gpu") {
    return MLComputeUnitsCPUAndGPU;
  }
  if (requested == "all") {
    return MLComputeUnitsAll;
  }
  if (requested == "cpuandneuralengine" || requested == "ane") {
    return MLComputeUnitsCPUAndNeuralEngine;
  }
  return MLComputeUnitsCPUAndNeuralEngine;
}

std::string compute_units_name(MLComputeUnits units) {
  switch (units) {
    case MLComputeUnitsCPUOnly:
      return "cpuOnly";
    case MLComputeUnitsCPUAndGPU:
      return "cpuAndGPU";
    case MLComputeUnitsAll:
      return "all";
    case MLComputeUnitsCPUAndNeuralEngine:
      return "cpuAndNeuralEngine";
    default:
      return "unknown";
  }
}

MLMultiArrayDataType coreml_dtype(int32_t dtype, std::string* error) {
  switch (dtype) {
    case DINF_DTYPE_FLOAT32:
      return MLMultiArrayDataTypeFloat32;
    case DINF_DTYPE_FLOAT64:
      return MLMultiArrayDataTypeDouble;
    case DINF_DTYPE_INT32:
      return MLMultiArrayDataTypeInt32;
    case DINF_DTYPE_FLOAT16:
      return MLMultiArrayDataTypeFloat16;
    default:
      if (error != nullptr) {
        *error = "Core ML supports float16, float32, float64, and int32 inputs";
      }
      return MLMultiArrayDataTypeFloat32;
  }
}

int32_t dinf_dtype(MLMultiArrayDataType dtype) {
  switch (dtype) {
    case MLMultiArrayDataTypeFloat32:
      return DINF_DTYPE_FLOAT32;
    case MLMultiArrayDataTypeDouble:
      return DINF_DTYPE_FLOAT64;
    case MLMultiArrayDataTypeInt32:
      return DINF_DTYPE_INT32;
    case MLMultiArrayDataTypeFloat16:
      return DINF_DTYPE_FLOAT16;
    default:
      return 0;
  }
}

size_t array_byte_length(MLMultiArray* array) {
  const size_t dtype_size = dinf_dtype_size(dinf_dtype(array.dataType));
  if (dtype_size == 0) {
    return 0;
  }
  size_t count = 1;
  for (NSNumber* dim in array.shape) {
    count *= static_cast<size_t>(dim.longLongValue);
  }
  return count * dtype_size;
}

MLMultiArray* make_multi_array(const DinfNamedTensor& input, std::string* error) {
  NSMutableArray<NSNumber*>* shape = [NSMutableArray array];
  for (int32_t i = 0; i < input.tensor.rank; ++i) {
    [shape addObject:@(input.tensor.shape[i])];
  }
  NSError* ns_error = nil;
  const auto data_type = coreml_dtype(input.tensor.dtype, error);
  if (error != nullptr && !error->empty()) {
    return nil;
  }
  MLMultiArray* array =
      [[MLMultiArray alloc] initWithShape:shape dataType:data_type error:&ns_error];
  if (array == nil) {
    if (error != nullptr) {
      *error = ns_error.localizedDescription.UTF8String;
    }
    return nil;
  }
  const size_t expected = array_byte_length(array);
  if (expected != static_cast<size_t>(input.tensor.byte_length)) {
    if (error != nullptr) {
      *error = "Core ML input byte length does not match tensor shape";
    }
    return nil;
  }
  std::memcpy(array.dataPointer, input.tensor.data, expected);
  return array;
}

std::vector<int64_t> shape_of(MLMultiArray* array) {
  std::vector<int64_t> shape;
  for (NSNumber* dim in array.shape) {
    shape.push_back(dim.longLongValue);
  }
  return shape;
}

bool is_coreml_model_path(const std::string& path) {
  return (path.size() >= 9 && path.rfind(".mlmodelc") == path.size() - 9) ||
         (path.size() >= 10 && path.rfind(".mlpackage") == path.size() - 10);
}

bool is_coreml_package_path(const std::string& path) {
  return path.size() >= 10 && path.rfind(".mlpackage") == path.size() - 10;
}

std::string loadable_coreml_model_path(
    const std::string& path,
    std::string* error) {
  if (!is_coreml_package_path(path)) {
    return path;
  }
  NSError* ns_error = nil;
  NSURL* url = [NSURL fileURLWithPath:ns_string(path.c_str())];
  NSURL* compiled_url = [MLModel compileModelAtURL:url error:&ns_error];
  if (compiled_url == nil) {
    if (error != nullptr) {
      *error = ns_error == nil ? "Core ML package compilation failed"
                               : ns_error.localizedDescription.UTF8String;
    }
    return "";
  }
  return compiled_url.path.UTF8String;
}

bool has_prefix(const std::string& value, const std::string& prefix) {
  return value.rfind(prefix, 0) == 0;
}

std::string strip_coreml_extension(const std::string& name) {
  if (name.size() >= 9 && name.rfind(".mlmodelc") == name.size() - 9) {
    return name.substr(0, name.size() - 9);
  }
  if (name.size() >= 10 && name.rfind(".mlpackage") == name.size() - 10) {
    return name.substr(0, name.size() - 10);
  }
  return name;
}

int chunk_index(const std::string& name, const std::string& prefix) {
  const std::string stem = strip_coreml_extension(name);
  if (prefix == "chunk" && stem == "chunk_head") {
    return 1000000;
  }
  if (!has_prefix(stem, prefix)) {
    return -1;
  }
  size_t cursor = prefix.size();
  if (cursor < stem.size() && stem[cursor] == '_') {
    ++cursor;
  }
  if (cursor >= stem.size() ||
      !std::isdigit(static_cast<unsigned char>(stem[cursor]))) {
    return -1;
  }
  int value = 0;
  while (cursor < stem.size() &&
         std::isdigit(static_cast<unsigned char>(stem[cursor]))) {
    value = value * 10 + (stem[cursor] - '0');
    ++cursor;
  }
  return cursor == stem.size() ? value : -1;
}

std::vector<std::string> discover_chunk_paths(
    const std::string& root,
    const std::string& prefix) {
  std::vector<std::string> paths;
  if (!std::filesystem::is_directory(root)) {
    return paths;
  }
  for (const auto& entry : std::filesystem::directory_iterator(root)) {
    if (!entry.is_directory()) {
      continue;
    }
    const std::string path = entry.path().string();
    const std::string name = basename(path);
    if (is_coreml_model_path(path) && chunk_index(name, prefix) >= 0) {
      paths.push_back(path);
    }
  }
  std::sort(paths.begin(), paths.end(), [&](const auto& left, const auto& right) {
    return chunk_index(basename(left), prefix) < chunk_index(basename(right), prefix);
  });
  return paths;
}

std::vector<std::string> discover_model_paths(
    const std::string& model_path,
    const DinfOptions* runtime_options,
    std::string* layout,
    std::string* mode) {
  if (is_coreml_model_path(model_path)) {
    *layout = "monolithic";
    *mode = "single";
    return {model_path};
  }
  if (!std::filesystem::is_directory(model_path)) {
    return {model_path};
  }

  const std::string requested_mode =
      dinf_option_string(runtime_options, "coremlMode",
          dinf_option_string(runtime_options, "mode", "decode"));
  const bool wants_prefill = requested_mode == "prefill";
  std::vector<std::string> chunks = wants_prefill
      ? discover_chunk_paths(model_path, "prefill_chunk")
      : discover_chunk_paths(model_path, "chunk");
  if (chunks.empty() && wants_prefill) {
    chunks = discover_chunk_paths(model_path, "chunk");
  }
  if (!chunks.empty()) {
    *layout = "chunked";
    *mode = wants_prefill ? "prefill" : "decode";
    return chunks;
  }

  for (const auto& candidate : {"model.mlmodelc", "model.mlpackage"}) {
    const auto path = (std::filesystem::path(model_path) / candidate).string();
    if (std::filesystem::exists(path)) {
      *layout = "monolithic";
      *mode = "single";
      return {path};
    }
  }

  *layout = "directory";
  *mode = requested_mode;
  return {model_path};
}

MLDictionaryFeatureProvider* provider_from_dict(
    NSMutableDictionary<NSString*, MLFeatureValue*>* dict,
    std::string* error) {
  NSError* ns_error = nil;
  MLDictionaryFeatureProvider* provider =
      [[MLDictionaryFeatureProvider alloc] initWithDictionary:dict
                                                        error:&ns_error];
  if (provider == nil && error != nullptr) {
    *error = ns_error.localizedDescription.UTF8String;
  }
  return provider;
}

void merge_prediction(
    id<MLFeatureProvider> prediction,
    NSMutableDictionary<NSString*, MLFeatureValue*>* dict) {
  for (NSString* name in prediction.featureNames) {
    MLFeatureValue* value = [prediction featureValueForName:name];
    if (value != nil) {
      dict[name] = value;
    }
  }
}

using StringMap = std::map<std::string, std::string>;

StringMap string_map_from_json(const json& value) {
  StringMap result;
  if (!value.is_object()) {
    return result;
  }
  for (const auto& item : value.items()) {
    if (item.value().is_string()) {
      result[item.key()] = item.value().get<std::string>();
    }
  }
  return result;
}

std::filesystem::path resolve_pipeline_path(
    const std::filesystem::path& spec_path,
    const std::string& value) {
  std::filesystem::path path(value);
  if (path.is_absolute()) {
    return path;
  }
  return spec_path.parent_path() / path;
}

bool is_pipeline_spec(const std::filesystem::path& path, json* spec) {
  if (path.extension() != ".json") {
    return false;
  }
  std::ifstream file(path);
  if (!file) {
    return false;
  }
  json parsed;
  file >> parsed;
  if (!parsed.is_object() || !parsed.contains("stages")) {
    return false;
  }
  const std::string format = parsed.value("format", "");
  if (!format.empty() && format != "dart_inference.coreml_pipeline.v1") {
    return false;
  }
  *spec = std::move(parsed);
  return true;
}

StringMap requested_outputs_from_json(const json& spec) {
  if (!spec.contains("outputs")) {
    return {};
  }
  const auto& outputs = spec.at("outputs");
  if (outputs.is_object()) {
    return string_map_from_json(outputs);
  }
  StringMap result;
  if (outputs.is_array()) {
    for (const auto& item : outputs) {
      if (item.is_string()) {
        const std::string name = item.get<std::string>();
        result[name] = name;
      }
    }
  }
  return result;
}

std::string device_name(id<MLComputeDeviceProtocol> device) {
  if (device == nil) {
    return "unknown";
  }
  if ([device isKindOfClass:NSClassFromString(@"MLNeuralEngineComputeDevice")]) {
    return "ane";
  }
  if ([device isKindOfClass:NSClassFromString(@"MLGPUComputeDevice")]) {
    return "gpu";
  }
  if ([device isKindOfClass:NSClassFromString(@"MLCPUComputeDevice")]) {
    return "cpu";
  }
  return NSStringFromClass([device class]).UTF8String;
}

void add_count(std::map<std::string, int>* counts, const std::string& key) {
  (*counts)[key] += 1;
}

void add_usage(
    MLComputePlanDeviceUsage* usage,
    std::map<std::string, int>* preferred,
    std::map<std::string, int>* supported) API_AVAILABLE(macos(14.4), ios(17.4)) {
  if (usage == nil) {
    add_count(preferred, "unknown");
    return;
  }
  add_count(preferred, device_name(usage.preferredComputeDevice));
  for (id<MLComputeDeviceProtocol> device in usage.supportedComputeDevices) {
    add_count(supported, device_name(device));
  }
}

void audit_block(
    MLComputePlan* plan,
    MLModelStructureProgramBlock* block,
    std::map<std::string, int>* preferred,
    std::map<std::string, int>* supported,
    std::map<std::string, int>* operators,
    int* operation_count) API_AVAILABLE(macos(14.4), ios(17.4)) {
  if (block == nil) {
    return;
  }
  for (MLModelStructureProgramOperation* operation in block.operations) {
    *operation_count += 1;
    add_count(operators, operation.operatorName.UTF8String);
    add_usage(
        [plan computeDeviceUsageForMLProgramOperation:operation],
        preferred,
        supported);
    for (MLModelStructureProgramBlock* nested in operation.blocks) {
      audit_block(plan, nested, preferred, supported, operators, operation_count);
    }
  }
}

std::string json_counts(const std::map<std::string, int>& counts) {
  std::string out = "{";
  bool first = true;
  for (const auto& entry : counts) {
    if (!first) {
      out += ",";
    }
    first = false;
    out += "\"";
    out += dinf_json_escape(entry.first);
    out += "\":";
    out += std::to_string(entry.second);
  }
  out += "}";
  return out;
}

void append_unique(std::vector<std::string>* values, const std::string& value) {
  if (std::find(values->begin(), values->end(), value) == values->end()) {
    values->push_back(value);
  }
}

void append_feature_names(
    MLModel* model,
    bool inputs,
    std::vector<std::string>* values) {
  NSDictionary<NSString*, MLFeatureDescription*>* descriptions = inputs
      ? model.modelDescription.inputDescriptionsByName
      : model.modelDescription.outputDescriptionsByName;
  for (NSString* name in descriptions.allKeys) {
    append_unique(values, name.UTF8String);
  }
  std::sort(values->begin(), values->end());
}

struct ComputePlanAudit {
  int audited_models = 0;
  int program_operations = 0;
  int neural_network_layers = 0;
  int error_count = 0;
  std::vector<std::string> errors;
  std::map<std::string, int> preferred;
  std::map<std::string, int> supported;
  std::map<std::string, int> operators;
};

MLComputePlan* load_compute_plan(
    const std::string& path,
    MLModelConfiguration* config,
    std::string* error) API_AVAILABLE(macos(14.4), ios(17.4)) {
  __block MLComputePlan* out = nil;
  __block NSError* out_error = nil;
  dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
  NSURL* url = [NSURL fileURLWithPath:ns_string(path.c_str())];
  [MLComputePlan loadContentsOfURL:url
                     configuration:config
                 completionHandler:^(MLComputePlan* plan, NSError* ns_error) {
                   out = plan;
                   out_error = ns_error;
                   dispatch_semaphore_signal(semaphore);
                 }];
  dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
  if (out == nil && error != nullptr) {
    *error = out_error == nil ? "MLComputePlan load failed"
                              : out_error.localizedDescription.UTF8String;
  }
  return out;
}

void audit_compute_plan(
    const std::string& path,
    MLModelConfiguration* config,
    ComputePlanAudit* audit) API_AVAILABLE(macos(14.4), ios(17.4)) {
  std::string error;
  MLComputePlan* plan = load_compute_plan(path, config, &error);
  if (plan == nil) {
    audit->error_count += 1;
    audit->errors.push_back(basename(path) + ": " + error);
    return;
  }
  audit->audited_models += 1;
  MLModelStructure* structure = plan.modelStructure;
  if (structure.program != nil) {
    for (MLModelStructureProgramFunction* function in structure.program.functions.allValues) {
      audit_block(
          plan,
          function.block,
          &audit->preferred,
          &audit->supported,
          &audit->operators,
          &audit->program_operations);
    }
  }
  if (structure.neuralNetwork != nil) {
    for (MLModelStructureNeuralNetworkLayer* layer in structure.neuralNetwork.layers) {
      audit->neural_network_layers += 1;
      add_usage(
          [plan computeDeviceUsageForNeuralNetworkLayer:layer],
          &audit->preferred,
          &audit->supported);
    }
  }
}

std::string compute_plan_audit_json(
    const std::vector<std::string>& paths,
    MLModelConfiguration* config,
    const DinfOptions* runtime_options) {
  const bool enabled = dinf_option_bool(
      runtime_options,
      "computePlanAudit",
      dinf_option_bool(runtime_options, "diagnostics", false));
  if (!enabled) {
    return "{\"enabled\":false}";
  }
  if (@available(macOS 14.4, iOS 17.4, *)) {
    ComputePlanAudit audit;
    for (const auto& path : paths) {
      audit_compute_plan(path, config, &audit);
    }
    std::string out = "{\"enabled\":true,\"available\":true";
    out += ",\"audited_models\":";
    out += std::to_string(audit.audited_models);
    out += ",\"program_operations\":";
    out += std::to_string(audit.program_operations);
    out += ",\"neural_network_layers\":";
    out += std::to_string(audit.neural_network_layers);
    out += ",\"error_count\":";
    out += std::to_string(audit.error_count);
    out += ",\"preferred_device_counts\":";
    out += json_counts(audit.preferred);
    out += ",\"supported_device_counts\":";
    out += json_counts(audit.supported);
    out += ",\"operator_counts\":";
    out += json_counts(audit.operators);
    out += ",\"errors\":";
    out += dinf_json_string_array(audit.errors);
    out += "}";
    return out;
  }
  return "{\"enabled\":true,\"available\":false,"
         "\"reason\":\"MLComputePlan requires macOS 14.4 or iOS 17.4\"}";
}

struct CoreMlStage {
  std::string name;
  std::string op;
  std::string path;
  MLModel* model = nil;
  StringMap inputs;
  StringMap outputs;
};

std::string mapped_name(
    const StringMap& values,
    const std::string& name,
    const std::string& fallback) {
  const auto found = values.find(name);
  return found == values.end() ? fallback : found->second;
}

MLMultiArray* feature_array(
    NSMutableDictionary<NSString*, MLFeatureValue*>* dict,
    const std::string& name,
    std::string* error) {
  MLFeatureValue* value = dict[ns_string(name.c_str())];
  if (value == nil || value.multiArrayValue == nil) {
    *error = "Core ML pipeline missing tensor '" + name + "'";
    return nil;
  }
  return value.multiArrayValue;
}

int64_t multi_array_index_value(MLMultiArray* array, size_t index) {
  if (array.dataType == MLMultiArrayDataTypeInt32) {
    return static_cast<const int32_t*>(array.dataPointer)[index];
  }
  if (array.dataType == MLMultiArrayDataTypeDouble) {
    return static_cast<int64_t>(
        static_cast<const double*>(array.dataPointer)[index]);
  }
  if (array.dataType == MLMultiArrayDataTypeFloat32) {
    return static_cast<int64_t>(
        static_cast<const float*>(array.dataPointer)[index]);
  }
  return -1;
}

bool scatter_embeddings(
    const CoreMlStage& stage,
    NSMutableDictionary<NSString*, MLFeatureValue*>* dict,
    NSMutableDictionary<NSString*, MLFeatureValue*>* last_outputs,
    std::string* error) {
  const std::string base_key =
      mapped_name(stage.inputs, "base", mapped_name(stage.inputs, "inputs_embeds", "inputs_embeds"));
  const std::string updates_key =
      mapped_name(stage.inputs, "updates", mapped_name(stage.inputs, "image_embeds", "image_embeds"));
  const std::string indices_key =
      mapped_name(stage.inputs, "indices", "image_token_indices");
  MLMultiArray* base = feature_array(dict, base_key, error);
  MLMultiArray* updates = feature_array(dict, updates_key, error);
  MLMultiArray* indices = feature_array(dict, indices_key, error);
  if (base == nil || updates == nil || indices == nil) {
    return false;
  }
  if (base.dataType != updates.dataType) {
    *error = "scatter_embeddings requires base and updates to share dtype";
    return false;
  }
  if (base.shape.count < 2 || updates.shape.count != 2) {
    *error = "scatter_embeddings expects base rank >= 2 and updates rank 2";
    return false;
  }
  if (indices.dataType != MLMultiArrayDataTypeInt32) {
    *error = "scatter_embeddings indices must be int32 for Core ML";
    return false;
  }
  const int64_t sequence = base.shape[base.shape.count - 2].longLongValue;
  const int64_t hidden = base.shape[base.shape.count - 1].longLongValue;
  int64_t leading = 1;
  for (NSUInteger i = 0; i + 2 < base.shape.count; ++i) {
    leading *= base.shape[i].longLongValue;
  }
  if (leading != 1) {
    *error = "scatter_embeddings currently supports a single leading batch";
    return false;
  }
  if (sequence <= 0 || hidden <= 0) {
    *error = "scatter_embeddings base shape must have positive sequence and hidden dims";
    return false;
  }
  if (updates.shape[1].longLongValue != hidden) {
    *error = "scatter_embeddings update hidden size does not match base";
    return false;
  }
  const size_t index_count =
      array_byte_length(indices) / dinf_dtype_size(dinf_dtype(indices.dataType));
  if (index_count != static_cast<size_t>(updates.shape[0].longLongValue)) {
    *error = "scatter_embeddings update count does not match index count";
    return false;
  }
  NSError* ns_error = nil;
  MLMultiArray* merged =
      [[MLMultiArray alloc] initWithShape:base.shape
                                 dataType:base.dataType
                                    error:&ns_error];
  if (merged == nil) {
    *error = ns_error.localizedDescription.UTF8String;
    return false;
  }
  const size_t total_bytes = array_byte_length(base);
  const size_t row_bytes = static_cast<size_t>(hidden) *
                           dinf_dtype_size(dinf_dtype(base.dataType));
  std::memcpy(merged.dataPointer, base.dataPointer, total_bytes);
  const auto* update_bytes = static_cast<const uint8_t*>(updates.dataPointer);
  auto* merged_bytes = static_cast<uint8_t*>(merged.dataPointer);
  for (size_t i = 0; i < index_count; ++i) {
    const int64_t raw_index = multi_array_index_value(indices, i);
    if (raw_index < 0 || raw_index >= sequence) {
      *error = "scatter_embeddings index is out of sequence bounds";
      return false;
    }
    const size_t offset = static_cast<size_t>(raw_index) * row_bytes;
    if (offset + row_bytes > total_bytes) {
      *error = "scatter_embeddings computed offset is out of bounds";
      return false;
    }
    std::memcpy(merged_bytes + offset, update_bytes + i * row_bytes, row_bytes);
  }
  const std::string output_name =
      mapped_name(stage.outputs, "output", "inputs_embeds");
  MLFeatureValue* value = [MLFeatureValue featureValueWithMultiArray:merged];
  NSString* name = ns_string(output_name.c_str());
  dict[name] = value;
  last_outputs[name] = value;
  return true;
}

class CoreMlSession final : public DinfRuntimeSession {
 public:
  CoreMlSession(
      std::vector<CoreMlStage> stages,
      std::string compute_units,
      std::string layout,
      std::string mode,
      std::vector<std::string> input_names,
      std::vector<std::string> output_names,
      std::string compute_plan_audit,
      StringMap requested_outputs = {})
      : stages_(std::move(stages)),
        compute_units_(std::move(compute_units)),
        layout_(std::move(layout)),
        mode_(std::move(mode)),
        input_names_(std::move(input_names)),
        output_names_(std::move(output_names)),
        compute_plan_audit_(std::move(compute_plan_audit)),
        requested_outputs_(std::move(requested_outputs)) {}

  int Run(
      const DinfNamedTensor* inputs,
      size_t input_count,
      DinfNamedTensor** outputs,
      size_t* output_count,
      std::string* error) override {
    NSMutableDictionary<NSString*, MLFeatureValue*>* dict =
        [NSMutableDictionary dictionary];
    for (size_t i = 0; i < input_count; ++i) {
      MLMultiArray* array = make_multi_array(inputs[i], error);
      if (array == nil) {
        return 1;
      }
      NSString* name = ns_string(inputs[i].name);
      dict[name] = [MLFeatureValue featureValueWithMultiArray:array];
    }

    NSMutableDictionary<NSString*, MLFeatureValue*>* last_outputs =
        [NSMutableDictionary dictionary];
    bool produced_outputs = false;
    for (const CoreMlStage& stage : stages_) {
      if (!stage.op.empty()) {
        [last_outputs removeAllObjects];
        if (stage.op == "scatter_embeddings") {
          if (!scatter_embeddings(stage, dict, last_outputs, error)) {
            return 1;
          }
          produced_outputs = true;
          continue;
        }
        *error = "Unsupported Core ML pipeline op: " + stage.op;
        return 1;
      }
      NSMutableDictionary<NSString*, MLFeatureValue*>* stage_dict =
          [NSMutableDictionary dictionary];
      NSDictionary<NSString*, MLFeatureDescription*>* descriptions =
          stage.model.modelDescription.inputDescriptionsByName;
      for (NSString* input_name in descriptions.allKeys) {
        const std::string input_key = input_name.UTF8String;
        const auto alias = stage.inputs.find(input_key);
        const std::string source =
            alias == stage.inputs.end() ? input_key : alias->second;
        MLFeatureValue* value = dict[ns_string(source.c_str())];
        if (value == nil) {
          *error = "Core ML pipeline stage '" + stage.name +
                   "' is missing input '" + source + "'";
          return 1;
        }
        stage_dict[input_name] = value;
      }
      MLDictionaryFeatureProvider* provider = provider_from_dict(stage_dict, error);
      if (provider == nil) {
        return 1;
      }
      NSError* ns_error = nil;
      id<MLFeatureProvider> prediction =
          [stage.model predictionFromFeatures:provider error:&ns_error];
      if (prediction == nil) {
        *error = ns_error.localizedDescription.UTF8String;
        return 1;
      }
      produced_outputs = true;
      [last_outputs removeAllObjects];
      for (NSString* name in prediction.featureNames) {
        MLFeatureValue* value = [prediction featureValueForName:name];
        if (value == nil) {
          continue;
        }
        const std::string output_name = name.UTF8String;
        const auto alias = stage.outputs.find(output_name);
        const std::string key =
            alias == stage.outputs.end() ? output_name : alias->second;
        NSString* mapped = ns_string(key.c_str());
        dict[mapped] = value;
        last_outputs[mapped] = value;
      }
    }
    if (!produced_outputs || last_outputs.count == 0) {
      *error = "Core ML session produced no outputs";
      return 1;
    }

    NSMutableDictionary<NSString*, MLFeatureValue*>* final_outputs =
        [NSMutableDictionary dictionary];
    if (requested_outputs_.empty()) {
      [final_outputs addEntriesFromDictionary:last_outputs];
    } else {
      for (const auto& item : requested_outputs_) {
        MLFeatureValue* value = dict[ns_string(item.second.c_str())];
        if (value == nil) {
          *error = "Core ML pipeline requested missing output '" + item.second + "'";
          return 1;
        }
        final_outputs[ns_string(item.first.c_str())] = value;
      }
    }
    NSMutableArray<NSString*>* names =
        [NSMutableArray arrayWithArray:final_outputs.allKeys];
    [names sortUsingSelector:@selector(compare:)];
    std::vector<DinfNamedTensor> produced;
    for (NSString* name in names) {
      MLFeatureValue* value = final_outputs[name];
      MLMultiArray* array = value.multiArrayValue;
      if (array == nil) {
        continue;
      }
      const int32_t dtype = dinf_dtype(array.dataType);
      if (dtype == 0) {
        continue;
      }
      produced.push_back(dinf_make_tensor(
          name.UTF8String,
          dtype,
          shape_of(array),
          array.dataPointer,
          array_byte_length(array)));
    }
    *output_count = produced.size();
    *outputs = static_cast<DinfNamedTensor*>(
        std::malloc(sizeof(DinfNamedTensor) * produced.size()));
    if (!produced.empty()) {
      std::memcpy(*outputs, produced.data(), sizeof(DinfNamedTensor) * produced.size());
    }
    return 0;
  }

  std::string DiagnosticsJson() const override {
    return std::string("{\"engine\":\"coreml\",\"compute_units\":\"") +
           dinf_json_escape(compute_units_) + "\",\"layout\":\"" +
           dinf_json_escape(layout_) + "\",\"mode\":\"" +
           dinf_json_escape(mode_) + "\",\"loaded_models\":" +
           std::to_string(loaded_model_count()) +
           ",\"stage_count\":" + std::to_string(stages_.size()) +
           ",\"input_names\":" + dinf_json_string_array(input_names_) +
           ",\"output_names\":" + dinf_json_string_array(output_names_) +
           ",\"stages\":" + stages_json() +
           ",\"compute_plan\":" +
           compute_plan_audit_ + "}";
  }

 private:
  std::string stages_json() const {
    std::string out = "[";
    for (size_t i = 0; i < stages_.size(); ++i) {
      if (i > 0) {
        out += ",";
      }
      out += "{\"name\":\"" + dinf_json_escape(stages_[i].name) +
             "\"";
      if (!stages_[i].op.empty()) {
        out += ",\"op\":\"" + dinf_json_escape(stages_[i].op) + "\"";
      } else {
        out += ",\"model\":\"" + dinf_json_escape(stages_[i].path) + "\"";
      }
      out += "}";
    }
    out += "]";
    return out;
  }

  size_t loaded_model_count() const {
    size_t count = 0;
    for (const auto& stage : stages_) {
      if (stage.model != nil) {
        count += 1;
      }
    }
    return count;
  }

  std::vector<CoreMlStage> stages_;
  std::string compute_units_;
  std::string layout_;
  std::string mode_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::string compute_plan_audit_;
  StringMap requested_outputs_;
};

}  // namespace

DinfRuntimeSession* dinf_create_coreml_session(
    const char* model_path,
    const DinfOptions* runtime_options,
    std::string* error) {
  @autoreleasepool {
    MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
    config.computeUnits = compute_units_from_options(runtime_options);
    std::string layout;
    std::string mode;
    std::vector<std::string> paths;
    std::vector<CoreMlStage> stages;
    StringMap requested_outputs;
    json pipeline;
    const bool is_pipeline = is_pipeline_spec(model_path, &pipeline);
    if (is_pipeline) {
      layout = "pipeline";
      mode = "pipeline";
      if (!pipeline.at("stages").is_array()) {
        *error = "Core ML pipeline stages must be an array.";
        return nullptr;
      }
      for (const auto& item : pipeline.at("stages")) {
        if (!item.is_object()) {
          *error = "Core ML pipeline stage must be an object.";
          return nullptr;
        }
        CoreMlStage stage;
        stage.name = item.value(
            "name",
            std::string("stage_") + std::to_string(stages.size()));
        stage.inputs = string_map_from_json(item.value("inputs", json::object()));
        stage.outputs = string_map_from_json(item.value("outputs", json::object()));
        if (item.contains("op")) {
          if (item.contains("model")) {
            *error = "Core ML pipeline stage must contain either op or model, not both.";
            return nullptr;
          }
          if (!item.at("op").is_string()) {
            *error = "Core ML pipeline op stage must contain a string op.";
            return nullptr;
          }
          stage.op = item.at("op").get<std::string>();
        } else {
          if (!item.contains("model")) {
            *error = "Core ML pipeline model stage must contain a model path.";
            return nullptr;
          }
          if (!item.at("model").is_string()) {
            *error = "Core ML pipeline model stage must contain a string model path.";
            return nullptr;
          }
          const auto path = resolve_pipeline_path(
              model_path,
              item.at("model").get<std::string>());
          stage.path = path.string();
          paths.push_back(stage.path);
        }
        stages.push_back(std::move(stage));
      }
      requested_outputs = requested_outputs_from_json(pipeline);
    } else {
      paths = discover_model_paths(model_path, runtime_options, &layout, &mode);
      for (const auto& path : paths) {
        CoreMlStage stage;
        stage.name = basename(path);
        stage.path = path;
        stages.push_back(std::move(stage));
      }
    }
    paths.clear();
    for (CoreMlStage& stage : stages) {
      if (!stage.op.empty()) {
        continue;
      }
      const std::string path = loadable_coreml_model_path(stage.path, error);
      if (path.empty()) {
        return nullptr;
      }
      stage.path = path;
      paths.push_back(stage.path);
    }
    const std::string compute_plan_audit =
        compute_plan_audit_json(paths, config, runtime_options);
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    for (CoreMlStage& stage : stages) {
      if (!stage.op.empty()) {
        continue;
      }
      NSError* ns_error = nil;
      NSURL* url = [NSURL fileURLWithPath:ns_string(stage.path.c_str())];
      MLModel* model = [MLModel modelWithContentsOfURL:url
                                        configuration:config
                                                error:&ns_error];
      if (model == nil) {
        *error = ns_error.localizedDescription.UTF8String;
        return nullptr;
      }
      stage.model = model;
      append_feature_names(model, /*inputs=*/true, &input_names);
      append_feature_names(model, /*inputs=*/false, &output_names);
    }
    return new CoreMlSession(
        std::move(stages),
        compute_units_name(config.computeUnits),
        layout,
        mode,
        std::move(input_names),
        std::move(output_names),
        compute_plan_audit,
        std::move(requested_outputs));
  }
}
