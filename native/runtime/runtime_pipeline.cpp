#include "runtime_pipeline.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using json = nlohmann::json;
using StringMap = std::map<std::string, std::string>;
using TensorMap = std::map<std::string, const DinfNamedTensor*>;

struct OwnedTensorArray {
  DinfNamedTensor* values = nullptr;
  size_t count = 0;

  ~OwnedTensorArray() {
    dinf_cpp_free_tensors(values, static_cast<intptr_t>(count));
  }
};

struct PipelineStage {
  std::string name;
  std::string op;
  std::string model_path;
  StringMap inputs;
  StringMap outputs;
  std::unique_ptr<DinfRuntimeSession> session;
};

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

std::filesystem::path resolve_pipeline_path(
    const std::filesystem::path& spec_path,
    const std::string& value) {
  std::filesystem::path path(value);
  if (path.is_absolute()) {
    return path;
  }
  return spec_path.parent_path() / path;
}

std::vector<int64_t> tensor_shape(const DinfTensor& tensor) {
  std::vector<int64_t> shape;
  shape.reserve(tensor.rank);
  for (int32_t i = 0; i < tensor.rank; ++i) {
    shape.push_back(tensor.shape[i]);
  }
  return shape;
}

std::string mapped_name(
    const StringMap& values,
    const std::string& name,
    const std::string& fallback) {
  const auto found = values.find(name);
  return found == values.end() ? fallback : found->second;
}

const DinfNamedTensor* find_tensor(
    const TensorMap& tensors,
    const std::string& name,
    std::string* error) {
  const auto found = tensors.find(name);
  if (found == tensors.end()) {
    *error = "Runtime pipeline missing tensor '" + name + "'";
    return nullptr;
  }
  return found->second;
}

int64_t index_value(const DinfTensor& tensor, size_t index) {
  if (tensor.dtype == DINF_DTYPE_INT64) {
    return static_cast<const int64_t*>(tensor.data)[index];
  }
  if (tensor.dtype == DINF_DTYPE_INT32) {
    return static_cast<const int32_t*>(tensor.data)[index];
  }
  return -1;
}

OwnedTensorArray* scatter_embeddings(
    const PipelineStage& stage,
    const TensorMap& tensors,
    std::string* error) {
  const std::string base_key = mapped_name(
      stage.inputs,
      "base",
      mapped_name(stage.inputs, "inputs_embeds", "inputs_embeds"));
  const std::string updates_key = mapped_name(
      stage.inputs,
      "updates",
      mapped_name(stage.inputs, "image_embeds", "image_embeds"));
  const std::string indices_key =
      mapped_name(stage.inputs, "indices", "image_token_indices");
  const auto* base = find_tensor(tensors, base_key, error);
  const auto* updates = find_tensor(tensors, updates_key, error);
  const auto* indices = find_tensor(tensors, indices_key, error);
  if (base == nullptr || updates == nullptr || indices == nullptr) {
    return nullptr;
  }
  if (base->tensor.dtype != updates->tensor.dtype) {
    *error = "scatter_embeddings requires base and updates to share dtype";
    return nullptr;
  }
  if (base->tensor.rank < 2 || updates->tensor.rank != 2) {
    *error = "scatter_embeddings expects base rank >= 2 and updates rank 2";
    return nullptr;
  }
  if (indices->tensor.dtype != DINF_DTYPE_INT64 &&
      indices->tensor.dtype != DINF_DTYPE_INT32) {
    *error = "scatter_embeddings indices must be int32 or int64";
    return nullptr;
  }
  const int64_t sequence = base->tensor.shape[base->tensor.rank - 2];
  const int64_t hidden = base->tensor.shape[base->tensor.rank - 1];
  int64_t leading = 1;
  for (int32_t i = 0; i < base->tensor.rank - 2; ++i) {
    leading *= base->tensor.shape[i];
  }
  if (leading != 1) {
    *error = "scatter_embeddings currently supports a single leading batch";
    return nullptr;
  }
  if (sequence <= 0 || hidden <= 0) {
    *error = "scatter_embeddings base shape must have positive dims";
    return nullptr;
  }
  if (updates->tensor.shape[1] != hidden) {
    *error = "scatter_embeddings update hidden size does not match base";
    return nullptr;
  }
  const size_t index_count =
      static_cast<size_t>(indices->tensor.byte_length /
                          dinf_dtype_size(indices->tensor.dtype));
  if (index_count != static_cast<size_t>(updates->tensor.shape[0])) {
    *error = "scatter_embeddings update count does not match index count";
    return nullptr;
  }
  const size_t row_bytes =
      static_cast<size_t>(hidden) * dinf_dtype_size(base->tensor.dtype);
  const auto* update_bytes = static_cast<const uint8_t*>(updates->tensor.data);
  std::vector<uint8_t> merged(static_cast<size_t>(base->tensor.byte_length));
  std::memcpy(merged.data(), base->tensor.data, merged.size());
  for (size_t i = 0; i < index_count; ++i) {
    const int64_t raw_index = index_value(indices->tensor, i);
    if (raw_index < 0 || raw_index >= sequence) {
      *error = "scatter_embeddings index is out of sequence bounds";
      return nullptr;
    }
    const size_t offset = static_cast<size_t>(raw_index) * row_bytes;
    if (offset + row_bytes > merged.size()) {
      *error = "scatter_embeddings computed offset is out of bounds";
      return nullptr;
    }
    std::memcpy(merged.data() + offset, update_bytes + i * row_bytes, row_bytes);
  }
  const std::string output_name =
      mapped_name(stage.outputs, "output", "inputs_embeds");
  auto* holder = new OwnedTensorArray();
  holder->count = 1;
  holder->values =
      static_cast<DinfNamedTensor*>(std::malloc(sizeof(DinfNamedTensor)));
  holder->values[0] = dinf_make_tensor(
      output_name.c_str(),
      base->tensor.dtype,
      tensor_shape(base->tensor),
      merged.data(),
      merged.size());
  return holder;
}

class RuntimePipelineSession final : public DinfRuntimeSession {
 public:
  RuntimePipelineSession(
      std::string engine,
      std::string spec_path,
      std::vector<PipelineStage> stages,
      StringMap requested_outputs)
      : engine_(std::move(engine)),
        spec_path_(std::move(spec_path)),
        stages_(std::move(stages)),
        requested_outputs_(std::move(requested_outputs)) {}

  int Run(
      const DinfNamedTensor* inputs,
      size_t input_count,
      DinfNamedTensor** outputs,
      size_t* output_count,
      std::string* error) override {
    TensorMap tensors;
    for (size_t i = 0; i < input_count; ++i) {
      tensors[inputs[i].name == nullptr ? "" : inputs[i].name] = &inputs[i];
    }
    std::vector<std::unique_ptr<OwnedTensorArray>> owned;
    std::vector<std::string> last_keys;
    for (auto& stage : stages_) {
      if (!stage.op.empty()) {
        std::unique_ptr<OwnedTensorArray> holder;
        if (stage.op == "scatter_embeddings") {
          holder.reset(scatter_embeddings(stage, tensors, error));
        } else {
          *error = "Unsupported runtime pipeline op: " + stage.op;
          return 1;
        }
        if (!holder) {
          return 1;
        }
        last_keys.clear();
        for (size_t i = 0; i < holder->count; ++i) {
          const std::string key =
              holder->values[i].name == nullptr ? "" : holder->values[i].name;
          tensors[key] = &holder->values[i];
          last_keys.push_back(key);
        }
        owned.push_back(std::move(holder));
        continue;
      }

      if (stage.inputs.empty()) {
        *error = engine_ + " pipeline stage '" + stage.name +
                 "' must declare its inputs explicitly";
        return 1;
      }
      std::vector<DinfNamedTensor> selected;
      std::vector<std::string> selected_names;
      selected.reserve(stage.inputs.size());
      selected_names.reserve(stage.inputs.size());
      for (const auto& item : stage.inputs) {
        const auto found = tensors.find(item.second);
        if (found == tensors.end()) {
          *error = engine_ + " pipeline stage '" + stage.name +
                   "' is missing input '" + item.second + "' for '" +
                   item.first + "'";
          return 1;
        }
        selected.push_back(*found->second);
        selected_names.push_back(item.first);
      }
      for (size_t i = 0; i < selected.size(); ++i) {
        selected[i].name = const_cast<char*>(selected_names[i].c_str());
      }

      DinfNamedTensor* stage_outputs = nullptr;
      size_t stage_output_count = 0;
      const int status = stage.session->Run(
          selected.data(),
          selected.size(),
          &stage_outputs,
          &stage_output_count,
          error);
      if (status != 0) {
        dinf_cpp_free_tensors(
            stage_outputs,
            static_cast<intptr_t>(stage_output_count));
        return status;
      }

      auto holder = std::make_unique<OwnedTensorArray>();
      holder->values = stage_outputs;
      holder->count = stage_output_count;
      last_keys.clear();
      for (size_t i = 0; i < stage_output_count; ++i) {
        const std::string output_name =
            stage_outputs[i].name == nullptr ? "" : stage_outputs[i].name;
        const std::string key =
            mapped_name(stage.outputs, output_name, output_name);
        tensors[key] = &stage_outputs[i];
        last_keys.push_back(key);
      }
      owned.push_back(std::move(holder));
    }

    StringMap requested = requested_outputs_;
    if (requested.empty()) {
      for (const auto& key : last_keys) {
        requested[key] = key;
      }
    }
    std::vector<DinfNamedTensor> produced;
    for (const auto& item : requested) {
      const auto found = tensors.find(item.second);
      if (found == tensors.end()) {
        *error = engine_ + " pipeline requested missing output '" +
                 item.second + "'";
        return 1;
      }
      const auto* source = found->second;
      produced.push_back(dinf_make_tensor(
          item.first.c_str(),
          source->tensor.dtype,
          tensor_shape(source->tensor),
          source->tensor.data,
          source->tensor.byte_length));
    }
    *output_count = produced.size();
    *outputs = static_cast<DinfNamedTensor*>(
        std::malloc(sizeof(DinfNamedTensor) * produced.size()));
    if (!produced.empty()) {
      std::memcpy(
          *outputs,
          produced.data(),
          sizeof(DinfNamedTensor) * produced.size());
    }
    return 0;
  }

  void Diagnostics(
      DinfDiagBuilder* out,
      const std::string& prefix) const override {
    out->AddString(dinf_diag_path(prefix, "engine"), engine_);
    out->AddBool(dinf_diag_path(prefix, "pipeline"), true);
    out->AddString(dinf_diag_path(prefix, "spec"), spec_path_);
    out->AddInt(dinf_diag_path(prefix, "stage_count"), stages_.size());
    const std::string stages_path = dinf_diag_path(prefix, "stages");
    out->AddList(stages_path);
    for (size_t i = 0; i < stages_.size(); ++i) {
      const std::string stage_path =
          dinf_diag_path(stages_path, std::to_string(i));
      out->AddMap(stage_path);
      out->AddString(dinf_diag_path(stage_path, "name"), stages_[i].name);
      if (!stages_[i].op.empty()) {
        out->AddString(dinf_diag_path(stage_path, "op"), stages_[i].op);
      } else {
        out->AddString(
            dinf_diag_path(stage_path, "model"),
            stages_[i].model_path);
        const std::string diag_path = dinf_diag_path(stage_path, "diagnostics");
        out->AddMap(diag_path);
        stages_[i].session->Diagnostics(out, diag_path);
      }
    }
  }

 private:
  std::string engine_;
  std::string spec_path_;
  std::vector<PipelineStage> stages_;
  StringMap requested_outputs_;
};

bool read_pipeline_spec(
    const std::filesystem::path& path,
    const char* expected_format,
    json* spec,
    std::string* error) {
  if (path.extension() != ".json") {
    return false;
  }
  std::ifstream file(path);
  if (!file) {
    return false;
  }
  json parsed;
  try {
    file >> parsed;
  } catch (const std::exception&) {
    return false;
  }
  if (!parsed.is_object() || !parsed.contains("stages")) {
    return false;
  }
  const std::string format = parsed.value("format", "");
  if (format != expected_format) {
    *error = "Unsupported runtime pipeline format '" + format + "' in " +
             path.string();
    return false;
  }
  *spec = std::move(parsed);
  return true;
}

DinfRuntimeSession* create_pipeline_session(
    const std::filesystem::path& spec_path,
    const json& spec,
    const DinfOptions* options,
    const char* engine_name,
    DinfPipelineStageFactory stage_factory,
    std::string* error) {
  if (!spec.at("stages").is_array()) {
    *error = std::string(engine_name) + " pipeline stages must be an array.";
    return nullptr;
  }
  std::vector<PipelineStage> stages;
  for (const auto& item : spec.at("stages")) {
    if (!item.is_object()) {
      *error = std::string(engine_name) + " pipeline stage must be an object.";
      return nullptr;
    }
    PipelineStage stage;
    stage.name =
        item.value("name", std::string("stage_") + std::to_string(stages.size()));
    stage.inputs = string_map_from_json(item.value("inputs", json::object()));
    stage.outputs = string_map_from_json(item.value("outputs", json::object()));
    if (item.contains("op")) {
      if (item.contains("model")) {
        *error = std::string(engine_name) +
                 " pipeline stage must contain either op or model, not both.";
        return nullptr;
      }
      if (!item.at("op").is_string()) {
        *error = std::string(engine_name) +
                 " pipeline op stage must contain a string op.";
        return nullptr;
      }
      stage.op = item.at("op").get<std::string>();
    } else {
      if (!item.contains("model") || !item.at("model").is_string()) {
        *error = std::string(engine_name) +
                 " pipeline model stage must contain a model path.";
        return nullptr;
      }
      const auto path =
          resolve_pipeline_path(spec_path, item.at("model").get<std::string>());
      stage.model_path = path.string();
      stage.session.reset(
          stage_factory(stage.model_path.c_str(), options, error));
      if (!stage.session) {
        return nullptr;
      }
    }
    stages.push_back(std::move(stage));
  }
  return new RuntimePipelineSession(
      engine_name,
      spec_path.string(),
      std::move(stages),
      requested_outputs_from_json(spec));
}

}  // namespace

DinfRuntimeSession* dinf_try_create_pipeline_session(
    const char* model_path,
    const DinfOptions* options,
    const char* engine_name,
    const char* pipeline_format,
    DinfPipelineStageFactory stage_factory,
    std::string* error) {
  json spec;
  if (!read_pipeline_spec(model_path, pipeline_format, &spec, error)) {
    return nullptr;
  }
  return create_pipeline_session(
      model_path,
      spec,
      options,
      engine_name,
      stage_factory,
      error);
}
