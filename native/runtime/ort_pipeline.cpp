#include "ort_pipeline.h"

#if DINF_ENABLE_ORT
#include "ort_session.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <utility>
#include <vector>

namespace dinf_ort {
namespace {

using json = nlohmann::json;
using TensorMap = std::map<std::string, const DinfNamedTensor*>;
using StringMap = std::map<std::string, std::string>;

struct OwnedTensorArray {
  DinfNamedTensor* values = nullptr;
  size_t count = 0;

  ~OwnedTensorArray() {
    dinf_cpp_free_tensors(values, static_cast<intptr_t>(count));
  }
};

StringMap StringMapFromJson(const json& value) {
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

std::filesystem::path ResolvePipelinePath(
    const std::filesystem::path& spec_path,
    const std::string& value) {
  std::filesystem::path path(value);
  if (path.is_absolute()) {
    return path;
  }
  return spec_path.parent_path() / path;
}

struct PipelineStage {
  std::string name;
  std::string op;
  std::string model_path;
  StringMap inputs;
  StringMap outputs;
  std::unique_ptr<Session> session;
};

std::string MappedName(
    const StringMap& values,
    const std::string& name,
    const std::string& fallback) {
  const auto found = values.find(name);
  return found == values.end() ? fallback : found->second;
}

const DinfNamedTensor* FindTensor(
    const TensorMap& tensors,
    const std::string& name,
    std::string* error) {
  const auto found = tensors.find(name);
  if (found == tensors.end()) {
    *error = "ONNX pipeline missing tensor '" + name + "'";
    return nullptr;
  }
  return found->second;
}

int64_t IndexValue(const DinfTensor& tensor, size_t index) {
  if (tensor.dtype == DINF_DTYPE_INT64) {
    return static_cast<const int64_t*>(tensor.data)[index];
  }
  if (tensor.dtype == DINF_DTYPE_INT32) {
    return static_cast<const int32_t*>(tensor.data)[index];
  }
  return -1;
}

OwnedTensorArray* ScatterEmbeddings(
    const PipelineStage& stage,
    const TensorMap& tensors,
    std::string* error) {
  const std::string base_key =
      MappedName(stage.inputs, "base", MappedName(stage.inputs, "inputs_embeds", "inputs_embeds"));
  const std::string updates_key =
      MappedName(stage.inputs, "updates", MappedName(stage.inputs, "image_embeds", "image_embeds"));
  const std::string indices_key =
      MappedName(stage.inputs, "indices", "image_token_indices");
  const auto* base = FindTensor(tensors, base_key, error);
  const auto* updates = FindTensor(tensors, updates_key, error);
  const auto* indices = FindTensor(tensors, indices_key, error);
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
    *error = "scatter_embeddings base shape must have positive sequence and hidden dims";
    return nullptr;
  }
  if (updates->tensor.shape[1] != hidden) {
    *error = "scatter_embeddings update hidden size does not match base";
    return nullptr;
  }
  const size_t index_count =
      static_cast<size_t>(indices->tensor.byte_length / dinf_dtype_size(indices->tensor.dtype));
  if (index_count != static_cast<size_t>(updates->tensor.shape[0])) {
    *error = "scatter_embeddings update count does not match index count";
    return nullptr;
  }
  const size_t row_bytes = static_cast<size_t>(hidden) * dinf_dtype_size(base->tensor.dtype);
  const auto* update_bytes = static_cast<const uint8_t*>(updates->tensor.data);
  std::vector<uint8_t> merged(static_cast<size_t>(base->tensor.byte_length));
  std::memcpy(merged.data(), base->tensor.data, merged.size());
  for (size_t i = 0; i < index_count; ++i) {
    const int64_t raw_index = IndexValue(indices->tensor, i);
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
      MappedName(stage.outputs, "output", "inputs_embeds");
  auto* holder = new OwnedTensorArray();
  holder->count = 1;
  holder->values = static_cast<DinfNamedTensor*>(std::malloc(sizeof(DinfNamedTensor)));
  holder->values[0] = dinf_make_tensor(
      output_name.c_str(),
      base->tensor.dtype,
      TensorShape(base->tensor),
      merged.data(),
      merged.size());
  return holder;
}

class PipelineSession final : public DinfRuntimeSession {
 public:
  PipelineSession(
      std::string spec_path,
      std::vector<PipelineStage> stages,
      StringMap requested_outputs)
      : spec_path_(std::move(spec_path)),
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
          holder.reset(ScatterEmbeddings(stage, tensors, error));
        } else {
          *error = "Unsupported ONNX pipeline op: " + stage.op;
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

      std::vector<DinfNamedTensor> selected;
      std::vector<std::string> selected_names;
      const auto& required = stage.session->InputNames();
      selected.reserve(required.size());
      selected_names.reserve(required.size());
      for (const auto& input_name : required) {
        const auto mapped = stage.inputs.find(input_name);
        const std::string source =
            mapped == stage.inputs.end() ? input_name : mapped->second;
        const auto found = tensors.find(source);
        if (found == tensors.end()) {
          *error = "ONNX pipeline stage '" + stage.name +
                   "' is missing input '" + source + "' for '" + input_name + "'";
          return 1;
        }
        selected.push_back(*found->second);
        selected_names.push_back(input_name);
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
        const auto mapped = stage.outputs.find(output_name);
        const std::string key =
            mapped == stage.outputs.end() ? output_name : mapped->second;
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
        *error = "ONNX pipeline requested missing output '" + item.second + "'";
        return 1;
      }
      const auto* source = found->second;
      produced.push_back(dinf_make_tensor(
          item.first.c_str(),
          source->tensor.dtype,
          TensorShape(source->tensor),
          source->tensor.data,
          source->tensor.byte_length));
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
    std::string stages = "[";
    for (size_t i = 0; i < stages_.size(); ++i) {
      if (i > 0) {
        stages += ",";
      }
      stages += "{\"name\":\"" + dinf_json_escape(stages_[i].name) + "\"";
      if (!stages_[i].op.empty()) {
        stages += ",\"op\":\"" + dinf_json_escape(stages_[i].op) + "\"";
      } else {
        stages += ",\"model\":\"" + dinf_json_escape(stages_[i].model_path) +
                  "\",\"diagnostics\":" + stages_[i].session->DiagnosticsJson();
      }
      stages += "}";
    }
    stages += "]";
    return "{\"engine\":\"onnx\",\"pipeline\":true,\"spec\":\"" +
           dinf_json_escape(spec_path_) + "\",\"stage_count\":" +
           std::to_string(stages_.size()) + ",\"stages\":" + stages + "}";
  }

 private:
  std::string spec_path_;
  std::vector<PipelineStage> stages_;
  StringMap requested_outputs_;
};

StringMap RequestedOutputsFromJson(const json& spec) {
  if (!spec.contains("outputs")) {
    return {};
  }
  const auto& outputs = spec.at("outputs");
  if (outputs.is_object()) {
    return StringMapFromJson(outputs);
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

}  // namespace

bool IsPipelineSpec(const std::filesystem::path& path, json* spec) {
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
  if (!format.empty() && format != "dart_inference.onnx_pipeline.v1") {
    return false;
  }
  *spec = std::move(parsed);
  return true;
}

std::unique_ptr<DinfRuntimeSession> CreatePipeline(
    const std::filesystem::path& spec_path,
    const json& spec,
    const char* options_json,
    std::string* error) {
  if (!spec.at("stages").is_array()) {
    *error = "ONNX pipeline stages must be an array.";
    return nullptr;
  }
  std::vector<PipelineStage> stages;
  for (const auto& item : spec.at("stages")) {
    if (!item.is_object()) {
      *error = "ONNX pipeline stage must be an object.";
      return nullptr;
    }
    PipelineStage stage;
    stage.name = item.value("name", std::string("stage_") + std::to_string(stages.size()));
    stage.inputs = StringMapFromJson(item.value("inputs", json::object()));
    stage.outputs = StringMapFromJson(item.value("outputs", json::object()));
    if (item.contains("op")) {
      if (item.contains("model")) {
        *error = "ONNX pipeline stage must contain either op or model, not both.";
        return nullptr;
      }
      if (!item.at("op").is_string()) {
        *error = "ONNX pipeline op stage must contain a string op.";
        return nullptr;
      }
      stage.op = item.at("op").get<std::string>();
    } else {
      if (!item.contains("model")) {
        *error = "ONNX pipeline model stage must contain a model path.";
        return nullptr;
      }
      if (!item.at("model").is_string()) {
        *error = "ONNX pipeline model stage must contain a string model path.";
        return nullptr;
      }
      const auto path = ResolvePipelinePath(spec_path, item.at("model").get<std::string>());
      stage.model_path = path.string();
      stage.session = CreateSession(stage.model_path.c_str(), options_json, error);
      if (!stage.session) {
        return nullptr;
      }
    }
    stages.push_back(std::move(stage));
  }
  return std::make_unique<PipelineSession>(
      spec_path.string(),
      std::move(stages),
      RequestedOutputsFromJson(spec));
}

}  // namespace dinf_ort

#endif
