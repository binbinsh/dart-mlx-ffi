void ReleaseOrtValueHandle(void* value, void* context) {
  if (value == nullptr || context == nullptr) {
    return;
  }
  const auto* api = static_cast<const OrtApi*>(context);
  api->ReleaseValue(static_cast<OrtValue*>(value));
}

DinfNamedTensor MakeHandleTensor(
    const OrtApi* api,
    const char* name,
    int32_t dtype,
    const std::vector<int64_t>& shape,
    size_t byte_length,
    OrtValue* value) {
  DinfNamedTensor tensor{};
  tensor.name = dinf_copy_string(name == nullptr ? "" : name);
  tensor.tensor.dtype = dtype;
  tensor.tensor.rank = static_cast<int32_t>(shape.size());
  tensor.tensor.byte_length = static_cast<intptr_t>(byte_length);
  tensor.tensor.memory_kind = DINF_TENSOR_MEMORY_HANDLE;
  tensor.tensor.shape = static_cast<int64_t*>(
      std::malloc(sizeof(int64_t) * shape.size()));
  if (!shape.empty() && tensor.tensor.shape != nullptr) {
    std::memcpy(
        tensor.tensor.shape,
        shape.data(),
        sizeof(int64_t) * shape.size());
  }
  tensor.tensor.handle = dinf_make_tensor_handle(
      value,
      const_cast<OrtApi*>(api),
      ReleaseOrtValueHandle);
  return tensor;
}

DinfNamedTensor MakeCpuViewTensor(
    const OrtApi* api,
    const char* name,
    int32_t dtype,
    const std::vector<int64_t>& shape,
    size_t byte_length,
    void* data,
    OrtValue* value) {
  DinfNamedTensor tensor{};
  tensor.name = dinf_copy_string(name == nullptr ? "" : name);
  tensor.tensor.dtype = dtype;
  tensor.tensor.rank = static_cast<int32_t>(shape.size());
  tensor.tensor.byte_length = static_cast<intptr_t>(byte_length);
  tensor.tensor.data = data;
  tensor.tensor.memory_kind = DINF_TENSOR_MEMORY_CPU_VIEW;
  tensor.tensor.shape = static_cast<int64_t*>(
      std::malloc(sizeof(int64_t) * shape.size()));
  if (!shape.empty() && tensor.tensor.shape != nullptr) {
    std::memcpy(
        tensor.tensor.shape,
        shape.data(),
        sizeof(int64_t) * shape.size());
  }
  tensor.tensor.handle = dinf_make_tensor_handle(
      value,
      const_cast<OrtApi*>(api),
      ReleaseOrtValueHandle);
  return tensor;
}

void FreeProduced(std::vector<DinfNamedTensor>* produced) {
  if (produced == nullptr) {
    return;
  }
  for (auto& tensor : *produced) {
    std::free(tensor.name);
    std::free(tensor.tensor.shape);
    if (tensor.tensor.memory_kind == DINF_TENSOR_MEMORY_CPU) {
      std::free(tensor.tensor.data);
    }
    dinf_release_tensor_handle(tensor.tensor.handle);
  }
  produced->clear();
}
