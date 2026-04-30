#pragma once

#include "mlx/c/mlx.h"

#include <cstdlib>
#include <cstring>

namespace {

struct DartMlxArrayHandle {
  mlx_array value;
};

struct DartMlxDeviceHandle {
  mlx_device value;
};

struct DartMlxStreamHandle {
  mlx_stream value;
};

struct DartMlxGroupHandle {
  mlx_distributed_group value;
};

struct DartMlxClosureHandle {
  mlx_closure value;
};

struct DartMlxKwHandle {
  mlx_closure_kwargs value;
};

struct DartMlxCustomHandle {
  mlx_closure_custom value;
};

struct DartMlxCustomJvpHandle {
  mlx_closure_custom_jvp value;
};

struct DartMlxImportedHandle {
  mlx_imported_function value;
};

struct DartMlxExporterHandle {
  mlx_function_exporter value;
};

struct DartMlxCudaConfigHandle {
  mlx_fast_cuda_kernel_config value;
};

struct DartMlxCudaKernelHandle {
  mlx_fast_cuda_kernel value;
};

struct DartMlxMetalConfigHandle {
  mlx_fast_metal_kernel_config value;
};

struct DartMlxMetalKernelHandle {
  mlx_fast_metal_kernel value;
};

using DartMlxClosureCallback = int (*)(
    DartMlxArrayHandle*** outputs,
    size_t* outputs_len,
    DartMlxArrayHandle** inputs,
    size_t inputs_len);

using DartMlxKwCallback = int (*)(
    DartMlxArrayHandle*** outputs,
    size_t* outputs_len,
    DartMlxArrayHandle** inputs,
    size_t inputs_len,
    char*** keys,
    DartMlxArrayHandle*** values,
    size_t* values_len);

using DartMlxCustomCallback = int (*)(
    DartMlxArrayHandle*** outputs,
    size_t* outputs_len,
    DartMlxArrayHandle** input0,
    size_t input0_len,
    DartMlxArrayHandle** input1,
    size_t input1_len,
    DartMlxArrayHandle** input2,
    size_t input2_len);

using DartMlxCustomJvpCallback = int (*)(
    DartMlxArrayHandle*** outputs,
    size_t* outputs_len,
    DartMlxArrayHandle** primals,
    size_t primals_len,
    DartMlxArrayHandle** tangents,
    size_t tangents_len,
    const int* argnums,
    size_t argnums_len);

struct DartMlxClosurePayload {
  DartMlxClosureCallback callback;
};

struct DartMlxKwPayload {
  DartMlxKwCallback callback;
};

struct DartMlxCustomPayload {
  DartMlxCustomCallback callback;
};

struct DartMlxCustomJvpPayload {
  DartMlxCustomJvpCallback callback;
};

extern "C" void dart_mlx_free_string_array(char** values, size_t len);

mlx_stream default_cpu_stream() {
  static mlx_stream stream = mlx_default_cpu_stream_new();
  return stream;
}

mlx_stream default_gpu_stream() {
  static mlx_stream stream = mlx_default_gpu_stream_new();
  return stream;
}

mlx_stream default_device_stream() {
  auto device = mlx_device_new();
  if (mlx_get_default_device(&device) != 0 || device.ctx == nullptr) {
    return default_cpu_stream();
  }

  mlx_device_type type = MLX_CPU;
  const auto status = mlx_device_get_type(&type, device);
  mlx_device_free(device);
  if (status != 0) {
    return default_cpu_stream();
  }
  return type == MLX_GPU ? default_gpu_stream() : default_cpu_stream();
}

mlx_dtype as_dtype(int dtype) {
  return static_cast<mlx_dtype>(dtype);
}

int normalized_axis(const DartMlxArrayHandle* input, int axis) {
  const auto ndim = static_cast<int>(mlx_array_ndim(input->value));
  return axis < 0 ? axis + ndim : axis;
}

int resolved_n(const DartMlxArrayHandle* input, int axis, int n) {
  if (n >= 0) {
    return n;
  }
  const auto resolved_axis = normalized_axis(input, axis);
  return mlx_array_dim(input->value, resolved_axis);
}

DartMlxArrayHandle* wrap_array(mlx_array value) {
  if (value.ctx == nullptr) {
    return nullptr;
  }
  auto* handle = new DartMlxArrayHandle;
  handle->value = value;
  return handle;
}

DartMlxArrayHandle* wrap_array_copy(mlx_array value) {
  auto copy = mlx_array_new();
  if (mlx_array_set(&copy, value) != 0) {
    return nullptr;
  }
  return wrap_array(copy);
}

DartMlxDeviceHandle* wrap_device(mlx_device value) {
  if (value.ctx == nullptr) {
    return nullptr;
  }
  auto* handle = new DartMlxDeviceHandle;
  handle->value = value;
  return handle;
}

DartMlxStreamHandle* wrap_stream(mlx_stream value) {
  if (value.ctx == nullptr) {
    return nullptr;
  }
  auto* handle = new DartMlxStreamHandle;
  handle->value = value;
  return handle;
}

DartMlxGroupHandle* wrap_group(mlx_distributed_group value) {
  if (value.ctx == nullptr) {
    return nullptr;
  }
  auto* handle = new DartMlxGroupHandle;
  handle->value = value;
  return handle;
}

DartMlxClosureHandle* wrap_closure(mlx_closure value) {
  if (value.ctx == nullptr) {
    return nullptr;
  }
  auto* handle = new DartMlxClosureHandle;
  handle->value = value;
  return handle;
}

DartMlxKwHandle* wrap_kw(mlx_closure_kwargs value) {
  if (value.ctx == nullptr) {
    return nullptr;
  }
  auto* handle = new DartMlxKwHandle;
  handle->value = value;
  return handle;
}

DartMlxCustomHandle* wrap_custom(mlx_closure_custom value) {
  if (value.ctx == nullptr) {
    return nullptr;
  }
  auto* handle = new DartMlxCustomHandle;
  handle->value = value;
  return handle;
}

DartMlxCustomJvpHandle* wrap_custom_jvp(mlx_closure_custom_jvp value) {
  if (value.ctx == nullptr) {
    return nullptr;
  }
  auto* handle = new DartMlxCustomJvpHandle;
  handle->value = value;
  return handle;
}

DartMlxImportedHandle* wrap_imported(mlx_imported_function value) {
  if (value.ctx == nullptr) {
    return nullptr;
  }
  auto* handle = new DartMlxImportedHandle;
  handle->value = value;
  return handle;
}

DartMlxExporterHandle* wrap_exporter(mlx_function_exporter value) {
  if (value.ctx == nullptr) {
    return nullptr;
  }
  auto* handle = new DartMlxExporterHandle;
  handle->value = value;
  return handle;
}

DartMlxCudaConfigHandle* wrap_cuda_config(mlx_fast_cuda_kernel_config value) {
  if (value.ctx == nullptr) {
    return nullptr;
  }
  auto* handle = new DartMlxCudaConfigHandle;
  handle->value = value;
  return handle;
}

DartMlxCudaKernelHandle* wrap_cuda_kernel(mlx_fast_cuda_kernel value) {
  if (value.ctx == nullptr) {
    return nullptr;
  }
  auto* handle = new DartMlxCudaKernelHandle;
  handle->value = value;
  return handle;
}

DartMlxMetalConfigHandle* wrap_metal_config(mlx_fast_metal_kernel_config value) {
  if (value.ctx == nullptr) {
    return nullptr;
  }
  auto* handle = new DartMlxMetalConfigHandle;
  handle->value = value;
  return handle;
}

DartMlxMetalKernelHandle* wrap_metal_kernel(mlx_fast_metal_kernel value) {
  if (value.ctx == nullptr) {
    return nullptr;
  }
  auto* handle = new DartMlxMetalKernelHandle;
  handle->value = value;
  return handle;
}

char* copy_c_string(const char* value) {
  if (value == nullptr) {
    return nullptr;
  }
  auto length = std::strlen(value);
  auto* copy = static_cast<char*>(std::malloc(length + 1));
  if (copy == nullptr) {
    return nullptr;
  }
  std::memcpy(copy, value, length + 1);
  return copy;
}

template <typename Callback>
DartMlxArrayHandle* unary_array_op(
    const DartMlxArrayHandle* input,
    Callback callback) {
  mlx_array out = mlx_array_new();
  if (callback(&out, input->value, default_device_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

template <typename Callback>
DartMlxArrayHandle* binary_array_op(
    const DartMlxArrayHandle* lhs,
    const DartMlxArrayHandle* rhs,
    Callback callback) {
  mlx_array out = mlx_array_new();
  if (callback(&out, lhs->value, rhs->value, default_device_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

template <typename CopyCallback>
int copy_array_data_contiguous(
    const DartMlxArrayHandle* handle,
    CopyCallback callback) {
  mlx_array contiguous = mlx_array_new();
  if (mlx_contiguous(&contiguous, handle->value, false, default_cpu_stream()) !=
      0) {
    return 1;
  }
  if (mlx_array_eval(contiguous) != 0) {
    mlx_array_free(contiguous);
    return 1;
  }
  const auto status = callback(contiguous);
  mlx_array_free(contiguous);
  return status;
}

mlx_vector_array build_array_vector(DartMlxArrayHandle** arrays, size_t len) {
  auto values = mlx_vector_array_new();
  for (size_t i = 0; i < len; ++i) {
    if (mlx_vector_array_append_value(values, arrays[i]->value) != 0) {
      mlx_vector_array_free(values);
      return mlx_vector_array_new();
    }
  }
  return values;
}

int export_vector_array(
    const mlx_vector_array values,
    DartMlxArrayHandle*** outputs_out,
    size_t* outputs_len_out) {
  const auto len = mlx_vector_array_size(values);
  auto** outputs = static_cast<DartMlxArrayHandle**>(
      std::malloc(sizeof(DartMlxArrayHandle*) * len));
  if (len > 0 && outputs == nullptr) {
    return 1;
  }
  auto temp = mlx_array_new();
  for (size_t i = 0; i < len; ++i) {
    if (mlx_vector_array_get(&temp, values, i) != 0) {
      std::free(outputs);
      return 1;
    }
    outputs[i] = wrap_array_copy(temp);
    if (outputs[i] == nullptr) {
      std::free(outputs);
      return 1;
    }
  }
  *outputs_out = outputs;
  *outputs_len_out = len;
  return 0;
}

void free_handle_array(DartMlxArrayHandle** handles, size_t len) {
  if (handles == nullptr) {
    return;
  }
  for (size_t i = 0; i < len; ++i) {
    if (handles[i] != nullptr) {
      mlx_array_free(handles[i]->value);
      delete handles[i];
    }
  }
  std::free(handles);
}

void free_closure_payload(void* payload) {
  delete static_cast<DartMlxClosurePayload*>(payload);
}

void free_kw_payload(void* payload) {
  delete static_cast<DartMlxKwPayload*>(payload);
}

void free_custom_payload(void* payload) {
  delete static_cast<DartMlxCustomPayload*>(payload);
}

void free_custom_jvp_payload(void* payload) {
  delete static_cast<DartMlxCustomJvpPayload*>(payload);
}

int dart_mlx_closure_trampoline(
    mlx_vector_array* res,
    const mlx_vector_array input,
    void* payload) {
  auto* closure_payload = static_cast<DartMlxClosurePayload*>(payload);
  const auto input_len = mlx_vector_array_size(input);
  auto** input_handles = static_cast<DartMlxArrayHandle**>(
      std::malloc(sizeof(DartMlxArrayHandle*) * input_len));
  if (input_len > 0 && input_handles == nullptr) {
    return 1;
  }

  auto temp = mlx_array_new();
  for (size_t i = 0; i < input_len; ++i) {
    if (mlx_vector_array_get(&temp, input, i) != 0) {
      std::free(input_handles);
      return 1;
    }
    input_handles[i] = wrap_array_copy(temp);
    if (input_handles[i] == nullptr) {
      std::free(input_handles);
      return 1;
    }
  }

  DartMlxArrayHandle** output_handles = nullptr;
  size_t output_len = 0;
  const auto status = closure_payload->callback(
      &output_handles, &output_len, input_handles, input_len);
  free_handle_array(input_handles, input_len);
  if (status != 0) {
    std::free(output_handles);
    return status;
  }

  auto outputs = mlx_vector_array_new();
  for (size_t i = 0; i < output_len; ++i) {
    if (mlx_vector_array_append_value(outputs, output_handles[i]->value) != 0) {
      mlx_vector_array_free(outputs);
      std::free(output_handles);
      return 1;
    }
  }
  *res = outputs;
  std::free(output_handles);
  return 0;
}

int export_map_string_to_array(
    const mlx_map_string_to_array values,
    char*** keys_out,
    DartMlxArrayHandle*** arrays_out,
    size_t* len_out) {
  size_t len = 0;
  {
    auto it = mlx_map_string_to_array_iterator_new(values);
    const char* key = nullptr;
    mlx_array value = mlx_array_new();
    while (mlx_map_string_to_array_iterator_next(&key, &value, it) == 0) {
      len++;
    }
    mlx_map_string_to_array_iterator_free(it);
  }

  auto** keys = static_cast<char**>(std::malloc(sizeof(char*) * len));
  auto** arrays =
      static_cast<DartMlxArrayHandle**>(
          std::malloc(sizeof(DartMlxArrayHandle*) * len));
  if ((len > 0) && (keys == nullptr || arrays == nullptr)) {
    std::free(keys);
    std::free(arrays);
    return 1;
  }

  {
    auto it = mlx_map_string_to_array_iterator_new(values);
    const char* key = nullptr;
    mlx_array value = mlx_array_new();
    size_t index = 0;
    while (mlx_map_string_to_array_iterator_next(&key, &value, it) == 0) {
      keys[index] = copy_c_string(key);
      arrays[index] = wrap_array_copy(value);
      index++;
    }
    mlx_map_string_to_array_iterator_free(it);
  }

  *keys_out = keys;
  *arrays_out = arrays;
  *len_out = len;
  return 0;
}

int dart_mlx_kw_trampoline(
    mlx_vector_array* res,
    const mlx_vector_array input_0,
    const mlx_map_string_to_array input_1,
    void* payload) {
  auto* kw_payload = static_cast<DartMlxKwPayload*>(payload);
  DartMlxArrayHandle** inputs = nullptr;
  char** keys = nullptr;
  DartMlxArrayHandle** values = nullptr;
  size_t inputs_len = 0;
  size_t values_len = 0;

  if (export_vector_array(input_0, &inputs, &inputs_len) != 0 ||
      export_map_string_to_array(input_1, &keys, &values, &values_len) != 0) {
    free_handle_array(inputs, inputs_len);
    free_handle_array(values, values_len);
    dart_mlx_free_string_array(keys, values_len);
    return 1;
  }

  DartMlxArrayHandle** output_handles = nullptr;
  size_t output_len = 0;
  const auto status = kw_payload->callback(
      &output_handles,
      &output_len,
      inputs,
      inputs_len,
      &keys,
      &values,
      &values_len);
  free_handle_array(inputs, inputs_len);
  free_handle_array(values, values_len);
  dart_mlx_free_string_array(keys, values_len);
  if (status != 0) {
    std::free(output_handles);
    return status;
  }

  auto outputs = mlx_vector_array_new();
  for (size_t i = 0; i < output_len; ++i) {
    if (mlx_vector_array_append_value(outputs, output_handles[i]->value) != 0) {
      mlx_vector_array_free(outputs);
      std::free(output_handles);
      return 1;
    }
  }
  *res = outputs;
  std::free(output_handles);
  return 0;
}

int dart_mlx_custom_trampoline(
    mlx_vector_array* res,
    const mlx_vector_array input_0,
    const mlx_vector_array input_1,
    const mlx_vector_array input_2,
    void* payload) {
  auto* custom_payload = static_cast<DartMlxCustomPayload*>(payload);
  DartMlxArrayHandle** handles0 = nullptr;
  DartMlxArrayHandle** handles1 = nullptr;
  DartMlxArrayHandle** handles2 = nullptr;
  size_t len0 = 0;
  size_t len1 = 0;
  size_t len2 = 0;

  if (export_vector_array(input_0, &handles0, &len0) != 0 ||
      export_vector_array(input_1, &handles1, &len1) != 0 ||
      export_vector_array(input_2, &handles2, &len2) != 0) {
    free_handle_array(handles0, len0);
    free_handle_array(handles1, len1);
    free_handle_array(handles2, len2);
    return 1;
  }

  DartMlxArrayHandle** output_handles = nullptr;
  size_t output_len = 0;
  const auto status = custom_payload->callback(
      &output_handles,
      &output_len,
      handles0,
      len0,
      handles1,
      len1,
      handles2,
      len2);
  free_handle_array(handles0, len0);
  free_handle_array(handles1, len1);
  free_handle_array(handles2, len2);
  if (status != 0) {
    std::free(output_handles);
    return status;
  }

  auto outputs = mlx_vector_array_new();
  for (size_t i = 0; i < output_len; ++i) {
    if (mlx_vector_array_append_value(outputs, output_handles[i]->value) != 0) {
      mlx_vector_array_free(outputs);
      std::free(output_handles);
      return 1;
    }
  }
  *res = outputs;
  std::free(output_handles);
  return 0;
}

int dart_mlx_custom_jvp_trampoline(
    mlx_vector_array* res,
    const mlx_vector_array primals,
    const mlx_vector_array tangents,
    const int* argnums,
    size_t argnums_len,
    void* payload) {
  auto* jvp_payload = static_cast<DartMlxCustomJvpPayload*>(payload);
  DartMlxArrayHandle** primal_handles = nullptr;
  DartMlxArrayHandle** tangent_handles = nullptr;
  size_t primal_len = 0;
  size_t tangent_len = 0;
  if (export_vector_array(primals, &primal_handles, &primal_len) != 0 ||
      export_vector_array(tangents, &tangent_handles, &tangent_len) != 0) {
    free_handle_array(primal_handles, primal_len);
    free_handle_array(tangent_handles, tangent_len);
    return 1;
  }

  DartMlxArrayHandle** output_handles = nullptr;
  size_t output_len = 0;
  const auto status = jvp_payload->callback(
      &output_handles,
      &output_len,
      primal_handles,
      primal_len,
      tangent_handles,
      tangent_len,
      argnums,
      argnums_len);
  free_handle_array(primal_handles, primal_len);
  free_handle_array(tangent_handles, tangent_len);
  if (status != 0) {
    std::free(output_handles);
    return status;
  }

  auto outputs = mlx_vector_array_new();
  for (size_t i = 0; i < output_len; ++i) {
    if (mlx_vector_array_append_value(outputs, output_handles[i]->value) != 0) {
      mlx_vector_array_free(outputs);
      std::free(output_handles);
      return 1;
    }
  }
  *res = outputs;
  std::free(output_handles);
  return 0;
}

} // namespace
