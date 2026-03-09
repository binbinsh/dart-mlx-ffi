#include "bridge.h"

extern "C" DartMlxCudaConfigHandle* dart_mlx_cuda_config_new() {
  return wrap_cuda_config(mlx_fast_cuda_kernel_config_new());
}

extern "C" void dart_mlx_cuda_config_free(DartMlxCudaConfigHandle* handle) {
  if (handle == nullptr) {
    return;
  }
  mlx_fast_cuda_kernel_config_free(handle->value);
  delete handle;
}

extern "C" int dart_mlx_cuda_config_add_output_arg(
    const DartMlxCudaConfigHandle* handle,
    const int* shape,
    size_t len,
    int dtype) {
  return mlx_fast_cuda_kernel_config_add_output_arg(
      handle->value, shape, len, as_dtype(dtype));
}

extern "C" int dart_mlx_cuda_config_set_grid(
    const DartMlxCudaConfigHandle* handle,
    int x,
    int y,
    int z) {
  return mlx_fast_cuda_kernel_config_set_grid(handle->value, x, y, z);
}

extern "C" int dart_mlx_cuda_config_set_thread_group(
    const DartMlxCudaConfigHandle* handle,
    int x,
    int y,
    int z) {
  return mlx_fast_cuda_kernel_config_set_thread_group(handle->value, x, y, z);
}

extern "C" int dart_mlx_cuda_config_set_init_value(
    const DartMlxCudaConfigHandle* handle,
    float value) {
  return mlx_fast_cuda_kernel_config_set_init_value(handle->value, value);
}

extern "C" int dart_mlx_cuda_config_set_verbose(
    const DartMlxCudaConfigHandle* handle,
    bool value) {
  return mlx_fast_cuda_kernel_config_set_verbose(handle->value, value);
}

extern "C" int dart_mlx_cuda_config_add_template_dtype(
    const DartMlxCudaConfigHandle* handle,
    const char* name,
    int dtype) {
  return mlx_fast_cuda_kernel_config_add_template_arg_dtype(
      handle->value, name, as_dtype(dtype));
}

extern "C" int dart_mlx_cuda_config_add_template_int(
    const DartMlxCudaConfigHandle* handle,
    const char* name,
    int value) {
  return mlx_fast_cuda_kernel_config_add_template_arg_int(handle->value, name, value);
}

extern "C" int dart_mlx_cuda_config_add_template_bool(
    const DartMlxCudaConfigHandle* handle,
    const char* name,
    bool value) {
  return mlx_fast_cuda_kernel_config_add_template_arg_bool(handle->value, name, value);
}

extern "C" DartMlxCudaKernelHandle* dart_mlx_cuda_kernel_new(
    const char* name,
    char** input_names,
    size_t input_names_len,
    char** output_names,
    size_t output_names_len,
    const char* source,
    const char* header,
    bool ensure_row_contiguous,
    int shared_memory) {
  auto inputs = mlx_vector_string_new_data(const_cast<const char**>(input_names), input_names_len);
  auto outputs =
      mlx_vector_string_new_data(const_cast<const char**>(output_names), output_names_len);
  auto kernel = mlx_fast_cuda_kernel_new(
      name,
      inputs,
      outputs,
      source,
      header,
      ensure_row_contiguous,
      shared_memory);
  mlx_vector_string_free(inputs);
  mlx_vector_string_free(outputs);
  return wrap_cuda_kernel(kernel);
}

extern "C" void dart_mlx_cuda_kernel_free(DartMlxCudaKernelHandle* handle) {
  if (handle == nullptr) {
    return;
  }
  mlx_fast_cuda_kernel_free(handle->value);
  delete handle;
}

extern "C" int dart_mlx_cuda_kernel_apply(
    const DartMlxCudaKernelHandle* kernel,
    DartMlxArrayHandle** inputs,
    size_t input_len,
    const DartMlxCudaConfigHandle* config,
    DartMlxArrayHandle*** outputs_out,
    size_t* outputs_len_out) {
  auto input_vec = build_array_vector(inputs, input_len);
  auto outputs = mlx_vector_array_new();
  auto status = mlx_fast_cuda_kernel_apply(
      &outputs, kernel->value, input_vec, config->value, default_cpu_stream());
  mlx_vector_array_free(input_vec);
  if (status != 0) {
    return status;
  }
  status = export_vector_array(outputs, outputs_out, outputs_len_out);
  mlx_vector_array_free(outputs);
  return status;
}

extern "C" DartMlxMetalConfigHandle* dart_mlx_metal_config_new() {
  return wrap_metal_config(mlx_fast_metal_kernel_config_new());
}

extern "C" void dart_mlx_metal_config_free(DartMlxMetalConfigHandle* handle) {
  if (handle == nullptr) {
    return;
  }
  mlx_fast_metal_kernel_config_free(handle->value);
  delete handle;
}

extern "C" int dart_mlx_metal_config_add_output_arg(
    const DartMlxMetalConfigHandle* handle,
    const int* shape,
    size_t len,
    int dtype) {
  return mlx_fast_metal_kernel_config_add_output_arg(
      handle->value, shape, len, as_dtype(dtype));
}

extern "C" int dart_mlx_metal_config_set_grid(
    const DartMlxMetalConfigHandle* handle,
    int x,
    int y,
    int z) {
  return mlx_fast_metal_kernel_config_set_grid(handle->value, x, y, z);
}

extern "C" int dart_mlx_metal_config_set_thread_group(
    const DartMlxMetalConfigHandle* handle,
    int x,
    int y,
    int z) {
  return mlx_fast_metal_kernel_config_set_thread_group(handle->value, x, y, z);
}

extern "C" int dart_mlx_metal_config_set_init_value(
    const DartMlxMetalConfigHandle* handle,
    float value) {
  return mlx_fast_metal_kernel_config_set_init_value(handle->value, value);
}

extern "C" int dart_mlx_metal_config_set_verbose(
    const DartMlxMetalConfigHandle* handle,
    bool value) {
  return mlx_fast_metal_kernel_config_set_verbose(handle->value, value);
}

extern "C" int dart_mlx_metal_config_add_template_dtype(
    const DartMlxMetalConfigHandle* handle,
    const char* name,
    int dtype) {
  return mlx_fast_metal_kernel_config_add_template_arg_dtype(
      handle->value, name, as_dtype(dtype));
}

extern "C" int dart_mlx_metal_config_add_template_int(
    const DartMlxMetalConfigHandle* handle,
    const char* name,
    int value) {
  return mlx_fast_metal_kernel_config_add_template_arg_int(handle->value, name, value);
}

extern "C" int dart_mlx_metal_config_add_template_bool(
    const DartMlxMetalConfigHandle* handle,
    const char* name,
    bool value) {
  return mlx_fast_metal_kernel_config_add_template_arg_bool(handle->value, name, value);
}

extern "C" DartMlxMetalKernelHandle* dart_mlx_metal_kernel_new(
    const char* name,
    char** input_names,
    size_t input_names_len,
    char** output_names,
    size_t output_names_len,
    const char* source,
    const char* header,
    bool ensure_row_contiguous,
    bool atomic_outputs) {
  auto inputs = mlx_vector_string_new_data(const_cast<const char**>(input_names), input_names_len);
  auto outputs =
      mlx_vector_string_new_data(const_cast<const char**>(output_names), output_names_len);
  auto kernel = mlx_fast_metal_kernel_new(
      name,
      inputs,
      outputs,
      source,
      header,
      ensure_row_contiguous,
      atomic_outputs);
  mlx_vector_string_free(inputs);
  mlx_vector_string_free(outputs);
  return wrap_metal_kernel(kernel);
}

extern "C" void dart_mlx_metal_kernel_free(DartMlxMetalKernelHandle* handle) {
  if (handle == nullptr) {
    return;
  }
  mlx_fast_metal_kernel_free(handle->value);
  delete handle;
}

extern "C" int dart_mlx_metal_kernel_apply(
    const DartMlxMetalKernelHandle* kernel,
    DartMlxArrayHandle** inputs,
    size_t input_len,
    const DartMlxMetalConfigHandle* config,
    DartMlxArrayHandle*** outputs_out,
    size_t* outputs_len_out) {
  auto input_vec = build_array_vector(inputs, input_len);
  auto outputs = mlx_vector_array_new();
  auto status = mlx_fast_metal_kernel_apply(
      &outputs, kernel->value, input_vec, config->value, default_cpu_stream());
  mlx_vector_array_free(input_vec);
  if (status != 0) {
    return status;
  }
  status = export_vector_array(outputs, outputs_out, outputs_len_out);
  mlx_vector_array_free(outputs);
  return status;
}
