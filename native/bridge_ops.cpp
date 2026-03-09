#include "bridge.h"

extern "C" DartMlxArrayHandle* dart_mlx_add(
    const DartMlxArrayHandle* lhs,
    const DartMlxArrayHandle* rhs) {
  return binary_array_op(lhs, rhs, mlx_add);
}

extern "C" DartMlxArrayHandle* dart_mlx_subtract(
    const DartMlxArrayHandle* lhs,
    const DartMlxArrayHandle* rhs) {
  return binary_array_op(lhs, rhs, mlx_subtract);
}

extern "C" DartMlxArrayHandle* dart_mlx_multiply(
    const DartMlxArrayHandle* lhs,
    const DartMlxArrayHandle* rhs) {
  return binary_array_op(lhs, rhs, mlx_multiply);
}

extern "C" DartMlxArrayHandle* dart_mlx_divide(
    const DartMlxArrayHandle* lhs,
    const DartMlxArrayHandle* rhs) {
  return binary_array_op(lhs, rhs, mlx_divide);
}

extern "C" DartMlxArrayHandle* dart_mlx_matmul(
    const DartMlxArrayHandle* lhs,
    const DartMlxArrayHandle* rhs) {
  return binary_array_op(lhs, rhs, mlx_matmul);
}

extern "C" DartMlxArrayHandle* dart_mlx_equal(
    const DartMlxArrayHandle* lhs,
    const DartMlxArrayHandle* rhs) {
  return binary_array_op(lhs, rhs, mlx_equal);
}

extern "C" DartMlxArrayHandle* dart_mlx_where(
    const DartMlxArrayHandle* condition,
    const DartMlxArrayHandle* lhs,
    const DartMlxArrayHandle* rhs) {
  mlx_array out = mlx_array_new();
  if (mlx_where(
          &out,
          condition->value,
          lhs->value,
          rhs->value,
          default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_abs(const DartMlxArrayHandle* input) {
  return unary_array_op(input, mlx_abs);
}

extern "C" DartMlxArrayHandle* dart_mlx_negative(
    const DartMlxArrayHandle* input) {
  return unary_array_op(input, mlx_negative);
}

extern "C" DartMlxArrayHandle* dart_mlx_exp(const DartMlxArrayHandle* input) {
  return unary_array_op(input, mlx_exp);
}

extern "C" DartMlxArrayHandle* dart_mlx_log(const DartMlxArrayHandle* input) {
  return unary_array_op(input, mlx_log);
}

extern "C" DartMlxArrayHandle* dart_mlx_sin(const DartMlxArrayHandle* input) {
  return unary_array_op(input, mlx_sin);
}

extern "C" DartMlxArrayHandle* dart_mlx_cos(const DartMlxArrayHandle* input) {
  return unary_array_op(input, mlx_cos);
}

extern "C" DartMlxArrayHandle* dart_mlx_zeros(
    const int* shape,
    int dim,
    int dtype) {
  mlx_array out = mlx_array_new();
  if (mlx_zeros(&out, shape, dim, as_dtype(dtype), default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_ones(
    const int* shape,
    int dim,
    int dtype) {
  mlx_array out = mlx_array_new();
  if (mlx_ones(&out, shape, dim, as_dtype(dtype), default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_full(
    const int* shape,
    int dim,
    double value,
    int dtype) {
  mlx_array scalar = mlx_array_new_double(value);
  mlx_array out = mlx_array_new();
  if (mlx_full(
          &out,
          shape,
          dim,
          scalar,
          as_dtype(dtype),
          default_cpu_stream()) != 0) {
    mlx_array_free(scalar);
    return nullptr;
  }
  mlx_array_free(scalar);
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_arange(
    double start,
    double stop,
    double step,
    int dtype) {
  mlx_array out = mlx_array_new();
  if (mlx_arange(
          &out,
          start,
          stop,
          step,
          as_dtype(dtype),
          default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_astype(
    const DartMlxArrayHandle* input,
    int dtype) {
  mlx_array out = mlx_array_new();
  if (mlx_astype(&out, input->value, as_dtype(dtype), default_device_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_reshape(
    const DartMlxArrayHandle* input,
    const int* shape,
    int dim) {
  mlx_array out = mlx_array_new();
  if (mlx_reshape(&out, input->value, shape, dim, default_device_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_transpose(
    const DartMlxArrayHandle* input) {
  mlx_array out = mlx_array_new();
  if (mlx_transpose(&out, input->value, default_device_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_sum(
    const DartMlxArrayHandle* input,
    bool keepdims) {
  mlx_array out = mlx_array_new();
  if (mlx_sum(&out, input->value, keepdims, default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_sum_axis(
    const DartMlxArrayHandle* input,
    int axis,
    bool keepdims) {
  mlx_array out = mlx_array_new();
  if (mlx_sum_axis(
          &out, input->value, axis, keepdims, default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_mean(
    const DartMlxArrayHandle* input,
    bool keepdims) {
  mlx_array out = mlx_array_new();
  if (mlx_mean(&out, input->value, keepdims, default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_mean_axis(
    const DartMlxArrayHandle* input,
    int axis,
    bool keepdims) {
  mlx_array out = mlx_array_new();
  if (mlx_mean_axis(
          &out, input->value, axis, keepdims, default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_random_uniform(
    double low,
    double high,
    const int* shape,
    int dim,
    int dtype) {
  mlx_array low_arr = mlx_array_new_double(low);
  mlx_array high_arr = mlx_array_new_double(high);
  mlx_array out = mlx_array_new();
  auto status = mlx_random_uniform(
      &out,
      low_arr,
      high_arr,
      shape,
      dim,
      as_dtype(dtype),
      mlx_array(),
      default_cpu_stream());
  mlx_array_free(low_arr);
  mlx_array_free(high_arr);
  if (status != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_random_normal(
    const int* shape,
    int dim,
    int dtype,
    double loc,
    double scale) {
  mlx_array out = mlx_array_new();
  if (mlx_random_normal(
          &out,
          shape,
          dim,
          as_dtype(dtype),
          loc,
          scale,
          mlx_array(),
          default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_fft_fft(
    const DartMlxArrayHandle* input,
    int n,
    int axis) {
  mlx_array out = mlx_array_new();
  const auto target_n = resolved_n(input, axis, n);
  const auto target_axis = normalized_axis(input, axis);
  if (mlx_fft_fft(
          &out, input->value, target_n, target_axis, default_cpu_stream()) !=
      0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_fft_ifft(
    const DartMlxArrayHandle* input,
    int n,
    int axis) {
  mlx_array out = mlx_array_new();
  const auto target_n = resolved_n(input, axis, n);
  const auto target_axis = normalized_axis(input, axis);
  if (mlx_fft_ifft(
          &out, input->value, target_n, target_axis, default_cpu_stream()) !=
      0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_fft_rfft(
    const DartMlxArrayHandle* input,
    int n,
    int axis) {
  mlx_array out = mlx_array_new();
  const auto target_n = resolved_n(input, axis, n);
  const auto target_axis = normalized_axis(input, axis);
  if (mlx_fft_rfft(
          &out, input->value, target_n, target_axis, default_cpu_stream()) !=
      0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_fft_irfft(
    const DartMlxArrayHandle* input,
    int n,
    int axis) {
  mlx_array out = mlx_array_new();
  const auto target_n = resolved_n(input, axis, n);
  const auto target_axis = normalized_axis(input, axis);
  if (mlx_fft_irfft(
          &out, input->value, target_n, target_axis, default_cpu_stream()) !=
      0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_linalg_inv(
    const DartMlxArrayHandle* input) {
  mlx_array out = mlx_array_new();
  if (mlx_linalg_inv(&out, input->value, default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_linalg_solve(
    const DartMlxArrayHandle* a,
    const DartMlxArrayHandle* b) {
  mlx_array out = mlx_array_new();
  if (mlx_linalg_solve(&out, a->value, b->value, default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" int dart_mlx_linalg_qr(
    const DartMlxArrayHandle* input,
    DartMlxArrayHandle** q,
    DartMlxArrayHandle** r) {
  mlx_array out_q = mlx_array_new();
  mlx_array out_r = mlx_array_new();
  if (mlx_linalg_qr(&out_q, &out_r, input->value, default_cpu_stream()) != 0) {
    return 1;
  }
  *q = wrap_array(out_q);
  *r = wrap_array(out_r);
  return (*q == nullptr || *r == nullptr) ? 1 : 0;
}

extern "C" int dart_mlx_linalg_eig(
    const DartMlxArrayHandle* input,
    DartMlxArrayHandle** values,
    DartMlxArrayHandle** vectors) {
  mlx_array out_values = mlx_array_new();
  mlx_array out_vectors = mlx_array_new();
  if (mlx_linalg_eig(
          &out_values, &out_vectors, input->value, default_cpu_stream()) != 0) {
    return 1;
  }
  *values = wrap_array(out_values);
  *vectors = wrap_array(out_vectors);
  return (*values == nullptr || *vectors == nullptr) ? 1 : 0;
}

extern "C" int dart_mlx_quantize(
    const DartMlxArrayHandle* weights,
    bool has_group_size,
    int group_size,
    bool has_bits,
    int bits,
    const char* mode,
    const DartMlxArrayHandle* global_scale,
    DartMlxArrayHandle*** outputs_out,
    size_t* outputs_len_out) {
  auto results = mlx_vector_array_new();
  mlx_optional_int opt_group_size = {.value = group_size, .has_value = has_group_size};
  mlx_optional_int opt_bits = {.value = bits, .has_value = has_bits};
  auto status = mlx_quantize(
      &results,
      weights->value,
      opt_group_size,
      opt_bits,
      mode,
      global_scale == nullptr ? mlx_array() : global_scale->value,
      default_device_stream());
  if (status != 0) {
    return status;
  }
  status = export_vector_array(results, outputs_out, outputs_len_out);
  mlx_vector_array_free(results);
  return status;
}

extern "C" DartMlxArrayHandle* dart_mlx_dequantize(
    const DartMlxArrayHandle* weights,
    const DartMlxArrayHandle* scales,
    const DartMlxArrayHandle* biases,
    bool has_group_size,
    int group_size,
    bool has_bits,
    int bits,
    const char* mode,
    const DartMlxArrayHandle* global_scale,
    bool has_dtype,
    int dtype) {
  mlx_array out = mlx_array_new();
  mlx_optional_int opt_group_size = {.value = group_size, .has_value = has_group_size};
  mlx_optional_int opt_bits = {.value = bits, .has_value = has_bits};
  mlx_optional_dtype opt_dtype = {.value = as_dtype(dtype), .has_value = has_dtype};
  auto status = mlx_dequantize(
      &out,
      weights->value,
      scales->value,
      biases == nullptr ? mlx_array() : biases->value,
      opt_group_size,
      opt_bits,
      mode,
      global_scale == nullptr ? mlx_array() : global_scale->value,
      opt_dtype,
      default_device_stream());
  if (status != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_quantized_matmul(
    const DartMlxArrayHandle* x,
    const DartMlxArrayHandle* weights,
    const DartMlxArrayHandle* scales,
    const DartMlxArrayHandle* biases,
    bool transpose,
    bool has_group_size,
    int group_size,
    bool has_bits,
    int bits,
    const char* mode) {
  mlx_array out = mlx_array_new();
  mlx_optional_int opt_group_size = {.value = group_size, .has_value = has_group_size};
  mlx_optional_int opt_bits = {.value = bits, .has_value = has_bits};
  auto status = mlx_quantized_matmul(
      &out,
      x->value,
      weights->value,
      scales->value,
      biases == nullptr ? mlx_array() : biases->value,
      transpose,
      opt_group_size,
      opt_bits,
      mode,
      default_device_stream());
  if (status != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_qqmm(
    const DartMlxArrayHandle* x,
    const DartMlxArrayHandle* weights,
    const DartMlxArrayHandle* weight_scales,
    bool has_group_size,
    int group_size,
    bool has_bits,
    int bits,
    const char* mode,
    const DartMlxArrayHandle* global_scale_x,
    const DartMlxArrayHandle* global_scale_w) {
  mlx_array out = mlx_array_new();
  mlx_optional_int opt_group_size = {.value = group_size, .has_value = has_group_size};
  mlx_optional_int opt_bits = {.value = bits, .has_value = has_bits};
  auto status = mlx_qqmm(
      &out,
      x->value,
      weights->value,
      weight_scales == nullptr ? mlx_array() : weight_scales->value,
      opt_group_size,
      opt_bits,
      mode,
      global_scale_x == nullptr ? mlx_array() : global_scale_x->value,
      global_scale_w == nullptr ? mlx_array() : global_scale_w->value,
      default_device_stream());
  if (status != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_gather_qmm(
    const DartMlxArrayHandle* x,
    const DartMlxArrayHandle* weights,
    const DartMlxArrayHandle* scales,
    const DartMlxArrayHandle* biases,
    const DartMlxArrayHandle* lhs_indices,
    const DartMlxArrayHandle* rhs_indices,
    bool transpose,
    bool has_group_size,
    int group_size,
    bool has_bits,
    int bits,
    const char* mode,
    bool sorted_indices) {
  mlx_array out = mlx_array_new();
  mlx_optional_int opt_group_size = {.value = group_size, .has_value = has_group_size};
  mlx_optional_int opt_bits = {.value = bits, .has_value = has_bits};
  auto status = mlx_gather_qmm(
      &out,
      x->value,
      weights->value,
      scales->value,
      biases == nullptr ? mlx_array() : biases->value,
      lhs_indices == nullptr ? mlx_array() : lhs_indices->value,
      rhs_indices == nullptr ? mlx_array() : rhs_indices->value,
      transpose,
      opt_group_size,
      opt_bits,
      mode,
      sorted_indices,
      default_device_stream());
  if (status != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_fast_layer_norm(
    const DartMlxArrayHandle* input,
    const DartMlxArrayHandle* weight,
    const DartMlxArrayHandle* bias,
    float eps) {
  mlx_array out = mlx_array_new();
  auto status = mlx_fast_layer_norm(
      &out,
      input->value,
      weight == nullptr ? mlx_array() : weight->value,
      bias == nullptr ? mlx_array() : bias->value,
      eps,
      default_device_stream());
  if (status != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_fast_rms_norm(
    const DartMlxArrayHandle* input,
    const DartMlxArrayHandle* weight,
    float eps) {
  mlx_array out = mlx_array_new();
  auto status = mlx_fast_rms_norm(
      &out,
      input->value,
      weight == nullptr ? mlx_array() : weight->value,
      eps,
      default_device_stream());
  if (status != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_fast_rope(
    const DartMlxArrayHandle* input,
    int dims,
    bool traditional,
    bool has_base,
    float base,
    float scale,
    int offset,
    const DartMlxArrayHandle* freqs) {
  mlx_array out = mlx_array_new();
  mlx_optional_float opt_base = {.value = base, .has_value = has_base};
  auto status = mlx_fast_rope(
      &out,
      input->value,
      dims,
      traditional,
      opt_base,
      scale,
      offset,
      freqs == nullptr ? mlx_array() : freqs->value,
      default_device_stream());
  if (status != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_fast_rope_dynamic(
    const DartMlxArrayHandle* input,
    int dims,
    bool traditional,
    bool has_base,
    float base,
    float scale,
    const DartMlxArrayHandle* offset,
    const DartMlxArrayHandle* freqs) {
  mlx_array out = mlx_array_new();
  mlx_optional_float opt_base = {.value = base, .has_value = has_base};
  auto status = mlx_fast_rope_dynamic(
      &out,
      input->value,
      dims,
      traditional,
      opt_base,
      scale,
      offset->value,
      freqs == nullptr ? mlx_array() : freqs->value,
      default_device_stream());
  if (status != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_fast_sdpa(
    const DartMlxArrayHandle* queries,
    const DartMlxArrayHandle* keys,
    const DartMlxArrayHandle* values,
    float scale,
    const char* mask_mode,
    const DartMlxArrayHandle* mask,
    const DartMlxArrayHandle* sinks) {
  mlx_array out = mlx_array_new();
  auto status = mlx_fast_scaled_dot_product_attention(
      &out,
      queries->value,
      keys->value,
      values->value,
      scale,
      mask_mode,
      mask == nullptr ? mlx_array() : mask->value,
      sinks == nullptr ? mlx_array() : sinks->value,
      default_device_stream());
  if (status != 0) {
    return nullptr;
  }
  return wrap_array(out);
}
