#include "bridge.h"

extern "C" DartMlxArrayHandle* dart_mlx_greater(
    const DartMlxArrayHandle* a,
    const DartMlxArrayHandle* b) {
  return binary_array_op(a, b, mlx_greater);
}

extern "C" DartMlxArrayHandle* dart_mlx_greater_equal(
    const DartMlxArrayHandle* a,
    const DartMlxArrayHandle* b) {
  return binary_array_op(a, b, mlx_greater_equal);
}

extern "C" DartMlxArrayHandle* dart_mlx_less(
    const DartMlxArrayHandle* a,
    const DartMlxArrayHandle* b) {
  return binary_array_op(a, b, mlx_less);
}

extern "C" DartMlxArrayHandle* dart_mlx_less_equal(
    const DartMlxArrayHandle* a,
    const DartMlxArrayHandle* b) {
  return binary_array_op(a, b, mlx_less_equal);
}

extern "C" DartMlxArrayHandle* dart_mlx_floor_divide(
    const DartMlxArrayHandle* a,
    const DartMlxArrayHandle* b) {
  return binary_array_op(a, b, mlx_floor_divide);
}

extern "C" DartMlxArrayHandle* dart_mlx_logaddexp(
    const DartMlxArrayHandle* a,
    const DartMlxArrayHandle* b) {
  return binary_array_op(a, b, mlx_logaddexp);
}

extern "C" DartMlxArrayHandle* dart_mlx_inner(
    const DartMlxArrayHandle* a,
    const DartMlxArrayHandle* b) {
  return binary_array_op(a, b, mlx_inner);
}

extern "C" DartMlxArrayHandle* dart_mlx_hadamard_transform(
    const DartMlxArrayHandle* input,
    bool has_scale,
    float scale) {
  auto out = mlx_array_new();
  mlx_optional_float opt_scale = {.value = scale, .has_value = has_scale};
  if (mlx_hadamard_transform(&out, input->value, opt_scale, default_device_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_floor(const DartMlxArrayHandle* input) {
  return unary_array_op(input, mlx_floor);
}

extern "C" DartMlxArrayHandle* dart_mlx_sqrt(const DartMlxArrayHandle* input) {
  return unary_array_op(input, mlx_sqrt);
}

extern "C" DartMlxArrayHandle* dart_mlx_rsqrt(const DartMlxArrayHandle* input) {
  return unary_array_op(input, mlx_rsqrt);
}

extern "C" DartMlxArrayHandle* dart_mlx_square(const DartMlxArrayHandle* input) {
  return unary_array_op(input, mlx_square);
}

extern "C" DartMlxArrayHandle* dart_mlx_reciprocal(
    const DartMlxArrayHandle* input) {
  return unary_array_op(input, mlx_reciprocal);
}

extern "C" DartMlxArrayHandle* dart_mlx_sigmoid(
    const DartMlxArrayHandle* input) {
  return unary_array_op(input, mlx_sigmoid);
}

extern "C" DartMlxArrayHandle* dart_mlx_degrees(
    const DartMlxArrayHandle* input) {
  return unary_array_op(input, mlx_degrees);
}

extern "C" DartMlxArrayHandle* dart_mlx_radians(
    const DartMlxArrayHandle* input) {
  return unary_array_op(input, mlx_radians);
}

extern "C" DartMlxArrayHandle* dart_mlx_expm1(const DartMlxArrayHandle* input) {
  return unary_array_op(input, mlx_expm1);
}

extern "C" DartMlxArrayHandle* dart_mlx_erf(const DartMlxArrayHandle* input) {
  return unary_array_op(input, mlx_erf);
}

extern "C" DartMlxArrayHandle* dart_mlx_erfinv(
    const DartMlxArrayHandle* input) {
  return unary_array_op(input, mlx_erfinv);
}

extern "C" DartMlxArrayHandle* dart_mlx_log1p(const DartMlxArrayHandle* input) {
  return unary_array_op(input, mlx_log1p);
}

extern "C" DartMlxArrayHandle* dart_mlx_log2(const DartMlxArrayHandle* input) {
  return unary_array_op(input, mlx_log2);
}

extern "C" DartMlxArrayHandle* dart_mlx_log10(const DartMlxArrayHandle* input) {
  return unary_array_op(input, mlx_log10);
}

extern "C" DartMlxArrayHandle* dart_mlx_round(
    const DartMlxArrayHandle* input,
    int decimals) {
  auto out = mlx_array_new();
  if (mlx_round(&out, input->value, decimals, default_device_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_stop_gradient(
    const DartMlxArrayHandle* input) {
  return unary_array_op(input, mlx_stop_gradient);
}

extern "C" DartMlxArrayHandle* dart_mlx_isfinite(
    const DartMlxArrayHandle* input) {
  return unary_array_op(input, mlx_isfinite);
}

extern "C" DartMlxArrayHandle* dart_mlx_isinf(
    const DartMlxArrayHandle* input) {
  return unary_array_op(input, mlx_isinf);
}

extern "C" DartMlxArrayHandle* dart_mlx_isnan(
    const DartMlxArrayHandle* input) {
  return unary_array_op(input, mlx_isnan);
}

extern "C" DartMlxArrayHandle* dart_mlx_isneginf(
    const DartMlxArrayHandle* input) {
  return unary_array_op(input, mlx_isneginf);
}

extern "C" DartMlxArrayHandle* dart_mlx_isposinf(
    const DartMlxArrayHandle* input) {
  return unary_array_op(input, mlx_isposinf);
}

extern "C" DartMlxArrayHandle* dart_mlx_zeros_like(
    const DartMlxArrayHandle* input) {
  return unary_array_op(input, mlx_zeros_like);
}

extern "C" DartMlxArrayHandle* dart_mlx_ones_like(
    const DartMlxArrayHandle* input) {
  return unary_array_op(input, mlx_ones_like);
}

extern "C" DartMlxArrayHandle* dart_mlx_full_like(
    const DartMlxArrayHandle* input,
    double value,
    int dtype) {
  auto scalar = mlx_array_new_double(value);
  auto out = mlx_array_new();
  auto status =
      mlx_full_like(&out, input->value, scalar, as_dtype(dtype), default_cpu_stream());
  mlx_array_free(scalar);
  if (status != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_to_fp8(const DartMlxArrayHandle* input) {
  return unary_array_op(input, mlx_to_fp8);
}

extern "C" DartMlxArrayHandle* dart_mlx_from_fp8(
    const DartMlxArrayHandle* input,
    int dtype) {
  auto out = mlx_array_new();
  if (mlx_from_fp8(&out, input->value, as_dtype(dtype), default_device_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_put_along_axis(
    const DartMlxArrayHandle* input,
    const DartMlxArrayHandle* indices,
    const DartMlxArrayHandle* values,
    int axis) {
  auto out = mlx_array_new();
  if (mlx_put_along_axis(
          &out, input->value, indices->value, values->value, axis, default_device_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_scatter_add_axis(
    const DartMlxArrayHandle* input,
    const DartMlxArrayHandle* indices,
    const DartMlxArrayHandle* values,
    int axis) {
  auto out = mlx_array_new();
  if (mlx_scatter_add_axis(
          &out, input->value, indices->value, values->value, axis, default_device_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}
