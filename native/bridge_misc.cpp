#include "bridge.h"

extern "C" DartMlxArrayHandle* dart_mlx_linspace(
    double start,
    double stop,
    int num,
    int dtype) {
  auto out = mlx_array_new();
  if (mlx_linspace(
          &out, start, stop, num, as_dtype(dtype), default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_outer(
    const DartMlxArrayHandle* a,
    const DartMlxArrayHandle* b) {
  return binary_array_op(a, b, mlx_outer);
}

extern "C" DartMlxArrayHandle* dart_mlx_isclose(
    const DartMlxArrayHandle* a,
    const DartMlxArrayHandle* b,
    double rtol,
    double atol,
    bool equal_nan) {
  auto out = mlx_array_new();
  if (mlx_isclose(
          &out, a->value, b->value, rtol, atol, equal_nan, default_cpu_stream()) !=
      0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_logical_and(
    const DartMlxArrayHandle* a,
    const DartMlxArrayHandle* b) {
  return binary_array_op(a, b, mlx_logical_and);
}

extern "C" DartMlxArrayHandle* dart_mlx_logical_or(
    const DartMlxArrayHandle* a,
    const DartMlxArrayHandle* b) {
  return binary_array_op(a, b, mlx_logical_or);
}

extern "C" DartMlxArrayHandle* dart_mlx_logical_not(
    const DartMlxArrayHandle* input) {
  return unary_array_op(input, mlx_logical_not);
}

extern "C" DartMlxArrayHandle* dart_mlx_repeat(
    const DartMlxArrayHandle* input,
    int repeats,
    int axis,
    bool has_axis) {
  auto out = mlx_array_new();
  auto status = has_axis
      ? mlx_repeat_axis(&out, input->value, repeats, axis, default_cpu_stream())
      : mlx_repeat(&out, input->value, repeats, default_cpu_stream());
  if (status != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_roll(
    const DartMlxArrayHandle* input,
    const int* shift,
    int shift_len,
    const int* axes,
    int axes_len,
    int axis,
    int mode) {
  auto out = mlx_array_new();
  int status = 0;
  switch (mode) {
    case 1:
      status = mlx_roll_axis(
          &out, input->value, shift, shift_len, axis, default_cpu_stream());
      break;
    case 2:
      status = mlx_roll_axes(
          &out, input->value, shift, shift_len, axes, axes_len, default_cpu_stream());
      break;
    default:
      status = mlx_roll(&out, input->value, shift, shift_len, default_cpu_stream());
      break;
  }
  if (status != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_median(
    const DartMlxArrayHandle* input,
    const int* axes,
    int axes_len,
    bool keepdims) {
  auto out = mlx_array_new();
  if (mlx_median(
          &out, input->value, axes, axes_len, keepdims, default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_masked_scatter(
    const DartMlxArrayHandle* input,
    const DartMlxArrayHandle* mask,
    const DartMlxArrayHandle* values) {
  auto out = mlx_array_new();
  if (mlx_masked_scatter(
          &out, input->value, mask->value, values->value, default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_nan_to_num(
    const DartMlxArrayHandle* input,
    float nan,
    bool has_pos_inf,
    float pos_inf,
    bool has_neg_inf,
    float neg_inf) {
  auto out = mlx_array_new();
  mlx_optional_float opt_pos = {.value = pos_inf, .has_value = has_pos_inf};
  mlx_optional_float opt_neg = {.value = neg_inf, .has_value = has_neg_inf};
  if (mlx_nan_to_num(
          &out, input->value, nan, opt_pos, opt_neg, default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" int dart_mlx_divmod(
    const DartMlxArrayHandle* a,
    const DartMlxArrayHandle* b,
    DartMlxArrayHandle*** outputs_out,
    size_t* outputs_len_out) {
  auto outputs = mlx_vector_array_new();
  auto status = mlx_divmod(&outputs, a->value, b->value, default_cpu_stream());
  if (status != 0) {
    return status;
  }
  status = export_vector_array(outputs, outputs_out, outputs_len_out);
  mlx_vector_array_free(outputs);
  return status;
}
