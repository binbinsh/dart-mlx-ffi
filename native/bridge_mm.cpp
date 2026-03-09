#include "bridge.h"

extern "C" int dart_mlx_broadcast_arrays(
    DartMlxArrayHandle** inputs,
    size_t inputs_len,
    DartMlxArrayHandle*** outputs_out,
    size_t* outputs_len_out) {
  auto input_vec = build_array_vector(inputs, inputs_len);
  auto outputs = mlx_vector_array_new();
  auto status = mlx_broadcast_arrays(&outputs, input_vec, default_cpu_stream());
  mlx_vector_array_free(input_vec);
  if (status != 0) {
    return status;
  }
  status = export_vector_array(outputs, outputs_out, outputs_len_out);
  mlx_vector_array_free(outputs);
  return status;
}

extern "C" int dart_mlx_split_sections(
    const DartMlxArrayHandle* input,
    const int* indices,
    size_t indices_len,
    int axis,
    DartMlxArrayHandle*** outputs_out,
    size_t* outputs_len_out) {
  auto outputs = mlx_vector_array_new();
  auto status = mlx_split_sections(
      &outputs, input->value, indices, indices_len, axis, default_cpu_stream());
  if (status != 0) {
    return status;
  }
  status = export_vector_array(outputs, outputs_out, outputs_len_out);
  mlx_vector_array_free(outputs);
  return status;
}

extern "C" DartMlxArrayHandle* dart_mlx_segmented_mm(
    const DartMlxArrayHandle* a,
    const DartMlxArrayHandle* b,
    const DartMlxArrayHandle* segments) {
  mlx_array out = mlx_array_new();
  if (mlx_segmented_mm(
          &out, a->value, b->value, segments->value, default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_block_masked_mm(
    const DartMlxArrayHandle* a,
    const DartMlxArrayHandle* b,
    int block_size,
    const DartMlxArrayHandle* mask_out,
    const DartMlxArrayHandle* mask_lhs,
    const DartMlxArrayHandle* mask_rhs) {
  mlx_array out = mlx_array_new();
  if (mlx_block_masked_mm(
          &out,
          a->value,
          b->value,
          block_size,
          mask_out == nullptr ? mlx_array() : mask_out->value,
          mask_lhs == nullptr ? mlx_array() : mask_lhs->value,
          mask_rhs == nullptr ? mlx_array() : mask_rhs->value,
          default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}
