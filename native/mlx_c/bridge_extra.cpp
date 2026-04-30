#include "bridge.h"

namespace {

int scatter_status(
    int op,
    mlx_array* out,
    mlx_array input,
    mlx_vector_array indices,
    mlx_array updates,
    const int* axes,
    size_t axes_len) {
  switch (op) {
    case 1:
      return mlx_scatter_add(
          out, input, indices, updates, axes, axes_len, default_device_stream());
    case 2:
      return mlx_scatter_max(
          out, input, indices, updates, axes, axes_len, default_device_stream());
    case 3:
      return mlx_scatter_min(
          out, input, indices, updates, axes, axes_len, default_device_stream());
    case 4:
      return mlx_scatter_prod(
          out, input, indices, updates, axes, axes_len, default_device_stream());
    default:
      return mlx_scatter(
          out, input, indices, updates, axes, axes_len, default_device_stream());
  }
}

int scatter_single_status(
    int op,
    mlx_array* out,
    mlx_array input,
    mlx_array indices,
    mlx_array updates,
    int axis) {
  switch (op) {
    case 1:
      return mlx_scatter_add_single(
          out, input, indices, updates, axis, default_device_stream());
    case 2:
      return mlx_scatter_max_single(
          out, input, indices, updates, axis, default_device_stream());
    case 3:
      return mlx_scatter_min_single(
          out, input, indices, updates, axis, default_device_stream());
    case 4:
      return mlx_scatter_prod_single(
          out, input, indices, updates, axis, default_device_stream());
    default:
      return mlx_scatter_single(
          out, input, indices, updates, axis, default_device_stream());
  }
}

} // namespace

extern "C" DartMlxArrayHandle* dart_mlx_diag(
    const DartMlxArrayHandle* input,
    int k) {
  auto out = mlx_array_new();
  if (mlx_diag(&out, input->value, k, default_device_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_diagonal(
    const DartMlxArrayHandle* input,
    int offset,
    int axis1,
    int axis2) {
  auto out = mlx_array_new();
  if (mlx_diagonal(
          &out, input->value, offset, axis1, axis2, default_device_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_kron(
    const DartMlxArrayHandle* a,
    const DartMlxArrayHandle* b) {
  auto out = mlx_array_new();
  if (mlx_kron(&out, a->value, b->value, default_device_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" int dart_mlx_meshgrid(
    DartMlxArrayHandle** inputs,
    size_t inputs_len,
    bool sparse,
    const char* indexing,
    DartMlxArrayHandle*** outputs_out,
    size_t* outputs_len_out) {
  auto input_vec = build_array_vector(inputs, inputs_len);
  auto outputs = mlx_vector_array_new();
  auto status =
      mlx_meshgrid(&outputs, input_vec, sparse, indexing, default_device_stream());
  mlx_vector_array_free(input_vec);
  if (status != 0) {
    return status;
  }
  status = export_vector_array(outputs, outputs_out, outputs_len_out);
  mlx_vector_array_free(outputs);
  return status;
}

extern "C" DartMlxArrayHandle* dart_mlx_partition(
    const DartMlxArrayHandle* input,
    int kth,
    int axis,
    bool has_axis) {
  auto out = mlx_array_new();
  auto status = has_axis
      ? mlx_partition_axis(&out, input->value, kth, axis, default_device_stream())
      : mlx_partition(&out, input->value, kth, default_device_stream());
  if (status != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_scatter(
    const DartMlxArrayHandle* input,
    DartMlxArrayHandle** indices,
    size_t indices_len,
    const DartMlxArrayHandle* updates,
    const int* axes,
    size_t axes_len,
    int op,
    int single_axis) {
  auto out = mlx_array_new();
  if (single_axis >= 0) {
    if (scatter_single_status(
            op,
            &out,
            input->value,
            indices[0]->value,
            updates->value,
            single_axis) != 0) {
      return nullptr;
    }
    return wrap_array(out);
  }
  auto index_vec = build_array_vector(indices, indices_len);
  auto status = scatter_status(
      op, &out, input->value, index_vec, updates->value, axes, axes_len);
  mlx_vector_array_free(index_vec);
  if (status != 0) {
    return nullptr;
  }
  return wrap_array(out);
}
