#include "bridge.h"

extern "C" int dart_mlx_eval_many(DartMlxArrayHandle** arrays, size_t len) {
  auto values = mlx_vector_array_new();
  for (size_t i = 0; i < len; ++i) {
    if (mlx_vector_array_append_value(values, arrays[i]->value) != 0) {
      mlx_vector_array_free(values);
      return 1;
    }
  }
  auto status = mlx_eval(values);
  mlx_vector_array_free(values);
  return status;
}

extern "C" int dart_mlx_async_eval_many(
    DartMlxArrayHandle** arrays,
    size_t len) {
  auto values = build_array_vector(arrays, len);
  auto status = mlx_async_eval(values);
  mlx_vector_array_free(values);
  return status;
}

extern "C" DartMlxArrayHandle* dart_mlx_concatenate(
    DartMlxArrayHandle** arrays,
    size_t len,
    int axis) {
  auto values = build_array_vector(arrays, len);
  mlx_array out = mlx_array_new();
  auto status = mlx_concatenate_axis(&out, values, axis, default_device_stream());
  mlx_vector_array_free(values);
  if (status != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_stack(
    DartMlxArrayHandle** arrays,
    size_t len,
    int axis) {
  auto values = build_array_vector(arrays, len);
  mlx_array out = mlx_array_new();
  auto status = mlx_stack_axis(&out, values, axis, default_device_stream());
  mlx_vector_array_free(values);
  if (status != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_broadcast_to(
    const DartMlxArrayHandle* input,
    const int* shape,
    int dim) {
  mlx_array out = mlx_array_new();
  if (mlx_broadcast_to(&out, input->value, shape, dim, default_device_stream()) !=
      0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_expand_dims(
    const DartMlxArrayHandle* input,
    int axis) {
  mlx_array out = mlx_array_new();
  if (mlx_expand_dims(&out, input->value, axis, default_device_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_squeeze(const DartMlxArrayHandle* input) {
  mlx_array out = mlx_array_new();
  if (mlx_squeeze(&out, input->value, default_device_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_clip_scalar(
    const DartMlxArrayHandle* input,
    bool has_min,
    double min_value,
    bool has_max,
    double max_value) {
  auto min_arr = mlx_array_new();
  auto max_arr = mlx_array_new();
  if (has_min) {
    min_arr = mlx_array_new_double(min_value);
  }
  if (has_max) {
    max_arr = mlx_array_new_double(max_value);
  }
  mlx_array out = mlx_array_new();
  auto status = mlx_clip(
      &out,
      input->value,
      has_min ? min_arr : mlx_array(),
      has_max ? max_arr : mlx_array(),
      default_cpu_stream());
  if (has_min) {
    mlx_array_free(min_arr);
  }
  if (has_max) {
    mlx_array_free(max_arr);
  }
  if (status != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_minimum(
    const DartMlxArrayHandle* lhs,
    const DartMlxArrayHandle* rhs) {
  return binary_array_op(lhs, rhs, mlx_minimum);
}

extern "C" DartMlxArrayHandle* dart_mlx_maximum(
    const DartMlxArrayHandle* lhs,
    const DartMlxArrayHandle* rhs) {
  return binary_array_op(lhs, rhs, mlx_maximum);
}

extern "C" DartMlxArrayHandle* dart_mlx_argmax(
    const DartMlxArrayHandle* input,
    int axis,
    bool has_axis,
    bool keepdims) {
  mlx_array out = mlx_array_new();
  auto status = has_axis
      ? mlx_argmax_axis(&out, input->value, axis, keepdims, default_device_stream())
      : mlx_argmax(&out, input->value, keepdims, default_device_stream());
  if (status != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_argmin(
    const DartMlxArrayHandle* input,
    int axis,
    bool has_axis,
    bool keepdims) {
  mlx_array out = mlx_array_new();
  auto status = has_axis
      ? mlx_argmin_axis(&out, input->value, axis, keepdims, default_device_stream())
      : mlx_argmin(&out, input->value, keepdims, default_device_stream());
  if (status != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_sort(
    const DartMlxArrayHandle* input,
    int axis,
    bool has_axis) {
  mlx_array out = mlx_array_new();
  auto status = has_axis ? mlx_sort_axis(&out, input->value, axis, default_device_stream())
                         : mlx_sort(&out, input->value, default_device_stream());
  if (status != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_argsort(
    const DartMlxArrayHandle* input,
    int axis,
    bool has_axis) {
  mlx_array out = mlx_array_new();
  auto status = has_axis ? mlx_argsort_axis(&out, input->value, axis, default_device_stream())
                         : mlx_argsort(&out, input->value, default_device_stream());
  if (status != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_flatten(
    const DartMlxArrayHandle* input,
    int start_axis,
    int end_axis) {
  mlx_array out = mlx_array_new();
  if (mlx_flatten(
          &out, input->value, start_axis, end_axis, default_device_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_moveaxis(
    const DartMlxArrayHandle* input,
    int source,
    int destination) {
  mlx_array out = mlx_array_new();
  if (mlx_moveaxis(
          &out, input->value, source, destination, default_device_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_swapaxes(
    const DartMlxArrayHandle* input,
    int axis1,
    int axis2) {
  mlx_array out = mlx_array_new();
  if (mlx_swapaxes(
          &out, input->value, axis1, axis2, default_device_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_transpose_axes(
    const DartMlxArrayHandle* input,
    const int* axes,
    size_t axes_len) {
  mlx_array out = mlx_array_new();
  if (mlx_transpose_axes(
          &out, input->value, axes, axes_len, default_device_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_tile(
    const DartMlxArrayHandle* input,
    const int* reps,
    size_t reps_len) {
  mlx_array out = mlx_array_new();
  if (mlx_tile(
          &out, input->value, reps, reps_len, default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_pad(
    const DartMlxArrayHandle* input,
    const int* axes,
    size_t axes_len,
    const int* low_pads,
    size_t low_pads_len,
    const int* high_pads,
    size_t high_pads_len,
    const DartMlxArrayHandle* pad_value,
    const char* mode) {
  mlx_array out = mlx_array_new();
  if (mlx_pad(
          &out,
          input->value,
          axes,
          axes_len,
          low_pads,
          low_pads_len,
          high_pads,
          high_pads_len,
          pad_value == nullptr ? mlx_array() : pad_value->value,
          mode,
          default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_pad_symmetric(
    const DartMlxArrayHandle* input,
    int pad_width,
    const DartMlxArrayHandle* pad_value,
    const char* mode) {
  mlx_array out = mlx_array_new();
  if (mlx_pad_symmetric(
          &out,
          input->value,
          pad_width,
          pad_value == nullptr ? mlx_array() : pad_value->value,
          mode,
          default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_unflatten(
    const DartMlxArrayHandle* input,
    int axis,
    const int* shape,
    size_t shape_len) {
  mlx_array out = mlx_array_new();
  if (mlx_unflatten(
          &out, input->value, axis, shape, shape_len, default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_logsumexp(
    const DartMlxArrayHandle* input,
    int axis,
    bool has_axis,
    bool keepdims) {
  mlx_array out = mlx_array_new();
  auto status = has_axis
      ? mlx_logsumexp_axis(
            &out, input->value, axis, keepdims, default_device_stream())
      : mlx_logsumexp(&out, input->value, keepdims, default_device_stream());
  if (status != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_softmax(
    const DartMlxArrayHandle* input,
    int axis,
    bool has_axis,
    bool precise) {
  mlx_array out = mlx_array_new();
  auto status = has_axis
      ? mlx_softmax_axis(&out, input->value, axis, precise, default_device_stream())
      : mlx_softmax(&out, input->value, precise, default_device_stream());
  if (status != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_topk(
    const DartMlxArrayHandle* input,
    int k,
    int axis,
    bool has_axis) {
  mlx_array out = mlx_array_new();
  auto status = has_axis
      ? mlx_topk_axis(&out, input->value, k, axis, default_device_stream())
      : mlx_topk(&out, input->value, k, default_device_stream());
  if (status != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_take(
    const DartMlxArrayHandle* input,
    const DartMlxArrayHandle* indices,
    int axis,
    bool has_axis) {
  mlx_array out = mlx_array_new();
  auto status = has_axis
      ? mlx_take_axis(&out, input->value, indices->value, axis, default_device_stream())
      : mlx_take(&out, input->value, indices->value, default_device_stream());
  if (status != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_take_along_axis(
    const DartMlxArrayHandle* input,
    const DartMlxArrayHandle* indices,
    int axis) {
  mlx_array out = mlx_array_new();
  if (mlx_take_along_axis(
          &out, input->value, indices->value, axis, default_device_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_gather(
    const DartMlxArrayHandle* input,
    DartMlxArrayHandle** indices,
    size_t indices_len,
    const int* axes,
    size_t axes_len,
    const int* slice_sizes,
    size_t slice_sizes_len) {
  auto index_vec = build_array_vector(indices, indices_len);
  mlx_array out = mlx_array_new();
  auto status = mlx_gather(
      &out,
      input->value,
      index_vec,
      axes,
      axes_len,
      slice_sizes,
      slice_sizes_len,
      default_cpu_stream());
  mlx_vector_array_free(index_vec);
  if (status != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_gather_single(
    const DartMlxArrayHandle* input,
    const DartMlxArrayHandle* indices,
    int axis,
    const int* slice_sizes,
    size_t slice_sizes_len) {
  mlx_array out = mlx_array_new();
  if (mlx_gather_single(
          &out,
          input->value,
          indices->value,
          axis,
          slice_sizes,
          slice_sizes_len,
          default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_gather_mm(
    const DartMlxArrayHandle* a,
    const DartMlxArrayHandle* b,
    const DartMlxArrayHandle* lhs_indices,
    const DartMlxArrayHandle* rhs_indices,
    bool sorted_indices) {
  mlx_array out = mlx_array_new();
  if (mlx_gather_mm(
          &out,
          a->value,
          b->value,
          lhs_indices == nullptr ? mlx_array() : lhs_indices->value,
          rhs_indices == nullptr ? mlx_array() : rhs_indices->value,
          sorted_indices,
          default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_slice(
    const DartMlxArrayHandle* input,
    const int* start,
    size_t start_len,
    const int* stop,
    size_t stop_len,
    const int* strides,
    size_t strides_len) {
  mlx_array out = mlx_array_new();
  if (mlx_slice(
          &out,
          input->value,
          start,
          start_len,
          stop,
          stop_len,
          strides,
          strides_len,
          default_device_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_slice_dynamic(
    const DartMlxArrayHandle* input,
    const DartMlxArrayHandle* start,
    const int* axes,
    size_t axes_len,
    const int* slice_size,
    size_t slice_size_len) {
  mlx_array out = mlx_array_new();
  if (mlx_slice_dynamic(
          &out,
          input->value,
          start->value,
          axes,
          axes_len,
          slice_size,
          slice_size_len,
          default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_slice_update(
    const DartMlxArrayHandle* source,
    const DartMlxArrayHandle* update,
    const int* start,
    size_t start_len,
    const int* stop,
    size_t stop_len,
    const int* strides,
    size_t strides_len) {
  mlx_array out = mlx_array_new();
  if (mlx_slice_update(
          &out,
          source->value,
          update->value,
          start,
          start_len,
          stop,
          stop_len,
          strides,
          strides_len,
          default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_slice_update_dynamic(
    const DartMlxArrayHandle* source,
    const DartMlxArrayHandle* update,
    const DartMlxArrayHandle* start,
    const int* axes,
    size_t axes_len) {
  mlx_array out = mlx_array_new();
  if (mlx_slice_update_dynamic(
          &out,
          source->value,
          update->value,
          start->value,
          axes,
          axes_len,
          default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_einsum(
    const char* subscripts,
    DartMlxArrayHandle** operands,
    size_t operands_len) {
  auto operand_vec = build_array_vector(operands, operands_len);
  mlx_array out = mlx_array_new();
  auto status =
      mlx_einsum(&out, subscripts, operand_vec, default_cpu_stream());
  mlx_vector_array_free(operand_vec);
  if (status != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_tensordot(
    const DartMlxArrayHandle* a,
    const DartMlxArrayHandle* b,
    const int* axes_a,
    size_t axes_a_len,
    const int* axes_b,
    size_t axes_b_len) {
  mlx_array out = mlx_array_new();
  if (mlx_tensordot(
          &out,
          a->value,
          b->value,
          axes_a,
          axes_a_len,
          axes_b,
          axes_b_len,
          default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_tensordot_axis(
    const DartMlxArrayHandle* a,
    const DartMlxArrayHandle* b,
    int axis) {
  mlx_array out = mlx_array_new();
  if (mlx_tensordot_axis(
          &out, a->value, b->value, axis, default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_random_key(uint64_t seed) {
  mlx_array out = mlx_array_new();
  if (mlx_random_key(&out, seed) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_random_bernoulli(
    const DartMlxArrayHandle* p,
    const int* shape,
    size_t shape_len,
    const DartMlxArrayHandle* key) {
  mlx_array out = mlx_array_new();
  auto status = mlx_random_bernoulli(
      &out,
      p->value,
      shape,
      shape_len,
      key == nullptr ? mlx_array() : key->value,
      default_cpu_stream());
  if (status != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" int dart_mlx_random_split(
    const DartMlxArrayHandle* key,
    DartMlxArrayHandle** first,
    DartMlxArrayHandle** second) {
  mlx_array out_first = mlx_array_new();
  mlx_array out_second = mlx_array_new();
  if (mlx_random_split(
          &out_first, &out_second, key->value, default_cpu_stream()) != 0) {
    return 1;
  }
  *first = wrap_array(out_first);
  *second = wrap_array(out_second);
  return (*first == nullptr || *second == nullptr) ? 1 : 0;
}

extern "C" DartMlxArrayHandle* dart_mlx_random_categorical(
    const DartMlxArrayHandle* logits,
    int axis,
    int mode,
    const int* shape,
    size_t shape_len,
    int num_samples,
    const DartMlxArrayHandle* key) {
  mlx_array out = mlx_array_new();
  auto native_key = key == nullptr ? mlx_array() : key->value;
  int status = 0;
  switch (mode) {
    case 1:
      status = mlx_random_categorical_shape(
          &out,
          logits->value,
          axis,
          shape,
          shape_len,
          native_key,
          default_cpu_stream());
      break;
    case 2:
      status = mlx_random_categorical_num_samples(
          &out,
          logits->value,
          axis,
          num_samples,
          native_key,
          default_cpu_stream());
      break;
    default:
      status = mlx_random_categorical(
          &out, logits->value, axis, native_key, default_cpu_stream());
      break;
  }
  if (status != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_random_permutation(
    const DartMlxArrayHandle* input,
    int axis,
    const DartMlxArrayHandle* key) {
  mlx_array out = mlx_array_new();
  auto status = mlx_random_permutation(
      &out,
      input->value,
      axis,
      key == nullptr ? mlx_array() : key->value,
      default_cpu_stream());
  if (status != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_random_permutation_arange(
    int x,
    const DartMlxArrayHandle* key) {
  mlx_array out = mlx_array_new();
  auto status = mlx_random_permutation_arange(
      &out, x, key == nullptr ? mlx_array() : key->value, default_cpu_stream());
  if (status != 0) {
    return nullptr;
  }
  return wrap_array(out);
}
