#include "bridge.h"
#include "mlx/c/private/array.h"
#include "mlx/ops.h"

#include <cstdio>
#include <limits>

namespace {

template <typename T>
int scan_argmax_typed(const T* data, int len, int32_t* out, float* best_value) {
  if (data == nullptr || len <= 0) {
    return 1;
  }
  auto best_index = 0;
  auto best = static_cast<float>(data[0]);
  for (int i = 1; i < len; ++i) {
    const auto value = static_cast<float>(data[i]);
    if (value > best) {
      best = value;
      best_index = i;
    }
  }
  *out = best_index;
  *best_value = best;
  return 0;
}

int scan_argmax_float32_contiguous(
    mlx_array contiguous,
    int32_t* out,
    float* best_value) {
  const auto len = static_cast<int>(mlx_array_size(contiguous));
  return scan_argmax_typed(
      mlx_array_data_float32(contiguous), len, out, best_value);
}

int scan_argmax_contiguous(
    mlx_array contiguous,
    int32_t* out,
    float* best_value) {
  const auto len = static_cast<int>(mlx_array_size(contiguous));
  if (len <= 0) {
    return 1;
  }
  switch (mlx_array_dtype(contiguous)) {
    case MLX_FLOAT32:
      return scan_argmax_typed(
          mlx_array_data_float32(contiguous), len, out, best_value);
    case MLX_FLOAT64:
      return scan_argmax_typed(
          mlx_array_data_float64(contiguous), len, out, best_value);
    case MLX_FLOAT16:
      return scan_argmax_typed(
          mlx_array_data_float16(contiguous), len, out, best_value);
    case MLX_BFLOAT16:
      return scan_argmax_typed(
          mlx_array_data_bfloat16(contiguous), len, out, best_value);
    default:
      return 1;
  }
}

template <typename T>
void scan_argmax_strided_recursive(
    const T* data,
    const int* shape,
    const size_t* strides,
    int ndim,
    int depth,
    size_t offset,
    int32_t* logical_index,
    bool* has_best,
    int32_t* out,
    float* best_value) {
  if (depth == ndim) {
    const auto value = static_cast<float>(data[offset]);
    if (!*has_best || value > *best_value) {
      *has_best = true;
      *best_value = value;
      *out = *logical_index;
    }
    (*logical_index)++;
    return;
  }
  for (int i = 0; i < shape[depth]; ++i) {
    scan_argmax_strided_recursive(
        data,
        shape,
        strides,
        ndim,
        depth + 1,
        offset + static_cast<size_t>(i) * strides[depth],
        logical_index,
        has_best,
        out,
        best_value);
  }
}

template <typename T>
int scan_argmax_strided(
    const T* data,
    const int* shape,
    const size_t* strides,
    int ndim,
    int32_t* out,
    float* best_value) {
  if (data == nullptr || shape == nullptr || strides == nullptr || ndim <= 0) {
    return 1;
  }
  auto logical_index = 0;
  auto has_best = false;
  scan_argmax_strided_recursive(
      data,
      shape,
      strides,
      ndim,
      0,
      0,
      &logical_index,
      &has_best,
      out,
      best_value);
  return has_best ? 0 : 1;
}

int scan_argmax_array_view(
    mlx_array array,
    int32_t* out,
    float* best_value) {
  const auto shape = mlx_array_shape(array);
  const auto strides = mlx_array_strides(array);
  const auto ndim = static_cast<int>(mlx_array_ndim(array));
  switch (mlx_array_dtype(array)) {
    case MLX_FLOAT32:
      return scan_argmax_strided(
          mlx_array_data_float32(array), shape, strides, ndim, out, best_value);
    case MLX_FLOAT64:
      return scan_argmax_strided(
          mlx_array_data_float64(array), shape, strides, ndim, out, best_value);
    case MLX_FLOAT16:
      return scan_argmax_strided(
          mlx_array_data_float16(array), shape, strides, ndim, out, best_value);
    case MLX_BFLOAT16:
      return scan_argmax_strided(
          mlx_array_data_bfloat16(array), shape, strides, ndim, out, best_value);
    default:
      return 1;
  }
}

int scan_argmax_handle(
    const DartMlxArrayHandle* handle,
    int32_t* out,
    float* best_value,
    const char** stage_out) {
  if (mlx_array_eval(handle->value) != 0) {
    *stage_out = "direct_eval";
  } else {
    const auto direct_status = scan_argmax_array_view(handle->value, out, best_value);
    if (direct_status == 0) {
      return 0;
    }
    *stage_out = "direct_data";
  }

  mlx_array contiguous = mlx_array_new();
  if (mlx_contiguous(&contiguous, handle->value, false, default_cpu_stream()) !=
      0) {
    *stage_out = "cpu_contiguous";
    return 1;
  }
  if (mlx_array_eval(contiguous) != 0) {
    *stage_out = "cpu_eval";
    mlx_array_free(contiguous);
    return 1;
  }
  const auto status = scan_argmax_contiguous(contiguous, out, best_value);
  if (status != 0) {
    *stage_out = "cpu_data";
  }
  mlx_array_free(contiguous);
  return status;
}

int scan_argmax_chunk_float32(
    const mlx::core::array& chunk,
    int32_t* out,
    float* best_value,
    int source_dtype,
    int total,
    int start,
    int end) {
  auto chunk32 = mlx::core::astype(chunk, mlx::core::float32);
  auto chunk_value = mlx_array_new_(std::move(chunk32));
  if (mlx_array_eval(chunk_value) != 0) {
    std::fprintf(
        stderr,
        "[argmax-flat-helper] eval chunk failed dtype=%d size=%d start=%d end=%d\n",
        source_dtype,
        total,
        start,
        end);
    std::fflush(stderr);
    mlx_array_free(chunk_value);
    return 1;
  }

  const auto status = scan_argmax_float32_contiguous(
      chunk_value, out, best_value);
  if (status != 0) {
    std::fprintf(
        stderr,
        "[argmax-flat-helper] scan/data failed dtype=%d size=%d start=%d end=%d chunk_dtype=%d\n",
        source_dtype,
        total,
        start,
        end,
        static_cast<int>(mlx_array_dtype(chunk_value)));
    std::fflush(stderr);
  }

  mlx_array_free(chunk_value);
  return status;
}

} // namespace

extern "C" void dart_mlx_device_free(DartMlxDeviceHandle* handle) {
  if (handle == nullptr) {
    return;
  }
  mlx_device_free(handle->value);
  delete handle;
}

extern "C" char* dart_mlx_device_tostring_copy(const DartMlxDeviceHandle* handle) {
  if (handle == nullptr) {
    return nullptr;
  }
  mlx_string value = mlx_string_new();
  if (mlx_device_tostring(&value, handle->value) != 0) {
    return nullptr;
  }
  auto* copy = copy_c_string(mlx_string_data(value));
  mlx_string_free(value);
  return copy;
}

extern "C" DartMlxArrayHandle* dart_mlx_array_from_bool(
    void* data,
    const int* shape,
    int dim) {
  return wrap_array(mlx_array_new_data_managed_payload(
      data, shape, dim, MLX_BOOL, data, std::free));
}

extern "C" DartMlxArrayHandle* dart_mlx_array_from_int32(
    void* data,
    const int* shape,
    int dim) {
  return wrap_array(mlx_array_new_data_managed_payload(
      data, shape, dim, MLX_INT32, data, std::free));
}

extern "C" DartMlxArrayHandle* dart_mlx_array_from_float32(
    void* data,
    const int* shape,
    int dim) {
  return wrap_array(mlx_array_new_data_managed_payload(
      data, shape, dim, MLX_FLOAT32, data, std::free));
}

extern "C" DartMlxArrayHandle* dart_mlx_array_from_float64(
    void* data,
    const int* shape,
    int dim) {
  return wrap_array(mlx_array_new_data_managed_payload(
      data, shape, dim, MLX_FLOAT64, data, std::free));
}

extern "C" DartMlxArrayHandle* dart_mlx_array_from_int64(
    void* data,
    const int* shape,
    int dim) {
  return wrap_array(mlx_array_new_data_managed_payload(
      data, shape, dim, MLX_INT64, data, std::free));
}

extern "C" DartMlxArrayHandle* dart_mlx_array_from_uint64(
    void* data,
    const int* shape,
    int dim) {
  return wrap_array(mlx_array_new_data_managed_payload(
      data, shape, dim, MLX_UINT64, data, std::free));
}

extern "C" void dart_mlx_array_free(DartMlxArrayHandle* handle) {
  if (handle == nullptr) {
    return;
  }
  mlx_array_free(handle->value);
  delete handle;
}

extern "C" int dart_mlx_array_eval(const DartMlxArrayHandle* handle) {
  return mlx_array_eval(handle->value);
}

extern "C" int dart_mlx_array_ndim(const DartMlxArrayHandle* handle) {
  return static_cast<int>(mlx_array_ndim(handle->value));
}

extern "C" int dart_mlx_array_size(const DartMlxArrayHandle* handle) {
  return static_cast<int>(mlx_array_size(handle->value));
}

extern "C" int dart_mlx_array_dtype(const DartMlxArrayHandle* handle) {
  return static_cast<int>(mlx_array_dtype(handle->value));
}

extern "C" int dart_mlx_array_copy_shape(
    const DartMlxArrayHandle* handle,
    int* out_shape,
    int out_shape_len) {
  const int ndim = dart_mlx_array_ndim(handle);
  if (out_shape_len < ndim) {
    return 1;
  }
  const int* shape = mlx_array_shape(handle->value);
  std::memcpy(out_shape, shape, sizeof(int) * ndim);
  return 0;
}

extern "C" int dart_mlx_array_copy_bool(
    const DartMlxArrayHandle* handle,
    uint8_t* out,
    int len) {
  return copy_array_data_contiguous(handle, [&](mlx_array contiguous) {
    auto* data = mlx_array_data_bool(contiguous);
    if (data == nullptr) {
      return 1;
    }
    std::memcpy(out, data, len);
    return 0;
  });
}

extern "C" int dart_mlx_array_copy_int32(
    const DartMlxArrayHandle* handle,
    int32_t* out,
    int len) {
  return copy_array_data_contiguous(handle, [&](mlx_array contiguous) {
    auto* data = mlx_array_data_int32(contiguous);
    if (data == nullptr) {
      return 1;
    }
    std::memcpy(out, data, sizeof(int32_t) * len);
    return 0;
  });
}

extern "C" int dart_mlx_array_copy_uint32(
    const DartMlxArrayHandle* handle,
    uint32_t* out,
    int len) {
  return copy_array_data_contiguous(handle, [&](mlx_array contiguous) {
    auto* data = mlx_array_data_uint32(contiguous);
    if (data == nullptr) {
      return 1;
    }
    std::memcpy(out, data, sizeof(uint32_t) * len);
    return 0;
  });
}

extern "C" int dart_mlx_array_item_uint32(
    const DartMlxArrayHandle* handle,
    uint32_t* out) {
  return mlx_array_item_uint32(out, handle->value);
}

extern "C" int dart_mlx_array_copy_int64(
    const DartMlxArrayHandle* handle,
    int64_t* out,
    int len) {
  return copy_array_data_contiguous(handle, [&](mlx_array contiguous) {
    auto* data = mlx_array_data_int64(contiguous);
    if (data == nullptr) {
      return 1;
    }
    std::memcpy(out, data, sizeof(int64_t) * len);
    return 0;
  });
}

extern "C" int dart_mlx_array_item_int32(
    const DartMlxArrayHandle* handle,
    int32_t* out) {
  return mlx_array_item_int32(out, handle->value);
}

extern "C" int dart_mlx_array_scalar_int32_relaxed(
    const DartMlxArrayHandle* handle,
    int32_t* out) {
  if (handle == nullptr || out == nullptr) {
    std::fprintf(stderr, "[scalar-int32-relaxed] null handle or output\n");
    std::fflush(stderr);
    return 1;
  }
  if (mlx_array_eval(handle->value) != 0) {
    std::fprintf(stderr, "[scalar-int32-relaxed] eval failed\n");
    std::fflush(stderr);
    mlx_error("scalar int32 relaxed eval failed");
    return 1;
  }
  const auto size = static_cast<int>(mlx_array_size(handle->value));
  if (size != 1) {
    std::fprintf(
        stderr,
        "[scalar-int32-relaxed] size mismatch size=%d dtype=%d\n",
        size,
        static_cast<int>(mlx_array_dtype(handle->value)));
    std::fflush(stderr);
    mlx_error("scalar int32 relaxed requires size=1 got size=%d", size);
    return 1;
  }
  switch (mlx_array_dtype(handle->value)) {
    case MLX_INT32: {
      auto* data = mlx_array_data_int32(handle->value);
      if (data == nullptr) {
        std::fprintf(stderr, "[scalar-int32-relaxed] data_int32 null\n");
        std::fflush(stderr);
        mlx_error("scalar int32 relaxed data_int32 failed");
        return 1;
      }
      *out = data[0];
      return 0;
    }
    case MLX_UINT32: {
      auto* data = mlx_array_data_uint32(handle->value);
      if (data == nullptr) {
        std::fprintf(stderr, "[scalar-int32-relaxed] data_uint32 null\n");
        std::fflush(stderr);
        mlx_error("scalar int32 relaxed data_uint32 failed");
        return 1;
      }
      *out = static_cast<int32_t>(data[0]);
      return 0;
    }
    default:
      std::fprintf(
          stderr,
          "[scalar-int32-relaxed] unsupported dtype=%d\n",
          static_cast<int>(mlx_array_dtype(handle->value)));
      std::fflush(stderr);
      mlx_error(
          "scalar int32 relaxed unsupported dtype=%d",
          static_cast<int>(mlx_array_dtype(handle->value)));
      return 1;
  }
}

extern "C" int dart_mlx_array_copy_uint64(
    const DartMlxArrayHandle* handle,
    uint64_t* out,
    int len) {
  return copy_array_data_contiguous(handle, [&](mlx_array contiguous) {
    auto* data = mlx_array_data_uint64(contiguous);
    if (data == nullptr) {
      return 1;
    }
    std::memcpy(out, data, sizeof(uint64_t) * len);
    return 0;
  });
}

extern "C" int dart_mlx_array_item_uint64(
    const DartMlxArrayHandle* handle,
    uint64_t* out) {
  return mlx_array_item_uint64(out, handle->value);
}

extern "C" int dart_mlx_array_copy_float32(
    const DartMlxArrayHandle* handle,
    float* out,
    int len) {
  return copy_array_data_contiguous(handle, [&](mlx_array contiguous) {
    auto* data = mlx_array_data_float32(contiguous);
    if (data == nullptr) {
      return 1;
    }
    std::memcpy(out, data, sizeof(float) * len);
    return 0;
  });
}

extern "C" int dart_mlx_array_item_float32(
    const DartMlxArrayHandle* handle,
    float* out) {
  return mlx_array_item_float32(out, handle->value);
}

extern "C" int dart_mlx_array_scalar_float32_relaxed(
    const DartMlxArrayHandle* handle,
    float* out) {
  if (handle == nullptr || out == nullptr) {
    std::fprintf(stderr, "[scalar-float32-relaxed] null handle or output\n");
    std::fflush(stderr);
    return 1;
  }
  const auto size = static_cast<int>(mlx_array_size(handle->value));
  if (size != 1) {
    std::fprintf(
        stderr,
        "[scalar-float32-relaxed] size mismatch size=%d dtype=%d\n",
        size,
        static_cast<int>(mlx_array_dtype(handle->value)));
    std::fflush(stderr);
    mlx_error("scalar float32 relaxed requires size=1 got size=%d", size);
    return 1;
  }
  switch (mlx_array_dtype(handle->value)) {
    case MLX_FLOAT32: {
      const auto status = copy_array_data_contiguous(
          handle, [&](mlx_array contiguous) {
            auto* data = mlx_array_data_float32(contiguous);
            if (data == nullptr) {
              return 1;
            }
            *out = data[0];
            return 0;
          });
      if (status != 0) {
        std::fprintf(
            stderr, "[scalar-float32-relaxed] contiguous/data_float32 failed\n");
        std::fflush(stderr);
        mlx_error("scalar float32 relaxed contiguous/data_float32 failed");
      }
      return status;
    }
    case MLX_FLOAT64: {
      const auto status = copy_array_data_contiguous(
          handle, [&](mlx_array contiguous) {
            auto* data = mlx_array_data_float64(contiguous);
            if (data == nullptr) {
              return 1;
            }
            *out = static_cast<float>(data[0]);
            return 0;
          });
      if (status != 0) {
        std::fprintf(
            stderr, "[scalar-float32-relaxed] contiguous/data_float64 failed\n");
        std::fflush(stderr);
        mlx_error("scalar float32 relaxed contiguous/data_float64 failed");
      }
      return status;
    }
    default:
      std::fprintf(
          stderr,
          "[scalar-float32-relaxed] unsupported dtype=%d\n",
          static_cast<int>(mlx_array_dtype(handle->value)));
      std::fflush(stderr);
      mlx_error(
          "scalar float32 relaxed unsupported dtype=%d",
          static_cast<int>(mlx_array_dtype(handle->value)));
      return 1;
  }
}

extern "C" int dart_mlx_array_item_int64(
    const DartMlxArrayHandle* handle,
    int64_t* out) {
  return mlx_array_item_int64(out, handle->value);
}

extern "C" int dart_mlx_array_item_float64(
    const DartMlxArrayHandle* handle,
    double* out) {
  return mlx_array_item_float64(out, handle->value);
}

extern "C" int dart_mlx_array_argmax_flat_index_value_float32(
    const DartMlxArrayHandle* handle,
    int32_t* out_index,
    float* out_value) {
  try {
    const char* stage = "unknown";
    const auto status = scan_argmax_handle(handle, out_index, out_value, &stage);
    if (status != 0) {
      const auto dtype = static_cast<int>(mlx_array_dtype(handle->value));
      const auto size = static_cast<int>(mlx_array_size(handle->value));
      std::fprintf(
          stderr,
          "[argmax-index-value-helper] failed stage=%s dtype=%d size=%d\n",
          stage,
          dtype,
          size);
      std::fflush(stderr);
      mlx_error(
          "argmax index/value helper failed stage=%s dtype=%d size=%d",
          stage,
          dtype,
          size);
    }
    return status;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int dart_mlx_array_argmax_flat_int32(
    const DartMlxArrayHandle* handle,
    int32_t* out) {
  try {
    constexpr int kChunkSize = 4096;
    const auto total = static_cast<int>(mlx_array_size(handle->value));
    auto flat = mlx::core::reshape(mlx_array_get_(handle->value), {
        static_cast<mlx::core::ShapeElem>(total),
    });

    auto best_index = 0;
    auto best_value = -std::numeric_limits<float>::infinity();
    auto has_best = false;

    for (int start = 0; start < total; start += kChunkSize) {
      const auto end = std::min(start + kChunkSize, total);
      auto chunk = mlx::core::slice(
          flat,
          {static_cast<mlx::core::ShapeElem>(start)},
          {static_cast<mlx::core::ShapeElem>(end)});
      auto local_index = 0;
      float local_best = 0.0f;
      const auto status = scan_argmax_chunk_float32(
          chunk,
          &local_index,
          &local_best,
          static_cast<int>(mlx_array_dtype(handle->value)),
          total,
          start,
          end);
      if (status != 0) {
        std::fprintf(
            stderr,
            "[argmax-flat-helper] chunk failure status=%d dtype=%d size=%d start=%d end=%d stage=float32_contiguous\n",
            status,
            static_cast<int>(mlx_array_dtype(handle->value)),
            total,
            start,
            end);
        std::fflush(stderr);
        mlx_error(
            "argmax flat chunked cpu scan failed status=%d dtype=%d size=%d start=%d end=%d",
            status,
            static_cast<int>(mlx_array_dtype(handle->value)),
            total,
            start,
            end);
        return 1;
      }
      if (!has_best || local_best > best_value) {
        has_best = true;
        best_value = local_best;
        best_index = start + local_index;
      }
    }

    if (!has_best) {
      std::fprintf(
          stderr,
          "[argmax-flat-helper] no values dtype=%d size=%d\n",
          static_cast<int>(mlx_array_dtype(handle->value)),
          total);
      std::fflush(stderr);
      mlx_error(
          "argmax flat chunked cpu scan found no values dtype=%d size=%d",
          static_cast<int>(mlx_array_dtype(handle->value)),
          total);
      return 1;
    }
    *out = best_index;
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int dart_mlx_array_copy_float64(
    const DartMlxArrayHandle* handle,
    double* out,
    int len) {
  return copy_array_data_contiguous(handle, [&](mlx_array contiguous) {
    auto* data = mlx_array_data_float64(contiguous);
    if (data == nullptr) {
      return 1;
    }
    std::memcpy(out, data, sizeof(double) * len);
    return 0;
  });
}

extern "C" char* dart_mlx_array_tostring_copy(const DartMlxArrayHandle* handle) {
  mlx_string value = mlx_string_new();
  if (mlx_array_tostring(&value, handle->value) != 0) {
    return nullptr;
  }
  auto* copy = copy_c_string(mlx_string_data(value));
  mlx_string_free(value);
  return copy;
}
