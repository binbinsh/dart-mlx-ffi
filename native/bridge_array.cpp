#include "bridge.h"

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

extern "C" int dart_mlx_array_item_int64(
    const DartMlxArrayHandle* handle,
    int64_t* out) {
  return mlx_array_item_int64(out, handle->value);
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
