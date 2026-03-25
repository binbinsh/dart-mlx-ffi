#include "bridge.h"

namespace {

int scan_status(
    int op,
    mlx_array* out,
    mlx_array input,
    int axis,
    bool reverse,
    bool inclusive) {
  switch (op) {
    case 1:
      return mlx_cumprod(out, input, axis, reverse, inclusive, default_device_stream());
    case 2:
      return mlx_cummax(out, input, axis, reverse, inclusive, default_device_stream());
    case 3:
      return mlx_cummin(out, input, axis, reverse, inclusive, default_device_stream());
    case 4:
      return mlx_logcumsumexp(out, input, axis, reverse, inclusive, default_device_stream());
    default:
      return mlx_cumsum(out, input, axis, reverse, inclusive, default_device_stream());
  }
}

} // namespace

extern "C" DartMlxArrayHandle* dart_mlx_scan(
    const DartMlxArrayHandle* input,
    int axis,
    bool reverse,
    bool inclusive,
    int op) {
  auto out = mlx_array_new();
  if (scan_status(op, &out, input->value, axis, reverse, inclusive) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_eye(
    int n,
    int m,
    int k,
    int dtype) {
  auto out = mlx_array_new();
  if (mlx_eye(&out, n, m, k, as_dtype(dtype), default_device_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_identity(
    int n,
    int dtype) {
  auto out = mlx_array_new();
  if (mlx_identity(&out, n, as_dtype(dtype), default_device_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_hamming(int m) {
  auto out = mlx_array_new();
  if (mlx_hamming(&out, m, default_device_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_hanning(int m) {
  auto out = mlx_array_new();
  if (mlx_hanning(&out, m, default_device_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_tri(
    int n,
    int m,
    int k,
    int dtype) {
  auto out = mlx_array_new();
  if (mlx_tri(&out, n, m, k, as_dtype(dtype), default_device_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_tril(
    const DartMlxArrayHandle* input,
    int k) {
  auto out = mlx_array_new();
  if (mlx_tril(&out, input->value, k, default_device_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_triu(
    const DartMlxArrayHandle* input,
    int k) {
  auto out = mlx_array_new();
  if (mlx_triu(&out, input->value, k, default_device_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_trace(
    const DartMlxArrayHandle* input,
    int offset,
    int axis1,
    int axis2,
    int dtype) {
  auto out = mlx_array_new();
  if (mlx_trace(
          &out, input->value, offset, axis1, axis2, as_dtype(dtype), default_device_stream()) !=
      0) {
    return nullptr;
  }
  return wrap_array(out);
}
