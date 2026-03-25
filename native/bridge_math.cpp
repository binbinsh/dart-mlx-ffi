#include "bridge.h"

extern "C" DartMlxArrayHandle* dart_mlx_fft_fft2(
    const DartMlxArrayHandle* input,
    const int* n,
    size_t n_len,
    const int* axes,
    size_t axes_len) {
  mlx_array out = mlx_array_new();
  if (mlx_fft_fft2(
          &out, input->value, n, n_len, axes, axes_len, default_device_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_fft_fftn(
    const DartMlxArrayHandle* input,
    const int* n,
    size_t n_len,
    const int* axes,
    size_t axes_len) {
  mlx_array out = mlx_array_new();
  if (mlx_fft_fftn(
          &out, input->value, n, n_len, axes, axes_len, default_device_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_fft_ifft2(
    const DartMlxArrayHandle* input,
    const int* n,
    size_t n_len,
    const int* axes,
    size_t axes_len) {
  mlx_array out = mlx_array_new();
  if (mlx_fft_ifft2(
          &out, input->value, n, n_len, axes, axes_len, default_device_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_fft_ifftn(
    const DartMlxArrayHandle* input,
    const int* n,
    size_t n_len,
    const int* axes,
    size_t axes_len) {
  mlx_array out = mlx_array_new();
  if (mlx_fft_ifftn(
          &out, input->value, n, n_len, axes, axes_len, default_device_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_fft_rfft2(
    const DartMlxArrayHandle* input,
    const int* n,
    size_t n_len,
    const int* axes,
    size_t axes_len) {
  mlx_array out = mlx_array_new();
  if (mlx_fft_rfft2(
          &out, input->value, n, n_len, axes, axes_len, default_device_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_fft_rfftn(
    const DartMlxArrayHandle* input,
    const int* n,
    size_t n_len,
    const int* axes,
    size_t axes_len) {
  mlx_array out = mlx_array_new();
  if (mlx_fft_rfftn(
          &out, input->value, n, n_len, axes, axes_len, default_device_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_fft_irfft2(
    const DartMlxArrayHandle* input,
    const int* n,
    size_t n_len,
    const int* axes,
    size_t axes_len) {
  mlx_array out = mlx_array_new();
  if (mlx_fft_irfft2(
          &out, input->value, n, n_len, axes, axes_len, default_device_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_fft_irfftn(
    const DartMlxArrayHandle* input,
    const int* n,
    size_t n_len,
    const int* axes,
    size_t axes_len) {
  mlx_array out = mlx_array_new();
  if (mlx_fft_irfftn(
          &out, input->value, n, n_len, axes, axes_len, default_device_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_fft_fftshift(
    const DartMlxArrayHandle* input,
    const int* axes,
    size_t axes_len) {
  mlx_array out = mlx_array_new();
  if (mlx_fft_fftshift(
          &out, input->value, axes, axes_len, default_device_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_fft_ifftshift(
    const DartMlxArrayHandle* input,
    const int* axes,
    size_t axes_len) {
  mlx_array out = mlx_array_new();
  if (mlx_fft_ifftshift(
          &out, input->value, axes, axes_len, default_device_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_linalg_cholesky(
    const DartMlxArrayHandle* input,
    bool upper) {
  mlx_array out = mlx_array_new();
  if (mlx_linalg_cholesky(
          &out, input->value, upper, default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_linalg_cross(
    const DartMlxArrayHandle* a,
    const DartMlxArrayHandle* b,
    int axis) {
  mlx_array out = mlx_array_new();
  if (mlx_linalg_cross(
          &out, a->value, b->value, axis, default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" int dart_mlx_linalg_eigh(
    const DartMlxArrayHandle* input,
    const char* uplo,
    DartMlxArrayHandle** values,
    DartMlxArrayHandle** vectors) {
  mlx_array out_values = mlx_array_new();
  mlx_array out_vectors = mlx_array_new();
  if (mlx_linalg_eigh(
          &out_values,
          &out_vectors,
          input->value,
          uplo,
          default_cpu_stream()) != 0) {
    return 1;
  }
  *values = wrap_array(out_values);
  *vectors = wrap_array(out_vectors);
  return (*values == nullptr || *vectors == nullptr) ? 1 : 0;
}

extern "C" DartMlxArrayHandle* dart_mlx_linalg_eigvals(
    const DartMlxArrayHandle* input) {
  mlx_array out = mlx_array_new();
  if (mlx_linalg_eigvals(&out, input->value, default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_linalg_eigvalsh(
    const DartMlxArrayHandle* input,
    const char* uplo) {
  mlx_array out = mlx_array_new();
  if (mlx_linalg_eigvalsh(&out, input->value, uplo, default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" int dart_mlx_linalg_lu(
    const DartMlxArrayHandle* input,
    DartMlxArrayHandle*** outputs_out,
    size_t* outputs_len_out) {
  auto results = mlx_vector_array_new();
  auto status = mlx_linalg_lu(&results, input->value, default_cpu_stream());
  if (status != 0) {
    return status;
  }
  status = export_vector_array(results, outputs_out, outputs_len_out);
  mlx_vector_array_free(results);
  return status;
}

extern "C" int dart_mlx_linalg_lu_factor(
    const DartMlxArrayHandle* input,
    DartMlxArrayHandle** lu,
    DartMlxArrayHandle** pivots) {
  mlx_array out_lu = mlx_array_new();
  mlx_array out_pivots = mlx_array_new();
  if (mlx_linalg_lu_factor(
          &out_lu, &out_pivots, input->value, default_cpu_stream()) != 0) {
    return 1;
  }
  *lu = wrap_array(out_lu);
  *pivots = wrap_array(out_pivots);
  return (*lu == nullptr || *pivots == nullptr) ? 1 : 0;
}

extern "C" DartMlxArrayHandle* dart_mlx_linalg_norm(
    const DartMlxArrayHandle* input,
    double ord,
    const int* axes,
    size_t axes_len,
    bool keepdims) {
  mlx_array out = mlx_array_new();
  if (mlx_linalg_norm(
          &out,
          input->value,
          ord,
          axes_len == 0 ? nullptr : axes,
          axes_len,
          keepdims,
          default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_linalg_norm_matrix(
    const DartMlxArrayHandle* input,
    const char* ord,
    const int* axes,
    size_t axes_len,
    bool keepdims) {
  mlx_array out = mlx_array_new();
  if (mlx_linalg_norm_matrix(
          &out,
          input->value,
          ord,
          axes_len == 0 ? nullptr : axes,
          axes_len,
          keepdims,
          default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_linalg_norm_l2(
    const DartMlxArrayHandle* input,
    const int* axes,
    size_t axes_len,
    bool keepdims) {
  mlx_array out = mlx_array_new();
  if (mlx_linalg_norm_l2(
          &out,
          input->value,
          axes_len == 0 ? nullptr : axes,
          axes_len,
          keepdims,
          default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_linalg_pinv(
    const DartMlxArrayHandle* input) {
  mlx_array out = mlx_array_new();
  if (mlx_linalg_pinv(&out, input->value, default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_linalg_solve_triangular(
    const DartMlxArrayHandle* a,
    const DartMlxArrayHandle* b,
    bool upper) {
  mlx_array out = mlx_array_new();
  if (mlx_linalg_solve_triangular(
          &out, a->value, b->value, upper, default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" int dart_mlx_linalg_svd(
    const DartMlxArrayHandle* input,
    bool compute_uv,
    DartMlxArrayHandle*** outputs_out,
    size_t* outputs_len_out) {
  auto results = mlx_vector_array_new();
  auto status =
      mlx_linalg_svd(&results, input->value, compute_uv, default_cpu_stream());
  if (status != 0) {
    return status;
  }
  status = export_vector_array(results, outputs_out, outputs_len_out);
  mlx_vector_array_free(results);
  return status;
}
