#include "bridge.h"

extern "C" DartMlxArrayHandle* dart_mlx_random_gumbel(
    const int* shape,
    int shape_len,
    int dtype,
    const DartMlxArrayHandle* key) {
  mlx_array out = mlx_array_new();
  if (mlx_random_gumbel(
          &out,
          shape,
          shape_len,
          as_dtype(dtype),
          key == nullptr ? mlx_array() : key->value,
          default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_random_laplace(
    const int* shape,
    int shape_len,
    int dtype,
    double loc,
    double scale,
    const DartMlxArrayHandle* key) {
  mlx_array out = mlx_array_new();
  if (mlx_random_laplace(
          &out,
          shape,
          shape_len,
          as_dtype(dtype),
          loc,
          scale,
          key == nullptr ? mlx_array() : key->value,
          default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_random_multivariate_normal(
    const DartMlxArrayHandle* mean,
    const DartMlxArrayHandle* cov,
    const int* shape,
    int shape_len,
    int dtype,
    const DartMlxArrayHandle* key) {
  mlx_array out = mlx_array_new();
  if (mlx_random_multivariate_normal(
          &out,
          mean->value,
          cov->value,
          shape,
          shape_len,
          as_dtype(dtype),
          key == nullptr ? mlx_array() : key->value,
          default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_random_randint(
    int low,
    int high,
    const int* shape,
    int shape_len,
    int dtype,
    const DartMlxArrayHandle* key) {
  auto low_arr = mlx_array_new_int(low);
  auto high_arr = mlx_array_new_int(high);
  auto out = mlx_array_new();
  auto status = mlx_random_randint(
      &out,
      low_arr,
      high_arr,
      shape,
      shape_len,
      as_dtype(dtype),
      key == nullptr ? mlx_array() : key->value,
      default_cpu_stream());
  mlx_array_free(low_arr);
  mlx_array_free(high_arr);
  if (status != 0) {
    return nullptr;
  }
  return wrap_array(out);
}
