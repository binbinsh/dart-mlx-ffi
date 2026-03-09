#include "bridge.h"

extern "C" DartMlxArrayHandle* dart_mlx_conv1d(
    const DartMlxArrayHandle* input,
    const DartMlxArrayHandle* weight,
    int stride,
    int padding,
    int dilation,
    int groups) {
  mlx_array out = mlx_array_new();
  if (mlx_conv1d(
          &out,
          input->value,
          weight->value,
          stride,
          padding,
          dilation,
          groups,
          default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_conv2d(
    const DartMlxArrayHandle* input,
    const DartMlxArrayHandle* weight,
    int stride0,
    int stride1,
    int padding0,
    int padding1,
    int dilation0,
    int dilation1,
    int groups) {
  mlx_array out = mlx_array_new();
  if (mlx_conv2d(
          &out,
          input->value,
          weight->value,
          stride0,
          stride1,
          padding0,
          padding1,
          dilation0,
          dilation1,
          groups,
          default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_conv3d(
    const DartMlxArrayHandle* input,
    const DartMlxArrayHandle* weight,
    int stride0,
    int stride1,
    int stride2,
    int padding0,
    int padding1,
    int padding2,
    int dilation0,
    int dilation1,
    int dilation2,
    int groups) {
  mlx_array out = mlx_array_new();
  if (mlx_conv3d(
          &out,
          input->value,
          weight->value,
          stride0,
          stride1,
          stride2,
          padding0,
          padding1,
          padding2,
          dilation0,
          dilation1,
          dilation2,
          groups,
          default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_conv_general(
    const DartMlxArrayHandle* input,
    const DartMlxArrayHandle* weight,
    const int* stride,
    size_t stride_len,
    const int* padding_lo,
    size_t padding_lo_len,
    const int* padding_hi,
    size_t padding_hi_len,
    const int* kernel_dilation,
    size_t kernel_dilation_len,
    const int* input_dilation,
    size_t input_dilation_len,
    int groups,
    bool flip) {
  mlx_array out = mlx_array_new();
  if (mlx_conv_general(
          &out,
          input->value,
          weight->value,
          stride,
          stride_len,
          padding_lo,
          padding_lo_len,
          padding_hi,
          padding_hi_len,
          kernel_dilation,
          kernel_dilation_len,
          input_dilation,
          input_dilation_len,
          groups,
          flip,
          default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_conv_transpose1d(
    const DartMlxArrayHandle* input,
    const DartMlxArrayHandle* weight,
    int stride,
    int padding,
    int dilation,
    int output_padding,
    int groups) {
  mlx_array out = mlx_array_new();
  if (mlx_conv_transpose1d(
          &out,
          input->value,
          weight->value,
          stride,
          padding,
          dilation,
          output_padding,
          groups,
          default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_conv_transpose2d(
    const DartMlxArrayHandle* input,
    const DartMlxArrayHandle* weight,
    int stride0,
    int stride1,
    int padding0,
    int padding1,
    int dilation0,
    int dilation1,
    int output_padding0,
    int output_padding1,
    int groups) {
  mlx_array out = mlx_array_new();
  if (mlx_conv_transpose2d(
          &out,
          input->value,
          weight->value,
          stride0,
          stride1,
          padding0,
          padding1,
          dilation0,
          dilation1,
          output_padding0,
          output_padding1,
          groups,
          default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}

extern "C" DartMlxArrayHandle* dart_mlx_conv_transpose3d(
    const DartMlxArrayHandle* input,
    const DartMlxArrayHandle* weight,
    int stride0,
    int stride1,
    int stride2,
    int padding0,
    int padding1,
    int padding2,
    int dilation0,
    int dilation1,
    int dilation2,
    int output_padding0,
    int output_padding1,
    int output_padding2,
    int groups) {
  mlx_array out = mlx_array_new();
  if (mlx_conv_transpose3d(
          &out,
          input->value,
          weight->value,
          stride0,
          stride1,
          stride2,
          padding0,
          padding1,
          padding2,
          dilation0,
          dilation1,
          dilation2,
          output_padding0,
          output_padding1,
          output_padding2,
          groups,
          default_cpu_stream()) != 0) {
    return nullptr;
  }
  return wrap_array(out);
}
