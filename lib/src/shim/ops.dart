// ignore_for_file: non_constant_identifier_names

part of '../shim_bindings.dart';

@ffi.Native<DartMlxArrayHandle Function(ffi.Pointer<ffi.Int>, ffi.Int, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_zeros(
  ffi.Pointer<ffi.Int> shape,
  int dim,
  int dtype,
);

@ffi.Native<DartMlxArrayHandle Function(ffi.Pointer<ffi.Int>, ffi.Int, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_ones(
  ffi.Pointer<ffi.Int> shape,
  int dim,
  int dtype,
);

@ffi.Native<
  DartMlxArrayHandle Function(ffi.Pointer<ffi.Int>, ffi.Int, ffi.Double, ffi.Int)
>()
external DartMlxArrayHandle dart_mlx_full(
  ffi.Pointer<ffi.Int> shape,
  int dim,
  double value,
  int dtype,
);

@ffi.Native<DartMlxArrayHandle Function(ffi.Double, ffi.Double, ffi.Double, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_arange(
  double start,
  double stop,
  double step,
  int dtype,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_astype(
  DartMlxArrayHandle input,
  int dtype,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, ffi.Pointer<ffi.Int>, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_reshape(
  DartMlxArrayHandle input,
  ffi.Pointer<ffi.Int> shape,
  int dim,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_transpose(DartMlxArrayHandle input);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, ffi.Bool)>()
external DartMlxArrayHandle dart_mlx_sum(
  DartMlxArrayHandle input,
  bool keepdims,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, ffi.Int, ffi.Bool)>()
external DartMlxArrayHandle dart_mlx_sum_axis(
  DartMlxArrayHandle input,
  int axis,
  bool keepdims,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, ffi.Bool)>()
external DartMlxArrayHandle dart_mlx_mean(
  DartMlxArrayHandle input,
  bool keepdims,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, ffi.Int, ffi.Bool)>()
external DartMlxArrayHandle dart_mlx_mean_axis(
  DartMlxArrayHandle input,
  int axis,
  bool keepdims,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, ffi.Int, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_fft_fft(
  DartMlxArrayHandle input,
  int n,
  int axis,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, ffi.Int, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_fft_ifft(
  DartMlxArrayHandle input,
  int n,
  int axis,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, ffi.Int, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_fft_rfft(
  DartMlxArrayHandle input,
  int n,
  int axis,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, ffi.Int, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_fft_irfft(
  DartMlxArrayHandle input,
  int n,
  int axis,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_linalg_inv(DartMlxArrayHandle input);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_linalg_solve(
  DartMlxArrayHandle a,
  DartMlxArrayHandle b,
);

@ffi.Native<
  ffi.Int Function(
    DartMlxArrayHandle,
    ffi.Pointer<DartMlxArrayHandle>,
    ffi.Pointer<DartMlxArrayHandle>,
  )
>()
external int dart_mlx_linalg_qr(
  DartMlxArrayHandle input,
  ffi.Pointer<DartMlxArrayHandle> q,
  ffi.Pointer<DartMlxArrayHandle> r,
);

@ffi.Native<
  ffi.Int Function(
    DartMlxArrayHandle,
    ffi.Pointer<DartMlxArrayHandle>,
    ffi.Pointer<DartMlxArrayHandle>,
  )
>()
external int dart_mlx_linalg_eig(
  DartMlxArrayHandle input,
  ffi.Pointer<DartMlxArrayHandle> values,
  ffi.Pointer<DartMlxArrayHandle> vectors,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    DartMlxArrayHandle,
    DartMlxArrayHandle,
    ffi.Float,
  )
>()
external DartMlxArrayHandle dart_mlx_fast_layer_norm(
  DartMlxArrayHandle input,
  DartMlxArrayHandle weight,
  DartMlxArrayHandle bias,
  double eps,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, DartMlxArrayHandle, ffi.Float)>()
external DartMlxArrayHandle dart_mlx_fast_rms_norm(
  DartMlxArrayHandle input,
  DartMlxArrayHandle weight,
  double eps,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    ffi.Int,
    ffi.Bool,
    ffi.Bool,
    ffi.Float,
    ffi.Float,
    ffi.Int,
    DartMlxArrayHandle,
  )
>()
external DartMlxArrayHandle dart_mlx_fast_rope(
  DartMlxArrayHandle input,
  int dims,
  bool traditional,
  bool hasBase,
  double base,
  double scale,
  int offset,
  DartMlxArrayHandle freqs,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    ffi.Int,
    ffi.Bool,
    ffi.Bool,
    ffi.Float,
    ffi.Float,
    DartMlxArrayHandle,
    DartMlxArrayHandle,
  )
>()
external DartMlxArrayHandle dart_mlx_fast_rope_dynamic(
  DartMlxArrayHandle input,
  int dims,
  bool traditional,
  bool hasBase,
  double base,
  double scale,
  DartMlxArrayHandle offset,
  DartMlxArrayHandle freqs,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    DartMlxArrayHandle,
    DartMlxArrayHandle,
    ffi.Float,
    ffi.Pointer<ffi.Char>,
    DartMlxArrayHandle,
    DartMlxArrayHandle,
  )
>()
external DartMlxArrayHandle dart_mlx_fast_sdpa(
  DartMlxArrayHandle queries,
  DartMlxArrayHandle keys,
  DartMlxArrayHandle values,
  double scale,
  ffi.Pointer<ffi.Char> maskMode,
  DartMlxArrayHandle mask,
  DartMlxArrayHandle sinks,
);

@ffi.Native<DartMlxArrayHandle Function(ffi.Pointer<ffi.Char>)>()
external DartMlxArrayHandle dart_mlx_load(ffi.Pointer<ffi.Char> file);

@ffi.Native<ffi.Int Function(ffi.Pointer<ffi.Char>, DartMlxArrayHandle)>()
external int dart_mlx_save(
  ffi.Pointer<ffi.Char> file,
  DartMlxArrayHandle input,
);

@ffi.Native<
  ffi.Int Function(
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>>,
    ffi.Pointer<ffi.Pointer<ffi.Pointer<ffi.Char>>>,
    ffi.Pointer<ffi.Size>,
    ffi.Pointer<ffi.Pointer<ffi.Pointer<ffi.Char>>>,
    ffi.Pointer<ffi.Pointer<ffi.Pointer<ffi.Char>>>,
    ffi.Pointer<ffi.Size>,
  )
>()
external int dart_mlx_load_safetensors(
  ffi.Pointer<ffi.Char> file,
  ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>> arraysOut,
  ffi.Pointer<ffi.Pointer<ffi.Pointer<ffi.Char>>> keysOut,
  ffi.Pointer<ffi.Size> arraysLen,
  ffi.Pointer<ffi.Pointer<ffi.Pointer<ffi.Char>>> metadataKeysOut,
  ffi.Pointer<ffi.Pointer<ffi.Pointer<ffi.Char>>> metadataValuesOut,
  ffi.Pointer<ffi.Size> metadataLen,
);

@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Pointer<ffi.Char>>, ffi.Size)>()
external void dart_mlx_free_string_array(
  ffi.Pointer<ffi.Pointer<ffi.Char>> values,
  int len,
);

@ffi.Native<
  ffi.Int Function(
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<DartMlxArrayHandle>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
    ffi.Size,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
    ffi.Size,
  )
>()
external int dart_mlx_save_safetensors(
  ffi.Pointer<ffi.Char> file,
  ffi.Pointer<DartMlxArrayHandle> arrays,
  ffi.Pointer<ffi.Pointer<ffi.Char>> keys,
  int arraysLen,
  ffi.Pointer<ffi.Pointer<ffi.Char>> metadataKeys,
  ffi.Pointer<ffi.Pointer<ffi.Char>> metadataValues,
  int metadataLen,
);

@ffi.Native<ffi.Int Function(ffi.Pointer<DartMlxArrayHandle>, ffi.Size)>()
external int dart_mlx_eval_many(
  ffi.Pointer<DartMlxArrayHandle> arrays,
  int len,
);

@ffi.Native<ffi.Int Function(ffi.Pointer<DartMlxArrayHandle>, ffi.Size)>()
external int dart_mlx_async_eval_many(
  ffi.Pointer<DartMlxArrayHandle> arrays,
  int len,
);

@ffi.Native<DartMlxArrayHandle Function(ffi.Pointer<DartMlxArrayHandle>, ffi.Size, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_concatenate(
  ffi.Pointer<DartMlxArrayHandle> arrays,
  int len,
  int axis,
);

@ffi.Native<DartMlxArrayHandle Function(ffi.Pointer<DartMlxArrayHandle>, ffi.Size, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_stack(
  ffi.Pointer<DartMlxArrayHandle> arrays,
  int len,
  int axis,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, ffi.Pointer<ffi.Int>, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_broadcast_to(
  DartMlxArrayHandle input,
  ffi.Pointer<ffi.Int> shape,
  int dim,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_expand_dims(
  DartMlxArrayHandle input,
  int axis,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_squeeze(DartMlxArrayHandle input);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    ffi.Bool,
    ffi.Double,
    ffi.Bool,
    ffi.Double,
  )
>()
external DartMlxArrayHandle dart_mlx_clip_scalar(
  DartMlxArrayHandle input,
  bool hasMin,
  double minValue,
  bool hasMax,
  double maxValue,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_minimum(
  DartMlxArrayHandle lhs,
  DartMlxArrayHandle rhs,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, DartMlxArrayHandle)>()
external DartMlxArrayHandle dart_mlx_maximum(
  DartMlxArrayHandle lhs,
  DartMlxArrayHandle rhs,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, ffi.Int, ffi.Bool, ffi.Bool)>()
external DartMlxArrayHandle dart_mlx_argmax(
  DartMlxArrayHandle input,
  int axis,
  bool hasAxis,
  bool keepdims,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, ffi.Int, ffi.Bool, ffi.Bool)>()
external DartMlxArrayHandle dart_mlx_argmin(
  DartMlxArrayHandle input,
  int axis,
  bool hasAxis,
  bool keepdims,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, ffi.Int, ffi.Bool)>()
external DartMlxArrayHandle dart_mlx_sort(
  DartMlxArrayHandle input,
  int axis,
  bool hasAxis,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, ffi.Int, ffi.Bool)>()
external DartMlxArrayHandle dart_mlx_argsort(
  DartMlxArrayHandle input,
  int axis,
  bool hasAxis,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, ffi.Int, ffi.Bool, ffi.Bool)>()
external DartMlxArrayHandle dart_mlx_logsumexp(
  DartMlxArrayHandle input,
  int axis,
  bool hasAxis,
  bool keepDims,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, ffi.Int, ffi.Bool, ffi.Bool)>()
external DartMlxArrayHandle dart_mlx_softmax(
  DartMlxArrayHandle input,
  int axis,
  bool hasAxis,
  bool precise,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, ffi.Int, ffi.Int, ffi.Bool)>()
external DartMlxArrayHandle dart_mlx_topk(
  DartMlxArrayHandle input,
  int k,
  int axis,
  bool hasAxis,
);

@ffi.Native<
  ffi.Int Function(
    DartMlxArrayHandle,
    ffi.Bool,
    ffi.Int,
    ffi.Bool,
    ffi.Int,
    ffi.Pointer<ffi.Char>,
    DartMlxArrayHandle,
    ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>>,
    ffi.Pointer<ffi.Size>,
  )
>()
external int dart_mlx_quantize(
  DartMlxArrayHandle weights,
  bool hasGroupSize,
  int groupSize,
  bool hasBits,
  int bits,
  ffi.Pointer<ffi.Char> mode,
  DartMlxArrayHandle globalScale,
  ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>> outputsOut,
  ffi.Pointer<ffi.Size> outputsLenOut,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    DartMlxArrayHandle,
    DartMlxArrayHandle,
    ffi.Bool,
    ffi.Int,
    ffi.Bool,
    ffi.Int,
    ffi.Pointer<ffi.Char>,
    DartMlxArrayHandle,
    ffi.Bool,
    ffi.Int,
  )
>()
external DartMlxArrayHandle dart_mlx_dequantize(
  DartMlxArrayHandle weights,
  DartMlxArrayHandle scales,
  DartMlxArrayHandle biases,
  bool hasGroupSize,
  int groupSize,
  bool hasBits,
  int bits,
  ffi.Pointer<ffi.Char> mode,
  DartMlxArrayHandle globalScale,
  bool hasDtype,
  int dtype,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    DartMlxArrayHandle,
    DartMlxArrayHandle,
    DartMlxArrayHandle,
    ffi.Bool,
    ffi.Bool,
    ffi.Int,
    ffi.Bool,
    ffi.Int,
    ffi.Pointer<ffi.Char>,
  )
>()
external DartMlxArrayHandle dart_mlx_quantized_matmul(
  DartMlxArrayHandle x,
  DartMlxArrayHandle weights,
  DartMlxArrayHandle scales,
  DartMlxArrayHandle biases,
  bool transpose,
  bool hasGroupSize,
  int groupSize,
  bool hasBits,
  int bits,
  ffi.Pointer<ffi.Char> mode,
);

typedef DartMlxCudaConfigHandle = ffi.Pointer<ffi.Void>;
typedef DartMlxCudaKernelHandle = ffi.Pointer<ffi.Void>;
typedef DartMlxMetalConfigHandle = ffi.Pointer<ffi.Void>;
typedef DartMlxMetalKernelHandle = ffi.Pointer<ffi.Void>;

@ffi.Native<DartMlxCudaConfigHandle Function()>()
external DartMlxCudaConfigHandle dart_mlx_cuda_config_new();

@ffi.Native<ffi.Void Function(DartMlxCudaConfigHandle)>()
external void dart_mlx_cuda_config_free(DartMlxCudaConfigHandle handle);

@ffi.Native<ffi.Int Function(DartMlxCudaConfigHandle, ffi.Pointer<ffi.Int>, ffi.Size, ffi.Int)>()
external int dart_mlx_cuda_config_add_output_arg(
  DartMlxCudaConfigHandle handle,
  ffi.Pointer<ffi.Int> shape,
  int len,
  int dtype,
);

@ffi.Native<ffi.Int Function(DartMlxCudaConfigHandle, ffi.Int, ffi.Int, ffi.Int)>()
external int dart_mlx_cuda_config_set_grid(
  DartMlxCudaConfigHandle handle,
  int x,
  int y,
  int z,
);

@ffi.Native<ffi.Int Function(DartMlxCudaConfigHandle, ffi.Int, ffi.Int, ffi.Int)>()
external int dart_mlx_cuda_config_set_thread_group(
  DartMlxCudaConfigHandle handle,
  int x,
  int y,
  int z,
);

@ffi.Native<ffi.Int Function(DartMlxCudaConfigHandle, ffi.Float)>()
external int dart_mlx_cuda_config_set_init_value(
  DartMlxCudaConfigHandle handle,
  double value,
);

@ffi.Native<ffi.Int Function(DartMlxCudaConfigHandle, ffi.Bool)>()
external int dart_mlx_cuda_config_set_verbose(
  DartMlxCudaConfigHandle handle,
  bool value,
);

@ffi.Native<ffi.Int Function(DartMlxCudaConfigHandle, ffi.Pointer<ffi.Char>, ffi.Int)>()
external int dart_mlx_cuda_config_add_template_dtype(
  DartMlxCudaConfigHandle handle,
  ffi.Pointer<ffi.Char> name,
  int dtype,
);

@ffi.Native<ffi.Int Function(DartMlxCudaConfigHandle, ffi.Pointer<ffi.Char>, ffi.Int)>()
external int dart_mlx_cuda_config_add_template_int(
  DartMlxCudaConfigHandle handle,
  ffi.Pointer<ffi.Char> name,
  int value,
);

@ffi.Native<ffi.Int Function(DartMlxCudaConfigHandle, ffi.Pointer<ffi.Char>, ffi.Bool)>()
external int dart_mlx_cuda_config_add_template_bool(
  DartMlxCudaConfigHandle handle,
  ffi.Pointer<ffi.Char> name,
  bool value,
);

@ffi.Native<
  DartMlxCudaKernelHandle Function(
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
    ffi.Size,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
    ffi.Size,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Char>,
    ffi.Bool,
    ffi.Int,
  )
>()
external DartMlxCudaKernelHandle dart_mlx_cuda_kernel_new(
  ffi.Pointer<ffi.Char> name,
  ffi.Pointer<ffi.Pointer<ffi.Char>> inputNames,
  int inputLen,
  ffi.Pointer<ffi.Pointer<ffi.Char>> outputNames,
  int outputLen,
  ffi.Pointer<ffi.Char> source,
  ffi.Pointer<ffi.Char> header,
  bool ensureRowContiguous,
  int sharedMemory,
);

@ffi.Native<ffi.Void Function(DartMlxCudaKernelHandle)>()
external void dart_mlx_cuda_kernel_free(DartMlxCudaKernelHandle handle);

@ffi.Native<
  ffi.Int Function(
    DartMlxCudaKernelHandle,
    ffi.Pointer<DartMlxArrayHandle>,
    ffi.Size,
    DartMlxCudaConfigHandle,
    ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>>,
    ffi.Pointer<ffi.Size>,
  )
>()
external int dart_mlx_cuda_kernel_apply(
  DartMlxCudaKernelHandle kernel,
  ffi.Pointer<DartMlxArrayHandle> inputs,
  int inputLen,
  DartMlxCudaConfigHandle config,
  ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>> outputsOut,
  ffi.Pointer<ffi.Size> outputsLenOut,
);

@ffi.Native<DartMlxMetalConfigHandle Function()>()
external DartMlxMetalConfigHandle dart_mlx_metal_config_new();

@ffi.Native<ffi.Void Function(DartMlxMetalConfigHandle)>()
external void dart_mlx_metal_config_free(DartMlxMetalConfigHandle handle);

@ffi.Native<ffi.Int Function(DartMlxMetalConfigHandle, ffi.Pointer<ffi.Int>, ffi.Size, ffi.Int)>()
external int dart_mlx_metal_config_add_output_arg(
  DartMlxMetalConfigHandle handle,
  ffi.Pointer<ffi.Int> shape,
  int len,
  int dtype,
);

@ffi.Native<ffi.Int Function(DartMlxMetalConfigHandle, ffi.Int, ffi.Int, ffi.Int)>()
external int dart_mlx_metal_config_set_grid(
  DartMlxMetalConfigHandle handle,
  int x,
  int y,
  int z,
);

@ffi.Native<ffi.Int Function(DartMlxMetalConfigHandle, ffi.Int, ffi.Int, ffi.Int)>()
external int dart_mlx_metal_config_set_thread_group(
  DartMlxMetalConfigHandle handle,
  int x,
  int y,
  int z,
);

@ffi.Native<ffi.Int Function(DartMlxMetalConfigHandle, ffi.Float)>()
external int dart_mlx_metal_config_set_init_value(
  DartMlxMetalConfigHandle handle,
  double value,
);

@ffi.Native<ffi.Int Function(DartMlxMetalConfigHandle, ffi.Bool)>()
external int dart_mlx_metal_config_set_verbose(
  DartMlxMetalConfigHandle handle,
  bool value,
);

@ffi.Native<ffi.Int Function(DartMlxMetalConfigHandle, ffi.Pointer<ffi.Char>, ffi.Int)>()
external int dart_mlx_metal_config_add_template_dtype(
  DartMlxMetalConfigHandle handle,
  ffi.Pointer<ffi.Char> name,
  int dtype,
);

@ffi.Native<ffi.Int Function(DartMlxMetalConfigHandle, ffi.Pointer<ffi.Char>, ffi.Int)>()
external int dart_mlx_metal_config_add_template_int(
  DartMlxMetalConfigHandle handle,
  ffi.Pointer<ffi.Char> name,
  int value,
);

@ffi.Native<ffi.Int Function(DartMlxMetalConfigHandle, ffi.Pointer<ffi.Char>, ffi.Bool)>()
external int dart_mlx_metal_config_add_template_bool(
  DartMlxMetalConfigHandle handle,
  ffi.Pointer<ffi.Char> name,
  bool value,
);

@ffi.Native<
  DartMlxMetalKernelHandle Function(
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
    ffi.Size,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
    ffi.Size,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Char>,
    ffi.Bool,
    ffi.Bool,
  )
>()
external DartMlxMetalKernelHandle dart_mlx_metal_kernel_new(
  ffi.Pointer<ffi.Char> name,
  ffi.Pointer<ffi.Pointer<ffi.Char>> inputNames,
  int inputLen,
  ffi.Pointer<ffi.Pointer<ffi.Char>> outputNames,
  int outputLen,
  ffi.Pointer<ffi.Char> source,
  ffi.Pointer<ffi.Char> header,
  bool ensureRowContiguous,
  bool atomicOutputs,
);

@ffi.Native<ffi.Void Function(DartMlxMetalKernelHandle)>()
external void dart_mlx_metal_kernel_free(DartMlxMetalKernelHandle handle);

@ffi.Native<
  ffi.Int Function(
    DartMlxMetalKernelHandle,
    ffi.Pointer<DartMlxArrayHandle>,
    ffi.Size,
    DartMlxMetalConfigHandle,
    ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>>,
    ffi.Pointer<ffi.Size>,
  )
>()
external int dart_mlx_metal_kernel_apply(
  DartMlxMetalKernelHandle kernel,
  ffi.Pointer<DartMlxArrayHandle> inputs,
  int inputLen,
  DartMlxMetalConfigHandle config,
  ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>> outputsOut,
  ffi.Pointer<ffi.Size> outputsLenOut,
);
