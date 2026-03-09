// ignore_for_file: non_constant_identifier_names

part of '../shim_bindings.dart';

typedef DartMlxStreamHandle = ffi.Pointer<ffi.Void>;
typedef DartMlxGroupHandle = ffi.Pointer<ffi.Void>;

@ffi.Native<ffi.Pointer<ffi.Char> Function(DartMlxDeviceHandle)>()
external ffi.Pointer<ffi.Char> dart_mlx_device_info_json_copy(
  DartMlxDeviceHandle handle,
);

@ffi.Native<ffi.Int Function(DartMlxDeviceHandle)>()
external int dart_mlx_set_default_device(DartMlxDeviceHandle handle);

@ffi.Native<DartMlxStreamHandle Function()>()
external DartMlxStreamHandle dart_mlx_stream_new();

@ffi.Native<DartMlxStreamHandle Function(DartMlxDeviceHandle)>()
external DartMlxStreamHandle dart_mlx_stream_new_device(DartMlxDeviceHandle device);

@ffi.Native<DartMlxStreamHandle Function(DartMlxDeviceHandle)>()
external DartMlxStreamHandle dart_mlx_get_default_stream(DartMlxDeviceHandle device);

@ffi.Native<DartMlxStreamHandle Function()>()
external DartMlxStreamHandle dart_mlx_default_cpu_stream();

@ffi.Native<DartMlxStreamHandle Function()>()
external DartMlxStreamHandle dart_mlx_default_gpu_stream();

@ffi.Native<ffi.Void Function(DartMlxStreamHandle)>()
external void dart_mlx_stream_free(DartMlxStreamHandle handle);

@ffi.Native<ffi.Pointer<ffi.Char> Function(DartMlxStreamHandle)>()
external ffi.Pointer<ffi.Char> dart_mlx_stream_tostring_copy(
  DartMlxStreamHandle handle,
);

@ffi.Native<ffi.Int Function(DartMlxStreamHandle)>()
external int dart_mlx_stream_get_index(DartMlxStreamHandle handle);

@ffi.Native<DartMlxDeviceHandle Function(DartMlxStreamHandle)>()
external DartMlxDeviceHandle dart_mlx_stream_get_device(DartMlxStreamHandle handle);

@ffi.Native<ffi.Bool Function(DartMlxStreamHandle, DartMlxStreamHandle)>()
external bool dart_mlx_stream_equal(DartMlxStreamHandle lhs, DartMlxStreamHandle rhs);

@ffi.Native<ffi.Int Function(DartMlxStreamHandle)>()
external int dart_mlx_stream_synchronize(DartMlxStreamHandle handle);

@ffi.Native<ffi.Int Function(DartMlxStreamHandle)>()
external int dart_mlx_set_default_stream(DartMlxStreamHandle handle);

@ffi.Native<ffi.Bool Function()>()
external bool dart_mlx_distributed_is_available();

@ffi.Native<DartMlxGroupHandle Function(ffi.Bool)>()
external DartMlxGroupHandle dart_mlx_distributed_init(bool strict);

@ffi.Native<ffi.Void Function(DartMlxGroupHandle)>()
external void dart_mlx_distributed_group_free(DartMlxGroupHandle handle);

@ffi.Native<ffi.Int Function(DartMlxGroupHandle)>()
external int dart_mlx_distributed_group_rank(DartMlxGroupHandle handle);

@ffi.Native<ffi.Int Function(DartMlxGroupHandle)>()
external int dart_mlx_distributed_group_size(DartMlxGroupHandle handle);

@ffi.Native<DartMlxGroupHandle Function(DartMlxGroupHandle, ffi.Int, ffi.Int)>()
external DartMlxGroupHandle dart_mlx_distributed_group_split(
  DartMlxGroupHandle group,
  int color,
  int key,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    DartMlxGroupHandle,
    DartMlxStreamHandle,
  )
>()
external DartMlxArrayHandle dart_mlx_distributed_all_gather(
  DartMlxArrayHandle input,
  DartMlxGroupHandle group,
  DartMlxStreamHandle stream,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    DartMlxGroupHandle,
    DartMlxStreamHandle,
  )
>()
external DartMlxArrayHandle dart_mlx_distributed_all_sum(
  DartMlxArrayHandle input,
  DartMlxGroupHandle group,
  DartMlxStreamHandle stream,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    DartMlxGroupHandle,
    DartMlxStreamHandle,
  )
>()
external DartMlxArrayHandle dart_mlx_distributed_all_max(
  DartMlxArrayHandle input,
  DartMlxGroupHandle group,
  DartMlxStreamHandle stream,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    DartMlxGroupHandle,
    DartMlxStreamHandle,
  )
>()
external DartMlxArrayHandle dart_mlx_distributed_all_min(
  DartMlxArrayHandle input,
  DartMlxGroupHandle group,
  DartMlxStreamHandle stream,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    DartMlxGroupHandle,
    DartMlxStreamHandle,
  )
>()
external DartMlxArrayHandle dart_mlx_distributed_sum_scatter(
  DartMlxArrayHandle input,
  DartMlxGroupHandle group,
  DartMlxStreamHandle stream,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    ffi.Int,
    DartMlxGroupHandle,
    DartMlxStreamHandle,
  )
>()
external DartMlxArrayHandle dart_mlx_distributed_send(
  DartMlxArrayHandle input,
  int dst,
  DartMlxGroupHandle group,
  DartMlxStreamHandle stream,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxArrayHandle,
    ffi.Int,
    DartMlxGroupHandle,
    DartMlxStreamHandle,
  )
>()
external DartMlxArrayHandle dart_mlx_distributed_recv_like(
  DartMlxArrayHandle like,
  int src,
  DartMlxGroupHandle group,
  DartMlxStreamHandle stream,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    ffi.Pointer<ffi.Int>,
    ffi.Size,
    ffi.Int,
    ffi.Int,
    DartMlxGroupHandle,
    DartMlxStreamHandle,
  )
>()
external DartMlxArrayHandle dart_mlx_distributed_recv(
  ffi.Pointer<ffi.Int> shape,
  int shapeLen,
  int dtype,
  int src,
  DartMlxGroupHandle group,
  DartMlxStreamHandle stream,
);
