// ignore_for_file: non_constant_identifier_names

part of '../shim_bindings.dart';

typedef DartMlxArrayHandle = ffi.Pointer<ffi.Void>;
typedef DartMlxDeviceHandle = ffi.Pointer<ffi.Void>;
typedef DartMlxClosureHandle = ffi.Pointer<ffi.Void>;
typedef DartMlxKwHandle = ffi.Pointer<ffi.Void>;
typedef DartMlxCustomHandle = ffi.Pointer<ffi.Void>;
typedef DartMlxCustomJvpHandle = ffi.Pointer<ffi.Void>;
typedef DartMlxImportedHandle = ffi.Pointer<ffi.Void>;
typedef DartMlxExporterHandle = ffi.Pointer<ffi.Void>;

typedef DartMlxClosureCallback =
    ffi.Int Function(
      ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>>,
      ffi.Pointer<ffi.Size>,
      ffi.Pointer<DartMlxArrayHandle>,
      ffi.Size,
    );

typedef DartMlxKwCallback =
    ffi.Int Function(
      ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>>,
      ffi.Pointer<ffi.Size>,
      ffi.Pointer<DartMlxArrayHandle>,
      ffi.Size,
      ffi.Pointer<ffi.Pointer<ffi.Pointer<ffi.Char>>>,
      ffi.Pointer<ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>>>,
      ffi.Pointer<ffi.Size>,
    );

typedef DartMlxCustomCallback =
    ffi.Int Function(
      ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>>,
      ffi.Pointer<ffi.Size>,
      ffi.Pointer<DartMlxArrayHandle>,
      ffi.Size,
      ffi.Pointer<DartMlxArrayHandle>,
      ffi.Size,
      ffi.Pointer<DartMlxArrayHandle>,
      ffi.Size,
    );

typedef DartMlxCustomJvpCallback =
    ffi.Int Function(
      ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>>,
      ffi.Pointer<ffi.Size>,
      ffi.Pointer<DartMlxArrayHandle>,
      ffi.Size,
      ffi.Pointer<DartMlxArrayHandle>,
      ffi.Size,
      ffi.Pointer<ffi.Int>,
      ffi.Size,
    );

@ffi.Native<ffi.Pointer<ffi.Char> Function()>()
external ffi.Pointer<ffi.Char> dart_mlx_version_copy();

@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Char>)>()
external void dart_mlx_string_free_copy(ffi.Pointer<ffi.Char> value);

@ffi.Native<DartMlxDeviceHandle Function()>()
external DartMlxDeviceHandle dart_mlx_default_device();

@ffi.Native<DartMlxDeviceHandle Function(ffi.Int, ffi.Int)>()
external DartMlxDeviceHandle dart_mlx_device_new_type(int type, int index);

@ffi.Native<ffi.Int Function(DartMlxDeviceHandle)>()
external int dart_mlx_device_is_available(DartMlxDeviceHandle handle);

@ffi.Native<ffi.Int Function(DartMlxDeviceHandle)>()
external int dart_mlx_device_get_index(DartMlxDeviceHandle handle);

@ffi.Native<ffi.Int Function(DartMlxDeviceHandle)>()
external int dart_mlx_device_get_type(DartMlxDeviceHandle handle);

@ffi.Native<ffi.Bool Function(DartMlxDeviceHandle, DartMlxDeviceHandle)>()
external bool dart_mlx_device_equal(
  DartMlxDeviceHandle lhs,
  DartMlxDeviceHandle rhs,
);

@ffi.Native<ffi.Int Function(ffi.Int)>()
external int dart_mlx_device_count(int type);

@ffi.Native<ffi.Void Function(DartMlxDeviceHandle)>()
external void dart_mlx_device_free(DartMlxDeviceHandle handle);

@ffi.Native<
  DartMlxClosureHandle Function(
    ffi.Pointer<ffi.NativeFunction<DartMlxClosureCallback>>,
  )
>()
external DartMlxClosureHandle dart_mlx_function_from_callback(
  ffi.Pointer<ffi.NativeFunction<DartMlxClosureCallback>> callback,
);

@ffi.Native<
  DartMlxKwHandle Function(ffi.Pointer<ffi.NativeFunction<DartMlxKwCallback>>)
>()
external DartMlxKwHandle dart_mlx_kw_function_from_callback(
  ffi.Pointer<ffi.NativeFunction<DartMlxKwCallback>> callback,
);

@ffi.Native<ffi.Void Function(DartMlxKwHandle)>()
external void dart_mlx_kw_function_free(DartMlxKwHandle handle);

@ffi.Native<ffi.Void Function(DartMlxClosureHandle)>()
external void dart_mlx_function_free(DartMlxClosureHandle handle);

@ffi.Native<
  ffi.Int Function(
    DartMlxKwHandle,
    ffi.Pointer<DartMlxArrayHandle>,
    ffi.Size,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
    ffi.Pointer<DartMlxArrayHandle>,
    ffi.Size,
    ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>>,
    ffi.Pointer<ffi.Size>,
  )
>()
external int dart_mlx_kw_function_apply(
  DartMlxKwHandle function,
  ffi.Pointer<DartMlxArrayHandle> inputs,
  int inputLen,
  ffi.Pointer<ffi.Pointer<ffi.Char>> keys,
  ffi.Pointer<DartMlxArrayHandle> values,
  int valuesLen,
  ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>> outputsOut,
  ffi.Pointer<ffi.Size> outputsLenOut,
);

@ffi.Native<
  DartMlxCustomHandle Function(
    ffi.Pointer<ffi.NativeFunction<DartMlxCustomCallback>>,
  )
>()
external DartMlxCustomHandle dart_mlx_custom_from_callback(
  ffi.Pointer<ffi.NativeFunction<DartMlxCustomCallback>> callback,
);

@ffi.Native<ffi.Void Function(DartMlxCustomHandle)>()
external void dart_mlx_custom_free(DartMlxCustomHandle handle);

@ffi.Native<
  DartMlxCustomJvpHandle Function(
    ffi.Pointer<ffi.NativeFunction<DartMlxCustomJvpCallback>>,
  )
>()
external DartMlxCustomJvpHandle dart_mlx_custom_jvp_from_callback(
  ffi.Pointer<ffi.NativeFunction<DartMlxCustomJvpCallback>> callback,
);

@ffi.Native<ffi.Void Function(DartMlxCustomJvpHandle)>()
external void dart_mlx_custom_jvp_free(DartMlxCustomJvpHandle handle);

@ffi.Native<
  DartMlxClosureHandle Function(DartMlxClosureHandle, DartMlxCustomHandle)
>()
external DartMlxClosureHandle dart_mlx_function_custom_vjp(
  DartMlxClosureHandle function,
  DartMlxCustomHandle custom,
);

@ffi.Native<
  DartMlxClosureHandle Function(
    DartMlxClosureHandle,
    DartMlxCustomHandle,
    DartMlxCustomJvpHandle,
  )
>()
external DartMlxClosureHandle dart_mlx_function_custom(
  DartMlxClosureHandle function,
  DartMlxCustomHandle customVjp,
  DartMlxCustomJvpHandle customJvp,
);

@ffi.Native<
  ffi.Int Function(
    ffi.Pointer<ffi.Char>,
    DartMlxKwHandle,
    ffi.Pointer<DartMlxArrayHandle>,
    ffi.Size,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
    ffi.Pointer<DartMlxArrayHandle>,
    ffi.Size,
    ffi.Bool,
  )
>()
external int dart_mlx_export_kw_function(
  ffi.Pointer<ffi.Char> file,
  DartMlxKwHandle function,
  ffi.Pointer<DartMlxArrayHandle> args,
  int argsLen,
  ffi.Pointer<ffi.Pointer<ffi.Char>> keys,
  ffi.Pointer<DartMlxArrayHandle> values,
  int valuesLen,
  bool shapeless,
);

@ffi.Native<
  ffi.Int Function(
    ffi.Pointer<ffi.Char>,
    DartMlxClosureHandle,
    ffi.Pointer<DartMlxArrayHandle>,
    ffi.Size,
    ffi.Bool,
  )
>()
external int dart_mlx_export_function(
  ffi.Pointer<ffi.Char> file,
  DartMlxClosureHandle function,
  ffi.Pointer<DartMlxArrayHandle> args,
  int argsLen,
  bool shapeless,
);

@ffi.Native<
  DartMlxExporterHandle Function(
    ffi.Pointer<ffi.Char>,
    DartMlxClosureHandle,
    ffi.Bool,
  )
>()
external DartMlxExporterHandle dart_mlx_function_exporter_new(
  ffi.Pointer<ffi.Char> file,
  DartMlxClosureHandle function,
  bool shapeless,
);

@ffi.Native<ffi.Void Function(DartMlxExporterHandle)>()
external void dart_mlx_function_exporter_free(DartMlxExporterHandle handle);

@ffi.Native<
  ffi.Int Function(
    DartMlxExporterHandle,
    ffi.Pointer<DartMlxArrayHandle>,
    ffi.Size,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
    ffi.Pointer<DartMlxArrayHandle>,
    ffi.Size,
  )
>()
external int dart_mlx_function_exporter_apply(
  DartMlxExporterHandle exporter,
  ffi.Pointer<DartMlxArrayHandle> inputs,
  int inputLen,
  ffi.Pointer<ffi.Pointer<ffi.Char>> keys,
  ffi.Pointer<DartMlxArrayHandle> values,
  int valuesLen,
);

@ffi.Native<DartMlxImportedHandle Function(ffi.Pointer<ffi.Char>)>()
external DartMlxImportedHandle dart_mlx_imported_function_new(
  ffi.Pointer<ffi.Char> file,
);

@ffi.Native<ffi.Void Function(DartMlxImportedHandle)>()
external void dart_mlx_imported_function_free(DartMlxImportedHandle handle);

@ffi.Native<
  ffi.Int Function(
    DartMlxImportedHandle,
    ffi.Pointer<DartMlxArrayHandle>,
    ffi.Size,
    ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>>,
    ffi.Pointer<ffi.Size>,
  )
>()
external int dart_mlx_imported_function_apply(
  DartMlxImportedHandle function,
  ffi.Pointer<DartMlxArrayHandle> inputs,
  int inputLen,
  ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>> outputsOut,
  ffi.Pointer<ffi.Size> outputsLenOut,
);

@ffi.Native<
  DartMlxArrayHandle Function(
    DartMlxImportedHandle,
    ffi.Pointer<DartMlxArrayHandle>,
    ffi.Size,
  )
>()
external DartMlxArrayHandle dart_mlx_imported_function_apply_one(
  DartMlxImportedHandle function,
  ffi.Pointer<DartMlxArrayHandle> inputs,
  int inputLen,
);

@ffi.Native<
  ffi.Int Function(
    DartMlxImportedHandle,
    ffi.Pointer<DartMlxArrayHandle>,
    ffi.Size,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
    ffi.Pointer<DartMlxArrayHandle>,
    ffi.Size,
    ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>>,
    ffi.Pointer<ffi.Size>,
  )
>()
external int dart_mlx_imported_function_apply_kwargs(
  DartMlxImportedHandle function,
  ffi.Pointer<DartMlxArrayHandle> inputs,
  int inputLen,
  ffi.Pointer<ffi.Pointer<ffi.Char>> keys,
  ffi.Pointer<DartMlxArrayHandle> values,
  int valuesLen,
  ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>> outputsOut,
  ffi.Pointer<ffi.Size> outputsLenOut,
);

@ffi.Native<
  ffi.Int Function(
    DartMlxClosureHandle,
    ffi.Pointer<DartMlxArrayHandle>,
    ffi.Size,
    ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>>,
    ffi.Pointer<ffi.Size>,
  )
>()
external int dart_mlx_function_apply(
  DartMlxClosureHandle function,
  ffi.Pointer<DartMlxArrayHandle> inputs,
  int inputLen,
  ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>> outputsOut,
  ffi.Pointer<ffi.Size> outputsLenOut,
);

@ffi.Native<DartMlxClosureHandle Function(DartMlxClosureHandle)>()
external DartMlxClosureHandle dart_mlx_function_checkpoint(
  DartMlxClosureHandle function,
);

@ffi.Native<DartMlxClosureHandle Function(DartMlxClosureHandle, ffi.Bool)>()
external DartMlxClosureHandle dart_mlx_function_compile(
  DartMlxClosureHandle function,
  bool shapeless,
);

@ffi.Native<
  ffi.Int Function(
    DartMlxClosureHandle,
    ffi.Pointer<DartMlxArrayHandle>,
    ffi.Size,
    ffi.Pointer<DartMlxArrayHandle>,
    ffi.Size,
    ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>>,
    ffi.Pointer<ffi.Size>,
    ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>>,
    ffi.Pointer<ffi.Size>,
  )
>()
external int dart_mlx_function_jvp(
  DartMlxClosureHandle function,
  ffi.Pointer<DartMlxArrayHandle> primals,
  int primalsLen,
  ffi.Pointer<DartMlxArrayHandle> tangents,
  int tangentsLen,
  ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>> outputsOut,
  ffi.Pointer<ffi.Size> outputsLenOut,
  ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>> tangentsOut,
  ffi.Pointer<ffi.Size> tangentsLenOut,
);

@ffi.Native<
  ffi.Int Function(
    DartMlxClosureHandle,
    ffi.Pointer<DartMlxArrayHandle>,
    ffi.Size,
    ffi.Pointer<DartMlxArrayHandle>,
    ffi.Size,
    ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>>,
    ffi.Pointer<ffi.Size>,
    ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>>,
    ffi.Pointer<ffi.Size>,
  )
>()
external int dart_mlx_function_vjp(
  DartMlxClosureHandle function,
  ffi.Pointer<DartMlxArrayHandle> primals,
  int primalsLen,
  ffi.Pointer<DartMlxArrayHandle> cotangents,
  int cotangentsLen,
  ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>> outputsOut,
  ffi.Pointer<ffi.Size> outputsLenOut,
  ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>> cotangentsOut,
  ffi.Pointer<ffi.Size> cotangentsLenOut,
);

@ffi.Native<
  ffi.Int Function(
    DartMlxClosureHandle,
    ffi.Pointer<ffi.Int>,
    ffi.Size,
    ffi.Pointer<DartMlxArrayHandle>,
    ffi.Size,
    ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>>,
    ffi.Pointer<ffi.Size>,
    ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>>,
    ffi.Pointer<ffi.Size>,
  )
>()
external int dart_mlx_function_value_and_grad(
  DartMlxClosureHandle function,
  ffi.Pointer<ffi.Int> argnums,
  int argnumsLen,
  ffi.Pointer<DartMlxArrayHandle> inputs,
  int inputsLen,
  ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>> valuesOut,
  ffi.Pointer<ffi.Size> valuesLenOut,
  ffi.Pointer<ffi.Pointer<DartMlxArrayHandle>> gradsOut,
  ffi.Pointer<ffi.Size> gradsLenOut,
);
