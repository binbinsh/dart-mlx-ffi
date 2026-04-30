// ignore_for_file: non_constant_identifier_names

part of '../shim_bindings.dart';

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, ffi.Int, ffi.Bool, ffi.Bool, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_scan(
  DartMlxArrayHandle input,
  int axis,
  bool reverse,
  bool inclusive,
  int op,
);

@ffi.Native<DartMlxArrayHandle Function(ffi.Int, ffi.Int, ffi.Int, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_eye(
  int n,
  int m,
  int k,
  int dtype,
);

@ffi.Native<DartMlxArrayHandle Function(ffi.Int, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_identity(
  int n,
  int dtype,
);

@ffi.Native<DartMlxArrayHandle Function(ffi.Int)>()
external DartMlxArrayHandle dart_mlx_hamming(int m);

@ffi.Native<DartMlxArrayHandle Function(ffi.Int)>()
external DartMlxArrayHandle dart_mlx_hanning(int m);

@ffi.Native<DartMlxArrayHandle Function(ffi.Int, ffi.Int, ffi.Int, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_tri(
  int n,
  int m,
  int k,
  int dtype,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_tril(
  DartMlxArrayHandle input,
  int k,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_triu(
  DartMlxArrayHandle input,
  int k,
);

@ffi.Native<DartMlxArrayHandle Function(DartMlxArrayHandle, ffi.Int, ffi.Int, ffi.Int, ffi.Int)>()
external DartMlxArrayHandle dart_mlx_trace(
  DartMlxArrayHandle input,
  int offset,
  int axis1,
  int axis2,
  int dtype,
);
