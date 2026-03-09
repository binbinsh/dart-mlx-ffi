// ignore_for_file: camel_case_types, non_constant_identifier_names

@ffi.DefaultAsset('package:dart_mlx_ffi/dart_mlx_ffi_bindings_generated.dart')
library;

import 'dart:ffi' as ffi;

import 'mlx_bindings.g.dart';

/// C `float _Complex` is laid out as two adjacent `float` values.
final class mlx_complex64_t extends ffi.Struct {
  @ffi.Float()
  external double real;

  @ffi.Float()
  external double imag;
}

@ffi.Native<ffi.Int Function(ffi.Pointer<mlx_complex64_t>, mlx_array)>()
external int mlx_array_item_complex64(
  ffi.Pointer<mlx_complex64_t> res,
  mlx_array arr,
);

@ffi.Native<ffi.Pointer<mlx_complex64_t> Function(mlx_array)>()
external ffi.Pointer<mlx_complex64_t> mlx_array_data_complex64(mlx_array arr);

@ffi.Native<ffi.Int Function(ffi.Pointer<ffi.Uint16>, mlx_array)>()
external int mlx_array_item_float16(ffi.Pointer<ffi.Uint16> res, mlx_array arr);

@ffi.Native<ffi.Int Function(ffi.Pointer<ffi.Uint16>, mlx_array)>()
external int mlx_array_item_bfloat16(
  ffi.Pointer<ffi.Uint16> res,
  mlx_array arr,
);

@ffi.Native<ffi.Pointer<ffi.Uint16> Function(mlx_array)>()
external ffi.Pointer<ffi.Uint16> mlx_array_data_float16(mlx_array arr);

@ffi.Native<ffi.Pointer<ffi.Uint16> Function(mlx_array)>()
external ffi.Pointer<ffi.Uint16> mlx_array_data_bfloat16(mlx_array arr);
