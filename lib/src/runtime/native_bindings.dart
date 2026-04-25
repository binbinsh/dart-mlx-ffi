// ignore_for_file: non_constant_identifier_names

@ffi.DefaultAsset(
  'package:dart_mlx_ffi/dart_mlx_ffi_runtime_bindings_generated.dart',
)
library;

import 'dart:ffi' as ffi;

final class DmfNativeTensor extends ffi.Struct {
  @ffi.Int32()
  external int dtype;

  @ffi.Int32()
  external int rank;

  external ffi.Pointer<ffi.Int64> shape;

  @ffi.IntPtr()
  external int byteLength;

  external ffi.Pointer<ffi.Void> data;
}

final class DmfNamedTensor extends ffi.Struct {
  external ffi.Pointer<ffi.Char> name;

  external DmfNativeTensor tensor;
}

@ffi.Native<
  ffi.Pointer<ffi.Void> Function(
    ffi.Int32,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dmf_runtime_create')
external ffi.Pointer<ffi.Void> dmf_runtime_create(
  int engine,
  ffi.Pointer<ffi.Char> modelPath,
  ffi.Pointer<ffi.Char> optionsJson,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Void>)>(
  symbol: 'dmf_runtime_free',
)
external void dmf_runtime_free(ffi.Pointer<ffi.Void> session);

@ffi.Native<
  ffi.Int32 Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<DmfNamedTensor>,
    ffi.IntPtr,
    ffi.Pointer<ffi.Pointer<DmfNamedTensor>>,
    ffi.Pointer<ffi.IntPtr>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dmf_runtime_run')
external int dmf_runtime_run(
  ffi.Pointer<ffi.Void> session,
  ffi.Pointer<DmfNamedTensor> inputs,
  int inputCount,
  ffi.Pointer<ffi.Pointer<DmfNamedTensor>> outputs,
  ffi.Pointer<ffi.IntPtr> outputCount,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<ffi.Void Function(ffi.Pointer<DmfNamedTensor>, ffi.IntPtr)>(
  symbol: 'dmf_runtime_free_tensors',
)
external void dmf_runtime_free_tensors(
  ffi.Pointer<DmfNamedTensor> tensors,
  int count,
);

@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Char>)>(
  symbol: 'dmf_runtime_free_string',
)
external void dmf_runtime_free_string(ffi.Pointer<ffi.Char> value);

@ffi.Native<ffi.Pointer<ffi.Char> Function()>(
  symbol: 'dmf_runtime_memory_info_json',
)
external ffi.Pointer<ffi.Char> dmf_runtime_memory_info_json();

@ffi.Native<ffi.Pointer<ffi.Char> Function(ffi.Pointer<ffi.Void>)>(
  symbol: 'dmf_runtime_diagnostics_json',
)
external ffi.Pointer<ffi.Char> dmf_runtime_diagnostics_json(
  ffi.Pointer<ffi.Void> session,
);
