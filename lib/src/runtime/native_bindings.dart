// ignore_for_file: non_constant_identifier_names

@ffi.DefaultAsset(
  'package:dart_inference/dart_inference_runtime_bindings_generated.dart',
)
library;

import 'dart:ffi' as ffi;

final class DartInferenceNativeTensor extends ffi.Struct {
  @ffi.Int32()
  external int dtype;

  @ffi.Int32()
  external int rank;

  external ffi.Pointer<ffi.Int64> shape;

  @ffi.IntPtr()
  external int byteLength;

  external ffi.Pointer<ffi.Void> data;
}

final class DartInferenceNamedTensor extends ffi.Struct {
  external ffi.Pointer<ffi.Char> name;

  external DartInferenceNativeTensor tensor;
}

@ffi.Native<
  ffi.Pointer<ffi.Void> Function(
    ffi.Int32,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dart_inference_runtime_create')
external ffi.Pointer<ffi.Void> dart_inference_runtime_create(
  int engine,
  ffi.Pointer<ffi.Char> modelPath,
  ffi.Pointer<ffi.Char> optionsJson,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Void>)>(
  symbol: 'dart_inference_runtime_free',
)
external void dart_inference_runtime_free(ffi.Pointer<ffi.Void> session);

@ffi.Native<
  ffi.Int32 Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<DartInferenceNamedTensor>,
    ffi.IntPtr,
    ffi.Pointer<ffi.Pointer<DartInferenceNamedTensor>>,
    ffi.Pointer<ffi.IntPtr>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dart_inference_runtime_run')
external int dart_inference_runtime_run(
  ffi.Pointer<ffi.Void> session,
  ffi.Pointer<DartInferenceNamedTensor> inputs,
  int inputCount,
  ffi.Pointer<ffi.Pointer<DartInferenceNamedTensor>> outputs,
  ffi.Pointer<ffi.IntPtr> outputCount,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<
  ffi.Void Function(ffi.Pointer<DartInferenceNamedTensor>, ffi.IntPtr)
>(symbol: 'dart_inference_runtime_free_tensors')
external void dart_inference_runtime_free_tensors(
  ffi.Pointer<DartInferenceNamedTensor> tensors,
  int count,
);

@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Char>)>(
  symbol: 'dart_inference_runtime_free_string',
)
external void dart_inference_runtime_free_string(ffi.Pointer<ffi.Char> value);

@ffi.Native<ffi.Pointer<ffi.Char> Function()>(
  symbol: 'dart_inference_runtime_backend_json',
)
external ffi.Pointer<ffi.Char> dart_inference_runtime_backend_json();

@ffi.Native<ffi.Pointer<ffi.Char> Function()>(
  symbol: 'dart_inference_runtime_memory_info_json',
)
external ffi.Pointer<ffi.Char> dart_inference_runtime_memory_info_json();

@ffi.Native<ffi.Pointer<ffi.Char> Function(ffi.Pointer<ffi.Void>)>(
  symbol: 'dart_inference_runtime_diagnostics_json',
)
external ffi.Pointer<ffi.Char> dart_inference_runtime_diagnostics_json(
  ffi.Pointer<ffi.Void> session,
);

@ffi.Native<ffi.Pointer<ffi.Void> Function(ffi.IntPtr)>(
  symbol: 'dart_inference_runtime_alloc',
)
external ffi.Pointer<ffi.Void> dart_inference_runtime_alloc(int byteLength);

@ffi.Native<
  ffi.Pointer<ffi.Void> Function(
    ffi.Int32,
    ffi.Pointer<ffi.Int64>,
    ffi.Int32,
    ffi.Pointer<ffi.IntPtr>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dart_inference_runtime_alloc_tensor_buffer')
external ffi.Pointer<ffi.Void> dart_inference_runtime_alloc_tensor_buffer(
  int dtype,
  ffi.Pointer<ffi.Int64> shape,
  int rank,
  ffi.Pointer<ffi.IntPtr> byteLength,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Void>)>(
  symbol: 'dart_inference_runtime_free_buffer',
)
external void dart_inference_runtime_free_buffer(ffi.Pointer<ffi.Void> value);
