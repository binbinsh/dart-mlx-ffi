part of '../stable_api.dart';

typedef MlxAneWeight = ({String path, Uint8List data});
typedef MlxAneWeightWithOffset = ({String path, Uint8List data, int offset});

Float32List _copyOwnedFloat32List(ffi.Pointer<ffi.Float> pointer, int count) {
  if (pointer == ffi.nullptr) {
    _throwAnePrivateError('float32 conversion');
  }
  try {
    return Float32List.fromList(pointer.asTypedList(count));
  } finally {
    shim.dart_mlx_free_buffer(pointer.cast());
  }
}

Uint8List _encodeFp16Bytes(Float32List values) {
  final nativeValues = calloc<ffi.Float>(values.length);
  try {
    nativeValues.asTypedList(values.length).setAll(0, values);
    _clearAnePrivateError();
    final pointer = shim.dart_mlx_ane_private_encode_fp32_to_fp16_bytes_copy(
      nativeValues,
      values.length,
    );
    if (pointer == ffi.nullptr) {
      _throwAnePrivateError('encodeFp16Bytes');
    }
    return _copyOwnedBytes(pointer, values.length * 2);
  } finally {
    calloc.free(nativeValues);
  }
}

Float32List _decodeFp16Bytes(Uint8List bytes) {
  final nativeBytes = calloc<ffi.Uint8>(bytes.length);
  final countOut = calloc<ffi.Size>();
  try {
    nativeBytes.asTypedList(bytes.length).setAll(0, bytes);
    _clearAnePrivateError();
    final pointer = shim.dart_mlx_ane_private_decode_fp16_bytes_to_fp32_copy(
      nativeBytes,
      bytes.length,
      countOut,
    );
    return _copyOwnedFloat32List(pointer, countOut.value);
  } finally {
    calloc.free(nativeBytes);
    calloc.free(countOut);
  }
}

Uint8List _encodeRawFloat32Bytes(Float32List values) {
  return Uint8List.view(
    values.buffer,
    values.offsetInBytes,
    values.lengthInBytes,
  );
}

Float32List _decodeRawFloat32Bytes(Uint8List bytes) {
  if (bytes.lengthInBytes % Float32List.bytesPerElement != 0) {
    throw ArgumentError.value(
      bytes.lengthInBytes,
      'bytes.lengthInBytes',
      'Expected a multiple of ${Float32List.bytesPerElement} bytes.',
    );
  }
  if (bytes.offsetInBytes % Float32List.bytesPerElement != 0) {
    final copy = Uint8List.fromList(bytes);
    return copy.buffer.asFloat32List();
  }
  return bytes.buffer.asFloat32List(
    bytes.offsetInBytes,
    bytes.lengthInBytes ~/ Float32List.bytesPerElement,
  );
}
