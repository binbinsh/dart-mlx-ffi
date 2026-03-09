part of '../stable_api.dart';

T _withShape<T>(List<int> shape, T Function(ffi.Pointer<ffi.Int>) callback) {
  final pointer = calloc<ffi.Int>(shape.length);
  try {
    for (var index = 0; index < shape.length; index++) {
      pointer[index] = shape[index];
    }
    return callback(pointer);
  } finally {
    calloc.free(pointer);
  }
}

T _withInts<T>(
  List<int> values,
  T Function(ffi.Pointer<ffi.Int>, int) callback,
) {
  final pointer = calloc<ffi.Int>(values.length);
  try {
    for (var index = 0; index < values.length; index++) {
      pointer[index] = values[index];
    }
    return callback(pointer, values.length);
  } finally {
    calloc.free(pointer);
  }
}

T _withArrayHandles<T>(
  List<MlxArray> arrays,
  T Function(ffi.Pointer<ffi.Pointer<ffi.Void>>, int) callback,
) {
  final handles = calloc<ffi.Pointer<ffi.Void>>(arrays.length);
  try {
    for (var index = 0; index < arrays.length; index++) {
      handles[index] = arrays[index]._handle;
    }
    return callback(handles, arrays.length);
  } finally {
    calloc.free(handles);
  }
}

T _withKwargHandles<T>(
  Map<String, MlxArray> kwargs,
  T Function(
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
    ffi.Pointer<ffi.Pointer<ffi.Void>>,
    int,
  )
  callback,
) {
  final keys = calloc<ffi.Pointer<ffi.Char>>(kwargs.length);
  final values = calloc<ffi.Pointer<ffi.Void>>(kwargs.length);
  final allocated = <ffi.Pointer<ffi.Char>>[];
  try {
    var index = 0;
    for (final entry in kwargs.entries) {
      final key = entry.key.toNativeUtf8().cast<ffi.Char>();
      allocated.add(key);
      keys[index] = key;
      values[index] = entry.value._handle;
      index++;
    }
    return callback(keys, values, kwargs.length);
  } finally {
    for (final key in allocated) {
      calloc.free(key);
    }
    calloc.free(keys);
    calloc.free(values);
  }
}

int _readSizeValue(
  String operation,
  int Function(ffi.Pointer<ffi.Size>) callback,
) {
  final result = calloc<ffi.Size>();
  try {
    _clearError();
    _checkStatus(operation, callback(result));
    return result.value;
  } finally {
    calloc.free(result);
  }
}

int _writeSizeValue(
  String operation,
  int Function(ffi.Pointer<ffi.Size>, int) callback,
  int value,
) {
  final result = calloc<ffi.Size>();
  try {
    _clearError();
    _checkStatus(operation, callback(result, value));
    return result.value;
  } finally {
    calloc.free(result);
  }
}

List<MlxArray> _readOutputArrayList(
  ffi.Pointer<ffi.Pointer<ffi.Void>> handles,
  int len,
) {
  return List<MlxArray>.generate(len, (index) => MlxArray._(handles[index]));
}

T _withNativePath<T>(
  String path,
  T Function(ffi.Pointer<ffi.Char>) callback,
) {
  final nativePath = path.toNativeUtf8().cast<ffi.Char>();
  try {
    return callback(nativePath);
  } finally {
    calloc.free(nativePath);
  }
}

T _withCString<T>(
  String value,
  T Function(ffi.Pointer<ffi.Char>) callback,
) {
  final pointer = value.toNativeUtf8().cast<ffi.Char>();
  try {
    return callback(pointer);
  } finally {
    calloc.free(pointer);
  }
}

Uint8List _copyOwnedBytes(ffi.Pointer<ffi.Uint8> pointer, int len) {
  if (pointer == ffi.nullptr) {
    throw const MlxException('Native byte buffer was unexpectedly null.');
  }
  try {
    return Uint8List.fromList(pointer.asTypedList(len));
  } finally {
    shim.dart_mlx_free_buffer(pointer.cast());
  }
}
