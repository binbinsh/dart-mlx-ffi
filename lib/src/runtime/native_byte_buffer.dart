import 'dart:convert';
import 'dart:ffi' as ffi;

import 'native_ffi.dart' as dz;

final class NativeByteBuffer {
  NativeByteBuffer._({
    required this.pointer,
    required this.length,
    required void Function()? close,
  }) : _close = close;

  factory NativeByteBuffer.utf8(String value, {dz.NativeFfi? ffiRuntime}) {
    return NativeByteBuffer.bytes(utf8.encode(value), ffiRuntime: ffiRuntime);
  }

  factory NativeByteBuffer.allocate(int length, {dz.NativeFfi? ffiRuntime}) {
    final buffer = dz.NativeByteArray.allocate(length, ffiRuntime: ffiRuntime);
    return NativeByteBuffer._(
      pointer: buffer.pointer,
      length: buffer.length,
      close: buffer.close,
    );
  }

  factory NativeByteBuffer.bytes(List<int> value, {dz.NativeFfi? ffiRuntime}) {
    if (value.isEmpty) {
      return NativeByteBuffer._(pointer: ffi.nullptr, length: 0, close: null);
    }
    final buffer = dz.NativeByteArray.bytes(value, ffiRuntime: ffiRuntime);
    return NativeByteBuffer._(
      pointer: buffer.pointer,
      length: buffer.length,
      close: buffer.close,
    );
  }

  final ffi.Pointer<ffi.Uint8> pointer;
  final int length;
  final void Function()? _close;

  void close() {
    _close?.call();
  }
}
