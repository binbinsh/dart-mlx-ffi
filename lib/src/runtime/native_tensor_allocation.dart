import 'dart:ffi' as ffi;

import 'native_ffi.dart' as dz;

import 'native_bindings.dart' as native;
import 'runtime.dart';

final nativeTensorAllocationFinalizer = Finalizer<NativeTensorAllocation>(
  (allocation) => allocation.release(),
);

final class NativeTensorAllocation {
  NativeTensorAllocation.runtime(ffi.Pointer<ffi.Void> pointer)
    : _pointer = pointer,
      _byteLength = 0,
      _nativeFfi = null;

  NativeTensorAllocation.nativeFfi(
    ffi.Pointer<ffi.Void> pointer, {
    required int byteLength,
    required dz.NativeFfi runtime,
  }) : _pointer = pointer,
       _byteLength = byteLength,
       _nativeFfi = runtime;

  ffi.Pointer<ffi.Void> _pointer;
  final int _byteLength;
  final dz.NativeFfi? _nativeFfi;

  ffi.Pointer<ffi.Void> get pointer => _pointer;

  void release() {
    final pointer = _pointer;
    if (pointer == ffi.nullptr) {
      return;
    }
    _pointer = ffi.nullptr;
    final nativeFfi = _nativeFfi;
    if (nativeFfi != null) {
      nativeFfi.free(pointer, _byteLength);
      return;
    }
    native.freeBuf(pointer);
  }
}

final class NativeFfiTensorAllocation {
  const NativeFfiTensorAllocation({
    required this.allocation,
    required this.byteLength,
  });

  final NativeTensorAllocation allocation;
  final int byteLength;
}

NativeFfiTensorAllocation allocateNativeFfiTensor({
  required dz.NativeFfi runtime,
  required RuntimeTensorDataType dtype,
  required List<int> shape,
}) {
  final byteLength = runtime.tensorByteLength(dtype, shape);
  final pointer = runtime.alloc(byteLength);
  return NativeFfiTensorAllocation(
    allocation: NativeTensorAllocation.nativeFfi(
      pointer,
      byteLength: byteLength,
      runtime: runtime,
    ),
    byteLength: byteLength,
  );
}
