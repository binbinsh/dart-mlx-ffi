import 'native_ffi.dart' as nf;

import 'runtime.dart';

RuntimeTensorDataType nativeFfiDtype(RuntimeTensorDataType dtype) => dtype;

nf.NativeTensorMemoryKind nativeFfiMemoryKind(RuntimeTensorMemoryKind kind) {
  return switch (kind) {
    RuntimeTensorMemoryKind.cpu => nf.NativeTensorMemoryKind.cpu,
    RuntimeTensorMemoryKind.nativeHandle =>
      nf.NativeTensorMemoryKind.nativeHandle,
  };
}
