part of '../stable_api.dart';

/// CUDA fast kernel configuration.
final class MlxCudaConfig {
  MlxCudaConfig() : _handle = _checkHandle('dart_mlx_cuda_config_new', shim.dart_mlx_cuda_config_new());

  final shim.DartMlxCudaConfigHandle _handle;
  bool _closed = false;

  void addOutputArg(List<int> shape, MlxDType dtype) {
    _ensureOpen();
    _withShape(shape, (pointer) {
      _clearError();
      _checkStatus(
        'dart_mlx_cuda_config_add_output_arg',
        shim.dart_mlx_cuda_config_add_output_arg(_handle, pointer, shape.length, dtype.value),
      );
    });
  }

  void setGrid(int x, int y, int z) {
    _ensureOpen();
    _clearError();
    _checkStatus('dart_mlx_cuda_config_set_grid', shim.dart_mlx_cuda_config_set_grid(_handle, x, y, z));
  }

  void setThreadGroup(int x, int y, int z) {
    _ensureOpen();
    _clearError();
    _checkStatus(
      'dart_mlx_cuda_config_set_thread_group',
      shim.dart_mlx_cuda_config_set_thread_group(_handle, x, y, z),
    );
  }

  void setInitValue(double value) {
    _ensureOpen();
    _clearError();
    _checkStatus(
      'dart_mlx_cuda_config_set_init_value',
      shim.dart_mlx_cuda_config_set_init_value(_handle, value),
    );
  }

  void setVerbose(bool value) {
    _ensureOpen();
    _clearError();
    _checkStatus(
      'dart_mlx_cuda_config_set_verbose',
      shim.dart_mlx_cuda_config_set_verbose(_handle, value),
    );
  }

  void addTemplateDtype(String name, MlxDType dtype) {
    _ensureOpen();
    _withCString(name, (pointer) {
      _clearError();
      _checkStatus(
        'dart_mlx_cuda_config_add_template_dtype',
        shim.dart_mlx_cuda_config_add_template_dtype(_handle, pointer, dtype.value),
      );
    });
  }

  void addTemplateInt(String name, int value) {
    _ensureOpen();
    _withCString(name, (pointer) {
      _clearError();
      _checkStatus(
        'dart_mlx_cuda_config_add_template_int',
        shim.dart_mlx_cuda_config_add_template_int(_handle, pointer, value),
      );
    });
  }

  void addTemplateBool(String name, bool value) {
    _ensureOpen();
    _withCString(name, (pointer) {
      _clearError();
      _checkStatus(
        'dart_mlx_cuda_config_add_template_bool',
        shim.dart_mlx_cuda_config_add_template_bool(_handle, pointer, value),
      );
    });
  }

  void close() {
    if (_closed) return;
    _closed = true;
    shim.dart_mlx_cuda_config_free(_handle);
  }

  void _ensureOpen() {
    if (_closed) throw StateError('MlxCudaConfig has been closed.');
  }
}

/// Metal fast kernel configuration.
final class MlxMetalConfig {
  MlxMetalConfig() : _handle = _checkHandle('dart_mlx_metal_config_new', shim.dart_mlx_metal_config_new());

  final shim.DartMlxMetalConfigHandle _handle;
  bool _closed = false;

  void addOutputArg(List<int> shape, MlxDType dtype) {
    _ensureOpen();
    _withShape(shape, (pointer) {
      _clearError();
      _checkStatus(
        'dart_mlx_metal_config_add_output_arg',
        shim.dart_mlx_metal_config_add_output_arg(_handle, pointer, shape.length, dtype.value),
      );
    });
  }

  void setGrid(int x, int y, int z) {
    _ensureOpen();
    _clearError();
    _checkStatus('dart_mlx_metal_config_set_grid', shim.dart_mlx_metal_config_set_grid(_handle, x, y, z));
  }

  void setThreadGroup(int x, int y, int z) {
    _ensureOpen();
    _clearError();
    _checkStatus(
      'dart_mlx_metal_config_set_thread_group',
      shim.dart_mlx_metal_config_set_thread_group(_handle, x, y, z),
    );
  }

  void setInitValue(double value) {
    _ensureOpen();
    _clearError();
    _checkStatus(
      'dart_mlx_metal_config_set_init_value',
      shim.dart_mlx_metal_config_set_init_value(_handle, value),
    );
  }

  void setVerbose(bool value) {
    _ensureOpen();
    _clearError();
    _checkStatus(
      'dart_mlx_metal_config_set_verbose',
      shim.dart_mlx_metal_config_set_verbose(_handle, value),
    );
  }

  void addTemplateDtype(String name, MlxDType dtype) {
    _ensureOpen();
    _withCString(name, (pointer) {
      _clearError();
      _checkStatus(
        'dart_mlx_metal_config_add_template_dtype',
        shim.dart_mlx_metal_config_add_template_dtype(_handle, pointer, dtype.value),
      );
    });
  }

  void addTemplateInt(String name, int value) {
    _ensureOpen();
    _withCString(name, (pointer) {
      _clearError();
      _checkStatus(
        'dart_mlx_metal_config_add_template_int',
        shim.dart_mlx_metal_config_add_template_int(_handle, pointer, value),
      );
    });
  }

  void addTemplateBool(String name, bool value) {
    _ensureOpen();
    _withCString(name, (pointer) {
      _clearError();
      _checkStatus(
        'dart_mlx_metal_config_add_template_bool',
        shim.dart_mlx_metal_config_add_template_bool(_handle, pointer, value),
      );
    });
  }

  void close() {
    if (_closed) return;
    _closed = true;
    shim.dart_mlx_metal_config_free(_handle);
  }

  void _ensureOpen() {
    if (_closed) throw StateError('MlxMetalConfig has been closed.');
  }
}

/// CUDA custom kernel.
final class MlxCudaKernel {
  MlxCudaKernel(
    String name,
    List<String> inputNames,
    List<String> outputNames,
    String source, {
    String header = '',
    bool ensureRowContiguous = true,
    int sharedMemory = 0,
  }) : _handle = _withStringLists(
         name,
         inputNames,
         outputNames,
         source,
         header,
         (namePtr, inputPtr, inputLen, outputPtr, outputLen, sourcePtr, headerPtr) =>
             _checkHandle(
               'dart_mlx_cuda_kernel_new',
               shim.dart_mlx_cuda_kernel_new(
                 namePtr,
                 inputPtr,
                 inputLen,
                 outputPtr,
                 outputLen,
                 sourcePtr,
                 headerPtr,
                 ensureRowContiguous,
                 sharedMemory,
               ),
             ),
       );

  final shim.DartMlxCudaKernelHandle _handle;
  bool _closed = false;

  List<MlxArray> apply(List<MlxArray> inputs, MlxCudaConfig config) {
    _ensureOpen();
    final inputHandles = calloc<ffi.Pointer<ffi.Void>>(inputs.length);
    final outputsOut = calloc<ffi.Pointer<ffi.Pointer<ffi.Void>>>();
    final outputsLen = calloc<ffi.Size>();
    try {
      for (var index = 0; index < inputs.length; index++) {
        inputHandles[index] = inputs[index]._handle;
      }
      _clearError();
      _checkStatus(
        'dart_mlx_cuda_kernel_apply',
        shim.dart_mlx_cuda_kernel_apply(
          _handle,
          inputHandles,
          inputs.length,
          config._handle,
          outputsOut,
          outputsLen,
        ),
      );
      return _readOutputArrayList(outputsOut.value, outputsLen.value);
    } finally {
      if (outputsOut.value != ffi.nullptr) {
        calloc.free(outputsOut.value);
      }
      calloc.free(outputsOut);
      calloc.free(outputsLen);
      calloc.free(inputHandles);
    }
  }

  void close() {
    if (_closed) return;
    _closed = true;
    shim.dart_mlx_cuda_kernel_free(_handle);
  }

  void _ensureOpen() {
    if (_closed) throw StateError('MlxCudaKernel has been closed.');
  }
}

/// Metal custom kernel.
final class MlxMetalKernel {
  MlxMetalKernel(
    String name,
    List<String> inputNames,
    List<String> outputNames,
    String source, {
    String header = '',
    bool ensureRowContiguous = true,
    bool atomicOutputs = false,
  }) : _handle = _withStringLists(
         name,
         inputNames,
         outputNames,
         source,
         header,
         (namePtr, inputPtr, inputLen, outputPtr, outputLen, sourcePtr, headerPtr) =>
             _checkHandle(
               'dart_mlx_metal_kernel_new',
               shim.dart_mlx_metal_kernel_new(
                 namePtr,
                 inputPtr,
                 inputLen,
                 outputPtr,
                 outputLen,
                 sourcePtr,
                 headerPtr,
                 ensureRowContiguous,
                 atomicOutputs,
               ),
             ),
       );

  final shim.DartMlxMetalKernelHandle _handle;
  bool _closed = false;

  List<MlxArray> apply(List<MlxArray> inputs, MlxMetalConfig config) {
    _ensureOpen();
    final inputHandles = calloc<ffi.Pointer<ffi.Void>>(inputs.length);
    final outputsOut = calloc<ffi.Pointer<ffi.Pointer<ffi.Void>>>();
    final outputsLen = calloc<ffi.Size>();
    try {
      for (var index = 0; index < inputs.length; index++) {
        inputHandles[index] = inputs[index]._handle;
      }
      _clearError();
      _checkStatus(
        'dart_mlx_metal_kernel_apply',
        shim.dart_mlx_metal_kernel_apply(
          _handle,
          inputHandles,
          inputs.length,
          config._handle,
          outputsOut,
          outputsLen,
        ),
      );
      return _readOutputArrayList(outputsOut.value, outputsLen.value);
    } finally {
      if (outputsOut.value != ffi.nullptr) {
        calloc.free(outputsOut.value);
      }
      calloc.free(outputsOut);
      calloc.free(outputsLen);
      calloc.free(inputHandles);
    }
  }

  void close() {
    if (_closed) return;
    _closed = true;
    shim.dart_mlx_metal_kernel_free(_handle);
  }

  void _ensureOpen() {
    if (_closed) throw StateError('MlxMetalKernel has been closed.');
  }
}

T _withStringLists<T>(
  String name,
  List<String> inputNames,
  List<String> outputNames,
  String source,
  String header,
  T Function(
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
    int,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
    int,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Char>,
  )
  callback,
) {
  return _withCString(name, (namePtr) {
    return _withCString(source, (sourcePtr) {
      return _withCString(header, (headerPtr) {
        return _withCStringList(inputNames, (inputPtr, inputLen) {
          return _withCStringList(outputNames, (outputPtr, outputLen) {
            return callback(
              namePtr,
              inputPtr,
              inputLen,
              outputPtr,
              outputLen,
              sourcePtr,
              headerPtr,
            );
          });
        });
      });
    });
  });
}

T _withCStringList<T>(
  List<String> values,
  T Function(ffi.Pointer<ffi.Pointer<ffi.Char>>, int) callback,
) {
  final ptr = calloc<ffi.Pointer<ffi.Char>>(values.length);
  final allocated = <ffi.Pointer<ffi.Char>>[];
  try {
    for (var index = 0; index < values.length; index++) {
      final s = values[index].toNativeUtf8().cast<ffi.Char>();
      allocated.add(s);
      ptr[index] = s;
    }
    return callback(ptr, values.length);
  } finally {
    for (final value in allocated) {
      calloc.free(value);
    }
    calloc.free(ptr);
  }
}
