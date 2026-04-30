part of '../stable_api.dart';

final class _Lease {
  _Lease(this._dispose);

  final void Function() _dispose;
  int _refCount = 1;
  bool _disposed = false;

  _Lease retain() {
    assert(!_disposed, 'Lease has already been disposed.');
    _refCount++;
    return this;
  }

  void release() {
    if (_disposed) {
      return;
    }
    _refCount--;
    if (_refCount == 0) {
      _disposed = true;
      _dispose();
    }
  }
}

/// Callable MLX function backed by an `mlx_closure`.
final class MlxFunction {
  MlxFunction._(
    this._handle, {
    required _Lease primaryLease,
    List<_Lease> retainedLeases = const [],
  }) : _primaryLease = primaryLease,
       _retainedLeases = retainedLeases;

  /// Creates an MLX function from a Dart callback.
  factory MlxFunction.fromCallback(MlxCallback callback) {
    late final ffi.NativeCallable<shim.DartMlxClosureCallback> nativeCallback;
    int trampoline(
      ffi.Pointer<ffi.Pointer<shim.DartMlxArrayHandle>> outputsPtr,
      ffi.Pointer<ffi.Size> outputsLenPtr,
      ffi.Pointer<shim.DartMlxArrayHandle> inputsPtr,
      int inputsLen,
    ) {
      try {
        final inputs = List<MlxArray>.generate(
          inputsLen,
          (index) => MlxArray._(inputsPtr[index], owned: false),
        );
        final outputs = callback(inputs);
        final outputHandles = calloc<ffi.Pointer<ffi.Void>>(outputs.length);
        for (var index = 0; index < outputs.length; index++) {
          outputHandles[index] = outputs[index]._handle;
        }
        outputsPtr.value = outputHandles.cast();
        outputsLenPtr.value = outputs.length;
        return 0;
      } catch (error) {
        _lastErrorMessage = error.toString();
        outputsPtr.value = ffi.nullptr;
        outputsLenPtr.value = 0;
        return 1;
      }
    }

    nativeCallback = ffi.NativeCallable<shim.DartMlxClosureCallback>.isolateLocal(
      trampoline,
      exceptionalReturn: 1,
    );
    nativeCallback.keepIsolateAlive = false;
    _clearError();
    final handle = _checkHandle(
      'dart_mlx_function_from_callback',
      shim.dart_mlx_function_from_callback(nativeCallback.nativeFunction),
    );
    return MlxFunction._(
      handle,
      primaryLease: _Lease(() => shim.dart_mlx_function_free(handle)),
      retainedLeases: [_Lease(nativeCallback.close)],
    );
  }

  final shim.DartMlxClosureHandle _handle;
  final _Lease _primaryLease;
  final List<_Lease> _retainedLeases;
  bool _closed = false;

  /// Invokes the function with MLX array inputs.
  List<MlxArray> call(List<MlxArray> inputs) {
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
        'dart_mlx_function_apply',
        shim.dart_mlx_function_apply(
          _handle,
          inputHandles,
          inputs.length,
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

  /// Returns a checkpointed function.
  MlxFunction checkpoint() {
    _ensureOpen();
    _clearError();
    final handle = _checkHandle(
      'dart_mlx_function_checkpoint',
      shim.dart_mlx_function_checkpoint(_handle),
    );
    return MlxFunction._(
      handle,
      primaryLease: _Lease(() => shim.dart_mlx_function_free(handle)),
      retainedLeases: _retainForChild(),
    );
  }

  /// Returns a compiled function.
  MlxFunction compile({bool shapeless = false}) {
    _ensureOpen();
    _clearError();
    final handle = _checkHandle(
      'dart_mlx_function_compile',
      shim.dart_mlx_function_compile(_handle, shapeless),
    );
    return MlxFunction._(
      handle,
      primaryLease: _Lease(() => shim.dart_mlx_function_free(handle)),
      retainedLeases: _retainForChild(),
    );
  }

  /// Returns a function with a custom VJP rule.
  MlxFunction customVjp(MlxCustomVjp custom) {
    _ensureOpen();
    custom._ensureOpen();
    _clearError();
    final handle = _checkHandle(
      'dart_mlx_function_custom_vjp',
      shim.dart_mlx_function_custom_vjp(_handle, custom._handle),
    );
    return MlxFunction._(
      handle,
      primaryLease: _Lease(() => shim.dart_mlx_function_free(handle)),
      retainedLeases: [..._retainForChild(), ...custom._retainForChild()],
    );
  }

  /// Returns a function with custom transform rules.
  MlxFunction customFunction({
    MlxCustomVjp? vjp,
    MlxCustomJvp? jvp,
  }) {
    _ensureOpen();
    vjp?._ensureOpen();
    jvp?._ensureOpen();
    _clearError();
    final handle = _checkHandle(
      'dart_mlx_function_custom',
      shim.dart_mlx_function_custom(
        _handle,
        vjp?._handle ?? ffi.nullptr,
        jvp?._handle ?? ffi.nullptr,
      ),
    );
    return MlxFunction._(
      handle,
      primaryLease: _Lease(() => shim.dart_mlx_function_free(handle)),
      retainedLeases: [
        ..._retainForChild(),
        if (vjp != null) ...vjp._retainForChild(),
        if (jvp != null) ...jvp._retainForChild(),
      ],
    );
  }

  /// Returns a reusable value-and-grad callable.
  MlxValueAndGradFunction valueAndGrad({List<int> argnums = const [0]}) {
    _ensureOpen();
    return MlxValueAndGradFunction._(
      _handle,
      argnums,
      retainedLeases: _retainForChild(),
    );
  }

  /// Releases the native closure and Dart callback resources.
  void close() {
    if (_closed) {
      return;
    }
    _closed = true;
    _primaryLease.release();
    for (final lease in _retainedLeases) {
      lease.release();
    }
  }

  void _ensureOpen() {
    if (_closed) {
      throw StateError('MlxFunction has been closed.');
    }
  }

  List<_Lease> _retainForChild() {
    final leases = <_Lease>[_primaryLease.retain()];
    leases.addAll(_retainedLeases.map((lease) => lease.retain()));
    return leases;
  }
}

/// Callable MLX function backed by an `mlx_closure_kwargs`.
final class MlxKwFunction {
  MlxKwFunction._(
    this._handle, {
    required _Lease primaryLease,
    List<_Lease> retainedLeases = const [],
  }) : _primaryLease = primaryLease,
       _retainedLeases = retainedLeases;

  /// Creates an MLX kwargs function from a Dart callback.
  factory MlxKwFunction.fromCallback(MlxKwCallback callback) {
    late final ffi.NativeCallable<shim.DartMlxKwCallback> nativeCallback;
    int trampoline(
      ffi.Pointer<ffi.Pointer<shim.DartMlxArrayHandle>> outputsPtr,
      ffi.Pointer<ffi.Size> outputsLenPtr,
      ffi.Pointer<shim.DartMlxArrayHandle> inputsPtr,
      int inputsLen,
      ffi.Pointer<ffi.Pointer<ffi.Pointer<ffi.Char>>> keysPtr,
      ffi.Pointer<ffi.Pointer<ffi.Pointer<shim.DartMlxArrayHandle>>> valuesPtr,
      ffi.Pointer<ffi.Size> valuesLenPtr,
    ) {
      try {
        final inputs = List<MlxArray>.generate(
          inputsLen,
          (index) => MlxArray._(inputsPtr[index], owned: false),
        );
        final kwargs = <String, MlxArray>{};
        for (var index = 0; index < valuesLenPtr.value; index++) {
          final key = keysPtr.value[index].cast<Utf8>().toDartString();
          kwargs[key] =
              MlxArray._(valuesPtr.value[index].cast(), owned: false);
        }
        final outputs = callback(inputs, kwargs);
        final outputHandles = calloc<ffi.Pointer<ffi.Void>>(outputs.length);
        for (var index = 0; index < outputs.length; index++) {
          outputHandles[index] = outputs[index]._handle;
        }
        outputsPtr.value = outputHandles.cast();
        outputsLenPtr.value = outputs.length;
        return 0;
      } catch (error) {
        _lastErrorMessage = error.toString();
        outputsPtr.value = ffi.nullptr;
        outputsLenPtr.value = 0;
        return 1;
      }
    }

    nativeCallback = ffi.NativeCallable<shim.DartMlxKwCallback>.isolateLocal(
      trampoline,
      exceptionalReturn: 1,
    );
    nativeCallback.keepIsolateAlive = false;
    _clearError();
    final handle = _checkHandle(
      'dart_mlx_kw_function_from_callback',
      shim.dart_mlx_kw_function_from_callback(nativeCallback.nativeFunction),
    );
    return MlxKwFunction._(
      handle,
      primaryLease: _Lease(() => shim.dart_mlx_kw_function_free(handle)),
      retainedLeases: [_Lease(nativeCallback.close)],
    );
  }

  final shim.DartMlxKwHandle _handle;
  final _Lease _primaryLease;
  final List<_Lease> _retainedLeases;
  bool _closed = false;

  /// Invokes the function with positional and keyword tensor arguments.
  List<MlxArray> call(
    List<MlxArray> inputs, {
    Map<String, MlxArray> kwargs = const {},
  }) {
    _ensureOpen();
    final outputsOut = calloc<ffi.Pointer<ffi.Pointer<ffi.Void>>>();
    final outputsLen = calloc<ffi.Size>();
    try {
      _withArrayHandles(inputs, (inputHandles, inputLen) {
        return _withKwargHandles(kwargs, (keys, values, valueLen) {
          _clearError();
          _checkStatus(
            'dart_mlx_kw_function_apply',
            shim.dart_mlx_kw_function_apply(
              _handle,
              inputHandles,
              inputLen,
              keys,
              values,
              valueLen,
              outputsOut,
              outputsLen,
            ),
          );
        });
      });
      return _readOutputArrayList(outputsOut.value, outputsLen.value);
    } finally {
      if (outputsOut.value != ffi.nullptr) {
        calloc.free(outputsOut.value);
      }
      calloc.free(outputsOut);
      calloc.free(outputsLen);
    }
  }

  /// Releases the native closure and Dart callback resources.
  void close() {
    if (_closed) {
      return;
    }
    _closed = true;
    _primaryLease.release();
    for (final lease in _retainedLeases) {
      lease.release();
    }
  }

  void _ensureOpen() {
    if (_closed) {
      throw StateError('MlxKwFunction has been closed.');
    }
  }
}

/// Custom VJP callback wrapper.
final class MlxCustomVjp {
  MlxCustomVjp._(
    this._handle, {
    required _Lease primaryLease,
    List<_Lease> retainedLeases = const [],
  }) : _primaryLease = primaryLease,
       _retainedLeases = retainedLeases;

  /// Creates a custom VJP callback from Dart code.
  factory MlxCustomVjp.fromCallback(MlxCustomVjpCallback callback) {
    late final ffi.NativeCallable<shim.DartMlxCustomCallback> nativeCallback;
    int trampoline(
      ffi.Pointer<ffi.Pointer<shim.DartMlxArrayHandle>> outputsPtr,
      ffi.Pointer<ffi.Size> outputsLenPtr,
      ffi.Pointer<shim.DartMlxArrayHandle> primalsPtr,
      int primalsLen,
      ffi.Pointer<shim.DartMlxArrayHandle> outputsInPtr,
      int outputsInLen,
      ffi.Pointer<shim.DartMlxArrayHandle> cotangentsPtr,
      int cotangentsLen,
    ) {
      try {
        final primals = List<MlxArray>.generate(
          primalsLen,
          (index) => MlxArray._(primalsPtr[index], owned: false),
        );
        final cotangents = List<MlxArray>.generate(
          cotangentsLen,
          (index) => MlxArray._(cotangentsPtr[index], owned: false),
        );
        final outputs = List<MlxArray>.generate(
          outputsInLen,
          (index) => MlxArray._(outputsInPtr[index], owned: false),
        );
        final grads = callback(primals, cotangents, outputs);
        final outputHandles = calloc<ffi.Pointer<ffi.Void>>(grads.length);
        for (var index = 0; index < grads.length; index++) {
          outputHandles[index] = grads[index]._handle;
        }
        outputsPtr.value = outputHandles.cast();
        outputsLenPtr.value = grads.length;
        return 0;
      } catch (error) {
        _lastErrorMessage = error.toString();
        outputsPtr.value = ffi.nullptr;
        outputsLenPtr.value = 0;
        return 1;
      }
    }

    nativeCallback = ffi.NativeCallable<shim.DartMlxCustomCallback>.isolateLocal(
      trampoline,
      exceptionalReturn: 1,
    );
    nativeCallback.keepIsolateAlive = false;
    _clearError();
    final handle = _checkHandle(
      'dart_mlx_custom_from_callback',
      shim.dart_mlx_custom_from_callback(nativeCallback.nativeFunction),
    );
    return MlxCustomVjp._(
      handle,
      primaryLease: _Lease(() => shim.dart_mlx_custom_free(handle)),
      retainedLeases: [_Lease(nativeCallback.close)],
    );
  }

  final shim.DartMlxCustomHandle _handle;
  final _Lease _primaryLease;
  final List<_Lease> _retainedLeases;
  bool _closed = false;

  /// Releases native callback resources.
  void close() {
    if (_closed) {
      return;
    }
    _closed = true;
    _primaryLease.release();
    for (final lease in _retainedLeases) {
      lease.release();
    }
  }

  void _ensureOpen() {
    if (_closed) {
      throw StateError('MlxCustomVjp has been closed.');
    }
  }

  List<_Lease> _retainForChild() {
    final leases = <_Lease>[_primaryLease.retain()];
    leases.addAll(_retainedLeases.map((lease) => lease.retain()));
    return leases;
  }
}

/// Custom JVP callback wrapper.
final class MlxCustomJvp {
  MlxCustomJvp._(
    this._handle, {
    required _Lease primaryLease,
    List<_Lease> retainedLeases = const [],
  }) : _primaryLease = primaryLease,
       _retainedLeases = retainedLeases;

  /// Creates a custom JVP callback from Dart code.
  factory MlxCustomJvp.fromCallback(MlxCustomJvpCallback callback) {
    late final ffi.NativeCallable<shim.DartMlxCustomJvpCallback> nativeCallback;
    int trampoline(
      ffi.Pointer<ffi.Pointer<shim.DartMlxArrayHandle>> outputsPtr,
      ffi.Pointer<ffi.Size> outputsLenPtr,
      ffi.Pointer<shim.DartMlxArrayHandle> primalsPtr,
      int primalsLen,
      ffi.Pointer<shim.DartMlxArrayHandle> tangentsPtr,
      int tangentsLen,
      ffi.Pointer<ffi.Int> argnumsPtr,
      int argnumsLen,
    ) {
      try {
        final primals = List<MlxArray>.generate(
          primalsLen,
          (index) => MlxArray._(primalsPtr[index], owned: false),
        );
        final tangents = List<MlxArray>.generate(
          tangentsLen,
          (index) => MlxArray._(tangentsPtr[index], owned: false),
        );
        final argnums = List<int>.generate(argnumsLen, (index) => argnumsPtr[index]);
        final outputs = callback(primals, tangents, argnums);
        final outputHandles = calloc<ffi.Pointer<ffi.Void>>(outputs.length);
        for (var index = 0; index < outputs.length; index++) {
          outputHandles[index] = outputs[index]._handle;
        }
        outputsPtr.value = outputHandles.cast();
        outputsLenPtr.value = outputs.length;
        return 0;
      } catch (error) {
        _lastErrorMessage = error.toString();
        outputsPtr.value = ffi.nullptr;
        outputsLenPtr.value = 0;
        return 1;
      }
    }

    nativeCallback =
        ffi.NativeCallable<shim.DartMlxCustomJvpCallback>.isolateLocal(
          trampoline,
          exceptionalReturn: 1,
        );
    nativeCallback.keepIsolateAlive = false;
    _clearError();
    final handle = _checkHandle(
      'dart_mlx_custom_jvp_from_callback',
      shim.dart_mlx_custom_jvp_from_callback(nativeCallback.nativeFunction),
    );
    return MlxCustomJvp._(
      handle,
      primaryLease: _Lease(() => shim.dart_mlx_custom_jvp_free(handle)),
      retainedLeases: [_Lease(nativeCallback.close)],
    );
  }

  final shim.DartMlxCustomJvpHandle _handle;
  final _Lease _primaryLease;
  final List<_Lease> _retainedLeases;
  bool _closed = false;

  /// Releases native callback resources.
  void close() {
    if (_closed) {
      return;
    }
    _closed = true;
    _primaryLease.release();
    for (final lease in _retainedLeases) {
      lease.release();
    }
  }

  void _ensureOpen() {
    if (_closed) {
      throw StateError('MlxCustomJvp has been closed.');
    }
  }

  List<_Lease> _retainForChild() {
    final leases = <_Lease>[_primaryLease.retain()];
    leases.addAll(_retainedLeases.map((lease) => lease.retain()));
    return leases;
  }
}

/// Reusable callable returned by [MlxFunction.valueAndGrad].
final class MlxValueAndGradFunction {
  MlxValueAndGradFunction._(
    this._functionHandle,
    this.argnums, {
    required List<_Lease> retainedLeases,
  }) : _retainedLeases = retainedLeases;

  final shim.DartMlxClosureHandle _functionHandle;
  final List<int> argnums;
  final List<_Lease> _retainedLeases;
  bool _closed = false;

  /// Invokes the reusable value-and-grad function.
  MlxValueAndGradResult call(List<MlxArray> inputs) {
    _ensureOpen();
    final inputHandles = calloc<ffi.Pointer<ffi.Void>>(inputs.length);
    final argnumsPtr = calloc<ffi.Int>(argnums.length);
    final valuesOut = calloc<ffi.Pointer<ffi.Pointer<ffi.Void>>>();
    final valuesLen = calloc<ffi.Size>();
    final gradsOut = calloc<ffi.Pointer<ffi.Pointer<ffi.Void>>>();
    final gradsLen = calloc<ffi.Size>();
    try {
      for (var index = 0; index < inputs.length; index++) {
        inputHandles[index] = inputs[index]._handle;
      }
      for (var index = 0; index < argnums.length; index++) {
        argnumsPtr[index] = argnums[index];
      }
      _clearError();
      _checkStatus(
        'dart_mlx_function_value_and_grad',
        shim.dart_mlx_function_value_and_grad(
          _functionHandle,
          argnumsPtr,
          argnums.length,
          inputHandles,
          inputs.length,
          valuesOut,
          valuesLen,
          gradsOut,
          gradsLen,
        ),
      );
      return (
        values: _readOutputArrayList(valuesOut.value, valuesLen.value),
        gradients: _readOutputArrayList(gradsOut.value, gradsLen.value),
      );
    } finally {
      if (valuesOut.value != ffi.nullptr) {
        calloc.free(valuesOut.value);
      }
      if (gradsOut.value != ffi.nullptr) {
        calloc.free(gradsOut.value);
      }
      calloc.free(valuesOut);
      calloc.free(valuesLen);
      calloc.free(gradsOut);
      calloc.free(gradsLen);
      calloc.free(inputHandles);
      calloc.free(argnumsPtr);
    }
  }

  /// Releases retained function resources.
  void close() {
    if (_closed) {
      return;
    }
    _closed = true;
    for (final lease in _retainedLeases) {
      lease.release();
    }
  }

  void _ensureOpen() {
    if (_closed) {
      throw StateError('MlxValueAndGradFunction has been closed.');
    }
  }
}

/// High-level arithmetic operations over [MlxArray].
