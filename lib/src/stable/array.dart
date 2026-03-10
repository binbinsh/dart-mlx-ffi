part of '../stable_api.dart';

final class MlxArray {
  MlxArray._(this._handle, {bool owned = true}) : _owned = owned {
    if (_owned) {
      _finalizer.attach(this, _handle, detach: this);
    }
  }

  /// Creates a zero-filled array.
  factory MlxArray.zeros(
    List<int> shape, {
    MlxDType dtype = raw.mlx_dtype_.MLX_FLOAT32,
  }) => _withShape(shape, (shapePointer) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_zeros',
        shim.dart_mlx_zeros(shapePointer, shape.length, dtype.value),
      ),
    );
  });

  /// Creates a one-filled array.
  factory MlxArray.ones(
    List<int> shape, {
    MlxDType dtype = raw.mlx_dtype_.MLX_FLOAT32,
  }) => _withShape(shape, (shapePointer) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_ones',
        shim.dart_mlx_ones(shapePointer, shape.length, dtype.value),
      ),
    );
  });

  /// Creates an array filled with a scalar value.
  factory MlxArray.full(
    List<int> shape,
    double value, {
    MlxDType dtype = raw.mlx_dtype_.MLX_FLOAT32,
  }) => _withShape(shape, (shapePointer) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_full',
        shim.dart_mlx_full(shapePointer, shape.length, value, dtype.value),
      ),
    );
  });

  /// Creates a 1D evenly spaced range.
  factory MlxArray.arange(
    double start,
    double stop,
    double step, {
    MlxDType dtype = raw.mlx_dtype_.MLX_FLOAT32,
  }) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_arange',
        shim.dart_mlx_arange(start, stop, step, dtype.value),
      ),
    );
  }

  /// Creates a bool array from a Dart list and an explicit shape.
  factory MlxArray.fromBoolList(
    List<bool> values, {
    required List<int> shape,
  }) {
    final data = calloc<ffi.Uint8>(values.length);
    for (var index = 0; index < values.length; index++) {
      data[index] = values[index] ? 1 : 0;
    }
    return _withShape(shape, (shapePointer) {
      _clearError();
      final handle =
          hooks.debugArrayFromBoolOverride?.call(
            data.cast(),
            shapePointer,
            shape.length,
          ) ??
          shim.dart_mlx_array_from_bool(
        data.cast(),
        shapePointer,
        shape.length,
      );
      if (handle == ffi.nullptr) {
        calloc.free(data);
      }
      return MlxArray._(_checkHandle('dart_mlx_array_from_bool', handle));
    });
  }

  /// Creates an `int32` array from a Dart list and an explicit shape.
  factory MlxArray.fromInt32List(
    List<int> values, {
    required List<int> shape,
  }) {
    final data = calloc<ffi.Int32>(values.length);
    for (var index = 0; index < values.length; index++) {
      data[index] = values[index];
    }
    return _withShape(shape, (shapePointer) {
      _clearError();
      final handle =
          hooks.debugArrayFromInt32Override?.call(
            data.cast(),
            shapePointer,
            shape.length,
          ) ??
          shim.dart_mlx_array_from_int32(
        data.cast(),
        shapePointer,
        shape.length,
      );
      if (handle == ffi.nullptr) {
        calloc.free(data);
      }
      return MlxArray._(_checkHandle('dart_mlx_array_from_int32', handle));
    });
  }

  /// Creates a `float32` array from a Dart list and an explicit shape.
  factory MlxArray.fromFloat32List(
    List<double> values, {
    required List<int> shape,
  }) {
    final data = calloc<ffi.Float>(values.length);
    for (var index = 0; index < values.length; index++) {
      data[index] = values[index];
    }
    return _withShape(shape, (shapePointer) {
      _clearError();
      final handle =
          hooks.debugArrayFromFloat32Override?.call(
            data.cast(),
            shapePointer,
            shape.length,
          ) ??
          shim.dart_mlx_array_from_float32(
        data.cast(),
        shapePointer,
        shape.length,
      );
      if (handle == ffi.nullptr) {
        calloc.free(data);
      }
      return MlxArray._(_checkHandle('dart_mlx_array_from_float32', handle));
    });
  }

  /// Creates a `float64` array from a Dart list and an explicit shape.
  factory MlxArray.fromFloat64List(
    List<double> values, {
    required List<int> shape,
  }) {
    final data = calloc<ffi.Double>(values.length);
    for (var index = 0; index < values.length; index++) {
      data[index] = values[index];
    }
    return _withShape(shape, (shapePointer) {
      _clearError();
      final handle =
          hooks.debugArrayFromFloat64Override?.call(
            data.cast(),
            shapePointer,
            shape.length,
          ) ??
          shim.dart_mlx_array_from_float64(
        data.cast(),
        shapePointer,
        shape.length,
      );
      if (handle == ffi.nullptr) {
        calloc.free(data);
      }
      return MlxArray._(_checkHandle('dart_mlx_array_from_float64', handle));
    });
  }

  /// Creates an `int64` array from a Dart list and an explicit shape.
  factory MlxArray.fromInt64List(
    List<int> values, {
    required List<int> shape,
  }) {
    final data = calloc<ffi.Int64>(values.length);
    for (var index = 0; index < values.length; index++) {
      data[index] = values[index];
    }
    return _withShape(shape, (shapePointer) {
      _clearError();
      final handle =
          hooks.debugArrayFromInt64Override?.call(
            data.cast(),
            shapePointer,
            shape.length,
          ) ??
          shim.dart_mlx_array_from_int64(
            data.cast(),
            shapePointer,
            shape.length,
          );
      if (handle == ffi.nullptr) {
        calloc.free(data);
      }
      return MlxArray._(_checkHandle('dart_mlx_array_from_int64', handle));
    });
  }

  /// Creates a `uint64` array from a Dart list and an explicit shape.
  factory MlxArray.fromUint64List(
    List<int> values, {
    required List<int> shape,
  }) {
    final data = calloc<ffi.Uint64>(values.length);
    for (var index = 0; index < values.length; index++) {
      data[index] = values[index];
    }
    return _withShape(shape, (shapePointer) {
      _clearError();
      final handle =
          hooks.debugArrayFromUint64Override?.call(
            data.cast(),
            shapePointer,
            shape.length,
          ) ??
          shim.dart_mlx_array_from_uint64(
            data.cast(),
            shapePointer,
            shape.length,
          );
      if (handle == ffi.nullptr) {
        calloc.free(data);
      }
      return MlxArray._(_checkHandle('dart_mlx_array_from_uint64', handle));
    });
  }

  static final Finalizer<ffi.Pointer<ffi.Void>> _finalizer =
      Finalizer<ffi.Pointer<ffi.Void>>(shim.dart_mlx_array_free);

  final ffi.Pointer<ffi.Void> _handle;
  final bool _owned;
  bool _closed = false;

  /// Returns `true` once [close] has been called.
  bool get isClosed => _closed;

  /// Element type of the underlying MLX array.
  MlxDType get dtype {
    _ensureOpen();
    return raw.mlx_dtype_.fromValue(shim.dart_mlx_array_dtype(_handle));
  }

  /// Rank of the array.
  int get ndim {
    _ensureOpen();
    return shim.dart_mlx_array_ndim(_handle);
  }

  /// Number of elements in the array.
  int get size {
    _ensureOpen();
    return shim.dart_mlx_array_size(_handle);
  }

  /// Logical shape of the array.
  List<int> get shape {
    _ensureOpen();
    final out = calloc<ffi.Int>(ndim);
    try {
      _clearError();
      _checkStatus(
        'dart_mlx_array_copy_shape',
        shim.dart_mlx_array_copy_shape(_handle, out, ndim),
      );
      return List<int>.generate(ndim, (index) => out[index]);
    } finally {
      calloc.free(out);
    }
  }

  /// Forces evaluation of the underlying lazy MLX array.
  void eval() {
    _ensureOpen();
    _clearError();
    _checkStatus('dart_mlx_array_eval', shim.dart_mlx_array_eval(_handle));
  }

  /// Copies the array contents into a Dart list.
  ///
  /// The high-level wrapper currently supports:
  /// `bool`, `int32`, `float32`, and `float64`.
  List<Object> toList() {
    _ensureOpen();
    return switch (dtype) {
      raw.mlx_dtype_.MLX_BOOL => _copyBoolData(),
      raw.mlx_dtype_.MLX_UINT32 => _copyUint32Data(),
      raw.mlx_dtype_.MLX_UINT64 => _copyUint64Data(),
      raw.mlx_dtype_.MLX_INT32 => _copyInt32Data(),
      raw.mlx_dtype_.MLX_INT64 => _copyInt64Data(),
      raw.mlx_dtype_.MLX_FLOAT32 => _copyFloat32Data(),
      raw.mlx_dtype_.MLX_FLOAT64 => _copyFloat64Data(),
      _ => throw UnsupportedError(
        'High-level list conversion currently supports bool/int32/float32/float64. '
        'Use package:dart_mlx_ffi/raw.dart for the full MLX C surface.',
      ),
    };
  }

  /// Returns a reshaped view-like array with the requested shape.
  MlxArray reshape(List<int> newShape) {
    _ensureOpen();
    return _withShape(newShape, (shapePointer) {
      _clearError();
      return MlxArray._(
        _checkHandle(
          'dart_mlx_reshape',
          shim.dart_mlx_reshape(_handle, shapePointer, newShape.length),
        ),
      );
    });
  }

  /// Returns the transposed array.
  MlxArray transpose() {
    _ensureOpen();
    _clearError();
    return MlxArray._(
      _checkHandle('dart_mlx_transpose', shim.dart_mlx_transpose(_handle)),
    );
  }

  /// Shorthand for [transpose], similar to Python MLX `array.T`.
  MlxArray get T => transpose();

  /// Transposes with explicit axes.
  MlxArray transposeAxes(List<int> axes) => MlxTensor.transposeAxes(this, axes);

  /// Flattens a range of axes.
  MlxArray flatten({int startAxis = 0, int endAxis = -1}) =>
      MlxTensor.flatten(this, startAxis: startAxis, endAxis: endAxis);

  /// Moves one axis to a new position.
  MlxArray moveaxis(int source, int destination) =>
      MlxTensor.moveaxis(this, source, destination);

  /// Swaps two axes.
  MlxArray swapaxes(int axis1, int axis2) =>
      MlxTensor.swapaxes(this, axis1, axis2);

  /// Repeats this array according to [reps].
  MlxArray tile(List<int> reps) => MlxTensor.tile(this, reps);

  /// Pads this array.
  MlxArray pad({
    List<int>? axes,
    required List<int> lowPads,
    required List<int> highPads,
    MlxArray? padValue,
    String mode = 'constant',
  }) => MlxTensor.pad(
    this,
    axes: axes,
    lowPads: lowPads,
    highPads: highPads,
    padValue: padValue,
    mode: mode,
  );

  /// Symmetric pad helper.
  MlxArray padSymmetric(
    int padWidth, {
    MlxArray? padValue,
    String mode = 'constant',
  }) => MlxTensor.padSymmetric(
    this,
    padWidth,
    padValue: padValue,
    mode: mode,
  );

  /// Unflattens [axis] into [shape].
  MlxArray unflatten({required int axis, required List<int> shape}) =>
      MlxTensor.unflatten(this, axis: axis, shape: shape);

  /// Absolute value.
  MlxArray abs() => MlxOps.abs(this);

  /// Negation.
  MlxArray negative() => MlxOps.negative(this);

  /// Exponential.
  MlxArray exp() => MlxOps.exp(this);

  /// Natural logarithm.
  MlxArray log() => MlxOps.log(this);

  /// Sine.
  MlxArray sin() => MlxOps.sin(this);

  /// Cosine.
  MlxArray cos() => MlxOps.cos(this);

  /// Hyperbolic tangent.
  MlxArray tanh() => MlxMore.tanh(this);

  /// Sum reduction.
  MlxArray sum({int? axis, bool keepDims = false}) =>
      MlxOps.sum(this, axis: axis, keepDims: keepDims);

  /// Mean reduction.
  MlxArray mean({int? axis, bool keepDims = false}) =>
      MlxOps.mean(this, axis: axis, keepDims: keepDims);

  /// Variance reduction.
  MlxArray variance({
    int? axis,
    List<int>? axes,
    bool keepDims = false,
    int ddof = 0,
  }) => MlxOps.variance(
    this,
    axis: axis,
    axes: axes,
    keepDims: keepDims,
    ddof: ddof,
  );

  /// Log-sum-exp reduction.
  MlxArray logSumExp({int? axis, bool keepDims = false}) =>
      MlxOps.logSumExp(this, axis: axis, keepDims: keepDims);

  /// Softmax.
  MlxArray softmax({int? axis, bool precise = false}) =>
      MlxOps.softmax(this, axis: axis, precise: precise);

  /// Top-k values.
  MlxArray topK(int k, {int? axis}) => MlxOps.topK(this, k, axis: axis);

  /// Takes values by flat or axis-based indices.
  MlxArray take(MlxArray indices, {int? axis}) =>
      MlxTensor.take(this, indices, axis: axis);

  /// Takes values along a specific axis.
  MlxArray takeAlongAxis(MlxArray indices, {required int axis}) =>
      MlxTensor.takeAlongAxis(this, indices, axis: axis);

  /// Gathers slices from this array using a single index tensor.
  MlxArray gatherSingle(
    MlxArray indices, {
    required int axis,
    required List<int> sliceSizes,
  }) => MlxTensor.gatherSingle(
    this,
    indices,
    axis: axis,
    sliceSizes: sliceSizes,
  );

  /// Slices this array with explicit start/stop/stride vectors.
  MlxArray slice({
    required List<int> start,
    required List<int> stop,
    List<int>? strides,
  }) => MlxTensor.slice(this, start: start, stop: stop, strides: strides);

  /// Slices this array using tensor start indices.
  MlxArray sliceDynamic({
    required MlxArray start,
    required List<int> axes,
    required List<int> sliceSize,
  }) => MlxTensor.sliceDynamic(
    this,
    start: start,
    axes: axes,
    sliceSize: sliceSize,
  );

  /// Updates a slice of this array.
  MlxArray sliceUpdate(
    MlxArray update, {
    required List<int> start,
    required List<int> stop,
    List<int>? strides,
  }) => MlxTensor.sliceUpdate(
    this,
    update,
    start: start,
    stop: stop,
    strides: strides,
  );

  /// Updates a dynamic slice of this array.
  MlxArray sliceUpdateDynamic(
    MlxArray update, {
    required MlxArray start,
    required List<int> axes,
  }) => MlxTensor.sliceUpdateDynamic(
    this,
    update,
    start: start,
    axes: axes,
  );

  /// Computes a tensor contraction with [other].
  MlxArray tensordot(
    MlxArray other, {
    int? axis,
    List<int>? axesA,
    List<int>? axesB,
  }) => MlxTensor.tensordot(
    this,
    other,
    axis: axis,
    axesA: axesA,
    axesB: axesB,
  );

  /// Matrix multiplication.
  MlxArray matmul(MlxArray other) => MlxOps.matmul(this, other);

  /// Matrix multiply with additive bias: `alpha * (a @ b) + beta * c`.
  MlxArray addmm(
    MlxArray a,
    MlxArray b, {
    double alpha = 1,
    double beta = 1,
  }) => MlxOps.addmm(this, a, b, alpha: alpha, beta: beta);

  /// Casts this array to a different dtype.
  MlxArray astype(MlxDType dtype) => MlxOps.astype(this, dtype);

  /// Elementwise equality comparison.
  MlxArray equal(MlxArray other) => MlxOps.equal(this, other);

  /// Broadcasts this array to [shape].
  MlxArray broadcastTo(List<int> shape) => MlxOps.broadcastTo(this, shape);

  /// Expands a dimension at [axis].
  MlxArray expandDims(int axis) => MlxOps.expandDims(this, axis);

  /// Removes singleton dimensions.
  MlxArray squeeze() => MlxOps.squeeze(this);

  /// Clips values to `[min, max]` when provided.
  MlxArray clip({double? min, double? max}) => MlxOps.clip(this, min: min, max: max);

  /// Elementwise minimum.
  MlxArray minimum(MlxArray other) => MlxOps.minimum(this, other);

  /// Elementwise maximum.
  MlxArray maximum(MlxArray other) => MlxOps.maximum(this, other);

  /// Returns argmax indices.
  MlxArray argmax({int? axis, bool keepDims = false}) =>
      MlxOps.argmax(this, axis: axis, keepDims: keepDims);

  /// Returns argmin indices.
  MlxArray argmin({int? axis, bool keepDims = false}) =>
      MlxOps.argmin(this, axis: axis, keepDims: keepDims);

  /// Returns sorted values.
  MlxArray sort({int? axis}) => MlxOps.sort(this, axis: axis);

  /// Returns sort indices.
  MlxArray argsort({int? axis}) => MlxOps.argsort(this, axis: axis);

  /// Elementwise addition.
  MlxArray operator +(MlxArray other) => MlxOps.add(this, other);

  /// Elementwise subtraction.
  MlxArray operator -(MlxArray other) => MlxOps.subtract(this, other);

  /// Elementwise multiplication.
  MlxArray operator *(MlxArray other) => MlxOps.multiply(this, other);

  /// Elementwise division.
  MlxArray operator /(MlxArray other) => MlxOps.divide(this, other);

  /// Unary negation.
  MlxArray operator -() => MlxOps.negative(this);

  @override
  String toString() {
    _ensureOpen();
    _clearError();
    return _copyOwnedString(shim.dart_mlx_array_tostring_copy(_handle));
  }

  /// Releases the native MLX array handle.
  ///
  /// This method is idempotent.
  void close() {
    if (_closed) {
      return;
    }
    _closed = true;
    if (_owned) {
      _finalizer.detach(this);
      shim.dart_mlx_array_free(_handle);
    }
  }

  void _ensureOpen() {
    if (_closed) {
      throw StateError('MlxArray has been closed.');
    }
  }

  List<Object> _copyBoolData() {
    final out = calloc<ffi.Uint8>(size);
    try {
      _clearError();
      _checkStatus(
        'dart_mlx_array_copy_bool',
        shim.dart_mlx_array_copy_bool(_handle, out, size),
      );
      return List<Object>.generate(size, (index) => out[index] != 0);
    } finally {
      calloc.free(out);
    }
  }

  List<Object> _copyInt32Data() {
    final out = calloc<ffi.Int32>(size);
    try {
      _clearError();
      _checkStatus(
        'dart_mlx_array_copy_int32',
        shim.dart_mlx_array_copy_int32(_handle, out, size),
      );
      return List<Object>.generate(size, (index) => out[index]);
    } finally {
      calloc.free(out);
    }
  }

  List<Object> _copyUint32Data() {
    final out = calloc<ffi.Uint32>(size);
    try {
      _clearError();
      _checkStatus(
        'dart_mlx_array_copy_uint32',
        shim.dart_mlx_array_copy_uint32(_handle, out, size),
      );
      return List<Object>.generate(size, (index) => out[index]);
    } finally {
      calloc.free(out);
    }
  }

  List<Object> _copyInt64Data() {
    final out = calloc<ffi.Int64>(size);
    try {
      _clearError();
      _checkStatus(
        'dart_mlx_array_copy_int64',
        shim.dart_mlx_array_copy_int64(_handle, out, size),
      );
      return List<Object>.generate(size, (index) => out[index]);
    } finally {
      calloc.free(out);
    }
  }

  List<Object> _copyUint64Data() {
    final out = calloc<ffi.Uint64>(size);
    try {
      _clearError();
      _checkStatus(
        'dart_mlx_array_copy_uint64',
        shim.dart_mlx_array_copy_uint64(_handle, out, size),
      );
      return List<Object>.generate(size, (index) => out[index]);
    } finally {
      calloc.free(out);
    }
  }

  List<Object> _copyFloat32Data() {
    final out = calloc<ffi.Float>(size);
    try {
      _clearError();
      _checkStatus(
        'dart_mlx_array_copy_float32',
        shim.dart_mlx_array_copy_float32(_handle, out, size),
      );
      return List<Object>.generate(size, (index) => out[index]);
    } finally {
      calloc.free(out);
    }
  }

  List<Object> _copyFloat64Data() {
    final out = calloc<ffi.Double>(size);
    try {
      _clearError();
      _checkStatus(
        'dart_mlx_array_copy_float64',
        shim.dart_mlx_array_copy_float64(_handle, out, size),
      );
      return List<Object>.generate(size, (index) => out[index]);
    } finally {
      calloc.free(out);
    }
  }
}

/// Managed wrapper for the current MLX default device.
