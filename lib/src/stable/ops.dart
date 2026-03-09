part of '../stable_api.dart';

abstract final class MlxOps {
  /// Elementwise addition.
  static MlxArray add(MlxArray a, MlxArray b) => _binary(
    'dart_mlx_add',
    a,
    b,
    shim.dart_mlx_add,
  );

  /// Elementwise subtraction.
  static MlxArray subtract(MlxArray a, MlxArray b) => _binary(
    'dart_mlx_subtract',
    a,
    b,
    shim.dart_mlx_subtract,
  );

  /// Elementwise multiplication.
  static MlxArray multiply(MlxArray a, MlxArray b) => _binary(
    'dart_mlx_multiply',
    a,
    b,
    shim.dart_mlx_multiply,
  );

  /// Elementwise division.
  static MlxArray divide(MlxArray a, MlxArray b) => _binary(
    'dart_mlx_divide',
    a,
    b,
    shim.dart_mlx_divide,
  );

  /// Matrix multiplication.
  static MlxArray matmul(MlxArray a, MlxArray b) => _binary(
    'dart_mlx_matmul',
    a,
    b,
    shim.dart_mlx_matmul,
  );

  /// Casts an array to a different dtype.
  static MlxArray astype(MlxArray input, MlxDType dtype) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_astype',
        shim.dart_mlx_astype(input._handle, dtype.value),
      ),
    );
  }

  /// Elementwise equality comparison.
  static MlxArray equal(MlxArray a, MlxArray b) => _binary(
    'dart_mlx_equal',
    a,
    b,
    shim.dart_mlx_equal,
  );

  /// Selects values from [x] and [y] according to [condition].
  static MlxArray where(MlxArray condition, MlxArray x, MlxArray y) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_where',
        shim.dart_mlx_where(condition._handle, x._handle, y._handle),
      ),
    );
  }

  /// Absolute value.
  static MlxArray abs(MlxArray input) =>
      _unary('dart_mlx_abs', input, shim.dart_mlx_abs);

  /// Negation.
  static MlxArray negative(MlxArray input) =>
      _unary('dart_mlx_negative', input, shim.dart_mlx_negative);

  /// Exponential.
  static MlxArray exp(MlxArray input) =>
      _unary('dart_mlx_exp', input, shim.dart_mlx_exp);

  /// Natural logarithm.
  static MlxArray log(MlxArray input) =>
      _unary('dart_mlx_log', input, shim.dart_mlx_log);

  /// Sine.
  static MlxArray sin(MlxArray input) =>
      _unary('dart_mlx_sin', input, shim.dart_mlx_sin);

  /// Cosine.
  static MlxArray cos(MlxArray input) =>
      _unary('dart_mlx_cos', input, shim.dart_mlx_cos);

  /// Sum reduction.
  static MlxArray sum(MlxArray input, {int? axis, bool keepDims = false}) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        axis == null ? 'dart_mlx_sum' : 'dart_mlx_sum_axis',
        axis == null
            ? shim.dart_mlx_sum(input._handle, keepDims)
            : shim.dart_mlx_sum_axis(input._handle, axis, keepDims),
      ),
    );
  }

  /// Mean reduction.
  static MlxArray mean(MlxArray input, {int? axis, bool keepDims = false}) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        axis == null ? 'dart_mlx_mean' : 'dart_mlx_mean_axis',
        axis == null
            ? shim.dart_mlx_mean(input._handle, keepDims)
            : shim.dart_mlx_mean_axis(input._handle, axis, keepDims),
      ),
    );
  }

  /// Log-sum-exp reduction.
  static MlxArray logSumExp(MlxArray input, {int? axis, bool keepDims = false}) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_logsumexp',
        shim.dart_mlx_logsumexp(input._handle, axis ?? 0, axis != null, keepDims),
      ),
    );
  }

  /// Softmax over the given axis or the flattened input when omitted.
  static MlxArray softmax(MlxArray input, {int? axis, bool precise = false}) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_softmax',
        shim.dart_mlx_softmax(input._handle, axis ?? 0, axis != null, precise),
      ),
    );
  }

  /// Returns the top-[k] values.
  static MlxArray topK(MlxArray input, int k, {int? axis}) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_topk',
        shim.dart_mlx_topk(input._handle, k, axis ?? 0, axis != null),
      ),
    );
  }

  /// Concatenates arrays along [axis].
  static MlxArray concatenate(List<MlxArray> arrays, {int axis = 0}) {
    final handles = calloc<ffi.Pointer<ffi.Void>>(arrays.length);
    try {
      for (var index = 0; index < arrays.length; index++) {
        handles[index] = arrays[index]._handle;
      }
      _clearError();
      return MlxArray._(
        _checkHandle(
          'dart_mlx_concatenate',
          shim.dart_mlx_concatenate(handles, arrays.length, axis),
        ),
      );
    } finally {
      calloc.free(handles);
    }
  }

  /// Stacks arrays along [axis].
  static MlxArray stack(List<MlxArray> arrays, {int axis = 0}) {
    final handles = calloc<ffi.Pointer<ffi.Void>>(arrays.length);
    try {
      for (var index = 0; index < arrays.length; index++) {
        handles[index] = arrays[index]._handle;
      }
      _clearError();
      return MlxArray._(
        _checkHandle(
          'dart_mlx_stack',
          shim.dart_mlx_stack(handles, arrays.length, axis),
        ),
      );
    } finally {
      calloc.free(handles);
    }
  }

  /// Broadcasts [input] to [shape].
  static MlxArray broadcastTo(MlxArray input, List<int> shape) =>
      _withShape(shape, (shapePointer) {
        _clearError();
        return MlxArray._(
          _checkHandle(
            'dart_mlx_broadcast_to',
            shim.dart_mlx_broadcast_to(input._handle, shapePointer, shape.length),
          ),
        );
      });

  /// Expands a dimension at [axis].
  static MlxArray expandDims(MlxArray input, int axis) {
    _clearError();
    return MlxArray._(
      _checkHandle('dart_mlx_expand_dims', shim.dart_mlx_expand_dims(input._handle, axis)),
    );
  }

  /// Removes singleton dimensions.
  static MlxArray squeeze(MlxArray input) {
    _clearError();
    return MlxArray._(
      _checkHandle('dart_mlx_squeeze', shim.dart_mlx_squeeze(input._handle)),
    );
  }

  /// Clips values to `[min, max]` when provided.
  static MlxArray clip(MlxArray input, {double? min, double? max}) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_clip_scalar',
        shim.dart_mlx_clip_scalar(
          input._handle,
          min != null,
          min ?? 0,
          max != null,
          max ?? 0,
        ),
      ),
    );
  }

  /// Elementwise minimum.
  static MlxArray minimum(MlxArray a, MlxArray b) => _binary(
    'dart_mlx_minimum',
    a,
    b,
    shim.dart_mlx_minimum,
  );

  /// Elementwise maximum.
  static MlxArray maximum(MlxArray a, MlxArray b) => _binary(
    'dart_mlx_maximum',
    a,
    b,
    shim.dart_mlx_maximum,
  );

  /// Returns argmax indices.
  static MlxArray argmax(MlxArray input, {int? axis, bool keepDims = false}) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_argmax',
        shim.dart_mlx_argmax(input._handle, axis ?? 0, axis != null, keepDims),
      ),
    );
  }

  /// Returns argmin indices.
  static MlxArray argmin(MlxArray input, {int? axis, bool keepDims = false}) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_argmin',
        shim.dart_mlx_argmin(input._handle, axis ?? 0, axis != null, keepDims),
      ),
    );
  }

  /// Returns sorted values.
  static MlxArray sort(MlxArray input, {int? axis}) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_sort',
        shim.dart_mlx_sort(input._handle, axis ?? 0, axis != null),
      ),
    );
  }

  /// Returns sort indices.
  static MlxArray argsort(MlxArray input, {int? axis}) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_argsort',
        shim.dart_mlx_argsort(input._handle, axis ?? 0, axis != null),
      ),
    );
  }

  static MlxArray _binary(
    String operation,
    MlxArray a,
    MlxArray b,
    ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
    )
    callback,
  ) {
    _clearError();
    return MlxArray._(_checkHandle(operation, callback(a._handle, b._handle)));
  }

  static MlxArray _unary(
    String operation,
    MlxArray input,
    ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>) callback,
  ) {
    _clearError();
    return MlxArray._(_checkHandle(operation, callback(input._handle)));
  }
}

/// High-level random-key helpers.
