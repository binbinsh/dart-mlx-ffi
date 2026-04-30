part of '../stable_api.dart';

abstract final class MlxRandom {
  /// Creates a new MLX random key from a seed.
  static MlxArray key(int seed) {
    _clearError();
    return MlxArray._(
      _checkHandle('dart_mlx_random_key', shim.dart_mlx_random_key(seed)),
    );
  }

  /// Splits a key into two derived keys.
  static MlxRandomSplit split(MlxArray key) {
    final first = calloc<ffi.Pointer<ffi.Void>>();
    final second = calloc<ffi.Pointer<ffi.Void>>();
    try {
      _clearError();
      _checkStatus(
        'dart_mlx_random_split',
        shim.dart_mlx_random_split(key._handle, first, second),
      );
      return (
        first: MlxArray._(_checkHandle('dart_mlx_random_split.first', first.value)),
        second: MlxArray._(
          _checkHandle('dart_mlx_random_split.second', second.value),
        ),
      );
    } finally {
      calloc.free(first);
      calloc.free(second);
    }
  }

  /// Samples a uniform random array.
  static MlxArray uniform(
    List<int> shape, {
    double low = 0,
    double high = 1,
    MlxDType dtype = raw.mlx_dtype_.MLX_FLOAT32,
  }) => _withShape(shape, (shapePointer) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_random_uniform',
        shim.dart_mlx_random_uniform(
          low,
          high,
          shapePointer,
          shape.length,
          dtype.value,
        ),
      ),
    );
  });

  /// Samples a normal random array.
  static MlxArray normal(
    List<int> shape, {
    double loc = 0,
    double scale = 1,
    MlxDType dtype = raw.mlx_dtype_.MLX_FLOAT32,
  }) => _withShape(shape, (shapePointer) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_random_normal',
        shim.dart_mlx_random_normal(
          shapePointer,
          shape.length,
          dtype.value,
          loc,
          scale,
        ),
      ),
    );
  });

  /// Samples Bernoulli values from [probability].
  static MlxArray bernoulli(
    MlxArray probability, {
    List<int>? shape,
    MlxArray? key,
  }) {
    final resolvedShape = shape ?? probability.shape;
    return _withShape(resolvedShape, (shapePointer) {
      _clearError();
      return MlxArray._(
        _checkHandle(
          'dart_mlx_random_bernoulli',
          shim.dart_mlx_random_bernoulli(
            probability._handle,
            shapePointer,
            resolvedShape.length,
            key?._handle ?? ffi.nullptr,
          ),
        ),
      );
    });
  }

  /// Samples categorical indices from [logits].
  static MlxArray categorical(
    MlxArray logits, {
    int axis = -1,
    List<int>? shape,
    int? numSamples,
    MlxArray? key,
  }) {
    if (shape != null && numSamples != null) {
      throw ArgumentError(
        'Only one of shape or numSamples may be provided to categorical().',
      );
    }
    if (shape != null) {
      return _withShape(shape, (shapePointer) {
        _clearError();
        return MlxArray._(
          _checkHandle(
            'dart_mlx_random_categorical',
            shim.dart_mlx_random_categorical(
              logits._handle,
              axis,
              1,
              shapePointer,
              shape.length,
              0,
              key?._handle ?? ffi.nullptr,
            ),
          ),
        );
      });
    }
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_random_categorical',
        shim.dart_mlx_random_categorical(
          logits._handle,
          axis,
          numSamples == null ? 0 : 2,
          ffi.nullptr,
          0,
          numSamples ?? 0,
          key?._handle ?? ffi.nullptr,
        ),
      ),
    );
  }

  /// Randomly permutes [input] along [axis].
  static MlxArray permutation(
    MlxArray input, {
    int axis = 0,
    MlxArray? key,
  }) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_random_permutation',
        shim.dart_mlx_random_permutation(
          input._handle,
          axis,
          key?._handle ?? ffi.nullptr,
        ),
      ),
    );
  }

  /// Returns a random permutation of `0..n-1`.
  static MlxArray permutationArange(int n, {MlxArray? key}) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_random_permutation_arange',
        shim.dart_mlx_random_permutation_arange(n, key?._handle ?? ffi.nullptr),
      ),
    );
  }

  /// Samples a Gumbel random array.
  static MlxArray gumbel(
    List<int> shape, {
    MlxDType dtype = raw.mlx_dtype_.MLX_FLOAT32,
    MlxArray? key,
  }) => _withShape(shape, (shapePointer) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_random_gumbel',
        shim.dart_mlx_random_gumbel(
          shapePointer,
          shape.length,
          dtype.value,
          key?._handle ?? ffi.nullptr,
        ),
      ),
    );
  });

  /// Samples a Laplace random array.
  static MlxArray laplace(
    List<int> shape, {
    MlxDType dtype = raw.mlx_dtype_.MLX_FLOAT32,
    double loc = 0,
    double scale = 1,
    MlxArray? key,
  }) => _withShape(shape, (shapePointer) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_random_laplace',
        shim.dart_mlx_random_laplace(
          shapePointer,
          shape.length,
          dtype.value,
          loc,
          scale,
          key?._handle ?? ffi.nullptr,
        ),
      ),
    );
  });

  /// Samples a multivariate normal random array.
  static MlxArray multivariateNormal(
    MlxArray mean,
    MlxArray cov, {
    List<int> shape = const [],
    MlxDType dtype = raw.mlx_dtype_.MLX_FLOAT32,
    MlxArray? key,
  }) => _withShape(shape, (shapePointer) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_random_multivariate_normal',
        shim.dart_mlx_random_multivariate_normal(
          mean._handle,
          cov._handle,
          shapePointer,
          shape.length,
          dtype.value,
          key?._handle ?? ffi.nullptr,
        ),
      ),
    );
  });

  /// Samples integer values in `[low, high)`.
  static MlxArray randint(
    int low,
    int high,
    List<int> shape, {
    MlxDType dtype = raw.mlx_dtype_.MLX_INT32,
    MlxArray? key,
  }) => _withShape(shape, (shapePointer) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_random_randint',
        shim.dart_mlx_random_randint(
          low,
          high,
          shapePointer,
          shape.length,
          dtype.value,
          key?._handle ?? ffi.nullptr,
        ),
      ),
    );
  });
}

/// Higher-order transforms over [MlxFunction].
abstract final class MlxTransforms {
  /// Computes the JVP of [function] at [primals] along [tangents].
  static MlxJvpResult jvp(
    MlxFunction function,
    List<MlxArray> primals,
    List<MlxArray> tangents,
  ) {
    function._ensureOpen();
    final primalsHandles = calloc<ffi.Pointer<ffi.Void>>(primals.length);
    final tangentsHandles = calloc<ffi.Pointer<ffi.Void>>(tangents.length);
    final outputsOut = calloc<ffi.Pointer<ffi.Pointer<ffi.Void>>>();
    final outputsLen = calloc<ffi.Size>();
    final tangentsOut = calloc<ffi.Pointer<ffi.Pointer<ffi.Void>>>();
    final tangentsLen = calloc<ffi.Size>();
    try {
      for (var index = 0; index < primals.length; index++) {
        primalsHandles[index] = primals[index]._handle;
      }
      for (var index = 0; index < tangents.length; index++) {
        tangentsHandles[index] = tangents[index]._handle;
      }
      _clearError();
      _checkStatus(
        'dart_mlx_function_jvp',
        shim.dart_mlx_function_jvp(
          function._handle,
          primalsHandles,
          primals.length,
          tangentsHandles,
          tangents.length,
          outputsOut,
          outputsLen,
          tangentsOut,
          tangentsLen,
        ),
      );
      return (
        outputs: _readOutputArrayList(outputsOut.value, outputsLen.value),
        tangents: _readOutputArrayList(tangentsOut.value, tangentsLen.value),
      );
    } finally {
      if (outputsOut.value != ffi.nullptr) {
        calloc.free(outputsOut.value);
      }
      if (tangentsOut.value != ffi.nullptr) {
        calloc.free(tangentsOut.value);
      }
      calloc.free(outputsOut);
      calloc.free(outputsLen);
      calloc.free(tangentsOut);
      calloc.free(tangentsLen);
      calloc.free(primalsHandles);
      calloc.free(tangentsHandles);
    }
  }

  /// Computes the VJP of [function] at [primals] with [cotangents].
  static MlxVjpResult vjp(
    MlxFunction function,
    List<MlxArray> primals,
    List<MlxArray> cotangents,
  ) {
    function._ensureOpen();
    final primalsHandles = calloc<ffi.Pointer<ffi.Void>>(primals.length);
    final cotangentsHandles = calloc<ffi.Pointer<ffi.Void>>(cotangents.length);
    final outputsOut = calloc<ffi.Pointer<ffi.Pointer<ffi.Void>>>();
    final outputsLen = calloc<ffi.Size>();
    final cotangentsOut = calloc<ffi.Pointer<ffi.Pointer<ffi.Void>>>();
    final cotangentsLen = calloc<ffi.Size>();
    try {
      for (var index = 0; index < primals.length; index++) {
        primalsHandles[index] = primals[index]._handle;
      }
      for (var index = 0; index < cotangents.length; index++) {
        cotangentsHandles[index] = cotangents[index]._handle;
      }
      _clearError();
      _checkStatus(
        'dart_mlx_function_vjp',
        shim.dart_mlx_function_vjp(
          function._handle,
          primalsHandles,
          primals.length,
          cotangentsHandles,
          cotangents.length,
          outputsOut,
          outputsLen,
          cotangentsOut,
          cotangentsLen,
        ),
      );
      return (
        outputs: _readOutputArrayList(outputsOut.value, outputsLen.value),
        cotangents: _readOutputArrayList(cotangentsOut.value, cotangentsLen.value),
      );
    } finally {
      if (outputsOut.value != ffi.nullptr) {
        calloc.free(outputsOut.value);
      }
      if (cotangentsOut.value != ffi.nullptr) {
        calloc.free(cotangentsOut.value);
      }
      calloc.free(outputsOut);
      calloc.free(outputsLen);
      calloc.free(cotangentsOut);
      calloc.free(cotangentsLen);
      calloc.free(primalsHandles);
      calloc.free(cotangentsHandles);
    }
  }

  /// Computes value and gradients of [function] with respect to [argnums].
  static MlxValueAndGradResult valueAndGrad(
    MlxFunction function,
    List<MlxArray> inputs, {
    List<int> argnums = const [0],
  }) {
    function._ensureOpen();
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
          function._handle,
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
}

/// Compilation controls for MLX compiled closures.
abstract final class MlxCompile {
  /// Compiles [function] and returns a callable compiled function.
  static MlxFunction compile(MlxFunction function, {bool shapeless = false}) =>
      function.compile(shapeless: shapeless);

  /// Sets the global compile mode.
  static void setMode(MlxCompileMode mode) {
    _clearError();
    _checkStatus('mlx_set_compile_mode', raw.mlx_set_compile_mode(mode));
  }

  /// Enables compile.
  static void enable() {
    _clearError();
    _checkStatus('mlx_enable_compile', raw.mlx_enable_compile());
  }

  /// Disables compile.
  static void disable() {
    _clearError();
    _checkStatus('mlx_disable_compile', raw.mlx_disable_compile());
  }

  /// Clears compile caches.
  static void clearCache() {
    _clearError();
    _checkStatus('mlx_detail_compile_clear_cache', raw.mlx_detail_compile_clear_cache());
  }
}

/// Module-style high-level MLX entrypoint.
