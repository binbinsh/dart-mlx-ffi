part of '../stable_api.dart';

abstract final class MlxFft {
  static MlxArray _callListFft(
    String op,
    MlxArray input,
    ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Int>,
      int,
      ffi.Pointer<ffi.Int>,
      int,
    )
    callback, {
    List<int> n = const [],
    List<int> axes = const [],
  }) => _withInts(n, (nPtr, nLen) {
    return _withInts(axes, (axesPtr, axesLen) {
      _clearError();
      return MlxArray._(
        _checkHandle(op, callback(input._handle, nPtr, nLen, axesPtr, axesLen)),
      );
    });
  });

  /// Complex FFT.
  static MlxArray fft(MlxArray input, {int n = -1, int axis = -1}) {
    _clearError();
    return MlxArray._(
      _checkHandle('dart_mlx_fft_fft', shim.dart_mlx_fft_fft(input._handle, n, axis)),
    );
  }

  /// Complex inverse FFT.
  static MlxArray ifft(MlxArray input, {int n = -1, int axis = -1}) {
    _clearError();
    return MlxArray._(
      _checkHandle('dart_mlx_fft_ifft', shim.dart_mlx_fft_ifft(input._handle, n, axis)),
    );
  }

  /// Real FFT.
  static MlxArray rfft(MlxArray input, {int n = -1, int axis = -1}) {
    _clearError();
    return MlxArray._(
      _checkHandle('dart_mlx_fft_rfft', shim.dart_mlx_fft_rfft(input._handle, n, axis)),
    );
  }

  /// Inverse real FFT.
  static MlxArray irfft(MlxArray input, {int n = -1, int axis = -1}) {
    _clearError();
    return MlxArray._(
      _checkHandle('dart_mlx_fft_irfft', shim.dart_mlx_fft_irfft(input._handle, n, axis)),
    );
  }

  static MlxArray fft2(MlxArray input, {List<int> n = const [], List<int> axes = const []}) =>
      _callListFft('dart_mlx_fft_fft2', input, shim.dart_mlx_fft_fft2, n: n, axes: axes);

  static MlxArray fftn(MlxArray input, {List<int> n = const [], List<int> axes = const []}) =>
      _callListFft('dart_mlx_fft_fftn', input, shim.dart_mlx_fft_fftn, n: n, axes: axes);

  static MlxArray ifft2(MlxArray input, {List<int> n = const [], List<int> axes = const []}) =>
      _callListFft('dart_mlx_fft_ifft2', input, shim.dart_mlx_fft_ifft2, n: n, axes: axes);

  static MlxArray ifftn(MlxArray input, {List<int> n = const [], List<int> axes = const []}) =>
      _callListFft('dart_mlx_fft_ifftn', input, shim.dart_mlx_fft_ifftn, n: n, axes: axes);

  static MlxArray rfft2(MlxArray input, {List<int> n = const [], List<int> axes = const []}) =>
      _callListFft('dart_mlx_fft_rfft2', input, shim.dart_mlx_fft_rfft2, n: n, axes: axes);

  static MlxArray rfftn(MlxArray input, {List<int> n = const [], List<int> axes = const []}) =>
      _callListFft('dart_mlx_fft_rfftn', input, shim.dart_mlx_fft_rfftn, n: n, axes: axes);

  static MlxArray irfft2(MlxArray input, {List<int> n = const [], List<int> axes = const []}) =>
      _callListFft('dart_mlx_fft_irfft2', input, shim.dart_mlx_fft_irfft2, n: n, axes: axes);

  static MlxArray irfftn(MlxArray input, {List<int> n = const [], List<int> axes = const []}) =>
      _callListFft('dart_mlx_fft_irfftn', input, shim.dart_mlx_fft_irfftn, n: n, axes: axes);

  static MlxArray fftshift(MlxArray input, {List<int> axes = const []}) =>
      _withInts(axes, (axesPtr, axesLen) {
        _clearError();
        return MlxArray._(
          _checkHandle(
            'dart_mlx_fft_fftshift',
            shim.dart_mlx_fft_fftshift(input._handle, axesPtr, axesLen),
          ),
        );
      });

  static MlxArray ifftshift(MlxArray input, {List<int> axes = const []}) =>
      _withInts(axes, (axesPtr, axesLen) {
        _clearError();
        return MlxArray._(
          _checkHandle(
            'dart_mlx_fft_ifftshift',
            shim.dart_mlx_fft_ifftshift(input._handle, axesPtr, axesLen),
          ),
        );
      });
}

/// Module-style FFT namespace.
final class MlxFftModule {
  const MlxFftModule._();

  /// Complex FFT.
  MlxArray fft(MlxArray input, {int n = -1, int axis = -1}) =>
      MlxFft.fft(input, n: n, axis: axis);

  /// Complex inverse FFT.
  MlxArray ifft(MlxArray input, {int n = -1, int axis = -1}) =>
      MlxFft.ifft(input, n: n, axis: axis);

  /// Real FFT.
  MlxArray rfft(MlxArray input, {int n = -1, int axis = -1}) =>
      MlxFft.rfft(input, n: n, axis: axis);

  /// Inverse real FFT.
  MlxArray irfft(MlxArray input, {int n = -1, int axis = -1}) =>
      MlxFft.irfft(input, n: n, axis: axis);

  MlxArray fft2(MlxArray input, {List<int> n = const [], List<int> axes = const []}) =>
      MlxFft.fft2(input, n: n, axes: axes);

  MlxArray fftn(MlxArray input, {List<int> n = const [], List<int> axes = const []}) =>
      MlxFft.fftn(input, n: n, axes: axes);

  MlxArray ifft2(MlxArray input, {List<int> n = const [], List<int> axes = const []}) =>
      MlxFft.ifft2(input, n: n, axes: axes);

  MlxArray ifftn(MlxArray input, {List<int> n = const [], List<int> axes = const []}) =>
      MlxFft.ifftn(input, n: n, axes: axes);

  MlxArray rfft2(MlxArray input, {List<int> n = const [], List<int> axes = const []}) =>
      MlxFft.rfft2(input, n: n, axes: axes);

  MlxArray rfftn(MlxArray input, {List<int> n = const [], List<int> axes = const []}) =>
      MlxFft.rfftn(input, n: n, axes: axes);

  MlxArray irfft2(MlxArray input, {List<int> n = const [], List<int> axes = const []}) =>
      MlxFft.irfft2(input, n: n, axes: axes);

  MlxArray irfftn(MlxArray input, {List<int> n = const [], List<int> axes = const []}) =>
      MlxFft.irfftn(input, n: n, axes: axes);

  MlxArray fftshift(MlxArray input, {List<int> axes = const []}) =>
      MlxFft.fftshift(input, axes: axes);

  MlxArray ifftshift(MlxArray input, {List<int> axes = const []}) =>
      MlxFft.ifftshift(input, axes: axes);
}

/// High-level linear algebra namespace.
abstract final class MlxLinalg {
  /// Matrix inverse.
  static MlxArray inv(MlxArray input) {
    _clearError();
    return MlxArray._(
      _checkHandle('dart_mlx_linalg_inv', shim.dart_mlx_linalg_inv(input._handle)),
    );
  }

  /// Solve `Ax = b`.
  static MlxArray solve(MlxArray a, MlxArray b) {
    _clearError();
    return MlxArray._(
      _checkHandle('dart_mlx_linalg_solve', shim.dart_mlx_linalg_solve(a._handle, b._handle)),
    );
  }

  /// QR decomposition.
  static MlxQrResult qr(MlxArray input) {
    final q = calloc<ffi.Pointer<ffi.Void>>();
    final r = calloc<ffi.Pointer<ffi.Void>>();
    try {
      _clearError();
      _checkStatus('dart_mlx_linalg_qr', shim.dart_mlx_linalg_qr(input._handle, q, r));
      return (
        q: MlxArray._(_checkHandle('dart_mlx_linalg_qr.q', q.value)),
        r: MlxArray._(_checkHandle('dart_mlx_linalg_qr.r', r.value)),
      );
    } finally {
      calloc.free(q);
      calloc.free(r);
    }
  }

  /// Eigenvalue decomposition.
  static MlxEigResult eig(MlxArray input) {
    final values = calloc<ffi.Pointer<ffi.Void>>();
    final vectors = calloc<ffi.Pointer<ffi.Void>>();
    try {
      _clearError();
      _checkStatus(
        'dart_mlx_linalg_eig',
        shim.dart_mlx_linalg_eig(input._handle, values, vectors),
      );
      return (
        values: MlxArray._(
          _checkHandle('dart_mlx_linalg_eig.values', values.value),
        ),
        vectors: MlxArray._(
          _checkHandle('dart_mlx_linalg_eig.vectors', vectors.value),
        ),
      );
    } finally {
      calloc.free(values);
      calloc.free(vectors);
    }
  }

  static MlxArray cholesky(MlxArray input, {bool upper = false}) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_linalg_cholesky',
        shim.dart_mlx_linalg_cholesky(input._handle, upper),
      ),
    );
  }

  static MlxArray cross(MlxArray a, MlxArray b, {int axis = -1}) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_linalg_cross',
        shim.dart_mlx_linalg_cross(a._handle, b._handle, axis),
      ),
    );
  }

  static MlxEigResult eigh(MlxArray input, {String uplo = 'L'}) {
    final values = calloc<ffi.Pointer<ffi.Void>>();
    final vectors = calloc<ffi.Pointer<ffi.Void>>();
    try {
      return _withCString(uplo, (uploPtr) {
        _clearError();
        _checkStatus(
          'dart_mlx_linalg_eigh',
          shim.dart_mlx_linalg_eigh(input._handle, uploPtr, values, vectors),
        );
        return (
          values: MlxArray._(_checkHandle('dart_mlx_linalg_eigh.values', values.value)),
          vectors: MlxArray._(_checkHandle('dart_mlx_linalg_eigh.vectors', vectors.value)),
        );
      });
    } finally {
      calloc.free(values);
      calloc.free(vectors);
    }
  }

  static MlxArray eigvals(MlxArray input) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_linalg_eigvals',
        shim.dart_mlx_linalg_eigvals(input._handle),
      ),
    );
  }

  static MlxArray eigvalsh(MlxArray input, {String uplo = 'L'}) =>
      _withCString(uplo, (uploPtr) {
        _clearError();
        return MlxArray._(
          _checkHandle(
            'dart_mlx_linalg_eigvalsh',
            shim.dart_mlx_linalg_eigvalsh(input._handle, uploPtr),
          ),
        );
      });

  static MlxLuResult lu(MlxArray input) {
    final outputsOut = calloc<ffi.Pointer<ffi.Pointer<ffi.Void>>>();
    final outputsLen = calloc<ffi.Size>();
    try {
      _clearError();
      _checkStatus(
        'dart_mlx_linalg_lu',
        shim.dart_mlx_linalg_lu(input._handle, outputsOut, outputsLen),
      );
      final outputs = _readOutputArrayList(outputsOut.value, outputsLen.value);
      return (rowPivots: outputs[0], l: outputs[1], u: outputs[2]);
    } finally {
      if (outputsOut.value != ffi.nullptr) {
        calloc.free(outputsOut.value);
      }
      calloc.free(outputsOut);
      calloc.free(outputsLen);
    }
  }

  static MlxLuFactorResult luFactor(MlxArray input) {
    final lu = calloc<ffi.Pointer<ffi.Void>>();
    final pivots = calloc<ffi.Pointer<ffi.Void>>();
    try {
      _clearError();
      _checkStatus(
        'dart_mlx_linalg_lu_factor',
        shim.dart_mlx_linalg_lu_factor(input._handle, lu, pivots),
      );
      return (
        lu: MlxArray._(_checkHandle('dart_mlx_linalg_lu_factor.lu', lu.value)),
        pivots: MlxArray._(
          _checkHandle('dart_mlx_linalg_lu_factor.pivots', pivots.value),
        ),
      );
    } finally {
      calloc.free(lu);
      calloc.free(pivots);
    }
  }

  static MlxArray norm(
    MlxArray input, {
    double ord = 2.0,
    List<int>? axes,
    bool keepDims = false,
  }) => _withInts(axes ?? const [], (axesPtr, axesLen) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_linalg_norm',
        shim.dart_mlx_linalg_norm(input._handle, ord, axesPtr, axesLen, keepDims),
      ),
    );
  });

  static MlxArray matrixNorm(
    MlxArray input, {
    String ord = 'fro',
    List<int>? axes,
    bool keepDims = false,
  }) => _withCString(ord, (ordPtr) {
    return _withInts(axes ?? const [], (axesPtr, axesLen) {
      _clearError();
      return MlxArray._(
        _checkHandle(
          'dart_mlx_linalg_norm_matrix',
          shim.dart_mlx_linalg_norm_matrix(
            input._handle,
            ordPtr,
            axesPtr,
            axesLen,
            keepDims,
          ),
        ),
      );
    });
  });

  static MlxArray l2Norm(
    MlxArray input, {
    List<int>? axes,
    bool keepDims = false,
  }) => _withInts(axes ?? const [], (axesPtr, axesLen) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_linalg_norm_l2',
        shim.dart_mlx_linalg_norm_l2(input._handle, axesPtr, axesLen, keepDims),
      ),
    );
  });

  static MlxArray pinv(MlxArray input) {
    _clearError();
    return MlxArray._(
      _checkHandle('dart_mlx_linalg_pinv', shim.dart_mlx_linalg_pinv(input._handle)),
    );
  }

  static MlxArray solveTriangular(MlxArray a, MlxArray b, {bool upper = false}) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_linalg_solve_triangular',
        shim.dart_mlx_linalg_solve_triangular(a._handle, b._handle, upper),
      ),
    );
  }

  static MlxSvdResult svd(MlxArray input, {bool computeUv = true}) {
    final outputsOut = calloc<ffi.Pointer<ffi.Pointer<ffi.Void>>>();
    final outputsLen = calloc<ffi.Size>();
    try {
      _clearError();
      _checkStatus(
        'dart_mlx_linalg_svd',
        shim.dart_mlx_linalg_svd(input._handle, computeUv, outputsOut, outputsLen),
      );
      final outputs = _readOutputArrayList(outputsOut.value, outputsLen.value);
      if (!computeUv) {
        return (u: null, s: outputs[0], vt: null);
      }
      return (u: outputs[0], s: outputs[1], vt: outputs[2]);
    } finally {
      if (outputsOut.value != ffi.nullptr) {
        calloc.free(outputsOut.value);
      }
      calloc.free(outputsOut);
      calloc.free(outputsLen);
    }
  }
}

/// Module-style linear algebra namespace.
final class MlxLinalgModule {
  const MlxLinalgModule._();

  /// Matrix inverse.
  MlxArray inv(MlxArray input) => MlxLinalg.inv(input);

  /// Solve `Ax = b`.
  MlxArray solve(MlxArray a, MlxArray b) => MlxLinalg.solve(a, b);

  /// QR decomposition.
  MlxQrResult qr(MlxArray input) => MlxLinalg.qr(input);

  /// Eigenvalue decomposition.
  MlxEigResult eig(MlxArray input) => MlxLinalg.eig(input);

  MlxArray cholesky(MlxArray input, {bool upper = false}) =>
      MlxLinalg.cholesky(input, upper: upper);

  MlxArray cross(MlxArray a, MlxArray b, {int axis = -1}) =>
      MlxLinalg.cross(a, b, axis: axis);

  MlxEigResult eigh(MlxArray input, {String uplo = 'L'}) =>
      MlxLinalg.eigh(input, uplo: uplo);

  MlxArray eigvals(MlxArray input) => MlxLinalg.eigvals(input);

  MlxArray eigvalsh(MlxArray input, {String uplo = 'L'}) =>
      MlxLinalg.eigvalsh(input, uplo: uplo);

  MlxLuResult lu(MlxArray input) => MlxLinalg.lu(input);

  MlxLuFactorResult luFactor(MlxArray input) => MlxLinalg.luFactor(input);

  MlxArray norm(MlxArray input, {double ord = 2.0, List<int>? axes, bool keepDims = false}) =>
      MlxLinalg.norm(input, ord: ord, axes: axes, keepDims: keepDims);

  MlxArray matrixNorm(
    MlxArray input, {
    String ord = 'fro',
    List<int>? axes,
    bool keepDims = false,
  }) => MlxLinalg.matrixNorm(input, ord: ord, axes: axes, keepDims: keepDims);

  MlxArray l2Norm(MlxArray input, {List<int>? axes, bool keepDims = false}) =>
      MlxLinalg.l2Norm(input, axes: axes, keepDims: keepDims);

  MlxArray pinv(MlxArray input) => MlxLinalg.pinv(input);

  MlxArray solveTriangular(MlxArray a, MlxArray b, {bool upper = false}) =>
      MlxLinalg.solveTriangular(a, b, upper: upper);

  MlxSvdResult svd(MlxArray input, {bool computeUv = true}) =>
      MlxLinalg.svd(input, computeUv: computeUv);
}

/// Runtime helpers that affect evaluation scheduling.
