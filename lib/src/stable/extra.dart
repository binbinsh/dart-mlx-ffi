part of '../stable_api.dart';

abstract final class MlxExtra {
  static MlxArray diag(MlxArray input, {int k = 0}) {
    _clearError();
    return MlxArray._(
      _checkHandle('dart_mlx_diag', shim.dart_mlx_diag(input._handle, k)),
    );
  }

  static MlxArray diagonal(
    MlxArray input, {
    int offset = 0,
    int axis1 = 0,
    int axis2 = 1,
  }) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_diagonal',
        shim.dart_mlx_diagonal(input._handle, offset, axis1, axis2),
      ),
    );
  }

  static MlxArray kron(MlxArray a, MlxArray b) {
    _clearError();
    return MlxArray._(
      _checkHandle('dart_mlx_kron', shim.dart_mlx_kron(a._handle, b._handle)),
    );
  }

  static List<MlxArray> meshgrid(
    List<MlxArray> inputs, {
    bool sparse = false,
    String indexing = 'xy',
  }) {
    final outputsOut = calloc<ffi.Pointer<ffi.Pointer<ffi.Void>>>();
    final outputsLen = calloc<ffi.Size>();
    try {
      return _withCString(indexing, (indexingPtr) {
        return _withArrayHandles(inputs, (inputHandles, inputLen) {
          _clearError();
          _checkStatus(
            'dart_mlx_meshgrid',
            shim.dart_mlx_meshgrid(
              inputHandles.cast(),
              inputLen,
              sparse,
              indexingPtr,
              outputsOut,
              outputsLen,
            ),
          );
          return _readOutputArrayList(outputsOut.value, outputsLen.value);
        });
      });
    } finally {
      if (outputsOut.value != ffi.nullptr) {
        calloc.free(outputsOut.value);
      }
      calloc.free(outputsOut);
      calloc.free(outputsLen);
    }
  }

  static MlxArray partition(MlxArray input, int kth, {int? axis}) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_partition',
        shim.dart_mlx_partition(input._handle, kth, axis ?? 0, axis != null),
      ),
    );
  }

  static MlxArray scatter(
    MlxArray input,
    List<MlxArray> indices,
    MlxArray updates, {
    required List<int> axes,
  }) => _scatterMany(input, indices, updates, axes: axes, op: 0);

  static MlxArray scatterAdd(
    MlxArray input,
    List<MlxArray> indices,
    MlxArray updates, {
    required List<int> axes,
  }) => _scatterMany(input, indices, updates, axes: axes, op: 1);

  static MlxArray scatterMax(
    MlxArray input,
    List<MlxArray> indices,
    MlxArray updates, {
    required List<int> axes,
  }) => _scatterMany(input, indices, updates, axes: axes, op: 2);

  static MlxArray scatterMin(
    MlxArray input,
    List<MlxArray> indices,
    MlxArray updates, {
    required List<int> axes,
  }) => _scatterMany(input, indices, updates, axes: axes, op: 3);

  static MlxArray scatterProd(
    MlxArray input,
    List<MlxArray> indices,
    MlxArray updates, {
    required List<int> axes,
  }) => _scatterMany(input, indices, updates, axes: axes, op: 4);

  static MlxArray scatterSingle(
    MlxArray input,
    MlxArray indices,
    MlxArray updates, {
    required int axis,
  }) => _scatterSingle(input, indices, updates, axis: axis, op: 0);

  static MlxArray scatterAddSingle(
    MlxArray input,
    MlxArray indices,
    MlxArray updates, {
    required int axis,
  }) => _scatterSingle(input, indices, updates, axis: axis, op: 1);

  static MlxArray scatterMaxSingle(
    MlxArray input,
    MlxArray indices,
    MlxArray updates, {
    required int axis,
  }) => _scatterSingle(input, indices, updates, axis: axis, op: 2);

  static MlxArray scatterMinSingle(
    MlxArray input,
    MlxArray indices,
    MlxArray updates, {
    required int axis,
  }) => _scatterSingle(input, indices, updates, axis: axis, op: 3);

  static MlxArray scatterProdSingle(
    MlxArray input,
    MlxArray indices,
    MlxArray updates, {
    required int axis,
  }) => _scatterSingle(input, indices, updates, axis: axis, op: 4);

  static MlxArray _scatterMany(
    MlxArray input,
    List<MlxArray> indices,
    MlxArray updates, {
    required List<int> axes,
    required int op,
  }) {
    if (indices.isEmpty) {
      throw ArgumentError('scatter requires at least one index array.');
    }
    return _withArrayHandles(indices, (indexHandles, indexLen) {
      return _withInts(axes, (axesPtr, axesLen) {
        _clearError();
        return MlxArray._(
          _checkHandle(
            'dart_mlx_scatter',
            shim.dart_mlx_scatter(
              input._handle,
              indexHandles.cast(),
              indexLen,
              updates._handle,
              axesPtr,
              axesLen,
              op,
              -1,
            ),
          ),
        );
      });
    });
  }

  static MlxArray _scatterSingle(
    MlxArray input,
    MlxArray indices,
    MlxArray updates, {
    required int axis,
    required int op,
  }) {
    final single = calloc<ffi.Pointer<ffi.Void>>(1);
    try {
      single[0] = indices._handle;
      _clearError();
      return MlxArray._(
        _checkHandle(
          'dart_mlx_scatter_single',
          shim.dart_mlx_scatter(
            input._handle,
            single.cast(),
            1,
            updates._handle,
            ffi.nullptr,
            0,
            op,
            axis,
          ),
        ),
      );
    } finally {
      calloc.free(single);
    }
  }
}
