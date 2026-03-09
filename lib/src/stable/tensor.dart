part of '../stable_api.dart';

/// Tensor indexing and algebra helpers beyond basic elementwise ops.
abstract final class MlxTensor {
  static MlxArray flatten(
    MlxArray input, {
    int startAxis = 0,
    int endAxis = -1,
  }) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_flatten',
        shim.dart_mlx_flatten(input._handle, startAxis, endAxis),
      ),
    );
  }

  static MlxArray moveaxis(MlxArray input, int source, int destination) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_moveaxis',
        shim.dart_mlx_moveaxis(input._handle, source, destination),
      ),
    );
  }

  static MlxArray swapaxes(MlxArray input, int axis1, int axis2) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_swapaxes',
        shim.dart_mlx_swapaxes(input._handle, axis1, axis2),
      ),
    );
  }

  static MlxArray transposeAxes(MlxArray input, List<int> axes) =>
      _withInts(axes, (axesPtr, axesLen) {
        _clearError();
        return MlxArray._(
          _checkHandle(
            'dart_mlx_transpose_axes',
            shim.dart_mlx_transpose_axes(input._handle, axesPtr, axesLen),
          ),
        );
      });

  static MlxArray tile(MlxArray input, List<int> reps) =>
      _withInts(reps, (repsPtr, repsLen) {
        _clearError();
        return MlxArray._(
          _checkHandle(
            'dart_mlx_tile',
            shim.dart_mlx_tile(input._handle, repsPtr, repsLen),
          ),
        );
      });

  static MlxArray pad(
    MlxArray input, {
    List<int>? axes,
    required List<int> lowPads,
    required List<int> highPads,
    MlxArray? padValue,
    String mode = 'constant',
  }) {
    if (lowPads.length != highPads.length) {
      throw ArgumentError('pad() requires lowPads and highPads to share a length.');
    }
    final resolvedAxes = axes ?? List<int>.generate(lowPads.length, (i) => i);
    if (resolvedAxes.length != lowPads.length) {
      throw ArgumentError('pad() requires axes, lowPads, and highPads to share a length.');
    }
    return _withInts(resolvedAxes, (axesPtr, axesLen) {
      return _withInts(lowPads, (lowPtr, lowLen) {
        return _withInts(highPads, (highPtr, highLen) {
          return _withCString(mode, (modePtr) {
            _clearError();
            return MlxArray._(
              _checkHandle(
                'dart_mlx_pad',
                shim.dart_mlx_pad(
                  input._handle,
                  axesPtr,
                  axesLen,
                  lowPtr,
                  lowLen,
                  highPtr,
                  highLen,
                  padValue?._handle ?? ffi.nullptr,
                  modePtr,
                ),
              ),
            );
          });
        });
      });
    });
  }

  static MlxArray padSymmetric(
    MlxArray input,
    int padWidth, {
    MlxArray? padValue,
    String mode = 'constant',
  }) => _withCString(mode, (modePtr) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_pad_symmetric',
        shim.dart_mlx_pad_symmetric(
          input._handle,
          padWidth,
          padValue?._handle ?? ffi.nullptr,
          modePtr,
        ),
      ),
    );
  });

  static MlxArray unflatten(
    MlxArray input, {
    required int axis,
    required List<int> shape,
  }) => _withInts(shape, (shapePtr, shapeLen) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_unflatten',
        shim.dart_mlx_unflatten(input._handle, axis, shapePtr, shapeLen),
      ),
    );
  });

  static MlxArray take(MlxArray input, MlxArray indices, {int? axis}) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_take',
        shim.dart_mlx_take(input._handle, indices._handle, axis ?? 0, axis != null),
      ),
    );
  }

  static MlxArray takeAlongAxis(MlxArray input, MlxArray indices, {required int axis}) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_take_along_axis',
        shim.dart_mlx_take_along_axis(input._handle, indices._handle, axis),
      ),
    );
  }

  static MlxArray gather(
    MlxArray input,
    List<MlxArray> indices, {
    required List<int> axes,
    required List<int> sliceSizes,
  }) {
    if (indices.isEmpty) {
      throw ArgumentError('gather() requires at least one index array.');
    }
    return _withArrayHandles(indices, (indexHandles, indexLen) {
      return _withInts(axes, (axesPtr, axesLen) {
        return _withInts(sliceSizes, (slicePtr, sliceLen) {
          _clearError();
          return MlxArray._(
            _checkHandle(
              'dart_mlx_gather',
              shim.dart_mlx_gather(
                input._handle,
                indexHandles.cast(),
                indexLen,
                axesPtr,
                axesLen,
                slicePtr,
                sliceLen,
              ),
            ),
          );
        });
      });
    });
  }

  static MlxArray gatherSingle(
    MlxArray input,
    MlxArray indices, {
    required int axis,
    required List<int> sliceSizes,
  }) => _withInts(sliceSizes, (slicePtr, sliceLen) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_gather_single',
        shim.dart_mlx_gather_single(
          input._handle,
          indices._handle,
          axis,
          slicePtr,
          sliceLen,
        ),
      ),
    );
  });

  static MlxArray gatherMm(
    MlxArray a,
    MlxArray b, {
    MlxArray? lhsIndices,
    MlxArray? rhsIndices,
    bool sortedIndices = false,
  }) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_gather_mm',
        shim.dart_mlx_gather_mm(
          a._handle,
          b._handle,
          lhsIndices?._handle ?? ffi.nullptr,
          rhsIndices?._handle ?? ffi.nullptr,
          sortedIndices,
        ),
      ),
    );
  }

  static List<MlxArray> broadcastArrays(List<MlxArray> inputs) {
    if (inputs.isEmpty) {
      return const [];
    }
    final outputsOut = calloc<ffi.Pointer<ffi.Pointer<ffi.Void>>>();
    final outputsLen = calloc<ffi.Size>();
    try {
      _withArrayHandles(inputs, (inputHandles, inputLen) {
        _clearError();
        _checkStatus(
          'dart_mlx_broadcast_arrays',
          shim.dart_mlx_broadcast_arrays(
            inputHandles.cast(),
            inputLen,
            outputsOut,
            outputsLen,
          ),
        );
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

  static List<MlxArray> splitSections(
    MlxArray input,
    List<int> indices, {
    int axis = 0,
  }) {
    final outputsOut = calloc<ffi.Pointer<ffi.Pointer<ffi.Void>>>();
    final outputsLen = calloc<ffi.Size>();
    try {
      _withInts(indices, (indicesPtr, indicesLen) {
        _clearError();
        _checkStatus(
          'dart_mlx_split_sections',
          shim.dart_mlx_split_sections(
            input._handle,
            indicesPtr,
            indicesLen,
            axis,
            outputsOut,
            outputsLen,
          ),
        );
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

  static MlxArray segmentedMm(MlxArray a, MlxArray b, MlxArray segments) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_segmented_mm',
        shim.dart_mlx_segmented_mm(a._handle, b._handle, segments._handle),
      ),
    );
  }

  static MlxArray blockMaskedMm(
    MlxArray a,
    MlxArray b, {
    required int blockSize,
    MlxArray? maskOut,
    MlxArray? maskLhs,
    MlxArray? maskRhs,
  }) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_block_masked_mm',
        shim.dart_mlx_block_masked_mm(
          a._handle,
          b._handle,
          blockSize,
          maskOut?._handle ?? ffi.nullptr,
          maskLhs?._handle ?? ffi.nullptr,
          maskRhs?._handle ?? ffi.nullptr,
        ),
      ),
    );
  }

  static MlxArray slice(
    MlxArray input, {
    required List<int> start,
    required List<int> stop,
    List<int>? strides,
  }) {
    final resolvedStrides = strides ?? List<int>.filled(start.length, 1);
    if (start.length != stop.length || start.length != resolvedStrides.length) {
      throw ArgumentError('slice() requires start, stop, and strides to share a length.');
    }
    return _withInts(start, (startPtr, startLen) {
      return _withInts(stop, (stopPtr, stopLen) {
        return _withInts(resolvedStrides, (stridePtr, strideLen) {
          _clearError();
          return MlxArray._(
            _checkHandle(
              'dart_mlx_slice',
              shim.dart_mlx_slice(
                input._handle,
                startPtr,
                startLen,
                stopPtr,
                stopLen,
                stridePtr,
                strideLen,
              ),
            ),
          );
        });
      });
    });
  }

  static MlxArray sliceDynamic(
    MlxArray input, {
    required MlxArray start,
    required List<int> axes,
    required List<int> sliceSize,
  }) => _withInts(axes, (axesPtr, axesLen) {
    return _withInts(sliceSize, (slicePtr, sliceLen) {
      _clearError();
      return MlxArray._(
        _checkHandle(
          'dart_mlx_slice_dynamic',
          shim.dart_mlx_slice_dynamic(
            input._handle,
            start._handle,
            axesPtr,
            axesLen,
            slicePtr,
            sliceLen,
          ),
        ),
      );
    });
  });

  static MlxArray sliceUpdate(
    MlxArray source,
    MlxArray update, {
    required List<int> start,
    required List<int> stop,
    List<int>? strides,
  }) {
    final resolvedStrides = strides ?? List<int>.filled(start.length, 1);
    if (start.length != stop.length || start.length != resolvedStrides.length) {
      throw ArgumentError(
        'sliceUpdate() requires start, stop, and strides to share a length.',
      );
    }
    return _withInts(start, (startPtr, startLen) {
      return _withInts(stop, (stopPtr, stopLen) {
        return _withInts(resolvedStrides, (stridePtr, strideLen) {
          _clearError();
          return MlxArray._(
            _checkHandle(
              'dart_mlx_slice_update',
              shim.dart_mlx_slice_update(
                source._handle,
                update._handle,
                startPtr,
                startLen,
                stopPtr,
                stopLen,
                stridePtr,
                strideLen,
              ),
            ),
          );
        });
      });
    });
  }

  static MlxArray sliceUpdateDynamic(
    MlxArray source,
    MlxArray update, {
    required MlxArray start,
    required List<int> axes,
  }) => _withInts(axes, (axesPtr, axesLen) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_slice_update_dynamic',
        shim.dart_mlx_slice_update_dynamic(
          source._handle,
          update._handle,
          start._handle,
          axesPtr,
          axesLen,
        ),
      ),
    );
  });

  static MlxArray einsum(String subscripts, List<MlxArray> operands) {
    if (operands.isEmpty) {
      throw ArgumentError('einsum() requires at least one operand.');
    }
    return _withCString(subscripts, (subscriptsPtr) {
      return _withArrayHandles(operands, (operandHandles, operandLen) {
        _clearError();
        return MlxArray._(
          _checkHandle(
            'dart_mlx_einsum',
            shim.dart_mlx_einsum(subscriptsPtr, operandHandles.cast(), operandLen),
          ),
        );
      });
    });
  }

  static MlxArray tensordot(
    MlxArray a,
    MlxArray b, {
    int? axis,
    List<int>? axesA,
    List<int>? axesB,
  }) {
    if (axis != null && (axesA != null || axesB != null)) {
      throw ArgumentError('tensordot() accepts either axis or axesA/axesB, not both.');
    }
    if (axesA != null || axesB != null) {
      if (axesA == null || axesB == null) {
        throw ArgumentError('tensordot() requires both axesA and axesB together.');
      }
      return _withInts(axesA, (axesAPtr, axesALen) {
        return _withInts(axesB, (axesBPtr, axesBLen) {
          _clearError();
          return MlxArray._(
            _checkHandle(
              'dart_mlx_tensordot',
              shim.dart_mlx_tensordot(
                a._handle,
                b._handle,
                axesAPtr,
                axesALen,
                axesBPtr,
                axesBLen,
              ),
            ),
          );
        });
      });
    }
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_tensordot_axis',
        shim.dart_mlx_tensordot_axis(a._handle, b._handle, axis ?? 2),
      ),
    );
  }
}
