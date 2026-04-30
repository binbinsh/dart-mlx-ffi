part of '../stable_api.dart';

/// Convolution helpers for 1D, 2D, 3D, and general convolution.
abstract final class MlxConv {
  static MlxArray conv1d(
    MlxArray input,
    MlxArray weight, {
    int stride = 1,
    int padding = 0,
    int dilation = 1,
    int groups = 1,
  }) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_conv1d',
        shim.dart_mlx_conv1d(
          input._handle,
          weight._handle,
          stride,
          padding,
          dilation,
          groups,
        ),
      ),
    );
  }

  static MlxArray conv2d(
    MlxArray input,
    MlxArray weight, {
    List<int> stride = const [1, 1],
    List<int> padding = const [0, 0],
    List<int> dilation = const [1, 1],
    int groups = 1,
  }) {
    _expectLen('conv2d.stride', stride, 2);
    _expectLen('conv2d.padding', padding, 2);
    _expectLen('conv2d.dilation', dilation, 2);
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_conv2d',
        shim.dart_mlx_conv2d(
          input._handle,
          weight._handle,
          stride[0],
          stride[1],
          padding[0],
          padding[1],
          dilation[0],
          dilation[1],
          groups,
        ),
      ),
    );
  }

  static MlxArray conv3d(
    MlxArray input,
    MlxArray weight, {
    List<int> stride = const [1, 1, 1],
    List<int> padding = const [0, 0, 0],
    List<int> dilation = const [1, 1, 1],
    int groups = 1,
  }) {
    _expectLen('conv3d.stride', stride, 3);
    _expectLen('conv3d.padding', padding, 3);
    _expectLen('conv3d.dilation', dilation, 3);
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_conv3d',
        shim.dart_mlx_conv3d(
          input._handle,
          weight._handle,
          stride[0],
          stride[1],
          stride[2],
          padding[0],
          padding[1],
          padding[2],
          dilation[0],
          dilation[1],
          dilation[2],
          groups,
        ),
      ),
    );
  }

  static MlxArray convGeneral(
    MlxArray input,
    MlxArray weight, {
    List<int> stride = const [],
    List<int>? padding,
    List<int>? paddingLo,
    List<int>? paddingHi,
    List<int> kernelDilation = const [],
    List<int> inputDilation = const [],
    int groups = 1,
    bool flip = false,
  }) {
    if (padding != null && (paddingLo != null || paddingHi != null)) {
      throw ArgumentError(
        'convGeneral() accepts either padding or paddingLo/paddingHi, not both.',
      );
    }
    final resolvedPaddingLo = paddingLo ?? padding ?? const <int>[];
    final resolvedPaddingHi = paddingHi ?? padding ?? const <int>[];
    return _withInts(stride, (stridePtr, strideLen) {
      return _withInts(resolvedPaddingLo, (paddingLoPtr, paddingLoLen) {
        return _withInts(resolvedPaddingHi, (paddingHiPtr, paddingHiLen) {
          return _withInts(kernelDilation, (kernelPtr, kernelLen) {
            return _withInts(inputDilation, (inputPtr, inputLen) {
              _clearError();
              return MlxArray._(
                _checkHandle(
                  'dart_mlx_conv_general',
                  shim.dart_mlx_conv_general(
                    input._handle,
                    weight._handle,
                    stridePtr,
                    strideLen,
                    paddingLoPtr,
                    paddingLoLen,
                    paddingHiPtr,
                    paddingHiLen,
                    kernelPtr,
                    kernelLen,
                    inputPtr,
                    inputLen,
                    groups,
                    flip,
                  ),
                ),
              );
            });
          });
        });
      });
    });
  }

  static MlxArray convTranspose1d(
    MlxArray input,
    MlxArray weight, {
    int stride = 1,
    int padding = 0,
    int dilation = 1,
    int outputPadding = 0,
    int groups = 1,
  }) {
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_conv_transpose1d',
        shim.dart_mlx_conv_transpose1d(
          input._handle,
          weight._handle,
          stride,
          padding,
          dilation,
          outputPadding,
          groups,
        ),
      ),
    );
  }

  static MlxArray convTranspose2d(
    MlxArray input,
    MlxArray weight, {
    List<int> stride = const [1, 1],
    List<int> padding = const [0, 0],
    List<int> dilation = const [1, 1],
    List<int> outputPadding = const [0, 0],
    int groups = 1,
  }) {
    _expectLen('convTranspose2d.stride', stride, 2);
    _expectLen('convTranspose2d.padding', padding, 2);
    _expectLen('convTranspose2d.dilation', dilation, 2);
    _expectLen('convTranspose2d.outputPadding', outputPadding, 2);
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_conv_transpose2d',
        shim.dart_mlx_conv_transpose2d(
          input._handle,
          weight._handle,
          stride[0],
          stride[1],
          padding[0],
          padding[1],
          dilation[0],
          dilation[1],
          outputPadding[0],
          outputPadding[1],
          groups,
        ),
      ),
    );
  }

  static MlxArray convTranspose3d(
    MlxArray input,
    MlxArray weight, {
    List<int> stride = const [1, 1, 1],
    List<int> padding = const [0, 0, 0],
    List<int> dilation = const [1, 1, 1],
    List<int> outputPadding = const [0, 0, 0],
    int groups = 1,
  }) {
    _expectLen('convTranspose3d.stride', stride, 3);
    _expectLen('convTranspose3d.padding', padding, 3);
    _expectLen('convTranspose3d.dilation', dilation, 3);
    _expectLen('convTranspose3d.outputPadding', outputPadding, 3);
    _clearError();
    return MlxArray._(
      _checkHandle(
        'dart_mlx_conv_transpose3d',
        shim.dart_mlx_conv_transpose3d(
          input._handle,
          weight._handle,
          stride[0],
          stride[1],
          stride[2],
          padding[0],
          padding[1],
          padding[2],
          dilation[0],
          dilation[1],
          dilation[2],
          outputPadding[0],
          outputPadding[1],
          outputPadding[2],
          groups,
        ),
      ),
    );
  }

  static void _expectLen(String label, List<int> values, int expected) {
    if (values.length != expected) {
      throw ArgumentError('$label must have length $expected.');
    }
  }
}
