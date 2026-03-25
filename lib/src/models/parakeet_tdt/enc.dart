import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

import 'bundle.dart';
import 'nn.dart';

final class ParakeetTdtPreEncoder {
  ParakeetTdtPreEncoder(this.bundle)
    : _conv0 = _Conv2dBias.load(bundle.tensors, 'encoder.pre_encode.conv.0'),
      _conv2 = _DepthwiseConv2dBias.load(
        bundle.tensors,
        'encoder.pre_encode.conv.2',
      ),
      _conv3 = _Conv2dBias.load(bundle.tensors, 'encoder.pre_encode.conv.3'),
      _conv5 = _DepthwiseConv2dBias.load(
        bundle.tensors,
        'encoder.pre_encode.conv.5',
      ),
      _conv6 = _Conv2dBias.load(bundle.tensors, 'encoder.pre_encode.conv.6'),
      _out = ParakeetDenseLinear.load(bundle.tensors, 'encoder.pre_encode.out');

  final ParakeetTdtBundle bundle;
  final _Conv2dBias _conv0;
  final _DepthwiseConv2dBias _conv2;
  final _Conv2dBias _conv3;
  final _DepthwiseConv2dBias _conv5;
  final _Conv2dBias _conv6;
  final ParakeetDenseLinear _out;

  ({MlxArray features, List<int> lengths}) call(
    MlxArray mel,
    List<int> lengths,
  ) {
    var nextLengths = List<int>.from(lengths);
    final input = mel.expandDims(1); // [B, 1, T, F]
    final layoutInput = input.transposeAxes(<int>[0, 2, 3, 1]); // [B, T, F, C]
    input.close();

    final conv0 = _conv0(
      layoutInput,
      stride: <int>[2, 2],
      padding: <int>[1, 1],
    );
    layoutInput.close();
    final act0 = parakeetRelu(conv0);
    conv0.close();
    nextLengths = nextLengths.map((len) => _convLen(len, 3, 2, 1)).toList();

    final conv2 = _conv2(act0, stride: <int>[2, 2], padding: <int>[1, 1]);
    act0.close();
    final conv3 = _conv3(conv2, stride: <int>[1, 1], padding: <int>[0, 0]);
    conv2.close();
    final act1 = parakeetRelu(conv3);
    conv3.close();
    nextLengths = nextLengths.map((len) => _convLen(len, 3, 2, 1)).toList();

    final conv5 = _conv5(act1, stride: <int>[2, 2], padding: <int>[1, 1]);
    act1.close();
    final conv6 = _conv6(conv5, stride: <int>[1, 1], padding: <int>[0, 0]);
    conv5.close();
    final act2 = parakeetRelu(conv6);
    conv6.close();
    nextLengths = nextLengths.map((len) => _convLen(len, 3, 2, 1)).toList();

    final batch = act2.shape[0];
    final time = act2.shape[1];
    final freq = act2.shape[2];
    final channels = act2.shape[3];
    final flattened = act2.transposeAxes(<int>[0, 1, 3, 2]).reshape(<int>[
      batch,
      time,
      channels * freq,
    ]);
    act2.close();
    final projected = _out(
      flattened.reshape(<int>[batch * time, channels * freq]),
    );
    flattened.close();
    final features = projected.reshape(<int>[
      batch,
      time,
      bundle.manifest.encoderHidden,
    ]);
    projected.close();
    return (features: features, lengths: nextLengths);
  }

  Map<String, MlxArray> debugStages(MlxArray mel) {
    final input = mel.expandDims(1);
    final layoutInput = input.transposeAxes(<int>[0, 2, 3, 1]);
    input.close();
    final conv0 = _conv0(
      layoutInput,
      stride: <int>[2, 2],
      padding: <int>[1, 1],
    );
    layoutInput.close();
    final act0 = parakeetRelu(conv0);
    final conv2 = _conv2(act0, stride: <int>[2, 2], padding: <int>[1, 1]);
    final conv3 = _conv3(conv2, stride: <int>[1, 1], padding: <int>[0, 0]);
    final act1 = parakeetRelu(conv3);
    final conv5 = _conv5(act1, stride: <int>[2, 2], padding: <int>[1, 1]);
    final conv6 = _conv6(conv5, stride: <int>[1, 1], padding: <int>[0, 0]);
    final act2 = parakeetRelu(conv6);
    return <String, MlxArray>{
      'conv0': conv0,
      'act0': act0,
      'conv2': conv2,
      'conv3': conv3,
      'act1': act1,
      'conv5': conv5,
      'conv6': conv6,
      'act2': act2,
    };
  }

  int _convLen(int length, int kernel, int stride, int padding) {
    return ((length + 2 * padding - kernel) ~/ stride) + 1;
  }
}

final class _Conv2dBias {
  const _Conv2dBias(this.weight, this.bias);

  final MlxArray weight;
  final MlxArray? bias;

  factory _Conv2dBias.load(Map<String, MlxArray> tensors, String prefix) {
    return _Conv2dBias(
      requireParakeetTensor(tensors, '$prefix.weight'),
      tensors['$prefix.bias'],
    );
  }

  MlxArray call(
    MlxArray input, {
    required List<int> stride,
    required List<int> padding,
  }) {
    final out = mx.conv2d(input, weight, stride: stride, padding: padding);
    if (bias == null) {
      return out;
    }
    final reshapedBias = bias!.reshape(<int>[1, 1, 1, bias!.shape[0]]);
    try {
      return mx.add(out, reshapedBias);
    } finally {
      reshapedBias.close();
      out.close();
    }
  }
}

final class _DepthwiseConv2dBias extends _Conv2dBias {
  const _DepthwiseConv2dBias(super.weight, super.bias);

  factory _DepthwiseConv2dBias.load(
    Map<String, MlxArray> tensors,
    String prefix,
  ) {
    return _DepthwiseConv2dBias(
      requireParakeetTensor(tensors, '$prefix.weight'),
      tensors['$prefix.bias'],
    );
  }

  @override
  MlxArray call(
    MlxArray input, {
    required List<int> stride,
    required List<int> padding,
  }) {
    final out = mx.conv2d(
      input,
      weight,
      stride: stride,
      padding: padding,
      groups: input.shape[3],
    );
    if (bias == null) {
      return out;
    }
    final reshapedBias = bias!.reshape(<int>[1, 1, 1, bias!.shape[0]]);
    try {
      return mx.add(out, reshapedBias);
    } finally {
      reshapedBias.close();
      out.close();
    }
  }
}
