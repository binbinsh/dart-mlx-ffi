library;

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

sealed class VlmLinearBase {
  const VlmLinearBase();

  MlxArray apply(MlxArray input, {required int groupSize, required int bits, required String mode});

  void close();

  static VlmLinearBase load(Map<String, MlxArray> tensors, String prefix) {
    final weight = tensors['$prefix.weight'];
    final scales = tensors['$prefix.scales'];
    final biases = tensors['$prefix.biases'];
    final bias = tensors['$prefix.bias'];
    if (weight == null) {
      throw StateError('Missing weight tensor for $prefix.');
    }
    if (scales != null) {
      return VlmQuantLinear(weight, scales, biases, bias);
    }
    return VlmDenseLinear(weight, bias);
  }
}

final class VlmDenseLinear extends VlmLinearBase {
  const VlmDenseLinear(this.weight, this.bias);

  final MlxArray weight;
  final MlxArray? bias;

  @override
  MlxArray apply(
    MlxArray input, {
    required int groupSize,
    required int bits,
    required String mode,
  }) {
    final y = mx.matmul(input, weight.transpose());
    if (bias == null) {
      return y;
    }
    final reshapedBias = bias!.reshape([1, bias!.shape[0]]);
    try {
      return mx.add(y, reshapedBias);
    } finally {
      reshapedBias.close();
      y.close();
    }
  }

  @override
  void close() {}
}

final class VlmQuantLinear extends VlmLinearBase {
  const VlmQuantLinear(this.weight, this.scales, this.biases, this.bias);

  final MlxArray weight;
  final MlxArray scales;
  final MlxArray? biases;
  final MlxArray? bias;

  MlxQuantizedMatrix get matrix => MlxQuantizedMatrix(weight, scales, biases);

  @override
  MlxArray apply(
    MlxArray input, {
    required int groupSize,
    required int bits,
    required String mode,
  }) {
    final y = mx.quant.matmul(
      input,
      matrix,
      transpose: true,
      groupSize: groupSize,
      bits: bits,
      mode: mode,
    );
    if (bias == null) {
      return y;
    }
    final reshapedBias = bias!.reshape([1, bias!.shape[0]]);
    try {
      return mx.add(y, reshapedBias);
    } finally {
      reshapedBias.close();
      y.close();
    }
  }

  @override
  void close() {}
}

sealed class VlmSwitchLinearBase {
  const VlmSwitchLinearBase();

  MlxArray applyExpert(
    MlxArray input,
    int expert, {
    required int groupSize,
    required int bits,
    required String mode,
  });

  static VlmSwitchLinearBase load(Map<String, MlxArray> tensors, String prefix) {
    final weight = tensors['$prefix.weight'];
    final scales = tensors['$prefix.scales'];
    final biases = tensors['$prefix.biases'];
    final bias = tensors['$prefix.bias'];
    if (weight == null) {
      throw StateError('Missing switch weight tensor for $prefix.');
    }
    if (scales != null) {
      return VlmQuantSwitchLinear(weight, scales, biases, bias);
    }
    return VlmDenseSwitchLinear(weight, bias);
  }
}

final class VlmDenseSwitchLinear extends VlmSwitchLinearBase {
  const VlmDenseSwitchLinear(this.weight, this.bias);

  final MlxArray weight;
  final MlxArray? bias;

  @override
  MlxArray applyExpert(
    MlxArray input,
    int expert, {
    required int groupSize,
    required int bits,
    required String mode,
  }) {
    final index = MlxArray.fromInt32List([expert], shape: [1]);
    try {
      final selected = weight.take(index, axis: 0).reshape([
        weight.shape[1],
        weight.shape[2],
      ]);
      try {
        final y = mx.matmul(input, selected.transpose());
        if (bias == null) {
          return y;
        }
        final selectedBias = bias!.take(index, axis: 0).reshape([
          1,
          bias!.shape[1],
        ]);
        try {
          return mx.add(y, selectedBias);
        } finally {
          selectedBias.close();
          y.close();
        }
      } finally {
        selected.close();
      }
    } finally {
      index.close();
    }
  }
}

final class VlmQuantSwitchLinear extends VlmSwitchLinearBase {
  const VlmQuantSwitchLinear(this.weight, this.scales, this.biases, this.bias);

  final MlxArray weight;
  final MlxArray scales;
  final MlxArray? biases;
  final MlxArray? bias;

  @override
  MlxArray applyExpert(
    MlxArray input,
    int expert, {
    required int groupSize,
    required int bits,
    required String mode,
  }) {
    final index = MlxArray.fromInt32List([expert], shape: [1]);
    try {
      final selectedWeight = weight.take(index, axis: 0).reshape([
        weight.shape[1],
        weight.shape[2],
        weight.shape[3],
      ]);
      final selectedScales = scales.take(index, axis: 0).reshape([
        scales.shape[1],
        scales.shape[2],
        scales.shape[3],
      ]);
      final selectedBiases = biases?.take(index, axis: 0).reshape([
        if (biases != null) biases!.shape[1],
        if (biases != null) biases!.shape[2],
        if (biases != null) biases!.shape[3],
      ]);
      final matrix = MlxQuantizedMatrix(
        selectedWeight,
        selectedScales,
        selectedBiases,
      );
      try {
        final y = mx.quant.matmul(
          input,
          matrix,
          transpose: true,
          groupSize: groupSize,
          bits: bits,
          mode: mode,
        );
        if (bias == null) {
          return y;
        }
        final selectedBias = bias!.take(index, axis: 0).reshape([
          1,
          bias!.shape[1],
        ]);
        try {
          return mx.add(y, selectedBias);
        } finally {
          selectedBias.close();
          y.close();
        }
      } finally {
        selectedBiases?.close();
        selectedScales.close();
        selectedWeight.close();
      }
    } finally {
      index.close();
    }
  }
}
