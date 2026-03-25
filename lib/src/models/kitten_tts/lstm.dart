library;

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

import 'quant.dart';

final class LstmState {
  const LstmState(this.hidden, this.cell);

  final MlxArray hidden;
  final MlxArray cell;
}

final class LstmResult {
  const LstmResult({
    required this.output,
    required this.forward,
    required this.backward,
  });

  final MlxArray output;
  final LstmState forward;
  final LstmState backward;
}

final class LSTM {
  const LSTM({
    required this.inputSize,
    required this.hiddenSize,
    required this.wxForward,
    required this.whForward,
    required this.biasIhForward,
    required this.biasHhForward,
    required this.wxBackward,
    required this.whBackward,
    required this.biasIhBackward,
    required this.biasHhBackward,
    this.activationQuant = false,
  });

  factory LSTM.load(
    Map<String, MlxArray> tensors,
    String prefix, {
    required int inputSize,
    required int hiddenSize,
    bool activationQuant = false,
  }) {
    return LSTM(
      inputSize: inputSize,
      hiddenSize: hiddenSize,
      wxForward: tensors['$prefix.Wx_forward']!,
      whForward: tensors['$prefix.Wh_forward']!,
      biasIhForward: tensors['$prefix.bias_ih_forward'],
      biasHhForward: tensors['$prefix.bias_hh_forward'],
      wxBackward: tensors['$prefix.Wx_backward']!,
      whBackward: tensors['$prefix.Wh_backward']!,
      biasIhBackward: tensors['$prefix.bias_ih_backward'],
      biasHhBackward: tensors['$prefix.bias_hh_backward'],
      activationQuant: activationQuant,
    );
  }

  final int inputSize;
  final int hiddenSize;
  final MlxArray wxForward;
  final MlxArray whForward;
  final MlxArray? biasIhForward;
  final MlxArray? biasHhForward;
  final MlxArray wxBackward;
  final MlxArray whBackward;
  final MlxArray? biasIhBackward;
  final MlxArray? biasHhBackward;
  final bool activationQuant;

  LstmResult call(
    MlxArray input, {
    MlxArray? hiddenForward,
    MlxArray? cellForward,
    MlxArray? hiddenBackward,
    MlxArray? cellBackward,
  }) {
    final x = input.ndim == 2 ? input.expandDims(0) : input;
    final forward = _runDirection(
      x,
      forward: true,
      hidden: hiddenForward,
      cell: cellForward,
    );
    final backward = _runDirection(
      x,
      forward: false,
      hidden: hiddenBackward,
      cell: cellBackward,
    );
    final output = mx.concatenate([forward.output, backward.output], axis: -1);
    if (!identical(x, input)) {
      x.close();
    }
    return LstmResult(
      output: output,
      forward: LstmState(forward.finalHidden, forward.finalCell),
      backward: LstmState(backward.finalHidden, backward.finalCell),
    );
  }

  _DirectionResult _runDirection(
    MlxArray input, {
    required bool forward,
    MlxArray? hidden,
    MlxArray? cell,
  }) {
    final wx = forward ? wxForward : wxBackward;
    final wh = forward ? whForward : whBackward;
    final biasIh = forward ? biasIhForward : biasIhBackward;
    final biasHh = forward ? biasHhForward : biasHhBackward;
    final xIn = maybeFakeQuant(input, activationQuant);
    final xProj = _projectInput(xIn, wx, biasIh, biasHh);
    if (!identical(xIn, input)) {
      xIn.close();
    }
    final batch = input.shape[0];
    final seqLen = input.shape[1];
    var hiddenState = hidden ?? MlxArray.zeros([batch, hiddenSize]);
    var cellState = cell ?? MlxArray.zeros([batch, hiddenSize]);
    final outputs = <MlxArray>[];
    final order = forward
        ? List<int>.generate(seqLen, (index) => index)
        : List<int>.generate(seqLen, (index) => seqLen - index - 1);
    for (final step in order) {
      final prevHidden = hiddenState;
      final prevCell = cellState;
      final proj = xProj
          .slice(start: [0, step, 0], stop: [batch, step + 1, 4 * hiddenSize])
          .reshape([batch, 4 * hiddenSize]);
      final hiddenIn = maybeFakeQuant(prevHidden, activationQuant);
      final hiddenProj = mx.matmul(hiddenIn, wh.transpose());
      if (!identical(hiddenIn, prevHidden)) {
        hiddenIn.close();
      }
      final gates = proj + hiddenProj;
      final parts = mx.splitSections(gates, [
        hiddenSize,
        hiddenSize * 2,
        hiddenSize * 3,
      ], axis: 1);
      final i = parts[0].sigmoid();
      final f = parts[1].sigmoid();
      final g = parts[2].tanh();
      final o = parts[3].sigmoid();
      final nextCell = (f * cellState) + (i * g);
      final nextHidden = o * nextCell.tanh();
      outputs.add(nextHidden);
      cellState = nextCell;
      hiddenState = nextHidden;
      if (!identical(prevCell, cell)) {
        prevCell.close();
      }
      proj.close();
      hiddenProj.close();
      gates.close();
      for (final part in parts) {
        part.close();
      }
      i.close();
      f.close();
      g.close();
      o.close();
    }
    if (!forward) {
      outputs.replaceRange(0, outputs.length, outputs.reversed);
    }
    final stacked = mx.stack(outputs, axis: 1);
    for (final output in outputs) {
      if (!identical(output, hiddenState)) {
        output.close();
      }
    }
    xProj.close();
    return _DirectionResult(
      output: stacked,
      finalHidden: hiddenState,
      finalCell: cellState,
    );
  }

  MlxArray _projectInput(
    MlxArray input,
    MlxArray weight,
    MlxArray? biasIh,
    MlxArray? biasHh,
  ) {
    if (biasIh == null || biasHh == null) {
      return mx.matmul(input, weight.transpose());
    }
    final bias = biasIh + biasHh;
    try {
      return mx.addmm(bias, input, weight.transpose());
    } finally {
      bias.close();
    }
  }
}

final class _DirectionResult {
  const _DirectionResult({
    required this.output,
    required this.finalHidden,
    required this.finalCell,
  });

  final MlxArray output;
  final MlxArray finalHidden;
  final MlxArray finalCell;
}
