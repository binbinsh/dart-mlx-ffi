library;

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

import 'args.dart';
import 'core.dart';
import 'lstm.dart';

final class TextEncoder {
  TextEncoder._({
    required this.embedding,
    required List<_TextConvBlock> blocks,
    required this.lstm,
  }) : _blocks = blocks;

  factory TextEncoder.load(
    Map<String, MlxArray> tensors, {
    required String prefix,
    required ModelConfig config,
    bool activationQuant = false,
  }) {
    final blocks = List<_TextConvBlock>.generate(
      config.nLayer,
      (index) => _TextConvBlock(
        conv: ConvWeighted.load(
          tensors,
          '$prefix.cnn.$index.0',
          padding: (config.textEncoderKernelSize - 1) ~/ 2,
          activationQuant: activationQuant,
        ),
        norm: LayerNorm.load(tensors, '$prefix.cnn.$index.1'),
      ),
      growable: false,
    );
    return TextEncoder._(
      embedding: Embedding.load(
        tensors,
        '$prefix.embedding',
        quant: config.quantization,
      ),
      blocks: blocks,
      lstm: LSTM.load(
        tensors,
        '$prefix.lstm',
        inputSize: config.hiddenDim,
        hiddenSize: config.hiddenDim ~/ 2,
        activationQuant: activationQuant,
      ),
    );
  }

  final Embedding embedding;
  final List<_TextConvBlock> _blocks;
  final LSTM lstm;

  MlxArray call(MlxArray inputIds, MlxArray textMask) {
    final maskB1T = textMask.expandDims(1);
    final embedded = embedding(inputIds);
    var current = mx.transposeAxes(embedded, [0, 2, 1]);
    embedded.close();
    final masked = maskFillZero(current, maskB1T);
    current.close();
    current = masked;
    for (final block in _blocks) {
      final convIn = mx.transposeAxes(current, [0, 2, 1]);
      final convOut = block.conv(convIn);
      convIn.close();
      final afterConv = mx.transposeAxes(convOut, [0, 2, 1]);
      convOut.close();
      current.close();
      current = maskFillZero(afterConv, maskB1T);
      afterConv.close();

      final normIn = mx.transposeAxes(current, [0, 2, 1]);
      final normOut = block.norm(normIn);
      normIn.close();
      final afterNorm = mx.transposeAxes(normOut, [0, 2, 1]);
      normOut.close();
      current.close();
      current = maskFillZero(afterNorm, maskB1T);
      afterNorm.close();

      final activated = leakyRelu(current);
      current.close();
      current = maskFillZero(activated, maskB1T);
      activated.close();
    }
    final lstmIn = mx.transposeAxes(current, [0, 2, 1]);
    current.close();
    final lstmOut = lstm(lstmIn).output;
    lstmIn.close();
    final transposed = mx.transposeAxes(lstmOut, [0, 2, 1]);
    lstmOut.close();
    final output = maskFillZero(transposed, maskB1T);
    transposed.close();
    maskB1T.close();
    return output;
  }
}

final class _TextConvBlock {
  const _TextConvBlock({required this.conv, required this.norm});

  final ConvWeighted conv;
  final LayerNorm norm;
}
