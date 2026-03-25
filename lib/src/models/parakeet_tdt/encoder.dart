import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

import 'bundle.dart';
import 'conf.dart';
import 'enc.dart';

typedef ParakeetEncoderTrace =
    void Function(int layerIndex, Duration elapsed, List<int> shape);

final class ParakeetTdtEncoder {
  ParakeetTdtEncoder(
    this.bundle, {
    this.maxLayers,
    this.onLayerComplete,
  })
    : _preEncoder = ParakeetTdtPreEncoder(bundle),
      _positionalEncoding = ParakeetTdtPositionalEncoding(
        dModel: bundle.manifest.encoderHidden,
        scaleInput: bundle.manifest.xScaling,
      ),
      _blocks = List<ParakeetTdtConformerBlock>.generate(
        bundle.manifest.encoderLayers,
        (int index) => ParakeetTdtConformerBlock.load(
          bundle.tensors,
          'encoder.layers.$index',
        ),
      );

  final ParakeetTdtBundle bundle;
  final int? maxLayers;
  final ParakeetEncoderTrace? onLayerComplete;
  final ParakeetTdtPreEncoder _preEncoder;
  final ParakeetTdtPositionalEncoding _positionalEncoding;
  final List<ParakeetTdtConformerBlock> _blocks;

  ({MlxArray features, List<int> lengths}) call(
    MlxArray mel,
    List<int> lengths,
  ) {
    final pre = _preEncoder(mel, lengths);
    final pos = _positionalEncoding(pre.features);
    var hidden = pos.scaledInput;
    final posEmb = pos.posEmb;
    try {
      if (!identical(hidden, pre.features)) {
        pre.features.close();
      }
      final limit = maxLayers == null
          ? _blocks.length
          : maxLayers!.clamp(0, _blocks.length);
      for (var index = 0; index < limit; index += 1) {
        final block = _blocks[index];
        final watch = Stopwatch()..start();
        final next = block(hidden, posEmb: posEmb);
        hidden.close();
        hidden = next;
        watch.stop();
        onLayerComplete?.call(index, watch.elapsed, hidden.shape);
      }
      MlxRuntime.evalAll(<MlxArray>[hidden]);
      return (features: hidden, lengths: pre.lengths);
    } finally {
      posEmb.close();
    }
  }
}
