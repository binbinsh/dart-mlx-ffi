import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

import 'bundle.dart';
import 'dec.dart';
import 'encoder.dart';
import 'nn.dart';

class ParakeetTdtPredictState {
  ParakeetTdtPredictState({
    required this.hiddenLayers,
    required this.cellLayers,
    this.ownsBuffers = true,
  });

  final List<MlxArray> hiddenLayers;
  final List<MlxArray> cellLayers;
  final bool ownsBuffers;

  MlxArray get hidden => mx.stack(hiddenLayers, axis: 0);
  MlxArray get cell => mx.stack(cellLayers, axis: 0);

  void close() {
    if (!ownsBuffers) {
      return;
    }
    for (final hidden in hiddenLayers) {
      hidden.close();
    }
    for (final cell in cellLayers) {
      cell.close();
    }
  }
}

class ParakeetTdtPredictResult {
  ParakeetTdtPredictResult({required this.output, required this.state});

  final MlxArray output;
  final ParakeetTdtPredictState state;
}

typedef ParakeetRuntimeTrace = void Function(String stage, String message);

class ParakeetTdtRuntime {
  ParakeetTdtRuntime(
    this.bundle, {
    this.maxEncoderLayers,
    this.onTrace,
  })
    : _embedding = ParakeetEmbedding.load(
        bundle.tensors,
        'decoder.prediction.embed',
      ),
      _jointEnc = ParakeetDenseLinear.load(bundle.tensors, 'joint.enc'),
      _jointPred = ParakeetDenseLinear.load(bundle.tensors, 'joint.pred'),
      _jointOut = ParakeetDenseLinear.load(bundle.tensors, 'joint.joint_net.2'),
      _predictLayers = List<ParakeetLstmCell>.generate(
        bundle.manifest.predictLayers,
        (int layer) => ParakeetLstmCell.load(
          bundle.tensors,
          'decoder.prediction.dec_rnn.lstm.$layer',
          hiddenSize: bundle.manifest.predictStateSize,
        ),
      );

  final ParakeetTdtBundle bundle;
  final int? maxEncoderLayers;
  final ParakeetRuntimeTrace? onTrace;
  final ParakeetEmbedding _embedding;
  final List<ParakeetLstmCell> _predictLayers;
  final ParakeetDenseLinear _jointEnc;
  final ParakeetDenseLinear _jointPred;
  final ParakeetDenseLinear _jointOut;
  late final MlxArray _zeroPredictInput = mx.zeros(<int>[
    1,
    1,
    bundle.manifest.predictHidden,
  ]);
  late final List<MlxArray> _zeroHiddenLayers = List<MlxArray>.generate(
    bundle.manifest.predictLayers,
    (_) => mx.zeros(<int>[1, bundle.manifest.predictStateSize]),
  );
  late final List<MlxArray> _zeroCellLayers = List<MlxArray>.generate(
    bundle.manifest.predictLayers,
    (_) => mx.zeros(<int>[1, bundle.manifest.predictStateSize]),
  );
  final Map<int, MlxArray> _predictTokenIds = <int, MlxArray>{};
  final Map<int, MlxArray> _predictEmbeddings = <int, MlxArray>{};
  late final ParakeetTdtEncoder _encoder = ParakeetTdtEncoder(
    bundle,
    maxLayers: maxEncoderLayers,
    onLayerComplete: (int layerIndex, Duration elapsed, List<int> shape) {
      onTrace?.call(
        'encoder',
        'layer=$layerIndex elapsedMs=${elapsed.inMilliseconds} shape=$shape',
      );
    },
  );
  late final ParakeetTdtGreedyDecoder _decoder = ParakeetTdtGreedyDecoder(
    bundle.manifest,
  );

  ParakeetTdtPredictState zeroPredictState() {
    return ParakeetTdtPredictState(
      hiddenLayers: _zeroHiddenLayers,
      cellLayers: _zeroCellLayers,
      ownsBuffers: false,
    );
  }

  ParakeetTdtPredictResult predictorStep({
    required int? tokenId,
    required ParakeetTdtPredictState state,
  }) {
    final sequenceInput = _predictInputFor(tokenId);

    final nextHidden = <MlxArray>[];
    final nextCell = <MlxArray>[];
    var layerInput = sequenceInput.reshape(<int>[
      1,
      bundle.manifest.predictHidden,
    ]);
    try {
      for (var layer = 0; layer < bundle.manifest.predictLayers; layer += 1) {
        final step = _predictLayers[layer](
          input: layerInput,
          hidden: state.hiddenLayers[layer],
          cell: state.cellLayers[layer],
        );
        nextHidden.add(step.hidden);
        nextCell.add(step.cell);
        layerInput.close();
        layerInput = layer == bundle.manifest.predictLayers - 1
            ? step.hidden
            : step.hidden.reshape(<int>[1, bundle.manifest.predictHidden]);
      }

      final output = layerInput.reshape(<int>[
        1,
        1,
        bundle.manifest.predictHidden,
      ]);
      return ParakeetTdtPredictResult(
        output: output,
        state: ParakeetTdtPredictState(
          hiddenLayers: nextHidden,
          cellLayers: nextCell,
        ),
      );
    } finally {
    }
  }

  ({MlxArray tokenLogits, MlxArray durationLogits}) jointStep({
    required MlxArray encoderFrame,
    required MlxArray predictorOutput,
  }) {
    final encProj = _jointEnc(
      encoderFrame.reshape(<int>[1, bundle.manifest.encoderHidden]),
    );
    final predProj = _jointPred(
      predictorOutput.reshape(<int>[1, bundle.manifest.predictHidden]),
    );
    final summed = mx.add(encProj, predProj);
    final activated = parakeetRelu(summed);
    final logits = _jointOut(activated);
    encProj.close();
    predProj.close();
    summed.close();
    activated.close();

    final tokenLogits = logits.slice(
      start: <int>[0, 0],
      stop: <int>[1, bundle.manifest.blankTokenId + 1],
    ).squeeze();
    final durationLogits = logits.slice(
      start: <int>[0, bundle.manifest.blankTokenId + 1],
      stop: <int>[
        1,
        bundle.manifest.blankTokenId + 1 + bundle.manifest.durations.length,
      ],
    ).squeeze();
    logits.close();
    return (tokenLogits: tokenLogits, durationLogits: durationLogits);
  }

  String transcribeMel(MlxArray mel, {List<int>? lengths}) {
    onTrace?.call('encoder', 'start lengths=${lengths ?? <int>[mel.shape[1]]}');
    final encoded = _encoder(mel, lengths ?? <int>[mel.shape[1]]);
    var predictState = zeroPredictState();
    final tokens = <ParakeetTdtToken>[];
    var state = const ParakeetTdtDecodeState();
    try {
      final timeSteps = encoded.lengths.first;
      onTrace?.call('encoder', 'done timeSteps=$timeSteps shape=${encoded.features.shape}');
      for (var frame = 0; frame < timeSteps;) {
        onTrace?.call('decode', 'frame=$frame token=${state.lastTokenId}');
        final encoderFrame = encoded.features.slice(
          start: <int>[0, frame, 0],
          stop: <int>[1, frame + 1, bundle.manifest.encoderHidden],
        );
        final predictor = predictorStep(
          tokenId: state.lastTokenId,
          state: predictState,
        );
        try {
          predictState.close();
          predictState = predictor.state;
          final joint = jointStep(
            encoderFrame: encoderFrame,
            predictorOutput: predictor.output,
          );
          try {
            final tokenIdArray = joint.tokenLogits.argmax();
            final durationIdArray = joint.durationLogits.argmax();
            try {
              final tokenId = tokenIdArray.toScalarInt();
              final durationId = durationIdArray.toScalarInt();
              final step = _decoder.pickStepFromIds(
                tokenId: tokenId,
                durationId: durationId,
                state: state,
              );
              onTrace?.call(
                'decode',
                'picked tokenId=$tokenId durationId=$durationId advance=${step.advanceFrames}',
              );
              if (step.emitToken) {
                tokens.add(_decoder.materializeToken(step));
              }
              state = _decoder.applyStep(state, step);
              frame = state.frameIndex;
            } finally {
              tokenIdArray.close();
              durationIdArray.close();
            }
          } finally {
            joint.tokenLogits.close();
            joint.durationLogits.close();
          }
        } finally {
          predictor.output.close();
          encoderFrame.close();
        }
      }
    } finally {
      encoded.features.close();
      predictState.close();
    }
    return tokens
        .map((item) => item.text.replaceAll('▁', ' '))
        .join()
        .replaceAll(RegExp(r'\s+'), ' ')
        .trim();
  }

  MlxArray _predictInputFor(int? tokenId) {
    if (tokenId == null) {
      return _zeroPredictInput;
    }
    return _predictEmbeddings.putIfAbsent(tokenId, () {
      final tokenArray = _predictTokenIds.putIfAbsent(
        tokenId,
        () => MlxArray.fromInt32List(<int>[tokenId], shape: <int>[1, 1]),
      );
      return _embedding(tokenArray);
    });
  }
}
