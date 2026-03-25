import 'bundle.dart';

class ParakeetTdtStep {
  const ParakeetTdtStep({
    required this.tokenId,
    required this.durationId,
    required this.frameIndex,
    required this.durationFrames,
    required this.isBlank,
    required this.emitToken,
    required this.advanceFrames,
  });

  final int tokenId;
  final int durationId;
  final int frameIndex;
  final int durationFrames;
  final bool isBlank;
  final bool emitToken;
  final int advanceFrames;
}

class ParakeetTdtToken {
  const ParakeetTdtToken({
    required this.id,
    required this.text,
    required this.startSeconds,
    required this.durationSeconds,
  });

  final int id;
  final String text;
  final double startSeconds;
  final double durationSeconds;
}

class ParakeetTdtDecodeState {
  const ParakeetTdtDecodeState({
    this.lastTokenId,
    this.emittedSymbols = 0,
    this.frameIndex = 0,
  });

  final int? lastTokenId;
  final int emittedSymbols;
  final int frameIndex;

  ParakeetTdtDecodeState copyWith({
    int? lastTokenId,
    int? emittedSymbols,
    int? frameIndex,
    bool clearLastToken = false,
  }) {
    return ParakeetTdtDecodeState(
      lastTokenId: clearLastToken ? null : (lastTokenId ?? this.lastTokenId),
      emittedSymbols: emittedSymbols ?? this.emittedSymbols,
      frameIndex: frameIndex ?? this.frameIndex,
    );
  }
}

class ParakeetTdtGreedyDecoder {
  const ParakeetTdtGreedyDecoder(this.manifest);

  final ParakeetTdtManifest manifest;

  ParakeetTdtStep pickStep({
    required List<double> tokenLogits,
    required List<double> durationLogits,
    required ParakeetTdtDecodeState state,
  }) {
    final tokenId = _argmax(tokenLogits);
    final durationId = _argmax(durationLogits);
    return pickStepFromIds(
      tokenId: tokenId,
      durationId: durationId,
      state: state,
    );
  }

  ParakeetTdtStep pickStepFromIds({
    required int tokenId,
    required int durationId,
    required ParakeetTdtDecodeState state,
  }) {
    final isBlank = tokenId == manifest.blankTokenId;
    final rawAdvance = manifest.durations[durationId];
    final hitMaxSymbols =
        !isBlank &&
        rawAdvance == 0 &&
        state.emittedSymbols + 1 >= manifest.maxSymbols;
    final advanceFrames = hitMaxSymbols ? 1 : rawAdvance;
    return ParakeetTdtStep(
      tokenId: tokenId,
      durationId: durationId,
      frameIndex: state.frameIndex,
      durationFrames: rawAdvance,
      isBlank: isBlank,
      emitToken: !isBlank,
      advanceFrames: advanceFrames,
    );
  }

  ParakeetTdtDecodeState applyStep(
    ParakeetTdtDecodeState state,
    ParakeetTdtStep step,
  ) {
    final resetSymbolCount = step.advanceFrames != 0;
    return state.copyWith(
      lastTokenId: step.isBlank ? state.lastTokenId : step.tokenId,
      emittedSymbols: resetSymbolCount ? 0 : state.emittedSymbols + 1,
      frameIndex: state.frameIndex + step.advanceFrames,
    );
  }

  ParakeetTdtToken materializeToken(ParakeetTdtStep step) {
    final tokenText =
        step.tokenId >= 0 && step.tokenId < manifest.vocabulary.length
        ? manifest.vocabulary[step.tokenId]
        : '';
    return ParakeetTdtToken(
      id: step.tokenId,
      text: tokenText,
      startSeconds: step.frameIndex * manifest.frameStepSeconds,
      durationSeconds: step.durationFrames * manifest.frameStepSeconds,
    );
  }

  int _argmax(List<double> values) {
    var bestIndex = 0;
    var bestValue = double.negativeInfinity;
    for (var index = 0; index < values.length; index += 1) {
      final value = values[index];
      if (value > bestValue) {
        bestValue = value;
        bestIndex = index;
      }
    }
    return bestIndex;
  }
}
