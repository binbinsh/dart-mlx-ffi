part of 'paddle_ocr_vl.dart';

/// Scatter pre-computed image hidden states into a text-embedding tensor at
/// the positions occupied by the image-token placeholder.
///
/// This is the pure tensor-only half of [PaddleOcrVlEmbed._buildMultimodalEmbedding]
/// — it has no runner / weight dependencies and is exposed for unit testing
/// and for the hybrid CoreML+MLX path (issue #1) which produces
/// [imageHidden] from a separate CoreML pipeline and only needs the MLX-side
/// scatter.
///
/// Inputs:
///   - [textEmbed]: `[1, totalLen, hiddenSize]` text-only embeddings. Not
///     closed by this function; the caller retains ownership.
///   - [imageHidden]: `[N, hiddenSize]` (or any shape whose first
///     `imagePositions.length * hiddenSize` elements correspond to the
///     placeholder positions in order). Must have at least
///     `imagePositions.length` rows.
///   - [imagePositions]: indices in `[0, totalLen)` of every image-token
///     placeholder, in ascending order. Must be non-empty.
///   - [tokenIds]: the `totalLen` prompt token ids (used to build the mask
///     for the sparse path).
///   - [imageTokenId]: the placeholder token id.
///   - [hiddenSize]: model hidden dim.
///
/// Returns a freshly allocated `[1, totalLen, hiddenSize]` array; the caller
/// owns it and must close it.
MlxArray paddleOcrVlScatterImageEmbeddings({
  required MlxArray textEmbed,
  required MlxArray imageHidden,
  required List<int> imagePositions,
  required List<int> tokenIds,
  required int imageTokenId,
  required int hiddenSize,
}) {
  if (imagePositions.isEmpty) {
    throw ArgumentError(
      'imagePositions must be non-empty; the caller should short-circuit the '
      'zero-image case.',
    );
  }
  final totalLen = tokenIds.length;
  final first = imagePositions.first;
  final last = imagePositions.last;
  final contiguous = (last - first + 1) == imagePositions.length;
  if (contiguous) {
    final image3d = imageHidden.astype(textEmbed.dtype).reshape([
      1,
      imagePositions.length,
      hiddenSize,
    ]);
    try {
      return textEmbed.sliceUpdate(
        image3d,
        start: [0, first, 0],
        stop: [1, first + imagePositions.length, hiddenSize],
      );
    } finally {
      image3d.close();
    }
  }
  // Sparse path: rebuild the equality mask from tokenIds and gather features
  // by cumulative-sum index.
  final ids = MlxArray.fromInt32List(tokenIds, shape: [1, totalLen]);
  final imageToken = MlxArray.full(
    [],
    imageTokenId.toDouble(),
    dtype: MlxDType.MLX_INT32,
  );
  final imageMask2d = ids.equal(imageToken);
  imageToken.close();
  ids.close();
  final imageMask = imageMask2d
      .reshape([totalLen])
      .astype(MlxDType.MLX_BOOL);
  imageMask2d.close();
  try {
    final numPositions = imagePositions.length;
    final batchFeatures = imageHidden
        .astype(textEmbed.dtype)
        .slice(start: [0, 0], stop: [numPositions, hiddenSize]);
    try {
      final imageMaskInt = imageMask.astype(MlxDType.MLX_INT32);
      final cumsum = mx.cumsum(imageMaskInt);
      imageMaskInt.close();
      final one = MlxArray.full([totalLen], 1, dtype: MlxDType.MLX_INT32);
      final shifted = cumsum - one;
      one.close();
      cumsum.close();
      final zero = MlxArray.zeros([totalLen], dtype: MlxDType.MLX_INT32);
      final featureIndices = mx.where(imageMask, shifted, zero);
      shifted.close();
      zero.close();
      final gatheredFeatures = batchFeatures.take(featureIndices, axis: 0);
      featureIndices.close();
      final imageMaskExpanded = imageMask.expandDims(1);
      final text2d = textEmbed.reshape([totalLen, hiddenSize]);
      final batchOutput = mx.where(
        imageMaskExpanded,
        gatheredFeatures,
        text2d,
      );
      text2d.close();
      imageMaskExpanded.close();
      gatheredFeatures.close();
      return batchOutput.reshape([1, totalLen, hiddenSize]);
    } finally {
      batchFeatures.close();
    }
  } finally {
    imageMask.close();
  }
}

extension PaddleOcrVlEmbed on PaddleOcrVlRunner {
  MlxArray _embed(MlxArray ids) {
    final seqLen = ids.shape[1];
    final rows = ids.reshape([seqLen]);
    try {
      if (_embedWeights case final _QuantLinear q) {
        final rowsW = q.weight.take(rows, axis: 0);
        final rowsS = q.scales.take(rows, axis: 0);
        final rowsB = q.biases?.take(rows, axis: 0);
        final gathered = MlxQuantizedMatrix(rowsW, rowsS, rowsB);
        try {
          final out = mx.quant.dequantize(
            gathered,
            groupSize: q.quantSpec.groupSize,
            bits: q.quantSpec.bits,
            mode: q.quantSpec.mode,
          );
          if (Platform.isIOS) {
            MlxRuntime.evalAll([out]);
          }
          return out.reshape([1, seqLen, config.hiddenSize]);
        } finally {
          rowsB?.close();
          rowsS.close();
          rowsW.close();
        }
      }
      if (_embedWeights case final _DenseLinear d) {
        final out = d.weight.take(rows, axis: 0);
        return out.reshape([1, seqLen, config.hiddenSize]);
      }
      throw StateError('Unsupported embedding type.');
    } finally {
      rows.close();
    }
  }

  List<int> _expandImageTokens(List<int> tokenIds, int numImageTokens) {
    final result = <int>[];
    for (final id in tokenIds) {
      if (id == config.imageTokenId) {
        for (var j = 0; j < numImageTokens; j++) {
          result.add(config.imageTokenId);
        }
      } else {
        result.add(id);
      }
    }
    return result;
  }

  MlxArray _buildMultimodalEmbedding(List<int> tokenIds, MlxArray imageHidden) {
    final totalLen = tokenIds.length;
    final imagePositions = <int>[];
    for (var i = 0; i < tokenIds.length; i++) {
      if (tokenIds[i] == config.imageTokenId) imagePositions.add(i);
    }
    final ids = MlxArray.fromInt32List(tokenIds, shape: [1, totalLen]);
    final textEmbed = _embed(ids);
    ids.close();
    if (imagePositions.isEmpty) {
      return textEmbed;
    }
    try {
      return paddleOcrVlScatterImageEmbeddings(
        textEmbed: textEmbed,
        imageHidden: imageHidden,
        imagePositions: imagePositions,
        tokenIds: tokenIds,
        imageTokenId: config.imageTokenId,
        hiddenSize: config.hiddenSize,
      );
    } finally {
      textEmbed.close();
    }
  }
}
