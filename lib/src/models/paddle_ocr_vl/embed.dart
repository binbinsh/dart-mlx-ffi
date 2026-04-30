part of 'paddle_ocr_vl.dart';

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
    MlxArray? result;
    try {
      if (imagePositions.isEmpty) {
        result = textEmbed;
        return result;
      }
      final first = imagePositions.first;
      final last = imagePositions.last;
      final contiguous = (last - first + 1) == imagePositions.length;
      if (contiguous) {
        final image3d = imageHidden.astype(textEmbed.dtype).reshape([
          1,
          imagePositions.length,
          config.hiddenSize,
        ]);
        try {
          final updated = textEmbed.sliceUpdate(
            image3d,
            start: [0, first, 0],
            stop: [1, first + imagePositions.length, config.hiddenSize],
          );
          result = updated;
          return updated;
        } finally {
          image3d.close();
        }
      }
      final imageToken = MlxArray.full(
        [],
        config.imageTokenId.toDouble(),
        dtype: MlxDType.MLX_INT32,
      );
      final imageMask2d = ids.equal(imageToken);
      imageToken.close();
      final imageMask = imageMask2d
          .reshape([totalLen])
          .astype(MlxDType.MLX_BOOL);
      imageMask2d.close();
      try {
        final numPositions = imagePositions.length;
        final batchFeatures = imageHidden
            .astype(textEmbed.dtype)
            .slice(start: [0, 0], stop: [numPositions, config.hiddenSize]);
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
          final text2d = textEmbed.reshape([totalLen, config.hiddenSize]);
          final batchOutput = mx.where(
            imageMaskExpanded,
            gatheredFeatures,
            text2d,
          );
          text2d.close();
          imageMaskExpanded.close();
          gatheredFeatures.close();
          final reshaped = batchOutput.reshape([
            1,
            totalLen,
            config.hiddenSize,
          ]);
          result = reshaped;
          return reshaped;
        } finally {
          batchFeatures.close();
        }
      } finally {
        imageMask.close();
      }
    } finally {
      if (!identical(result, textEmbed)) {
        textEmbed.close();
      }
      ids.close();
    }
  }
}
