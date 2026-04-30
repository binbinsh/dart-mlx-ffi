part of 'paddle_ocr_vl.dart';

const _decodeLmHeadChunkSize = 128;

int _nextPowerOfTwo(int value) {
  var n = 1;
  while (n < value) {
    n <<= 1;
  }
  return n;
}

MlxArray _sliceFirstAxis(MlxArray array, int start, int end) {
  final sliceStart = List<int>.filled(array.shape.length, 0);
  final sliceStop = List<int>.from(array.shape);
  sliceStart[0] = start;
  sliceStop[0] = end;
  return array.slice(start: sliceStart, stop: sliceStop);
}

({MlxArray values, MlxArray ids}) _reduceMaxIdsTree(
  MlxArray values,
  MlxArray ids,
) {
  var currentValues = values;
  var currentIds = ids;
  var currentSize = values.size;
  while (currentSize > 1) {
    final half = currentSize ~/ 2;
    final leftValues = currentValues.slice(start: [0], stop: [half]);
    final rightValues = currentValues.slice(start: [half], stop: [currentSize]);
    final leftIds = currentIds.slice(start: [0], stop: [half]);
    final rightIds = currentIds.slice(start: [half], stop: [currentSize]);
    final pickRight = MlxMore.greater(rightValues, leftValues);
    final nextValues = mx.where(pickRight, rightValues, leftValues);
    final nextIds = mx.where(pickRight, rightIds, leftIds);

    leftValues.close();
    rightValues.close();
    leftIds.close();
    rightIds.close();
    pickRight.close();
    currentValues.close();
    currentIds.close();

    currentValues = nextValues;
    currentIds = nextIds;
    currentSize = half;
  }
  return (values: currentValues, ids: currentIds);
}

int _sampleLinearChunkedArgmax(
  PaddleOcrVlRunner runner,
  _LinearBase linear,
  MlxArray hidden,
  int vocabSize,
) {
  final sampleHidden = hidden.dtype == MlxDType.MLX_FLOAT32
      ? hidden
      : hidden.astype(MlxDType.MLX_FLOAT32);
  MlxArray? bestIndex;
  MlxArray? bestValue;
  try {
    for (var start = 0; start < vocabSize; start += _decodeLmHeadChunkSize) {
      final end = math.min(start + _decodeLmHeadChunkSize, vocabSize);
      var stage = 'start';
      var linearPrefix = '';
      var linearKind = '';
      final traceFirstChunk = start == 0;
      void setStage(String value) {
        stage = value;
        if (!traceFirstChunk ||
            !runner.config.enableDecoderTailTraceForCurrentPlatform) {
          return;
        }
        PaddleOcrVlDebugOverrides.traceSink?.call(
          'sample_argmax chunk=[$start,$end) stage=$stage '
          'linearKind=$linearKind prefix=$linearPrefix '
          'sampleHiddenDType=${sampleHidden.dtype} hiddenDType=${hidden.dtype}',
        );
      }

      MlxArray? weightSlice;
      MlxArray? weightSliceF32;
      MlxArray? scaleSlice;
      MlxArray? qBiasSlice;
      MlxArray? biasSlice;
      MlxArray? biasSliceF32;
      MlxArray? dequantizedWeight;
      MlxArray? product;
      MlxArray? reduced;
      MlxArray? bias2d;
      MlxArray? chunk;
      MlxArray? flat;
      MlxArray? flatF32;
      MlxArray? chunkIds;
      MlxArray? paddedValues;
      MlxArray? paddedIds;
      MlxArray? localIndexF32;
      MlxArray? globalIndex;
      MlxArray? localValue;
      MlxArray? localValueF32;
      MlxArray? shouldUpdate;
      MlxArray? nextBestIndex;
      MlxArray? nextBestValue;
      try {
        switch (linear) {
          case _DenseLinear dense:
            linearPrefix = dense.prefix;
            linearKind = 'dense';
            setStage('slice_dense');
            weightSlice = _sliceFirstAxis(dense.weight, start, end);
            biasSlice = dense.bias == null
                ? null
                : _sliceFirstAxis(dense.bias!, start, end);
            setStage('cast_weight_dense');
            weightSliceF32 = weightSlice.dtype == MlxDType.MLX_FLOAT32
                ? weightSlice
                : weightSlice.astype(MlxDType.MLX_FLOAT32);
            setStage('multiply_dense');
            product = mx.multiply(weightSliceF32, sampleHidden);
            setStage('sum_dense');
            reduced = product.sum(axis: 1);
            if (biasSlice != null) {
              setStage('cast_bias_dense');
              biasSliceF32 = biasSlice.dtype == MlxDType.MLX_FLOAT32
                  ? biasSlice
                  : biasSlice.astype(MlxDType.MLX_FLOAT32);
              setStage('reshape_chunk_dense');
              chunk = reduced.reshape([1, end - start]);
              setStage('reshape_bias_dense');
              bias2d = biasSliceF32.reshape([1, biasSliceF32.shape[0]]);
              setStage('add_bias_dense');
              final biased = mx.add(chunk, bias2d);
              chunk.close();
              chunk = biased;
            } else {
              setStage('reshape_chunk_dense');
              chunk = reduced.reshape([1, end - start]);
            }
          case _QuantLinear quant:
            linearPrefix = quant.prefix;
            linearKind = 'quant';
            setStage('slice_quant');
            weightSlice = _sliceFirstAxis(quant.weight, start, end);
            scaleSlice = _sliceFirstAxis(quant.scales, start, end);
            qBiasSlice = quant.biases == null
                ? null
                : _sliceFirstAxis(quant.biases!, start, end);
            biasSlice = quant.bias == null
                ? null
                : _sliceFirstAxis(quant.bias!, start, end);
            setStage('dequantize_weight_quant');
            dequantizedWeight = mx.quant.dequantize(
              MlxQuantizedMatrix(weightSlice, scaleSlice, qBiasSlice),
              groupSize: quant.quantSpec.groupSize,
              bits: quant.quantSpec.bits,
              mode: quant.quantSpec.mode,
              dtype: MlxDType.MLX_FLOAT32,
            );
            setStage('multiply_quant');
            product = mx.multiply(dequantizedWeight, sampleHidden);
            setStage('sum_quant');
            reduced = product.sum(axis: 1);
            if (biasSlice != null) {
              setStage('cast_bias_quant');
              biasSliceF32 = biasSlice.dtype == MlxDType.MLX_FLOAT32
                  ? biasSlice
                  : biasSlice.astype(MlxDType.MLX_FLOAT32);
              setStage('reshape_chunk_quant');
              chunk = reduced.reshape([1, end - start]);
              setStage('reshape_bias_quant');
              bias2d = biasSliceF32.reshape([1, biasSliceF32.shape[0]]);
              setStage('add_bias_quant');
              final biased = mx.add(chunk, bias2d);
              chunk.close();
              chunk = biased;
            } else {
              setStage('reshape_chunk_quant');
              chunk = reduced.reshape([1, end - start]);
            }
        }
        setStage('reshape_chunk');
        flat = chunk.reshape([end - start]);
        setStage('cast_flat_f32');
        flatF32 = flat.dtype == MlxDType.MLX_FLOAT32
            ? flat
            : flat.astype(MlxDType.MLX_FLOAT32);
        setStage('chunk_ids');
        chunkIds = MlxArray.arange(
          start.toDouble(),
          end.toDouble(),
          1,
          dtype: MlxDType.MLX_FLOAT32,
        );
        final targetSize = _nextPowerOfTwo(end - start);
        if (targetSize == (end - start)) {
          paddedValues = flatF32;
          paddedIds = chunkIds;
          flatF32 = null;
          chunkIds = null;
        } else {
          setStage('pad_values');
          final valuePad = MlxArray.full(
            [targetSize - (end - start)],
            double.negativeInfinity,
            dtype: MlxDType.MLX_FLOAT32,
          );
          final idPad = MlxArray.zeros([
            targetSize - (end - start),
          ], dtype: MlxDType.MLX_FLOAT32);
          paddedValues = mx.concatenate([flatF32, valuePad], axis: 0);
          paddedIds = mx.concatenate([chunkIds, idPad], axis: 0);
          valuePad.close();
          idPad.close();
        }
        setStage('reduce_tree');
        final reducedPair = _reduceMaxIdsTree(paddedValues, paddedIds);
        paddedValues = null;
        paddedIds = null;
        localValue = reducedPair.values;
        localValueF32 = localValue;
        localIndexF32 = reducedPair.ids;
        globalIndex = localIndexF32;
        localIndexF32 = null;

        if (bestIndex == null || bestValue == null) {
          setStage('seed_best');
          bestIndex = globalIndex;
          bestValue = localValueF32;
          if (identical(localValue, localValueF32)) {
            localValue = null;
          }
          globalIndex = null;
          localValueF32 = null;
        } else {
          setStage('compare_best');
          shouldUpdate = MlxMore.greater(localValueF32, bestValue);
          setStage('where_best_index');
          nextBestIndex = mx.where(shouldUpdate, globalIndex, bestIndex);
          setStage('where_best_value');
          nextBestValue = mx.where(shouldUpdate, localValueF32, bestValue);
          bestIndex.close();
          bestValue.close();
          bestIndex = nextBestIndex;
          bestValue = nextBestValue;
          nextBestIndex = null;
          nextBestValue = null;
        }
      } on MlxException catch (e) {
        PaddleOcrVlDebugOverrides.traceSink?.call(
          'sample_token chunked_lm_head_failed='
          '${e.message} chunk=[$start,$end) '
          'stage=$stage linearKind=$linearKind prefix=$linearPrefix '
          'chunkDType=${chunk?.dtype} sampleHiddenDType=${sampleHidden.dtype} '
          'hiddenDType=${hidden.dtype}',
        );
        rethrow;
      } finally {
        nextBestValue?.close();
        nextBestIndex?.close();
        shouldUpdate?.close();
        reduced?.close();
        product?.close();
        paddedIds?.close();
        paddedValues?.close();
        if (!identical(localValueF32, localValue)) {
          localValueF32?.close();
        }
        localValue?.close();
        chunkIds?.close();
        globalIndex?.close();
        localIndexF32?.close();
        if (!identical(flatF32, flat)) {
          flatF32?.close();
        }
        flat?.close();
        chunk?.close();
        bias2d?.close();
        dequantizedWeight?.close();
        if (!identical(biasSliceF32, biasSlice)) {
          biasSliceF32?.close();
        }
        biasSlice?.close();
        qBiasSlice?.close();
        scaleSlice?.close();
        if (!identical(weightSliceF32, weightSlice)) {
          weightSliceF32?.close();
        }
        weightSlice?.close();
      }
    }
    if (bestIndex == null) {
      throw StateError('chunked lm_head argmax found no values.');
    }
    if (runner.config.enableDecoderTailTraceForCurrentPlatform) {
      PaddleOcrVlDebugOverrides.traceSink?.call(
        'sample_argmax final_scalar_read dtype=${bestIndex.dtype}',
      );
    }
    return bestIndex.scalarFloat32Relaxed().round();
  } finally {
    if (!identical(sampleHidden, hidden)) {
      sampleHidden.close();
    }
    bestValue?.close();
    bestIndex?.close();
  }
}

MlxArray _decodeLastHiddenWithCache(
  PaddleOcrVlRunner runner,
  MlxArray hidden,
  MlxArray positionIds,
  _ModelCache cache,
) {
  const seqLen = 1;
  var h = hidden;
  try {
    for (var i = 0; i < runner._lmLayers.length; i++) {
      final trace = runner._beginDecoderLayerTrace(i, seqLen, cache);
      var next = runner._decoderLayer(
        runner._lmLayers[i],
        h,
        seqLen,
        positionIds,
        layerIndex: i,
        cache: cache.layers[i],
      );
      if (runner.config.enableDecoderLayerwiseEvalForCurrentPlatform) {
        final evalTrace = _beginDecoderLayerEventTrace(
          runner,
          i,
          seqLen,
          cache,
          'eval_next',
        );
        MlxRuntime.evalAll([next]);
        _endDecoderLayerEventTrace(runner, evalTrace);
        if (runner
            .config
            .enableForceGpuSynchronizeAfterLayerEvalForCurrentPlatform) {
          final syncTrace = _beginDecoderLayerEventTrace(
            runner,
            i,
            seqLen,
            cache,
            'sync_after_eval',
          );
          runner._maybeSynchronizeGpuPerToken();
          _endDecoderLayerEventTrace(runner, syncTrace);
        }
      }
      if (h != hidden) {
        final closeTrace = _beginDecoderLayerEventTrace(
          runner,
          i,
          seqLen,
          cache,
          'close_prev_h',
        );
        h.close();
        _endDecoderLayerEventTrace(runner, closeTrace);
      }
      h = next;
      if (runner.config.enableAggressiveCacheClearingForCurrentPlatform) {
        final clearTrace = _beginDecoderLayerEventTrace(
          runner,
          i,
          seqLen,
          cache,
          'clear_cache',
        );
        try {
          MlxMemory.clearCache();
        } catch (_) {}
        _endDecoderLayerEventTrace(runner, clearTrace);
      }
      runner._endDecoderLayerTrace(trace, cache, cache.layers[i]);
    }

    final norm = _traceDecoderTailArray(
      runner,
      seqLen,
      cache,
      'sample_final_norm',
      () => _lmRmsNormCompat(
        h,
        weight: runner._finalNorm,
        eps: runner.config.rmsNormEps,
      ),
    );
    h.close();

    final last = _traceDecoderTailArray(
      runner,
      seqLen,
      cache,
      'sample_last_slice',
      () =>
          norm.slice(start: [0, 0, 0], stop: [1, 1, runner.config.hiddenSize]),
    );
    norm.close();

    final last2d = _traceDecoderTailArray(
      runner,
      seqLen,
      cache,
      'sample_last_reshape',
      () => last.reshape([1, runner.config.hiddenSize]),
    );
    last.close();

    return last2d;
  } catch (_) {
    if (h != hidden) h.close();
    rethrow;
  }
}

int _sampleNextTokenWithCache(
  PaddleOcrVlRunner runner,
  int tokenId,
  int textPosition,
  _ModelCache cache,
) {
  final stepArr = MlxArray.fromInt32List([tokenId], shape: [1, 1]);
  final stepPos = runner._textPositionIds(1, offset: textPosition);
  _maybeStartDecoderMetalCapture(cache);
  try {
    final hidden = _traceDecodeLoopValue(runner, cache, 'sample_embed', () {
      return runner._embed(stepArr);
    });
    final last2d = _traceDecodeLoopValue(
      runner,
      cache,
      'sample_decode_hidden',
      () => _decodeLastHiddenWithCache(runner, hidden, stepPos, cache),
    );
    try {
      final linear = runner.config.tieWordEmbeddings
          ? runner._embedWeights
          : runner._lmHead!;
      final next = _traceDecodeLoopValue(runner, cache, 'sample_argmax', () {
        return _sampleLinearChunkedArgmax(
          runner,
          linear,
          last2d,
          runner.config.vocabSize,
        );
      });
      _traceDecodeLoopValue(runner, cache, 'sample_postprocess_cache', () {
        cache.maybeQuantize(config: runner.config);
        final evalInterval =
            runner.config.decodeCacheStateEvalIntervalForCurrentPlatform;
        if (runner.config.enableDecodeCacheStateEvalForCurrentPlatform &&
            (evalInterval <= 1 || (cache.offset % evalInterval) == 0)) {
          cache.evalStates();
        }
        cache.maybeCompact(config: runner.config);
        return 0;
      });
      return next;
    } finally {
      _traceDecodeLoopValue(runner, cache, 'sample_close_last2d', () {
        last2d.close();
        return 0;
      });
    }
  } finally {
    _traceDecodeLoopValue(runner, cache, 'sample_stop_capture', () {
      _maybeStopDecoderMetalCapture(cache);
      return 0;
    });
    _traceDecodeLoopValue(runner, cache, 'sample_close_step_pos', () {
      stepPos.close();
      return 0;
    });
    _traceDecodeLoopValue(runner, cache, 'sample_close_step_arr', () {
      stepArr.close();
      return 0;
    });
  }
}
