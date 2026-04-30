part of 'paddle_ocr_vl.dart';

typedef _DecoderTrace = ({
  int activeBefore,
  int cacheBefore,
  int peakBefore,
  int resourceBefore,
  int commitBefore,
  int setDataBefore,
  int sharedCopyBefore,
  int contiguousBefore,
  int quantizedContigXBefore,
  int layerIndex,
  int offset,
  Stopwatch watch,
});

typedef _DecoderSubstepTrace = ({
  int activeBefore,
  int cacheBefore,
  int peakBefore,
  int resourceBefore,
  int commitBefore,
  int setDataBefore,
  int sharedCopyBefore,
  int contiguousBefore,
  int quantizedContigXBefore,
  int layerIndex,
  int offset,
  String step,
  Stopwatch watch,
});

typedef _DecoderTailTrace = ({
  int activeBefore,
  int cacheBefore,
  int peakBefore,
  int resourceBefore,
  int commitBefore,
  int setDataBefore,
  int sharedCopyBefore,
  int contiguousBefore,
  int quantizedContigXBefore,
  int offset,
  String step,
  Stopwatch watch,
});

typedef _DecoderLayerEventTrace = ({
  int activeBefore,
  int cacheBefore,
  int peakBefore,
  int resourceBefore,
  int commitBefore,
  int setDataBefore,
  int sharedCopyBefore,
  int contiguousBefore,
  int quantizedContigXBefore,
  int layerIndex,
  int offset,
  String step,
  Stopwatch watch,
});

bool _decoderMetalCaptureRunning = false;

void _maybeStartDecoderMetalCapture(_ModelCache cache) {
  if (_decoderMetalCaptureRunning) return;
  final path = PaddleOcrVlDebugOverrides.metalCapturePath;
  final start = PaddleOcrVlDebugOverrides.metalCaptureStartOffset;
  if (path == null || start == null) return;
  if (cache.offset != start) return;
  final sink = PaddleOcrVlDebugOverrides.traceSink;
  try {
    MlxMetal.startCapture(path);
    _decoderMetalCaptureRunning = true;
    sink?.call('metalCapture start path=$path offset=${cache.offset}');
  } catch (error) {
    sink?.call('metalCapture start failed: $error');
  }
}

void _maybeStopDecoderMetalCapture(_ModelCache cache) {
  if (!_decoderMetalCaptureRunning) return;
  final stop = PaddleOcrVlDebugOverrides.metalCaptureStopOffset;
  if (stop == null) return;
  if (cache.offset < stop) return;
  final sink = PaddleOcrVlDebugOverrides.traceSink;
  try {
    MlxMetal.stopCapture();
    sink?.call('metalCapture stop offset=${cache.offset}');
  } catch (error) {
    sink?.call('metalCapture stop failed: $error');
  }
  _decoderMetalCaptureRunning = false;
}

extension PaddleOcrVlTrace on PaddleOcrVlRunner {
  _DecoderTrace? _beginDecoderLayerTrace(
    int layerIndex,
    int seqLen,
    _ModelCache cache,
  ) {
    if (!config.enableDecoderPerLayerTraceForCurrentPlatform) return null;
    if (seqLen != 1) return null;
    final interval = config.decoderPerLayerTraceIntervalForCurrentPlatform;
    final offset = cache.offset;
    if (offset > 0 && interval > 1 && (offset % interval) != 0) return null;
    return (
      activeBefore: _safeActiveBytes(),
      cacheBefore: _safeCacheBytes(),
      peakBefore: _safePeakBytesTrace(),
      resourceBefore: _safeResourceCountTrace(),
      commitBefore: _safeCommitCountTrace(),
      setDataBefore: _safeSetDataCountTrace(),
      sharedCopyBefore: _safeSharedBufferCopyCountTrace(),
      contiguousBefore: _safeContiguousCopyCountTrace(),
      quantizedContigXBefore: _safeQuantizedContiguousXCountTrace(),
      layerIndex: layerIndex,
      offset: offset,
      watch: Stopwatch()..start(),
    );
  }

  void _endDecoderLayerTrace(
    _DecoderTrace? trace,
    _ModelCache cache,
    _LayerCache layerCache,
  ) {
    if (trace == null) return;
    trace.watch.stop();
    final sink = PaddleOcrVlDebugOverrides.traceSink;
    final activeAfter = _safeActiveBytes();
    final cacheAfter = _safeCacheBytes();
    final peakAfter = _safePeakBytesTrace();
    final resourceAfter = _safeResourceCountTrace();
    final commitAfter = _safeCommitCountTrace();
    final setDataAfter = _safeSetDataCountTrace();
    final sharedCopyAfter = _safeSharedBufferCopyCountTrace();
    final contiguousAfter = _safeContiguousCopyCountTrace();
    final quantizedContigXAfter = _safeQuantizedContiguousXCountTrace();
    final deltaActive = activeAfter - trace.activeBefore;
    final deltaCache = cacheAfter - trace.cacheBefore;
    final deltaPeak = peakAfter - trace.peakBefore;
    final msg =
        'decoderTrace '
        'offset=${trace.offset} '
        'layer=${trace.layerIndex + 1}/${_lmLayers.length} '
        'ms=${trace.watch.elapsedMilliseconds} '
        'active=${_formatApproxBytes(activeAfter)} '
        'cache=${_formatApproxBytes(cacheAfter)} '
        'peak=${_formatApproxBytes(peakAfter)} '
        'dActive=${_formatSignedBytes(deltaActive)} '
        'dCache=${_formatSignedBytes(deltaCache)} '
        'dPeak=${_formatSignedBytes(deltaPeak)} '
        'dRsrc=${resourceAfter - trace.resourceBefore} '
        'dCommits=${commitAfter - trace.commitBefore} '
        'dSetData=${setDataAfter - trace.setDataBefore} '
        'dSharedCopy=${sharedCopyAfter - trace.sharedCopyBefore} '
        'dContig=${contiguousAfter - trace.contiguousBefore} '
        'dQContigX=${quantizedContigXAfter - trace.quantizedContigXBefore} '
        'layerApprox=${_formatApproxBytes(layerCache.approxBytes())} '
        'cacheApprox=${_formatApproxBytes(cache.approxBytes())}'
        '${_formatAllocatorStatsSuffix()}';
    if (sink != null) {
      sink(msg);
    } else {
      stdout.writeln(msg);
    }
  }
}

_DecoderSubstepTrace? _beginDecoderSubstepTrace(
  PaddleOcrVlRunner runner,
  int layerIndex,
  int seqLen,
  _LayerCache? cache,
  String step,
) {
  if (!runner.config.enableDecoderSubstepTraceForCurrentPlatform) return null;
  if (seqLen != 1 || cache == null) return null;
  if (layerIndex != runner.config.decoderSubstepTraceLayerForCurrentPlatform) {
    return null;
  }
  return (
    activeBefore: _safeActiveBytes(),
    cacheBefore: _safeCacheBytes(),
    peakBefore: _safePeakBytesTrace(),
    resourceBefore: _safeResourceCountTrace(),
    commitBefore: _safeCommitCountTrace(),
    setDataBefore: _safeSetDataCountTrace(),
    sharedCopyBefore: _safeSharedBufferCopyCountTrace(),
    contiguousBefore: _safeContiguousCopyCountTrace(),
    quantizedContigXBefore: _safeQuantizedContiguousXCountTrace(),
    layerIndex: layerIndex,
    offset: cache.offset,
    step: step,
    watch: Stopwatch()..start(),
  );
}

void _endDecoderSubstepTrace(
  PaddleOcrVlRunner runner,
  _DecoderSubstepTrace? trace,
) {
  if (trace == null) return;
  trace.watch.stop();
  final sink = PaddleOcrVlDebugOverrides.traceSink;
  final activeAfter = _safeActiveBytes();
  final cacheAfter = _safeCacheBytes();
  final peakAfter = _safePeakBytesTrace();
  final resourceAfter = _safeResourceCountTrace();
  final commitAfter = _safeCommitCountTrace();
  final setDataAfter = _safeSetDataCountTrace();
  final sharedCopyAfter = _safeSharedBufferCopyCountTrace();
  final contiguousAfter = _safeContiguousCopyCountTrace();
  final quantizedContigXAfter = _safeQuantizedContiguousXCountTrace();
  final msg =
      'decoderSubstep '
      'offset=${trace.offset} '
      'layer=${trace.layerIndex + 1}/${runner.config.numHiddenLayers} '
      'step=${trace.step} '
      'ms=${trace.watch.elapsedMilliseconds} '
      'active=${_formatApproxBytes(activeAfter)} '
      'cache=${_formatApproxBytes(cacheAfter)} '
      'peak=${_formatApproxBytes(peakAfter)} '
      'dActive=${_formatSignedBytes(activeAfter - trace.activeBefore)} '
      'dCache=${_formatSignedBytes(cacheAfter - trace.cacheBefore)} '
      'dPeak=${_formatSignedBytes(peakAfter - trace.peakBefore)}'
      ' dRsrc=${resourceAfter - trace.resourceBefore}'
      ' dCommits=${commitAfter - trace.commitBefore}'
      ' dSetData=${setDataAfter - trace.setDataBefore}'
      ' dSharedCopy=${sharedCopyAfter - trace.sharedCopyBefore}'
      ' dContig=${contiguousAfter - trace.contiguousBefore}'
      ' dQContigX=${quantizedContigXAfter - trace.quantizedContigXBefore}'
      '${_formatAllocatorStatsSuffix()}';
  if (sink != null) {
    sink(msg);
  } else {
    stdout.writeln(msg);
  }
}

_DecoderTailTrace? _beginDecoderTailTrace(
  PaddleOcrVlRunner runner,
  int seqLen,
  _ModelCache cache,
  String step,
) {
  if (!runner.config.enableDecoderTailTraceForCurrentPlatform) return null;
  if (seqLen != 1) return null;
  return (
    activeBefore: _safeActiveBytes(),
    cacheBefore: _safeCacheBytes(),
    peakBefore: _safePeakBytesTrace(),
    resourceBefore: _safeResourceCountTrace(),
    commitBefore: _safeCommitCountTrace(),
    setDataBefore: _safeSetDataCountTrace(),
    sharedCopyBefore: _safeSharedBufferCopyCountTrace(),
    contiguousBefore: _safeContiguousCopyCountTrace(),
    quantizedContigXBefore: _safeQuantizedContiguousXCountTrace(),
    offset: cache.offset,
    step: step,
    watch: Stopwatch()..start(),
  );
}

void _endDecoderTailTrace(PaddleOcrVlRunner runner, _DecoderTailTrace? trace) {
  if (trace == null) return;
  trace.watch.stop();
  final sink = PaddleOcrVlDebugOverrides.traceSink;
  final activeAfter = _safeActiveBytes();
  final cacheAfter = _safeCacheBytes();
  final peakAfter = _safePeakBytesTrace();
  final resourceAfter = _safeResourceCountTrace();
  final commitAfter = _safeCommitCountTrace();
  final setDataAfter = _safeSetDataCountTrace();
  final sharedCopyAfter = _safeSharedBufferCopyCountTrace();
  final contiguousAfter = _safeContiguousCopyCountTrace();
  final quantizedContigXAfter = _safeQuantizedContiguousXCountTrace();
  final msg =
      'decoderTail '
      'offset=${trace.offset} '
      'step=${trace.step} '
      'ms=${trace.watch.elapsedMilliseconds} '
      'active=${_formatApproxBytes(activeAfter)} '
      'cache=${_formatApproxBytes(cacheAfter)} '
      'peak=${_formatApproxBytes(peakAfter)} '
      'dActive=${_formatSignedBytes(activeAfter - trace.activeBefore)} '
      'dCache=${_formatSignedBytes(cacheAfter - trace.cacheBefore)} '
      'dPeak=${_formatSignedBytes(peakAfter - trace.peakBefore)}'
      ' dRsrc=${resourceAfter - trace.resourceBefore}'
      ' dCommits=${commitAfter - trace.commitBefore}'
      ' dSetData=${setDataAfter - trace.setDataBefore}'
      ' dSharedCopy=${sharedCopyAfter - trace.sharedCopyBefore}'
      ' dContig=${contiguousAfter - trace.contiguousBefore}'
      ' dQContigX=${quantizedContigXAfter - trace.quantizedContigXBefore}'
      '${_formatAllocatorStatsSuffix()}';
  if (sink != null) {
    sink(msg);
  } else {
    stdout.writeln(msg);
  }
}

_DecoderLayerEventTrace? _beginDecoderLayerEventTrace(
  PaddleOcrVlRunner runner,
  int layerIndex,
  int seqLen,
  _ModelCache cache,
  String step,
) {
  if (!runner.config.enableDecoderPerLayerTraceForCurrentPlatform) return null;
  if (seqLen != 1) return null;
  final interval = runner.config.decoderPerLayerTraceIntervalForCurrentPlatform;
  final offset = cache.offset;
  if (offset > 0 && interval > 1 && (offset % interval) != 0) return null;
  return (
    activeBefore: _safeActiveBytes(),
    cacheBefore: _safeCacheBytes(),
    peakBefore: _safePeakBytesTrace(),
    resourceBefore: _safeResourceCountTrace(),
    commitBefore: _safeCommitCountTrace(),
    setDataBefore: _safeSetDataCountTrace(),
    sharedCopyBefore: _safeSharedBufferCopyCountTrace(),
    contiguousBefore: _safeContiguousCopyCountTrace(),
    quantizedContigXBefore: _safeQuantizedContiguousXCountTrace(),
    layerIndex: layerIndex,
    offset: offset,
    step: step,
    watch: Stopwatch()..start(),
  );
}

void _endDecoderLayerEventTrace(
  PaddleOcrVlRunner runner,
  _DecoderLayerEventTrace? trace,
) {
  if (trace == null) return;
  trace.watch.stop();
  final sink = PaddleOcrVlDebugOverrides.traceSink;
  final activeAfter = _safeActiveBytes();
  final cacheAfter = _safeCacheBytes();
  final peakAfter = _safePeakBytesTrace();
  final resourceAfter = _safeResourceCountTrace();
  final commitAfter = _safeCommitCountTrace();
  final setDataAfter = _safeSetDataCountTrace();
  final sharedCopyAfter = _safeSharedBufferCopyCountTrace();
  final contiguousAfter = _safeContiguousCopyCountTrace();
  final quantizedContigXAfter = _safeQuantizedContiguousXCountTrace();
  final msg =
      'decoderLayerEvent '
      'offset=${trace.offset} '
      'layer=${trace.layerIndex + 1}/${runner.config.numHiddenLayers} '
      'step=${trace.step} '
      'ms=${trace.watch.elapsedMilliseconds} '
      'active=${_formatApproxBytes(activeAfter)} '
      'cache=${_formatApproxBytes(cacheAfter)} '
      'peak=${_formatApproxBytes(peakAfter)} '
      'dActive=${_formatSignedBytes(activeAfter - trace.activeBefore)} '
      'dCache=${_formatSignedBytes(cacheAfter - trace.cacheBefore)} '
      'dPeak=${_formatSignedBytes(peakAfter - trace.peakBefore)}'
      ' dRsrc=${resourceAfter - trace.resourceBefore}'
      ' dCommits=${commitAfter - trace.commitBefore}'
      ' dSetData=${setDataAfter - trace.setDataBefore}'
      ' dSharedCopy=${sharedCopyAfter - trace.sharedCopyBefore}'
      ' dContig=${contiguousAfter - trace.contiguousBefore}'
      ' dQContigX=${quantizedContigXAfter - trace.quantizedContigXBefore}'
      '${_formatAllocatorStatsSuffix()}';
  if (sink != null) {
    sink(msg);
  } else {
    stdout.writeln(msg);
  }
}

MlxArray _traceDecoderTailArray(
  PaddleOcrVlRunner runner,
  int seqLen,
  _ModelCache cache,
  String step,
  MlxArray Function() fn,
) {
  final trace = _beginDecoderTailTrace(runner, seqLen, cache, step);
  final out = fn();
  _endDecoderTailTrace(runner, trace);
  return out;
}

void _traceDecoderTailVoid(
  PaddleOcrVlRunner runner,
  int seqLen,
  _ModelCache cache,
  String step,
  void Function() fn,
) {
  final trace = _beginDecoderTailTrace(runner, seqLen, cache, step);
  fn();
  _endDecoderTailTrace(runner, trace);
}

T _traceDecodeLoopValue<T>(
  PaddleOcrVlRunner runner,
  _ModelCache cache,
  String step,
  T Function() fn,
) {
  if (!runner.config.enableDecoderTailTraceForCurrentPlatform) {
    return fn();
  }
  final trace = (
    activeBefore: _safeActiveBytes(),
    cacheBefore: _safeCacheBytes(),
    peakBefore: _safePeakBytesTrace(),
    resourceBefore: _safeResourceCountTrace(),
    commitBefore: _safeCommitCountTrace(),
    setDataBefore: _safeSetDataCountTrace(),
    sharedCopyBefore: _safeSharedBufferCopyCountTrace(),
    contiguousBefore: _safeContiguousCopyCountTrace(),
    quantizedContigXBefore: _safeQuantizedContiguousXCountTrace(),
    offset: cache.offset,
    step: step,
    watch: Stopwatch()..start(),
  );
  final value = fn();
  _endDecoderTailTrace(runner, trace);
  return value;
}

int _safeActiveBytes() {
  try {
    return MlxMemory.activeBytes();
  } catch (_) {
    return -1;
  }
}

int _safeCacheBytes() {
  try {
    return MlxMemory.cacheBytes();
  } catch (_) {
    return -1;
  }
}

int _safePeakBytesTrace() {
  try {
    return MlxMemory.peakBytes();
  } catch (_) {
    return -1;
  }
}

int _safeResourceCountTrace() {
  try {
    return MlxMemory.resourceCount();
  } catch (_) {
    return -1;
  }
}

int _safeCommitCountTrace() {
  try {
    return MlxMemory.commandBufferCommitCount();
  } catch (_) {
    return -1;
  }
}

int _safeSetDataCountTrace() {
  try {
    return MlxMemory.setDataCount();
  } catch (_) {
    return -1;
  }
}

int _safeSharedBufferCopyCountTrace() {
  try {
    return MlxMemory.sharedBufferCopyCount();
  } catch (_) {
    return -1;
  }
}

int _safeContiguousCopyCountTrace() {
  try {
    return MlxMemory.gpuContiguousCopyCount();
  } catch (_) {
    return -1;
  }
}

int _safeQuantizedContiguousXCountTrace() {
  try {
    return MlxMemory.quantizedContiguousXCount();
  } catch (_) {
    return -1;
  }
}

String _formatSignedBytes(int bytes) {
  if (bytes < 0) return '-${_formatApproxBytes(-bytes)}';
  return '+${_formatApproxBytes(bytes)}';
}

String _formatAllocatorStatsSuffix() {
  try {
    final stats = MlxMemory.allocatorStats();
    final parts = <String>[];
    if (stats.resourceLimit > 0) {
      parts.add('rsrc=${stats.resourceCount}/${stats.resourceLimit}');
    }
    if (stats.cacheCount > 0) {
      parts.add('cacheCount=${stats.cacheCount}');
    }
    if (stats.commandBufferCommitCount > 0) {
      parts.add('commits=${stats.commandBufferCommitCount}');
    }
    if (stats.pendingOutputCount > 0) {
      parts.add('pendingOut=${stats.pendingOutputCount}');
    }
    if (stats.temporaryCount > 0) {
      parts.add('temps=${stats.temporaryCount}');
    }
    if (stats.bufferOpCount > 0) {
      parts.add('bufOps=${stats.bufferOpCount}');
    }
    if (stats.bufferSizeBytes > 0) {
      parts.add('bufSize=${_formatApproxBytes(stats.bufferSizeBytes)}');
    }
    if (stats.streamCount > 0) {
      parts.add('streams=${stats.streamCount}');
    }
    if (stats.setDataCount > 0) {
      parts.add('setData=${stats.setDataCount}');
    }
    if (stats.sharedBufferCopyCount > 0) {
      parts.add('sharedCopy=${stats.sharedBufferCopyCount}');
    }
    if (stats.allocationRequestCount > 0) {
      parts.add('allocReq=${stats.allocationRequestCount}');
    }
    if (stats.cacheReuseHitCount > 0) {
      parts.add('cacheHit=${stats.cacheReuseHitCount}');
    }
    if (stats.newAllocationCount > 0) {
      parts.add('newAlloc=${stats.newAllocationCount}');
    }
    if (stats.heapAllocationCount > 0) {
      parts.add('heapAlloc=${stats.heapAllocationCount}');
    }
    if (stats.deviceAllocationCount > 0) {
      parts.add('deviceAlloc=${stats.deviceAllocationCount}');
    }
    if (stats.commonBinaryAllocationCount > 0 ||
        stats.commonBinarySharedCopyCount > 0) {
      parts.add(
        'bin=${stats.commonBinaryAllocationCount}/${stats.commonBinarySharedCopyCount}',
      );
    }
    if (stats.commonUnaryAllocationCount > 0 ||
        stats.commonUnarySharedCopyCount > 0) {
      parts.add(
        'una=${stats.commonUnaryAllocationCount}/${stats.commonUnarySharedCopyCount}',
      );
    }
    if (stats.commonCopyAllocationCount > 0 ||
        stats.commonCopySharedCopyCount > 0) {
      parts.add(
        'cpy=${stats.commonCopyAllocationCount}/${stats.commonCopySharedCopyCount}',
      );
    }
    if (stats.commonCopyScalarAllocationCount > 0 ||
        stats.commonCopyScalarSharedCopyCount > 0 ||
        stats.commonCopyVectorAllocationCount > 0 ||
        stats.commonCopyVectorSharedCopyCount > 0 ||
        stats.commonCopyGeneralAllocationCount > 0 ||
        stats.commonCopyGeneralSharedCopyCount > 0 ||
        stats.commonCopyGeneralGeneralAllocationCount > 0 ||
        stats.commonCopyGeneralGeneralSharedCopyCount > 0) {
      parts.add(
        'cpyT=${stats.commonCopyScalarAllocationCount}/${stats.commonCopyScalarSharedCopyCount}'
        ',${stats.commonCopyVectorAllocationCount}/${stats.commonCopyVectorSharedCopyCount}'
        ',${stats.commonCopyGeneralAllocationCount}/${stats.commonCopyGeneralSharedCopyCount}'
        ',${stats.commonCopyGeneralGeneralAllocationCount}/${stats.commonCopyGeneralGeneralSharedCopyCount}',
      );
    }
    if (stats.commonCopyGpriAllocationCount > 0 ||
        stats.commonCopyGpriSharedCopyCount > 0 ||
        stats.commonCopyIdxAllocationCount > 0 ||
        stats.commonCopyIdxSharedCopyCount > 0 ||
        stats.commonCopyRopeAllocationCount > 0 ||
        stats.commonCopyRopeSharedCopyCount > 0 ||
        stats.commonCopyMatmulAllocationCount > 0 ||
        stats.commonCopyMatmulSharedCopyCount > 0 ||
        stats.commonCopyHadamardAllocationCount > 0 ||
        stats.commonCopyHadamardSharedCopyCount > 0) {
      parts.add(
        'cpyG=${stats.commonCopyGpriAllocationCount}/${stats.commonCopyGpriSharedCopyCount}'
        ',${stats.commonCopyIdxAllocationCount}/${stats.commonCopyIdxSharedCopyCount}'
        ',${stats.commonCopyRopeAllocationCount}/${stats.commonCopyRopeSharedCopyCount}'
        ',${stats.commonCopyMatmulAllocationCount}/${stats.commonCopyMatmulSharedCopyCount}'
        ',${stats.commonCopyHadamardAllocationCount}/${stats.commonCopyHadamardSharedCopyCount}',
      );
    }
    if (stats.commonTernaryAllocationCount > 0 ||
        stats.commonTernarySharedCopyCount > 0) {
      parts.add(
        'ter=${stats.commonTernaryAllocationCount}/${stats.commonTernarySharedCopyCount}',
      );
    }
    if (stats.gpuPrimitiveAllocationCount > 0 ||
        stats.gpuPrimitiveSharedCopyCount > 0) {
      parts.add(
        'gpri=${stats.gpuPrimitiveAllocationCount}/${stats.gpuPrimitiveSharedCopyCount}',
      );
    }
    if (stats.commonCopyGpriAstypeAllocationCount > 0 ||
        stats.commonCopyGpriAstypeSharedCopyCount > 0 ||
        stats.commonCopyGpriContiguousAllocationCount > 0 ||
        stats.commonCopyGpriContiguousSharedCopyCount > 0 ||
        stats.commonCopyGpriFullAllocationCount > 0 ||
        stats.commonCopyGpriFullSharedCopyCount > 0 ||
        stats.commonCopyGpriSliceUpdateAllocationCount > 0 ||
        stats.commonCopyGpriSliceUpdateSharedCopyCount > 0 ||
        stats.commonCopyGpriDynamicSliceUpdateAllocationCount > 0 ||
        stats.commonCopyGpriDynamicSliceUpdateSharedCopyCount > 0) {
      parts.add(
        'gpriT=${stats.commonCopyGpriAstypeAllocationCount}/${stats.commonCopyGpriAstypeSharedCopyCount}'
        ',${stats.commonCopyGpriContiguousAllocationCount}/${stats.commonCopyGpriContiguousSharedCopyCount}'
        ',${stats.commonCopyGpriFullAllocationCount}/${stats.commonCopyGpriFullSharedCopyCount}'
        ',${stats.commonCopyGpriSliceUpdateAllocationCount}/${stats.commonCopyGpriSliceUpdateSharedCopyCount}'
        ',${stats.commonCopyGpriDynamicSliceUpdateAllocationCount}/${stats.commonCopyGpriDynamicSliceUpdateSharedCopyCount}',
      );
    }
    if (stats.gpuContiguousCopyCount > 0) {
      parts.add('contig=${stats.gpuContiguousCopyCount}');
    }
    if (stats.quantizedContiguousXCount > 0 ||
        stats.quantizedContiguousWCount > 0 ||
        stats.quantizedContiguousScalesCount > 0 ||
        stats.quantizedContiguousBiasesCount > 0 ||
        stats.quantizedContiguousIndicesCount > 0) {
      parts.add(
        'qcontig=${stats.quantizedContiguousXCount}/${stats.quantizedContiguousWCount}/${stats.quantizedContiguousScalesCount}/${stats.quantizedContiguousBiasesCount}/${stats.quantizedContiguousIndicesCount}',
      );
    }
    if (stats.metalNormAllocationCount > 0 ||
        stats.metalNormSharedCopyCount > 0) {
      parts.add(
        'norm=${stats.metalNormAllocationCount}/${stats.metalNormSharedCopyCount}',
      );
    }
    if (stats.metalMatmulAllocationCount > 0 ||
        stats.metalMatmulSharedCopyCount > 0) {
      parts.add(
        'mm=${stats.metalMatmulAllocationCount}/${stats.metalMatmulSharedCopyCount}',
      );
    }
    if (stats.metalQuantizedAllocationCount > 0 ||
        stats.metalQuantizedSharedCopyCount > 0) {
      parts.add(
        'qq=${stats.metalQuantizedAllocationCount}/${stats.metalQuantizedSharedCopyCount}',
      );
    }
    if (stats.metalSdpaAllocationCount > 0 ||
        stats.metalSdpaSharedCopyCount > 0) {
      parts.add(
        'sdpa=${stats.metalSdpaAllocationCount}/${stats.metalSdpaSharedCopyCount}',
      );
    }
    if (stats.metalReduceAllocationCount > 0 ||
        stats.metalReduceSharedCopyCount > 0) {
      parts.add(
        'red=${stats.metalReduceAllocationCount}/${stats.metalReduceSharedCopyCount}',
      );
    }
    if (stats.metalIndexingAllocationCount > 0 ||
        stats.metalIndexingSharedCopyCount > 0) {
      parts.add(
        'idx=${stats.metalIndexingAllocationCount}/${stats.metalIndexingSharedCopyCount}',
      );
    }
    if (stats.metalIndexConcatAllocationCount > 0 ||
        stats.metalIndexConcatSharedCopyCount > 0 ||
        stats.metalIndexGatherAllocationCount > 0 ||
        stats.metalIndexGatherSharedCopyCount > 0 ||
        stats.metalIndexGatherAxisAllocationCount > 0 ||
        stats.metalIndexGatherAxisSharedCopyCount > 0 ||
        stats.metalIndexDynamicOffsetAllocationCount > 0 ||
        stats.metalIndexDynamicOffsetSharedCopyCount > 0) {
      parts.add(
        'idxT=${stats.metalIndexConcatAllocationCount}/${stats.metalIndexConcatSharedCopyCount}'
        ',${stats.metalIndexGatherAllocationCount}/${stats.metalIndexGatherSharedCopyCount}'
        ',${stats.metalIndexGatherAxisAllocationCount}/${stats.metalIndexGatherAxisSharedCopyCount}'
        ',${stats.metalIndexDynamicOffsetAllocationCount}/${stats.metalIndexDynamicOffsetSharedCopyCount}',
      );
    }
    if (stats.metalCopyAllocationCount > 0 ||
        stats.metalCopySharedCopyCount > 0) {
      parts.add(
        'cpy2=${stats.metalCopyAllocationCount}/${stats.metalCopySharedCopyCount}',
      );
    }
    if (stats.metalDirectCopyAllocationCount > 0 ||
        stats.metalDirectCopySharedCopyCount > 0 ||
        stats.metalRopeCopyAllocationCount > 0 ||
        stats.metalRopeCopySharedCopyCount > 0 ||
        stats.metalScanCopyAllocationCount > 0 ||
        stats.metalScanCopySharedCopyCount > 0 ||
        stats.metalPrimitiveCopyAllocationCount > 0 ||
        stats.metalPrimitiveCopySharedCopyCount > 0) {
      parts.add(
        'cpy2T=${stats.metalDirectCopyAllocationCount}/${stats.metalDirectCopySharedCopyCount}'
        ',${stats.metalRopeCopyAllocationCount}/${stats.metalRopeCopySharedCopyCount}'
        ',${stats.metalScanCopyAllocationCount}/${stats.metalScanCopySharedCopyCount}'
        ',${stats.metalPrimitiveCopyAllocationCount}/${stats.metalPrimitiveCopySharedCopyCount}',
      );
    }
    if (stats.metalReshapeCopyCount > 0 || stats.metalReshapeSharedCount > 0) {
      parts.add('reshape=${stats.metalReshapeCopyCount}/${stats.metalReshapeSharedCount}');
    }
    if (stats.donationRejectNotUniqueCount > 0 ||
        stats.donationRejectDescNotUniqueCount > 0 ||
        stats.donationRejectDataNotUniqueCount > 0 ||
        stats.donationRejectItemsizeCount > 0 ||
        stats.donationRejectOversizeCount > 0 ||
        stats.donationRejectLayoutCount > 0) {
      parts.add(
        'donReject=${stats.donationRejectNotUniqueCount}/${stats.donationRejectDescNotUniqueCount}/${stats.donationRejectDataNotUniqueCount}/${stats.donationRejectItemsizeCount}/${stats.donationRejectOversizeCount}/${stats.donationRejectLayoutCount}',
      );
    }
    if (stats.commonCopyRejectDescNotUniqueCount > 0 ||
        stats.commonCopyRejectDataNotUniqueCount > 0) {
      parts.add(
        'rejCpy=${stats.commonCopyRejectDescNotUniqueCount}/${stats.commonCopyRejectDataNotUniqueCount}',
      );
    }
    if (stats.commonBinaryRejectDescNotUniqueCount > 0 ||
        stats.commonBinaryRejectDataNotUniqueCount > 0) {
      parts.add(
        'rejBin=${stats.commonBinaryRejectDescNotUniqueCount}/${stats.commonBinaryRejectDataNotUniqueCount}',
      );
    }
    if (stats.commonUnaryRejectDescNotUniqueCount > 0 ||
        stats.commonUnaryRejectDataNotUniqueCount > 0) {
      parts.add(
        'rejUna=${stats.commonUnaryRejectDescNotUniqueCount}/${stats.commonUnaryRejectDataNotUniqueCount}',
      );
    }
    if (stats.commonBinaryDataNotUniqueScalarVectorCount > 0 ||
        stats.commonBinaryDataNotUniqueVectorScalarCount > 0 ||
        stats.commonBinaryDataNotUniqueVectorVectorCount > 0 ||
        stats.commonBinaryDataNotUniqueGeneralCount > 0) {
      parts.add(
        'binData=${stats.commonBinaryDataNotUniqueScalarVectorCount}/${stats.commonBinaryDataNotUniqueVectorScalarCount}/${stats.commonBinaryDataNotUniqueVectorVectorCount}/${stats.commonBinaryDataNotUniqueGeneralCount}',
      );
    }
    if (stats.commonBinaryAddDataNotUniqueVectorVectorCount > 0 ||
        stats.commonBinaryAddDataNotUniqueGeneralCount > 0) {
      parts.add(
        'addData=${stats.commonBinaryAddDataNotUniqueVectorVectorCount}/${stats.commonBinaryAddDataNotUniqueGeneralCount}',
      );
    }
    if (stats.commonBinaryMultiplyDataNotUniqueVectorVectorCount > 0 ||
        stats.commonBinaryMultiplyDataNotUniqueGeneralCount > 0) {
      parts.add(
        'mulData=${stats.commonBinaryMultiplyDataNotUniqueVectorVectorCount}/${stats.commonBinaryMultiplyDataNotUniqueGeneralCount}',
      );
    }
    if (stats.cacheLimitBytes > 0) {
      parts.add('cacheLimit=${_formatApproxBytes(stats.cacheLimitBytes)}');
    }
    if (stats.wiredLimitBytes > 0) {
      parts.add('wiredLimit=${_formatApproxBytes(stats.wiredLimitBytes)}');
    }
    if (stats.memoryLimitBytes > 0) {
      parts.add('memLimit=${_formatApproxBytes(stats.memoryLimitBytes)}');
    }
    return parts.isEmpty ? '' : ' ${parts.join(' ')}';
  } catch (_) {
    return '';
  }
}
