part of 'paddle_ocr_vl.dart';

extension PaddleOcrVlVisionMlp on PaddleOcrVlRunner {
  // -----------------------------------------------------------------------
  // Vision MLP (fc1 → GELU → fc2)
  // -----------------------------------------------------------------------

  MlxArray _visionMlp(_VisionBlockWeights block, MlxArray input) {
    return _chunkedTokenMlp(
      input,
      chunkSize: config.visionMlpChunkSizeForCurrentPlatform,
      applyChunk: (chunk) {
        final h = block.fc1.apply(chunk);
        final activated = _gelu(h);
        h.close();
        final out = block.fc2.apply(activated);
        activated.close();
        return out;
      },
    );
  }

  // -----------------------------------------------------------------------
  // Spatial-merge projector
  // -----------------------------------------------------------------------

  /// Performs 2×2 spatial merging then projects to LM hidden size.
  ///
  /// Input `hidden`: `[gridH*gridW, visionHidden]`
  /// Output:         `[mergedTokens, lmHidden]`
  MlxArray _spatialMergeProject(
    MlxArray hidden,
    int gridH,
    int gridW,
    _VisionConfig vCfg,
  ) {
    final m = vCfg.spatialMergeSize; // 2
    final mergedH = gridH ~/ m;
    final mergedW = gridW ~/ m;
    final proj = _visionWeights.projector;

    final normed = _visionLayerNorm(
      hidden,
      weight: proj.preNormWeight,
      bias: proj.preNormBias,
      eps: vCfg.layerNormEps,
    );

    final grid = normed.reshape([1, gridH, gridW, vCfg.hiddenSize]);
    normed.close();
    final reshaped = grid.reshape([1, mergedH, m, mergedW, m, vCfg.hiddenSize]);
    grid.close();
    final transposed = reshaped.transposeAxes([0, 1, 3, 2, 4, 5]);
    reshaped.close();
    final flat = transposed.reshape([
      mergedH * mergedW,
      m * m * vCfg.hiddenSize,
    ]);
    transposed.close();

    final out = _chunkedTokenMlp(
      flat,
      chunkSize: config.visionProjectorChunkSizeForCurrentPlatform,
      applyChunk: (chunk) {
        final h = proj.linear1.apply(chunk);
        final activated = _geluDefault(h);
        h.close();
        final projected = proj.linear2.apply(activated);
        activated.close();
        return projected;
      },
    );
    flat.close();
    MlxRuntime.evalAll([out]);
    return out;
  }

  MlxArray _chunkedTokenMlp(
    MlxArray input, {
    required int chunkSize,
    required MlxArray Function(MlxArray chunk) applyChunk,
  }) {
    final total = input.shape[0];
    if (chunkSize <= 0 || total <= chunkSize) {
      return applyChunk(input);
    }

    MlxArray? combined;
    for (var start = 0; start < total; start += chunkSize) {
      final end = math.min(start + chunkSize, total);
      final chunk = input.slice(start: [start, 0], stop: [end, input.shape[1]]);
      final out = applyChunk(chunk);
      chunk.close();
      combined ??= MlxArray.zeros([total, out.shape[1]], dtype: out.dtype);
      final updated = combined.sliceUpdate(
        out,
        start: [start, 0],
        stop: [end, out.shape[1]],
      );
      out.close();
      combined.close();
      combined = updated;
    }
    return combined!;
  }

  // -----------------------------------------------------------------------
  // GELU activation — MLX "precise" approximation
  //
  // Matches `nn.GELU(approx="precise")` in the Python MLX runtime:
  //   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
  // -----------------------------------------------------------------------

  MlxArray _gelu(MlxArray x) {
    final lut = _geluLut(x);
    if (lut != null) return lut;

    final cubicCoeff = MlxArray.fromFloat32List(
      [0.044715],
      shape: [1],
    ).astype(x.dtype);
    final scale = MlxArray.fromFloat32List(
      [math.sqrt(2 / math.pi)],
      shape: [1],
    ).astype(x.dtype);
    final half = MlxArray.fromFloat32List([0.5], shape: [1]).astype(x.dtype);
    final one = MlxArray.fromFloat32List([1.0], shape: [1]).astype(x.dtype);
    try {
      final xSquared = x * x;
      final xCubed = xSquared * x;
      xSquared.close();
      final cubicTerm = xCubed * cubicCoeff;
      xCubed.close();
      final inner = mx.add(x, cubicTerm);
      cubicTerm.close();
      final scaled = inner * scale;
      inner.close();
      final tanhVal = scaled.tanh();
      scaled.close();
      final sum = mx.add(one, tanhVal);
      tanhVal.close();
      final left = x * half;
      final result = left * sum;
      left.close();
      sum.close();
      return result;
    } finally {
      cubicCoeff.close();
      scale.close();
      half.close();
      one.close();
    }
  }

  MlxArray _geluDefault(MlxArray x) {
    final sqrt2 = MlxArray.fromFloat32List(
      [math.sqrt(2.0)],
      shape: [1],
    ).astype(x.dtype);
    final two = MlxArray.fromFloat32List([2.0], shape: [1]).astype(x.dtype);
    final one = MlxArray.fromFloat32List([1.0], shape: [1]).astype(x.dtype);
    try {
      final scaled = x / sqrt2;
      final erfVal = scaled.erf();
      scaled.close();
      final sum = mx.add(one, erfVal);
      erfVal.close();
      final numer = x * sum;
      sum.close();
      final result = numer / two;
      numer.close();
      return result;
    } finally {
      sqrt2.close();
      two.close();
      one.close();
    }
  }

  MlxArray? _geluLut(MlxArray x) {
    if (!MlxMetal.isAvailable() || x.dtype != MlxDType.MLX_BFLOAT16) {
      return null;
    }
    final flat = x.reshape([x.size]);
    final kernel = _visionGeluLutKernel ??= mx.fast.metalKernel(
      'vision_gelu_bf16_lut',
      ['x', 'lut'],
      ['out'],
      r'''
auto idx = thread_position_in_grid.x;
if (idx >= x_shape[0]) return;
ushort bits = as_type<ushort>(x[idx]);
out[idx] = lut[bits];
''',
    );
    final config = mx.fast.metalConfig();
    config.addOutputArg([x.size], MlxDType.MLX_FLOAT32);
    config.setGrid(x.size, 1, 1);
    config.setThreadGroup(math.min(256, math.max(1, x.size)), 1, 1);
    final outputs = kernel.apply([flat, _visionGeluLutArray()], config);
    flat.close();
    final out32 = outputs.first.reshape(x.shape);
    outputs.first.close();
    final cast = out32.astype(x.dtype);
    out32.close();
    return cast;
  }

  MlxArray _visionLayerNorm(
    MlxArray input, {
    required MlxArray weight,
    required MlxArray bias,
    required double eps,
  }) {
    return mx.fast.layerNorm(input, weight: weight, bias: bias, eps: eps);
  }
}

MlxMetalKernel? _visionGeluLutKernel;
