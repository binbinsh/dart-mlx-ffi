part of '../stable_api.dart';

final class MlxModule {
  const MlxModule._();

  /// Random submodule, similar to Python `mx.random`.
  MlxRandomModule get random => const MlxRandomModule._();

  /// FFT submodule.
  MlxFftModule get fft => const MlxFftModule._();

  /// Linear algebra submodule.
  MlxLinalgModule get linalg => const MlxLinalgModule._();

  /// IO submodule.
  MlxIoModule get io => const MlxIoModule._();

  /// Fast submodule.
  MlxFastModule get fast => const MlxFastModule._();

  /// Quantization submodule.
  MlxQuantModule get quant => const MlxQuantModule._();

  /// Stream helpers.
  MlxStreamModule get stream => const MlxStreamModule._();

  /// Distributed helpers.
  MlxDistributedModule get distributed => const MlxDistributedModule._();

  /// Metal runtime submodule.
  MlxMetalModule get metal => const MlxMetalModule._();

  /// Memory/runtime submodule.
  MlxMemoryModule get memory => const MlxMemoryModule._();

  /// Creates a zero-filled array.
  MlxArray zeros(
    List<int> shape, {
    MlxDType dtype = raw.mlx_dtype_.MLX_FLOAT32,
  }) => MlxArray.zeros(shape, dtype: dtype);

  /// Creates a one-filled array.
  MlxArray ones(
    List<int> shape, {
    MlxDType dtype = raw.mlx_dtype_.MLX_FLOAT32,
  }) => MlxArray.ones(shape, dtype: dtype);

  /// Creates a scalar-filled array.
  MlxArray full(
    List<int> shape,
    double value, {
    MlxDType dtype = raw.mlx_dtype_.MLX_FLOAT32,
  }) => MlxArray.full(shape, value, dtype: dtype);

  /// Creates a 1D evenly spaced range.
  MlxArray arange(
    double start,
    double stop,
    double step, {
    MlxDType dtype = raw.mlx_dtype_.MLX_FLOAT32,
  }) => MlxArray.arange(start, stop, step, dtype: dtype);

  /// Elementwise addition.
  MlxArray add(MlxArray a, MlxArray b) => MlxOps.add(a, b);

  /// Elementwise subtraction.
  MlxArray subtract(MlxArray a, MlxArray b) => MlxOps.subtract(a, b);

  /// Elementwise multiplication.
  MlxArray multiply(MlxArray a, MlxArray b) => MlxOps.multiply(a, b);

  /// Elementwise division.
  MlxArray divide(MlxArray a, MlxArray b) => MlxOps.divide(a, b);

  /// Matrix multiplication.
  MlxArray matmul(MlxArray a, MlxArray b) => MlxOps.matmul(a, b);

  /// Matrix multiply with additive bias: `alpha * (a @ b) + beta * c`.
  MlxArray addmm(
    MlxArray c,
    MlxArray a,
    MlxArray b, {
    double alpha = 1,
    double beta = 1,
  }) => MlxOps.addmm(c, a, b, alpha: alpha, beta: beta);

  /// Casts an array to a different dtype.
  MlxArray astype(MlxArray input, MlxDType dtype) =>
      MlxOps.astype(input, dtype);

  /// Absolute value.
  MlxArray abs(MlxArray input) => MlxOps.abs(input);

  /// Exponential.
  MlxArray exp(MlxArray input) => MlxOps.exp(input);

  /// Natural logarithm.
  MlxArray log(MlxArray input) => MlxOps.log(input);

  /// Sine.
  MlxArray sin(MlxArray input) => MlxOps.sin(input);

  /// Cosine.
  MlxArray cos(MlxArray input) => MlxOps.cos(input);

  /// Sum reduction.
  MlxArray sum(MlxArray input, {int? axis, bool keepDims = false}) =>
      MlxOps.sum(input, axis: axis, keepDims: keepDims);

  /// Mean reduction.
  MlxArray mean(MlxArray input, {int? axis, bool keepDims = false}) =>
      MlxOps.mean(input, axis: axis, keepDims: keepDims);

  /// Variance reduction.
  MlxArray variance(
    MlxArray input, {
    int? axis,
    List<int>? axes,
    bool keepDims = false,
    int ddof = 0,
  }) => MlxOps.variance(
    input,
    axis: axis,
    axes: axes,
    keepDims: keepDims,
    ddof: ddof,
  );

  /// Log-sum-exp reduction.
  MlxArray logSumExp(MlxArray input, {int? axis, bool keepDims = false}) =>
      MlxOps.logSumExp(input, axis: axis, keepDims: keepDims);

  /// Softmax.
  MlxArray softmax(MlxArray input, {int? axis, bool precise = false}) =>
      MlxOps.softmax(input, axis: axis, precise: precise);

  /// Top-k values.
  MlxArray topK(MlxArray input, int k, {int? axis}) =>
      MlxOps.topK(input, k, axis: axis);

  /// Takes values by flat or axis-based indices.
  MlxArray take(MlxArray input, MlxArray indices, {int? axis}) =>
      MlxTensor.take(input, indices, axis: axis);

  /// Takes values along a specific axis.
  MlxArray takeAlongAxis(
    MlxArray input,
    MlxArray indices, {
    required int axis,
  }) => MlxTensor.takeAlongAxis(input, indices, axis: axis);

  /// General gather over multiple index tensors.
  MlxArray gather(
    MlxArray input,
    List<MlxArray> indices, {
    required List<int> axes,
    required List<int> sliceSizes,
  }) => MlxTensor.gather(input, indices, axes: axes, sliceSizes: sliceSizes);

  /// Single-index gather helper.
  MlxArray gatherSingle(
    MlxArray input,
    MlxArray indices, {
    required int axis,
    required List<int> sliceSizes,
  }) => MlxTensor.gatherSingle(
    input,
    indices,
    axis: axis,
    sliceSizes: sliceSizes,
  );

  /// Matrix product with optional matrix-level gather.
  MlxArray gatherMm(
    MlxArray a,
    MlxArray b, {
    MlxArray? lhsIndices,
    MlxArray? rhsIndices,
    bool sortedIndices = false,
  }) => MlxTensor.gatherMm(
    a,
    b,
    lhsIndices: lhsIndices,
    rhsIndices: rhsIndices,
    sortedIndices: sortedIndices,
  );

  /// Broadcasts arrays to a shared shape.
  List<MlxArray> broadcastArrays(List<MlxArray> inputs) =>
      MlxTensor.broadcastArrays(inputs);

  /// Splits an array at explicit section boundaries.
  List<MlxArray> splitSections(
    MlxArray input,
    List<int> indices, {
    int axis = 0,
  }) => MlxTensor.splitSections(input, indices, axis: axis);

  /// Segmented matrix multiply.
  MlxArray segmentedMm(MlxArray a, MlxArray b, MlxArray segments) =>
      MlxTensor.segmentedMm(a, b, segments);

  /// Tile-masked matrix multiply.
  MlxArray blockMaskedMm(
    MlxArray a,
    MlxArray b, {
    required int blockSize,
    MlxArray? maskOut,
    MlxArray? maskLhs,
    MlxArray? maskRhs,
  }) => MlxTensor.blockMaskedMm(
    a,
    b,
    blockSize: blockSize,
    maskOut: maskOut,
    maskLhs: maskLhs,
    maskRhs: maskRhs,
  );

  /// Slices an array with explicit vectors.
  MlxArray slice(
    MlxArray input, {
    required List<int> start,
    required List<int> stop,
    List<int>? strides,
  }) => MlxTensor.slice(input, start: start, stop: stop, strides: strides);

  /// Dynamic slice using tensor start indices.
  MlxArray sliceDynamic(
    MlxArray input, {
    required MlxArray start,
    required List<int> axes,
    required List<int> sliceSize,
  }) => MlxTensor.sliceDynamic(
    input,
    start: start,
    axes: axes,
    sliceSize: sliceSize,
  );

  /// Slice update helper.
  MlxArray sliceUpdate(
    MlxArray source,
    MlxArray update, {
    required List<int> start,
    required List<int> stop,
    List<int>? strides,
  }) => MlxTensor.sliceUpdate(
    source,
    update,
    start: start,
    stop: stop,
    strides: strides,
  );

  /// Dynamic slice update helper.
  MlxArray sliceUpdateDynamic(
    MlxArray source,
    MlxArray update, {
    required MlxArray start,
    required List<int> axes,
  }) => MlxTensor.sliceUpdateDynamic(source, update, start: start, axes: axes);

  /// Einstein summation.
  MlxArray einsum(String subscripts, List<MlxArray> operands) =>
      MlxTensor.einsum(subscripts, operands);

  /// Tensor contraction helper.
  MlxArray tensordot(
    MlxArray a,
    MlxArray b, {
    int? axis,
    List<int>? axesA,
    List<int>? axesB,
  }) => MlxTensor.tensordot(a, b, axis: axis, axesA: axesA, axesB: axesB);

  MlxArray diag(MlxArray input, {int k = 0}) => MlxExtra.diag(input, k: k);

  MlxArray diagonal(
    MlxArray input, {
    int offset = 0,
    int axis1 = 0,
    int axis2 = 1,
  }) => MlxExtra.diagonal(input, offset: offset, axis1: axis1, axis2: axis2);

  MlxArray kron(MlxArray a, MlxArray b) => MlxExtra.kron(a, b);

  List<MlxArray> meshgrid(
    List<MlxArray> inputs, {
    bool sparse = false,
    String indexing = 'xy',
  }) => MlxExtra.meshgrid(inputs, sparse: sparse, indexing: indexing);

  MlxArray partition(MlxArray input, int kth, {int? axis}) =>
      MlxExtra.partition(input, kth, axis: axis);

  MlxArray scatter(
    MlxArray input,
    List<MlxArray> indices,
    MlxArray updates, {
    required List<int> axes,
  }) => MlxExtra.scatter(input, indices, updates, axes: axes);

  MlxArray scatterAdd(
    MlxArray input,
    List<MlxArray> indices,
    MlxArray updates, {
    required List<int> axes,
  }) => MlxExtra.scatterAdd(input, indices, updates, axes: axes);

  MlxArray scatterMax(
    MlxArray input,
    List<MlxArray> indices,
    MlxArray updates, {
    required List<int> axes,
  }) => MlxExtra.scatterMax(input, indices, updates, axes: axes);

  MlxArray scatterMin(
    MlxArray input,
    List<MlxArray> indices,
    MlxArray updates, {
    required List<int> axes,
  }) => MlxExtra.scatterMin(input, indices, updates, axes: axes);

  MlxArray scatterProd(
    MlxArray input,
    List<MlxArray> indices,
    MlxArray updates, {
    required List<int> axes,
  }) => MlxExtra.scatterProd(input, indices, updates, axes: axes);

  MlxArray scatterSingle(
    MlxArray input,
    MlxArray indices,
    MlxArray updates, {
    required int axis,
  }) => MlxExtra.scatterSingle(input, indices, updates, axis: axis);

  MlxArray scatterAddSingle(
    MlxArray input,
    MlxArray indices,
    MlxArray updates, {
    required int axis,
  }) => MlxExtra.scatterAddSingle(input, indices, updates, axis: axis);

  MlxArray scatterMaxSingle(
    MlxArray input,
    MlxArray indices,
    MlxArray updates, {
    required int axis,
  }) => MlxExtra.scatterMaxSingle(input, indices, updates, axis: axis);

  MlxArray scatterMinSingle(
    MlxArray input,
    MlxArray indices,
    MlxArray updates, {
    required int axis,
  }) => MlxExtra.scatterMinSingle(input, indices, updates, axis: axis);

  MlxArray scatterProdSingle(
    MlxArray input,
    MlxArray indices,
    MlxArray updates, {
    required int axis,
  }) => MlxExtra.scatterProdSingle(input, indices, updates, axis: axis);

  /// Flattens a range of axes.
  MlxArray flatten(MlxArray input, {int startAxis = 0, int endAxis = -1}) =>
      MlxTensor.flatten(input, startAxis: startAxis, endAxis: endAxis);

  /// Moves one axis to a new position.
  MlxArray moveaxis(MlxArray input, int source, int destination) =>
      MlxTensor.moveaxis(input, source, destination);

  /// Swaps two axes.
  MlxArray swapaxes(MlxArray input, int axis1, int axis2) =>
      MlxTensor.swapaxes(input, axis1, axis2);

  /// Explicit axis transpose.
  MlxArray transposeAxes(MlxArray input, List<int> axes) =>
      MlxTensor.transposeAxes(input, axes);

  /// Repeats an array according to [reps].
  MlxArray tile(MlxArray input, List<int> reps) => MlxTensor.tile(input, reps);

  /// Pads an array.
  MlxArray pad(
    MlxArray input, {
    List<int>? axes,
    required List<int> lowPads,
    required List<int> highPads,
    MlxArray? padValue,
    String mode = 'constant',
  }) => MlxTensor.pad(
    input,
    axes: axes,
    lowPads: lowPads,
    highPads: highPads,
    padValue: padValue,
    mode: mode,
  );

  /// Symmetric pad helper.
  MlxArray padSymmetric(
    MlxArray input,
    int padWidth, {
    MlxArray? padValue,
    String mode = 'constant',
  }) => MlxTensor.padSymmetric(input, padWidth, padValue: padValue, mode: mode);

  /// Unflattens an axis into a shape.
  MlxArray unflatten(
    MlxArray input, {
    required int axis,
    required List<int> shape,
  }) => MlxTensor.unflatten(input, axis: axis, shape: shape);

  /// 1D convolution.
  MlxArray conv1d(
    MlxArray input,
    MlxArray weight, {
    int stride = 1,
    int padding = 0,
    int dilation = 1,
    int groups = 1,
  }) => MlxConv.conv1d(
    input,
    weight,
    stride: stride,
    padding: padding,
    dilation: dilation,
    groups: groups,
  );

  /// 2D convolution.
  MlxArray conv2d(
    MlxArray input,
    MlxArray weight, {
    List<int> stride = const [1, 1],
    List<int> padding = const [0, 0],
    List<int> dilation = const [1, 1],
    int groups = 1,
  }) => MlxConv.conv2d(
    input,
    weight,
    stride: stride,
    padding: padding,
    dilation: dilation,
    groups: groups,
  );

  /// 3D convolution.
  MlxArray conv3d(
    MlxArray input,
    MlxArray weight, {
    List<int> stride = const [1, 1, 1],
    List<int> padding = const [0, 0, 0],
    List<int> dilation = const [1, 1, 1],
    int groups = 1,
  }) => MlxConv.conv3d(
    input,
    weight,
    stride: stride,
    padding: padding,
    dilation: dilation,
    groups: groups,
  );

  /// General convolution.
  MlxArray convGeneral(
    MlxArray input,
    MlxArray weight, {
    List<int> stride = const [],
    List<int>? padding,
    List<int>? paddingLo,
    List<int>? paddingHi,
    List<int> kernelDilation = const [],
    List<int> inputDilation = const [],
    int groups = 1,
    bool flip = false,
  }) => MlxConv.convGeneral(
    input,
    weight,
    stride: stride,
    padding: padding,
    paddingLo: paddingLo,
    paddingHi: paddingHi,
    kernelDilation: kernelDilation,
    inputDilation: inputDilation,
    groups: groups,
    flip: flip,
  );

  /// 1D transposed convolution.
  MlxArray convTranspose1d(
    MlxArray input,
    MlxArray weight, {
    int stride = 1,
    int padding = 0,
    int dilation = 1,
    int outputPadding = 0,
    int groups = 1,
  }) => MlxConv.convTranspose1d(
    input,
    weight,
    stride: stride,
    padding: padding,
    dilation: dilation,
    outputPadding: outputPadding,
    groups: groups,
  );

  /// 2D transposed convolution.
  MlxArray convTranspose2d(
    MlxArray input,
    MlxArray weight, {
    List<int> stride = const [1, 1],
    List<int> padding = const [0, 0],
    List<int> dilation = const [1, 1],
    List<int> outputPadding = const [0, 0],
    int groups = 1,
  }) => MlxConv.convTranspose2d(
    input,
    weight,
    stride: stride,
    padding: padding,
    dilation: dilation,
    outputPadding: outputPadding,
    groups: groups,
  );

  /// 3D transposed convolution.
  MlxArray convTranspose3d(
    MlxArray input,
    MlxArray weight, {
    List<int> stride = const [1, 1, 1],
    List<int> padding = const [0, 0, 0],
    List<int> dilation = const [1, 1, 1],
    List<int> outputPadding = const [0, 0, 0],
    int groups = 1,
  }) => MlxConv.convTranspose3d(
    input,
    weight,
    stride: stride,
    padding: padding,
    dilation: dilation,
    outputPadding: outputPadding,
    groups: groups,
  );

  /// Elementwise equality.
  MlxArray equal(MlxArray a, MlxArray b) => MlxOps.equal(a, b);

  /// Select values from [x] or [y] according to [condition].
  MlxArray where(MlxArray condition, MlxArray x, MlxArray y) =>
      MlxOps.where(condition, x, y);

  /// Concatenates arrays along [axis].
  MlxArray concatenate(List<MlxArray> arrays, {int axis = 0}) =>
      MlxOps.concatenate(arrays, axis: axis);

  /// Stacks arrays along [axis].
  MlxArray stack(List<MlxArray> arrays, {int axis = 0}) =>
      MlxOps.stack(arrays, axis: axis);

  /// Broadcasts [input] to [shape].
  MlxArray broadcastTo(MlxArray input, List<int> shape) =>
      MlxOps.broadcastTo(input, shape);

  /// Expands a dimension at [axis].
  MlxArray expandDims(MlxArray input, int axis) =>
      MlxOps.expandDims(input, axis);

  /// Removes singleton dimensions.
  MlxArray squeeze(MlxArray input) => MlxOps.squeeze(input);

  /// Clips values to `[min, max]`.
  MlxArray clip(MlxArray input, {double? min, double? max}) =>
      MlxOps.clip(input, min: min, max: max);

  /// Elementwise minimum.
  MlxArray minimum(MlxArray a, MlxArray b) => MlxOps.minimum(a, b);

  /// Elementwise maximum.
  MlxArray maximum(MlxArray a, MlxArray b) => MlxOps.maximum(a, b);

  /// Returns argmax indices.
  MlxArray argmax(MlxArray input, {int? axis, bool keepDims = false}) =>
      MlxOps.argmax(input, axis: axis, keepDims: keepDims);

  /// Returns argmin indices.
  MlxArray argmin(MlxArray input, {int? axis, bool keepDims = false}) =>
      MlxOps.argmin(input, axis: axis, keepDims: keepDims);

  /// Returns sorted values.
  MlxArray sort(MlxArray input, {int? axis}) => MlxOps.sort(input, axis: axis);

  /// Returns sort indices.
  MlxArray argsort(MlxArray input, {int? axis}) =>
      MlxOps.argsort(input, axis: axis);

  /// Batch-evaluates a list of arrays.
  void evalAll(List<MlxArray> arrays) => MlxRuntime.evalAll(arrays);

  /// Schedules asynchronous evaluation for a list of arrays.
  void asyncEvalAll(List<MlxArray> arrays) => MlxRuntime.asyncEvalAll(arrays);
}

/// Module-style random namespace.
final class MlxRandomModule {
  const MlxRandomModule._();

  /// Creates a random key.
  MlxArray key(int seed) => MlxRandom.key(seed);

  /// Splits a key into two keys.
  MlxRandomSplit split(MlxArray key) => MlxRandom.split(key);

  /// Samples a uniform random array.
  MlxArray uniform(
    List<int> shape, {
    double low = 0,
    double high = 1,
    MlxDType dtype = raw.mlx_dtype_.MLX_FLOAT32,
  }) => MlxRandom.uniform(shape, low: low, high: high, dtype: dtype);

  /// Samples a normal random array.
  MlxArray normal(
    List<int> shape, {
    double loc = 0,
    double scale = 1,
    MlxDType dtype = raw.mlx_dtype_.MLX_FLOAT32,
  }) => MlxRandom.normal(shape, loc: loc, scale: scale, dtype: dtype);

  /// Samples Bernoulli values.
  MlxArray bernoulli(MlxArray probability, {List<int>? shape, MlxArray? key}) =>
      MlxRandom.bernoulli(probability, shape: shape, key: key);

  /// Samples categorical indices.
  MlxArray categorical(
    MlxArray logits, {
    int axis = -1,
    List<int>? shape,
    int? numSamples,
    MlxArray? key,
  }) => MlxRandom.categorical(
    logits,
    axis: axis,
    shape: shape,
    numSamples: numSamples,
    key: key,
  );

  /// Permutes an input along [axis].
  MlxArray permutation(MlxArray input, {int axis = 0, MlxArray? key}) =>
      MlxRandom.permutation(input, axis: axis, key: key);

  /// Returns a permutation of `0..n-1`.
  MlxArray permutationArange(int n, {MlxArray? key}) =>
      MlxRandom.permutationArange(n, key: key);

  /// Samples a Gumbel random array.
  MlxArray gumbel(
    List<int> shape, {
    MlxDType dtype = raw.mlx_dtype_.MLX_FLOAT32,
    MlxArray? key,
  }) => MlxRandom.gumbel(shape, dtype: dtype, key: key);

  /// Samples a Laplace random array.
  MlxArray laplace(
    List<int> shape, {
    MlxDType dtype = raw.mlx_dtype_.MLX_FLOAT32,
    double loc = 0,
    double scale = 1,
    MlxArray? key,
  }) =>
      MlxRandom.laplace(shape, dtype: dtype, loc: loc, scale: scale, key: key);

  /// Samples a multivariate normal random array.
  MlxArray multivariateNormal(
    MlxArray mean,
    MlxArray cov, {
    List<int> shape = const [],
    MlxDType dtype = raw.mlx_dtype_.MLX_FLOAT32,
    MlxArray? key,
  }) => MlxRandom.multivariateNormal(
    mean,
    cov,
    shape: shape,
    dtype: dtype,
    key: key,
  );

  /// Samples integer values in `[low, high)`.
  MlxArray randint(
    int low,
    int high,
    List<int> shape, {
    MlxDType dtype = raw.mlx_dtype_.MLX_INT32,
    MlxArray? key,
  }) => MlxRandom.randint(low, high, shape, dtype: dtype, key: key);
}

/// High-level file IO namespace.
