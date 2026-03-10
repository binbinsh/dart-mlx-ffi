import 'dart:convert';
import 'dart:io';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

void main(List<String> args) {
  final results = <String, Object?>{
    'runtime': 'dart_mlx_ffi',
    'mlx_version': MlxVersion.current(),
    'cases': parityCases(),
  };
  final groups = tryReadStringArg(args, '--groups');
  if (groups != null) {
    final selected = groups.split(',').toSet();
    results['cases'] = Map<String, Object?>.fromEntries(
      (results['cases'] as Map<String, Object?>).entries.where(
        (entry) => selected.contains(entry.key),
      ),
    );
  }
  stdout.writeln(jsonEncode(results));
}

Map<String, Object?> parityCases() {
  final cases = <String, Object?>{};

  final a22 = f32([2, 2], 3);
  final b22 = f32([2, 2], 5);
  final a23 = f32([2, 3], 7);
  final a32 = f32([3, 2], 11);
  final v4 = f32([4], 13);
  final v3 = f32([3], 15);
  final bool22 = bvec([2, 2], 17);
  final idx2 = ivec([2], 19, 3);
  final idx22 = ivec([2, 2], 21, 3);

  cases['arith'] = {
    'add': enc(mx.add(a22, b22)),
    'subtract': enc(mx.subtract(a22, b22)),
    'multiply': enc(mx.multiply(a22, b22)),
    'divide': enc(mx.divide(a22, absPos(b22))),
    'matmul': enc(mx.matmul(a23, a32)),
  };

  final expInput = f32([4], 25);
  final logInput = pos([4], 27);
  final trigInput = f32([4], 29);
  final reduceInput = f32([2, 3], 31);
  final topkInput = f32([6], 33);
  cases['unary_reduce'] = {
    'abs': enc(mx.abs(a22)),
    'negative': enc(a22.negative()),
    'exp': enc(mx.exp(expInput)),
    'log': enc(mx.log(logInput)),
    'sin': enc(mx.sin(trigInput)),
    'cos': enc(mx.cos(trigInput)),
    'equal': enc(mx.equal(a22, a22)),
    'where': enc(mx.where(bool22, a22, b22)),
    'sum_all': enc(mx.sum(reduceInput)),
    'sum_axis': enc(mx.sum(reduceInput, axis: 1, keepDims: true)),
    'mean_all': enc(mx.mean(reduceInput)),
    'mean_axis': enc(mx.mean(reduceInput, axis: 0)),
    'logsumexp': enc(mx.logSumExp(reduceInput, axis: 1)),
    'softmax': enc(mx.softmax(reduceInput, axis: 1)),
    'topk': enc(mx.topK(topkInput, 3)),
  };

  final broadcastIn = f32([2], 35);
  final expandIn = f32([2], 37);
  final clipIn = f32([6], 39);
  final sortIn = f32([6], 41);
  final cubeA = f32([2, 2, 2], 43);
  final cubeB = f32([2, 2, 2], 45);
  final cubeC = f32([2, 2, 2], 47);
  cases['tensor'] = {
    'concatenate': enc(mx.concatenate([a22, b22], axis: 0)),
    'stack': enc(mx.stack([a22, b22], axis: 0)),
    'broadcast_to': enc(mx.broadcastTo(broadcastIn, [2, 2])),
    'expand_dims': enc(mx.expandDims(expandIn, 0)),
    'squeeze': enc(mx.squeeze(mx.expandDims(expandIn, 0))),
    'clip': enc(mx.clip(clipIn, min: -0.5, max: 0.5)),
    'minimum': enc(mx.minimum(a22, b22)),
    'maximum': enc(mx.maximum(a22, b22)),
    'argmax': enc(mx.argmax(a22)),
    'argmin_axis': enc(mx.argmin(a22, axis: 1, keepDims: true)),
    'sort': enc(mx.sort(sortIn)),
    'argsort': enc(mx.argsort(sortIn)),
    'flatten': enc(mx.flatten(a22)),
    'moveaxis': enc(mx.moveaxis(cubeA, 0, 2)),
    'swapaxes': enc(mx.swapaxes(cubeB, 0, 1)),
    'transpose_axes': enc(mx.transposeAxes(cubeC, [2, 0, 1])),
    'tile': enc(mx.tile(a22, [2, 1])),
    'unflatten': enc(mx.unflatten(mx.arange(0, 8, 1), axis: 0, shape: [2, 4])),
  };

  final takeInput = f32([2, 3], 49);
  final sliceInput = f32([6], 51);
  final diagVec = f32([3], 53);
  final diagMat = f32([3, 3], 55);
  final kronA = f32([2], 57);
  final kronB = f32([3], 59);
  final meshA = f32([2], 61);
  final meshB = f32([3], 63);
  final partitionIn = f32([6], 65);
  cases['index_extra'] = {
    'take': enc(mx.take(sliceInput, idx2)),
    'take_axis': enc(mx.take(takeInput, idx2, axis: 1)),
    'take_along_axis': enc(mx.takeAlongAxis(takeInput, idx22, axis: 1)),
    'slice': enc(mx.slice(sliceInput, start: [1], stop: [6], strides: [2])),
    'einsum': enc(mx.einsum('ij,jk->ik', [a23, a32])),
    'tensordot': enc(mx.tensordot(a23, a32, axis: 1)),
    'diag': enc(mx.diag(diagVec)),
    'diagonal': enc(mx.diagonal(diagMat)),
    'kron': enc(mx.kron(kronA, kronB)),
    'meshgrid0': enc(mx.meshgrid([meshA, meshB], indexing: 'ij')[0]),
    'meshgrid1': enc(mx.meshgrid([meshA, meshB], indexing: 'ij')[1]),
    'partition': enc(mx.partition(partitionIn, 2)),
  };

  final miscIn = f32([4], 67);
  final miscPos = pos([4], 69);
  final roundIn = f32([4], 71);
  final erfinvIn = MlxArray.fromFloat32List(
    [-0.75, -0.25, 0.25, 0.75],
    shape: [4],
  );
  final special = MlxArray.fromFloat32List(
    [double.nan, double.infinity, double.negativeInfinity, 1.0],
    shape: [4],
  );
  cases['misc'] = {
    'floor_divide': enc(mx.floorDivide(pos([4], 73), full([4], 1.5))),
    'logaddexp': enc(mx.logaddexp(v4, v4)),
    'inner': enc(mx.inner(v4, v4)),
    'floor': enc(miscIn.floor()),
    'sqrt': enc(miscPos.sqrt()),
    'rsqrt': enc(miscPos.rsqrt()),
    'square': enc(miscIn.square()),
    'reciprocal': enc(miscPos.reciprocal()),
    'sigmoid': enc(miscIn.sigmoid()),
    'degrees': enc(miscIn.degrees()),
    'radians': enc(miscIn.radians()),
    'expm1': enc(miscIn.expm1()),
    'erf': enc(miscIn.erf()),
    'erfinv': enc(mx.erfinv(erfinvIn)),
    'log1p': enc(miscPos.log1p()),
    'log2': enc(miscPos.log2()),
    'log10': enc(miscPos.log10()),
    'round': enc(roundIn.round(decimals: 2)),
    'linspace': enc(mx.linspace(-1.0, 1.0, 7)),
    'outer': enc(mx.outer(v3, mx.slice(v4, start: [0], stop: [3]))),
    'isclose': enc(mx.isClose(a22, mx.add(a22, full([2, 2], 1e-6)))),
    'repeat': enc(mx.repeat(v3, 2)),
    'roll': enc(mx.roll(v4, [1])),
    'median': enc(mx.median(miscIn)),
    'nan_to_num': enc(
      mx.nanToNum(special, nan: 0.0, posInf: 9.0, negInf: -9.0),
    ),
    'divmod_q': enc(
      mx
          .divmod(
            MlxArray.fromFloat32List([5.0, 7.0], shape: [2]),
            MlxArray.fromFloat32List([2.0, 2.0], shape: [2]),
          )
          .quotient,
    ),
    'divmod_r': enc(
      mx
          .divmod(
            MlxArray.fromFloat32List([5.0, 7.0], shape: [2]),
            MlxArray.fromFloat32List([2.0, 2.0], shape: [2]),
          )
          .remainder,
    ),
  };

  final scanInput = f32([6], 75);
  final triInput = f32([3, 3], 77);
  cases['scan'] = {
    'cumsum': enc(mx.cumsum(scanInput)),
    'cumprod': enc(mx.cumprod(pos([6], 79))),
    'cummax': enc(mx.cummax(scanInput)),
    'cummin': enc(mx.cummin(scanInput)),
    'logcumsumexp': enc(mx.logcumsumexp(scanInput)),
    'eye': enc(mx.eye(3, m: 3)),
    'identity': enc(mx.identity(3)),
    'tri': enc(mx.tri(3, m: 3)),
    'tril': enc(mx.tril(triInput)),
    'triu': enc(mx.triu(triInput)),
    'trace': enc(mx.trace(triInput)),
  };

  final conv1In = f32([1, 6, 2], 81);
  final conv1W = f32([4, 3, 2], 83);
  final conv2In = f32([1, 4, 4, 2], 85);
  final conv2W = f32([3, 3, 3, 2], 87);
  final conv3In = f32([1, 3, 3, 3, 2], 89);
  final conv3W = f32([2, 2, 2, 2, 2], 91);
  cases['conv'] = {
    'conv1d': enc(mx.conv1d(conv1In, conv1W, padding: 1)),
    'conv2d': enc(mx.conv2d(conv2In, conv2W, padding: [1, 1])),
    'conv3d': enc(mx.conv3d(conv3In, conv3W, padding: [1, 1, 1])),
    'conv_transpose1d': enc(mx.convTranspose1d(conv1In, conv1W, padding: 1)),
    'conv_transpose2d': enc(
      mx.convTranspose2d(conv2In, conv2W, padding: [1, 1]),
    ),
  };

  final base = f32([3, 3], 93);
  final spd = mx.add(
    mx.matmul(base.transpose(), base),
    mx.multiply(mx.eye(3, m: 3), full([3, 3], 0.5)),
  );
  final rhs = f32([3, 2], 95);
  final tri = mx.tril(spd);
  final eigSpd = mx.add(spd, mx.multiply(mx.eye(3, m: 3), full([3, 3], 0.25)));
  final svdIn = f32([3, 3], 97);
  final qr = mx.linalg.qr(svdIn);
  final eigh = mx.linalg.eigh(eigSpd);
  final svd = mx.linalg.svd(svdIn);
  cases['linalg'] = {
    'inv': enc(mx.linalg.inv(spd)),
    'solve': enc(mx.linalg.solve(spd, rhs)),
    'cholesky': enc(mx.linalg.cholesky(spd)),
    'pinv': enc(mx.linalg.pinv(spd)),
    'norm': enc(mx.linalg.norm(v4)),
    'cross': enc(mx.linalg.cross(f32([3], 99), f32([3], 101))),
    'qr_reconstruct': enc(mx.matmul(qr.q, qr.r)),
    'eigh_values': enc(eigh.values),
    'eigh_reconstruct': enc(
      mx.matmul(
        mx.matmul(eigh.vectors, mx.diag(eigh.values)),
        eigh.vectors.transpose(),
      ),
    ),
    'svd_values': enc(svd.s),
    'svd_reconstruct': enc(
      mx.matmul(mx.matmul(svd.u!, mx.diag(svd.s)), svd.vt!),
    ),
    'solve_triangular': enc(mx.linalg.solveTriangular(tri, rhs, upper: false)),
  };

  final lnIn = f32([1, 4], 103);
  final lnW = mx.ones([4]);
  final lnB = mx.zeros([4]);
  final ropeIn = f32([1, 1, 4, 64], 105);
  final q = f32([1, 2, 4, 64], 107);
  final k = f32([1, 2, 4, 64], 109);
  final v = f32([1, 2, 4, 64], 111);
  cases['fast'] = {
    'layer_norm': enc(mx.fast.layerNorm(lnIn, weight: lnW, bias: lnB)),
    'rms_norm': enc(mx.fast.rmsNorm(lnIn, weight: lnW)),
    'rope': enc(mx.fast.rope(ropeIn, dims: 64, base: 1000000.0)),
    'sdpa': enc(mx.fast.scaledDotProductAttention(q, k, v, scale: 1 / 8)),
  };

  final qWeights = f32([4, 32], 113);
  final qInput = f32([2, 32], 115);
  final quantized = mx.quant.quantize(
    qWeights,
    groupSize: 32,
    bits: 8,
    mode: 'affine',
  );
  cases['quant'] = {
    'dequantize': enc(
      mx.quant.dequantize(
        quantized,
        groupSize: 32,
        bits: 8,
        mode: 'affine',
        dtype: MlxDType.MLX_FLOAT32,
      ),
    ),
    'quantized_matmul': enc(
      mx.quant.matmul(
        qInput,
        quantized,
        transpose: true,
        groupSize: 32,
        bits: 8,
        mode: 'affine',
      ),
    ),
    'gather_qmm': enc(
      mx.quant.gatherQmm(
        qInput,
        quantized,
        groupSize: 32,
        bits: 8,
        mode: 'affine',
      ),
    ),
  };

  final p = MlxArray.fromFloat32List([0.2, 0.7], shape: [2]);
  final logits = MlxArray.fromFloat32List([1.0, 2.0, 3.0], shape: [1, 3]);
  final permInput = MlxArray.fromInt32List([1, 2, 3, 4], shape: [4]);
  final key = mx.random.key(123);
  final split = mx.random.split(key);
  MlxRuntime.seed(1001);
  final seedUniform = mx.random.uniform([2, 2], low: -1.0, high: 1.0);
  MlxRuntime.seed(1002);
  final seedNormal = mx.random.normal([2, 2], loc: 0.0, scale: 1.0);
  cases['random'] = {
    'key': enc(key),
    'split_first': enc(split.first),
    'split_second': enc(split.second),
    'seed_uniform': enc(seedUniform),
    'seed_normal': enc(seedNormal),
    'bernoulli': enc(
      mx.random.bernoulli(p, shape: [2], key: mx.random.key(2001)),
    ),
    'categorical': enc(
      mx.random.categorical(
        logits,
        axis: -1,
        numSamples: 4,
        key: mx.random.key(2002),
      ),
    ),
    'permutation': enc(
      mx.random.permutation(permInput, axis: 0, key: mx.random.key(2003)),
    ),
    'permutation_arange': enc(
      mx.random.permutationArange(4, key: mx.random.key(2004)),
    ),
    'gumbel': enc(mx.random.gumbel([2, 2], key: mx.random.key(2005))),
    'laplace': enc(
      mx.random.laplace([2, 2], loc: 1.0, scale: 0.5, key: mx.random.key(2006)),
    ),
    'randint': enc(mx.random.randint(0, 10, [4], key: mx.random.key(2007))),
  };

  return cases;
}

Map<String, Object?> enc(MlxArray x) {
  MlxArray y = x;
  var owned = false;
  if (x.dtype == MlxDType.MLX_FLOAT16 || x.dtype == MlxDType.MLX_BFLOAT16) {
    y = x.astype(MlxDType.MLX_FLOAT32);
    owned = true;
  }
  try {
    y.eval();
    return {'shape': y.shape, 'values': y.toList()};
  } finally {
    if (owned) {
      y.close();
    }
  }
}

MlxArray f32(List<int> shape, int seed, {int divisor = 64}) =>
    MlxArray.fromFloat32List(
      f32List(numel(shape), seed, divisor),
      shape: shape,
    );

List<double> f32List(int count, int seed, int divisor) =>
    List<double>.generate(count, (index) {
      final numerator = ((index * (seed * 2 + 1) + seed * 7 + 13) % 257) - 128;
      return numerator / divisor;
    });

MlxArray pos(List<int> shape, int seed) => MlxArray.fromFloat32List(
  f32List(numel(shape), seed, 32).map((v) => v.abs() + 0.25).toList(),
  shape: shape,
);

MlxArray absPos(MlxArray x) => mx.add(mx.abs(x), full(x.shape, 0.25));

MlxArray ivec(List<int> shape, int seed, int mod) => MlxArray.fromInt32List(
  List<int>.generate(
    numel(shape),
    (index) => ((index * (seed + 3)) + seed) % mod,
  ),
  shape: shape,
);

MlxArray bvec(List<int> shape, int seed) => MlxArray.fromBoolList(
  List<bool>.generate(
    numel(shape),
    (index) => (((index * (seed + 5)) + seed) % 2) == 0,
  ),
  shape: shape,
);

MlxArray full(List<int> shape, double value) =>
    MlxArray.full(shape, value, dtype: MlxDType.MLX_FLOAT32);

int numel(List<int> shape) => shape.fold(1, (a, b) => a * b);

String? tryReadStringArg(List<String> args, String name) {
  final prefix = '$name=';
  for (var index = 0; index < args.length; index++) {
    final arg = args[index];
    if (arg.startsWith(prefix)) {
      return arg.substring(prefix.length);
    }
    if (arg == name && index + 1 < args.length) {
      return args[index + 1];
    }
  }
  return null;
}
