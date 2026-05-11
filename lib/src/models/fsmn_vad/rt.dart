import 'dart:typed_data';

import 'package:dart_inference/dart_mlx_ffi.dart';

import 'bundle.dart';

class FsmnVadState {
  FsmnVadState({required this.caches});

  final List<MlxArray> caches;

  void close() {
    for (final cache in caches) {
      cache.close();
    }
  }
}

class FsmnVadFrameResult {
  const FsmnVadFrameResult({
    required this.speechProbabilities,
    required this.state,
  });

  final Float32List speechProbabilities;
  final FsmnVadState state;
}

final class FsmnVadRuntime {
  FsmnVadRuntime(FsmnVadBundle bundle)
    : _bundle = bundle,
      _manifest = bundle.manifest,
      _inLinear1W = _require(
        bundle.tensors,
        'encoder.in_linear1.linear.weight',
      ),
      _inLinear1B = _require(bundle.tensors, 'encoder.in_linear1.linear.bias'),
      _inLinear2W = _require(
        bundle.tensors,
        'encoder.in_linear2.linear.weight',
      ),
      _inLinear2B = _require(bundle.tensors, 'encoder.in_linear2.linear.bias'),
      _outLinear1W = _require(
        bundle.tensors,
        'encoder.out_linear1.linear.weight',
      ),
      _outLinear1B = _require(
        bundle.tensors,
        'encoder.out_linear1.linear.bias',
      ),
      _outLinear2W = _require(
        bundle.tensors,
        'encoder.out_linear2.linear.weight',
      ),
      _outLinear2B = _require(
        bundle.tensors,
        'encoder.out_linear2.linear.bias',
      ),
      _layers = List<_FsmnLayer>.generate(
        bundle.manifest.fsmnLayers,
        (index) => _FsmnLayer(
          linearW: _require(
            bundle.tensors,
            'encoder.fsmn.$index.linear.linear.weight',
          ),
          convLeftW: _require(
            bundle.tensors,
            'encoder.fsmn.$index.fsmn_block.conv_left.weight',
          ),
          affineW: _require(
            bundle.tensors,
            'encoder.fsmn.$index.affine.linear.weight',
          ),
          affineB: _require(
            bundle.tensors,
            'encoder.fsmn.$index.affine.linear.bias',
          ),
        ),
      );

  final FsmnVadBundle _bundle;
  final FsmnVadManifest _manifest;
  final MlxArray _inLinear1W;
  final MlxArray _inLinear1B;
  final MlxArray _inLinear2W;
  final MlxArray _inLinear2B;
  final MlxArray _outLinear1W;
  final MlxArray _outLinear1B;
  final MlxArray _outLinear2W;
  final MlxArray _outLinear2B;
  final List<_FsmnLayer> _layers;

  int get inputDim => _manifest.inputDim;
  int get cacheFrames => _manifest.cacheFrames;
  int get outputDim => _manifest.outputDim;
  FsmnVadManifest get manifest => _manifest;
  FsmnVadCmvn get cmvn => _bundle.cmvn;

  FsmnVadState createState({int batch = 1}) {
    return FsmnVadState(
      caches: List<MlxArray>.generate(
        _manifest.fsmnLayers,
        (_) => MlxArray.zeros([batch, cacheFrames, 1, _manifest.projDim]),
      ),
    );
  }

  FsmnVadFrameResult processFeatures({
    required Float32List features,
    required int frames,
    required FsmnVadState state,
  }) {
    if (frames <= 0 || features.isEmpty) {
      return FsmnVadFrameResult(
        speechProbabilities: Float32List(0),
        state: state,
      );
    }

    final input = MlxArray.fromFloat32List(
      features,
      shape: [1, frames, _manifest.inputDim],
    );
    MlxArray? x1;
    MlxArray? x2;
    MlxArray? x;
    MlxArray? xOut1;
    MlxArray? xOut2;
    MlxArray? logits;
    MlxArray? silence;
    try {
      x1 = _linearWithBias(input, _inLinear1W, _inLinear1B);
      x2 = _linearWithBias(x1, _inLinear2W, _inLinear2B);
      x = _relu(x2);
      var current = x;
      x = null;

      final nextCaches = <MlxArray>[];
      for (var index = 0; index < _layers.length; index += 1) {
        final result = _runLayer(_layers[index], current, state.caches[index]);
        current.close();
        current = result.output;
        nextCaches.add(result.cache);
      }

      xOut1 = _linearWithBias(current, _outLinear1W, _outLinear1B);
      current.close();
      xOut2 = _linearWithBias(xOut1, _outLinear2W, _outLinear2B);
      logits = mx.softmax(xOut2, axis: -1, precise: true);
      silence = logits.slice(start: [0, 0, 0], stop: [1, frames, 1]).reshape([
        frames,
      ]);
      MlxRuntime.evalAll([silence]);
      final silenceValues = silence.toFloat32List();
      final speechValues = Float32List(frames);
      for (var i = 0; i < frames; i += 1) {
        speechValues[i] = 1.0 - silenceValues[i];
      }

      state.close();
      return FsmnVadFrameResult(
        speechProbabilities: speechValues,
        state: FsmnVadState(caches: nextCaches),
      );
    } finally {
      input.close();
      x1?.close();
      x2?.close();
      x?.close();
      xOut1?.close();
      xOut2?.close();
      logits?.close();
      silence?.close();
    }
  }

  _LayerResult _runLayer(_FsmnLayer layer, MlxArray input, MlxArray cache) {
    MlxArray? proj;
    MlxArray? proj4d;
    MlxArray? padded;
    MlxArray? conv;
    MlxArray? residual;
    MlxArray? flat;
    MlxArray? affine;
    try {
      proj = _linearNoBias(input, layer.linearW);
      proj4d = proj.reshape([1, proj.shape[1], 1, _manifest.projDim]);
      padded = mx.concatenate([cache, proj4d], axis: 1);
      final nextCache = padded.slice(
        start: [0, padded.shape[1] - cacheFrames, 0, 0],
        stop: [1, padded.shape[1], 1, _manifest.projDim],
      );
      conv = mx.conv2d(padded, layer.convLeftW, groups: _manifest.projDim);
      residual = mx.add(proj4d, conv);
      flat = residual.reshape([1, proj.shape[1], _manifest.projDim]);
      affine = _linearWithBias(flat, layer.affineW, layer.affineB);
      final activated = _relu(affine);
      return _LayerResult(output: activated, cache: nextCache);
    } finally {
      proj?.close();
      proj4d?.close();
      padded?.close();
      conv?.close();
      residual?.close();
      flat?.close();
      affine?.close();
    }
  }

  static MlxArray _require(Map<String, MlxArray> tensors, String key) {
    final tensor = tensors[key];
    if (tensor == null) {
      throw StateError('Missing FSMN-VAD tensor: $key');
    }
    return tensor;
  }
}

class _FsmnLayer {
  const _FsmnLayer({
    required this.linearW,
    required this.convLeftW,
    required this.affineW,
    required this.affineB,
  });

  final MlxArray linearW;
  final MlxArray convLeftW;
  final MlxArray affineW;
  final MlxArray affineB;
}

class _LayerResult {
  const _LayerResult({required this.output, required this.cache});

  final MlxArray output;
  final MlxArray cache;
}

MlxArray _linearWithBias(MlxArray input, MlxArray weight, MlxArray bias) {
  final projected = mx.matmul(input, weight.transpose());
  final biasReshaped = bias.reshape([1, 1, bias.shape[0]]);
  try {
    final result = mx.add(projected, biasReshaped);
    projected.close();
    return result;
  } finally {
    biasReshaped.close();
  }
}

MlxArray _linearNoBias(MlxArray input, MlxArray weight) {
  return mx.matmul(input, weight.transpose());
}

MlxArray _relu(MlxArray input) {
  final zero = MlxArray.full([], 0.0);
  try {
    return mx.maximum(input, zero);
  } finally {
    zero.close();
  }
}
