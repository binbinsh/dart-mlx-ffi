import 'dart:math' as math;

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

/// Interleaved Multi-dimensional Rotary Position Embedding (MRoPE).
///
/// Qwen3-ASR uses stride-3 interleaved frequency assignment across
/// 3 spatial dimensions (temporal, height, width) with sections [24, 20, 20].
///
/// Frequency assignment:
///   - freq index 0 -> temporal
///   - freq index 1 -> height
///   - freq index 2 -> width
///   - freq index 3 -> temporal
///   - ...
///
/// Total: (24 + 20 + 20) = 64 = head_dim / 2.
/// Each frequency maps to 2 rotation dims, so 64 * 2 = 128 = head_dim.
///
/// Reference: moona3k/mlx-qwen3-asr mrope.py
final class AsrMRoPE {
  AsrMRoPE({
    required int headDim,
    required double base,
    required List<int> sections,
  }) : _headDim = headDim,
       _halfDim = headDim ~/ 2,
       _base = base,
       _sections = sections {
    assert(
      sections.fold<int>(0, (a, b) => a + b) == _halfDim,
      'Sum of mrope sections must equal head_dim / 2',
    );
    _invFreq = _buildInvFreq();
    _overwriteMasks = _buildOverwriteMasks();
  }

  final int _headDim;
  final int _halfDim;
  final double _base;
  final List<int> _sections;
  late final MlxArray _invFreq; // shape: [halfDim]
  late final List<MlxArray> _overwriteMasks; // 2 masks, each [1, 1, halfDim]

  /// Build inverse frequencies: 1 / (base ^ (2i / headDim)).
  MlxArray _buildInvFreq() {
    final values = <double>[
      for (var i = 0; i < _headDim; i += 2) 1.0 / math.pow(_base, i / _headDim),
    ];
    return MlxArray.fromFloat32List(
      values.map((v) => v.toDouble()).toList(),
      shape: [_halfDim],
    );
  }

  /// Build overwrite masks for height (dim=1) and width (dim=2).
  ///
  /// For each spatial dim, the mask marks freq indices that should be
  /// overwritten from that dimension's positions. Indices are:
  ///   dim=1 (height): 1, 4, 7, 10, ... up to sections[1]*3
  ///   dim=2 (width):  2, 5, 8, 11, ... up to sections[2]*3
  List<MlxArray> _buildOverwriteMasks() {
    final masks = <MlxArray>[];
    for (var dim = 1; dim <= 2; dim++) {
      final length = _sections[dim] * 3;
      final stop = length < _halfDim ? length : _halfDim;
      final boolMask = List<bool>.filled(_halfDim, false);
      for (var idx = dim; idx < stop; idx += 3) {
        boolMask[idx] = true;
      }
      masks.add(MlxArray.fromBoolList(boolMask, shape: [1, 1, _halfDim]));
    }
    return masks;
  }

  /// Compute interleaved MRoPE cos/sin embeddings.
  ///
  /// [positionIds]: shape (batch, 3, seqLen) — one row per spatial dimension.
  /// Returns (cos, sin) each of shape (batch, seqLen, headDim).
  ({MlxArray cos, MlxArray sin}) compute(MlxArray positionIds, MlxDType dtype) {
    final batch = positionIds.shape[0];
    final seqLen = positionIds.shape[2];

    // positionIds: (B, 3, L) -> transpose to (3, B, L)
    final pos3bl = positionIds.astype(MlxDType.MLX_FLOAT32).transposeAxes([
      1,
      0,
      2,
    ]);
    try {
      // pos3bl: (3, B, L) -> (3, B, L, 1)
      final posExp = pos3bl.expandDims(3);
      try {
        // invFreq: (halfDim,) -> (1, 1, 1, halfDim) for broadcast
        final invExp = _invFreq.reshape([1, 1, 1, _halfDim]);
        try {
          // freqs: (3, B, L, halfDim) = pos * inv_freq
          final freqs = posExp * invExp;
          try {
            // Start with temporal freqs (dim=0), then overwrite H/W indices.
            var freqsT = freqs
                .slice(start: [0, 0, 0, 0], stop: [1, batch, seqLen, _halfDim])
                .reshape([batch, seqLen, _halfDim]);
            try {
              for (var dim = 1; dim <= 2; dim++) {
                final dimFreqs = freqs
                    .slice(
                      start: [dim, 0, 0, 0],
                      stop: [dim + 1, batch, seqLen, _halfDim],
                    )
                    .reshape([batch, seqLen, _halfDim]);
                try {
                  final mask = _overwriteMasks[dim - 1]; // (1, 1, halfDim)
                  final merged = mx.where(mask, dimFreqs, freqsT);
                  freqsT.close();
                  freqsT = merged;
                } finally {
                  dimFreqs.close();
                }
              }

              // emb: (B, L, headDim) = [freqsT, freqsT] along last axis
              final emb = mx.concatenate([freqsT, freqsT], axis: -1);
              try {
                final cosArr = emb.cos().astype(dtype);
                final sinArr = emb.sin().astype(dtype);
                return (cos: cosArr, sin: sinArr);
              } finally {
                emb.close();
              }
            } finally {
              freqsT.close();
            }
          } finally {
            freqs.close();
          }
        } finally {
          invExp.close();
        }
      } finally {
        posExp.close();
      }
    } finally {
      pos3bl.close();
    }
  }

  void close() {
    _invFreq.close();
    for (final m in _overwriteMasks) {
      m.close();
    }
  }
}

/// Apply rotary position embeddings to query and key tensors.
///
/// Uses the standard rotation formula:
///   x_rot = x * cos + rotate_half(x) * sin
///
/// [q], [k]: shape (B, nHeads, seqLen, headDim).
/// [cos], [sin]: shape (B, seqLen, headDim).
///
/// Returns (qEmbed, kEmbed) with same shapes as inputs.
({MlxArray q, MlxArray k}) applyRotaryPosEmb(
  MlxArray q,
  MlxArray k,
  MlxArray cos,
  MlxArray sin,
) {
  // cos/sin: (B, L, D) -> (B, 1, L, D) for multi-head broadcast
  final cos4 = cos.expandDims(1);
  final sin4 = sin.expandDims(1);
  try {
    final qRot = _rotateHalf(q);
    final kRot = _rotateHalf(k);
    try {
      final qEmbed = (q * cos4) + (qRot * sin4);
      final kEmbed = (k * cos4) + (kRot * sin4);
      return (q: qEmbed, k: kEmbed);
    } finally {
      kRot.close();
      qRot.close();
    }
  } finally {
    sin4.close();
    cos4.close();
  }
}

/// Rotate half: split last dim in half, return [-x2, x1].
MlxArray _rotateHalf(MlxArray x) {
  final mid = x.shape[3] ~/ 2;
  final x1 = x.slice(
    start: [0, 0, 0, 0],
    stop: [x.shape[0], x.shape[1], x.shape[2], mid],
  );
  final x2 = x.slice(
    start: [0, 0, 0, mid],
    stop: [x.shape[0], x.shape[1], x.shape[2], x.shape[3]],
  );
  try {
    final negX2 = x2.negative();
    try {
      return mx.concatenate([negX2, x1], axis: 3);
    } finally {
      negX2.close();
    }
  } finally {
    x2.close();
    x1.close();
  }
}

/// Build 3D MRoPE position IDs for ASR streaming.
///
/// For ASR, all 3 spatial dimensions use the same positions
/// (temporal = height = width).
///
/// Returns shape (1, 3, length).
MlxArray buildAsrPositionIds(int start, int length) {
  final positions = MlxArray.fromInt32List(
    [for (var i = 0; i < length; i++) start + i],
    shape: [1, length],
  );
  try {
    return mx.stack([positions, positions, positions], axis: 1);
  } finally {
    positions.close();
  }
}
