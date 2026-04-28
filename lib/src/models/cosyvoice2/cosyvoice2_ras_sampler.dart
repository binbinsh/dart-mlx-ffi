// Repetition-Aware Sampling (RAS) for CosyVoice2's autoregressive
// speech-token decode loop.
//
// Faithful port of cosyvoice.utils.common.ras_sampling:
//   1. Nucleus sample (top-p with top-k cap) from softmax(logits).
//   2. If the candidate appears in the last `winSize` decoded tokens
//      with frequency >= `winSize * tauR`, fall back to a plain
//      multinomial draw over softmax(logits).
//
// This is intentionally pure-Dart (no Zig FFI): the inner loops are
// O(V log V) for softmax-sort, and at V=6564 that's microseconds even
// in unoptimised Dart, well below an ONNX decode step.

import 'dart:math';
import 'dart:typed_data';

/// RAS configuration. Defaults match the upstream Python reference.
class RasConfig {
  const RasConfig({
    this.topP = 0.8,
    this.topK = 25,
    this.winSize = 10,
    this.tauR = 0.1,
    this.fallbackSamples = 10,
  });

  final double topP;
  final int topK;
  final int winSize;
  final double tauR;

  /// Upstream `random_sampling` does a single multinomial draw. The
  /// reference inference loop wraps that call in retry logic when the
  /// drawn token is invalid — we don't replicate the validity gate
  /// here (it's cosyvoice-specific), but we keep an upper bound so a
  /// pathological RNG can't hang the loop. Default of 10 matches the
  /// upstream value.
  final int fallbackSamples;
}

/// Stateless RAS sampler: takes `[V]` float32 logits + the prior
/// decoded history and returns a single token id.
final class RasSampler {
  RasSampler({Random? rng, this.config = const RasConfig()})
      : _rng = rng ?? Random();

  final Random _rng;
  final RasConfig config;

  int sample(Float32List logits, List<int> decodedTokens) {
    final probs = _softmax(logits);
    // 1. nucleus
    final candidate = _nucleus(probs);
    // 2. repetition check on last winSize tokens
    final win = config.winSize;
    final history = decodedTokens.length <= win
        ? decodedTokens
        : decodedTokens.sublist(decodedTokens.length - win);
    var rep = 0;
    for (final t in history) {
      if (t == candidate) rep += 1;
    }
    if (rep >= win * config.tauR) {
      return _multinomial(probs);
    }
    return candidate;
  }

  // --- internals ---

  Float32List _softmax(Float32List logits) {
    var maxLogit = logits[0];
    for (var i = 1; i < logits.length; i += 1) {
      if (logits[i] > maxLogit) maxLogit = logits[i];
    }
    final out = Float32List(logits.length);
    var sum = 0.0;
    for (var i = 0; i < logits.length; i += 1) {
      final v = exp(logits[i] - maxLogit);
      out[i] = v;
      sum += v;
    }
    final inv = 1.0 / sum;
    for (var i = 0; i < out.length; i += 1) {
      out[i] *= inv;
    }
    return out;
  }

  int _nucleus(Float32List probs) {
    // Build (idx, prob) pairs, sort descending. The upstream reference
    // uses a stable sort; replicate that by tie-breaking on ascending
    // index so equal-probability tokens preserve insertion order.
    final n = probs.length;
    final order = List<int>.generate(n, (i) => i, growable: false);
    order.sort((a, b) {
      final cmp = probs[b].compareTo(probs[a]);
      if (cmp != 0) return cmp;
      return a.compareTo(b);
    });

    final keptIdx = <int>[];
    final keptProb = <double>[];
    var cum = 0.0;
    for (var i = 0; i < n; i += 1) {
      if (cum < config.topP && keptIdx.length < config.topK) {
        cum += probs[order[i]];
        keptIdx.add(order[i]);
        keptProb.add(probs[order[i]]);
      } else {
        break;
      }
    }
    if (keptIdx.isEmpty) {
      // Defensive: should never happen with positive probs, but if it
      // does, fall back to argmax.
      return order[0];
    }
    return _drawFrom(keptIdx, keptProb);
  }

  int _multinomial(Float32List probs) {
    // Single full-distribution multinomial draw. Inverse-CDF over the
    // dense distribution.
    final r = _rng.nextDouble();
    var cum = 0.0;
    for (var i = 0; i < probs.length; i += 1) {
      cum += probs[i];
      if (r < cum) return i;
    }
    return probs.length - 1;
  }

  int _drawFrom(List<int> ids, List<double> probs) {
    var sum = 0.0;
    for (final p in probs) {
      sum += p;
    }
    final r = _rng.nextDouble() * sum;
    var cum = 0.0;
    for (var i = 0; i < ids.length; i += 1) {
      cum += probs[i];
      if (r < cum) return ids[i];
    }
    return ids.last;
  }
}
