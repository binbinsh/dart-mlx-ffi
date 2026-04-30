// Repetition-Aware Sampling (RAS) for CosyVoice2's autoregressive
// speech-token decode loop.

import 'dart:ffi' as ffi;
import 'dart:math';
import 'dart:typed_data';

import '../../runtime/native_float32_source.dart';
import '../../runtime/native_runtime.dart' show NativeTensorBuffer;
import '../../runtime/runtime.dart' show RuntimeTensor;

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
  final int fallbackSamples;
}

final class RasSampler {
  RasSampler({Random? rng, this.config = const RasConfig()})
    : _rng = rng ?? Random();

  final Random _rng;
  final RasConfig config;

  int sample(
    Object logits,
    List<int> decodedTokens, {
    int eosToken = 6561,
    bool ignoreEos = false,
  }) {
    final values = _float32Values(logits);
    final effectiveEosToken = eosToken < values.length
        ? eosToken
        : values.length - 1;
    final candidate = _nucleusSample(
      values,
      eosToken: effectiveEosToken,
      ignoreEos: ignoreEos,
      topP: config.topP,
      topK: config.topK,
      randomDraw: _rng.nextDouble(),
    );
    if (_shouldFallback(
      candidate,
      decodedTokens,
      winSize: config.winSize,
      tauR: config.tauR,
    )) {
      return _multinomialSample(
        values,
        eosToken: effectiveEosToken,
        ignoreEos: ignoreEos,
        randomDraw: _rng.nextDouble(),
      );
    }
    return candidate;
  }
}

final class RasDecodeBuffer {
  RasDecodeBuffer({
    required int maxTokens,
    Random? rng,
    this.config = const RasConfig(),
  }) : _rng = rng ?? Random(),
       _history = NativeTensorBuffer.int32([_positiveMaxTokens(maxTokens)]);

  final Random _rng;
  final RasConfig config;
  final NativeTensorBuffer _history;
  var _count = 0;

  int get length => _count;
  int get capacity => _history.byteLength ~/ 4;

  int sample(Object logits, {int eosToken = 6561, bool ignoreEos = false}) {
    _checkOpen();
    final values = _float32Values(logits);
    final effectiveEosToken = eosToken < values.length
        ? eosToken
        : values.length - 1;
    final history = _history.asInt32List().sublist(0, _count);
    final candidate = _nucleusSample(
      values,
      eosToken: effectiveEosToken,
      ignoreEos: ignoreEos,
      topP: config.topP,
      topK: config.topK,
      randomDraw: _rng.nextDouble(),
    );
    if (_shouldFallback(
      candidate,
      history,
      winSize: config.winSize,
      tauR: config.tauR,
    )) {
      return _multinomialSample(
        values,
        eosToken: effectiveEosToken,
        ignoreEos: ignoreEos,
        randomDraw: _rng.nextDouble(),
      );
    }
    return candidate;
  }

  int sampleAndAppendNonEos(
    Object logits, {
    int eosToken = 6561,
    bool ignoreEos = false,
  }) {
    _checkOpen();
    final token = sample(logits, eosToken: eosToken, ignoreEos: ignoreEos);
    final effectiveEosToken = eosToken < _float32Length(logits)
        ? eosToken
        : _float32Length(logits) - 1;
    if (token != effectiveEosToken) {
      append(token);
    }
    return token;
  }

  void append(int token) {
    _checkOpen();
    if (_count >= capacity) {
      throw StateError('RAS decode buffer capacity exceeded.');
    }
    _history.asInt32List()[_count] = token;
    _count += 1;
  }

  void appendAll(Iterable<int> tokens) {
    for (final token in tokens) {
      append(token);
    }
  }

  List<int> toList() {
    _checkOpen();
    return _history.asInt32List().sublist(0, _count);
  }

  RuntimeTensor tokensTensor() {
    _checkOpen();
    return _history.tensorView(shape: [_count], byteLength: _count * 4);
  }

  void close() {
    _history.close();
  }

  void _checkOpen() {
    if (_history.isClosed) {
      throw StateError('RAS decode buffer is closed.');
    }
  }
}

final class _Candidate {
  _Candidate(this.id, this.logit);
  final int id;
  final double logit;
  double prob = 0;
}

Float32List _float32Values(Object source) {
  return withNativeFloat32Source(source, (pointer, length) {
    if (length == 0 || pointer == ffi.nullptr) return Float32List(0);
    return Float32List.fromList(pointer.asTypedList(length));
  });
}

int _float32Length(Object value) {
  return nativeFloat32SourceLength(value);
}

int _nucleusSample(
  Float32List logits, {
  required int eosToken,
  required bool ignoreEos,
  required double topP,
  required int topK,
  required double randomDraw,
}) {
  if (logits.isEmpty ||
      eosToken < 0 ||
      eosToken >= logits.length ||
      topK <= 0 ||
      topP < 0 ||
      randomDraw < 0) {
    throw StateError('CosyVoice2 RAS helper received invalid input.');
  }
  if (topP == 0) {
    return _bestLogit(logits, eosToken: eosToken, ignoreEos: ignoreEos);
  }
  final maxId = eosToken;
  final validIds = [
    for (var id = 0; id <= maxId; id += 1)
      if (!(ignoreEos && id == eosToken)) id,
  ];
  if (validIds.isEmpty) {
    throw StateError('CosyVoice2 RAS helper received invalid input.');
  }
  var maxLogit = logits[validIds.first].toDouble();
  for (final id in validIds.skip(1)) {
    final logit = logits[id].toDouble();
    if (logit > maxLogit) maxLogit = logit;
  }
  var sum = 0.0;
  final all = <_Candidate>[];
  for (final id in validIds) {
    final logit = logits[id].toDouble();
    final prob = exp(logit - maxLogit);
    sum += prob;
    all.add(_Candidate(id, logit)..prob = prob);
  }
  for (final candidate in all) {
    candidate.prob /= sum;
  }
  all.sort((a, b) {
    final byLogit = b.logit.compareTo(a.logit);
    return byLogit != 0 ? byLogit : a.id.compareTo(b.id);
  });
  final top = all.take(min(topK, all.length)).toList();
  top.sort((a, b) {
    final byProb = b.prob.compareTo(a.prob);
    return byProb != 0 ? byProb : a.id.compareTo(b.id);
  });
  final nucleusP = topP > 1 ? 1.0 : topP;
  var count = 0;
  var cumulative = 0.0;
  while (count < top.length && cumulative < nucleusP) {
    cumulative += top[count].prob;
    count += 1;
  }
  final selected = top.take(max(1, count)).toList();
  return _draw(selected, randomDraw);
}

int _multinomialSample(
  Float32List logits, {
  required int eosToken,
  required bool ignoreEos,
  required double randomDraw,
}) {
  if (logits.isEmpty ||
      eosToken < 0 ||
      eosToken >= logits.length ||
      randomDraw < 0) {
    throw StateError('CosyVoice2 RAS helper received invalid input.');
  }
  final validIds = [
    for (var id = 0; id <= eosToken; id += 1)
      if (!(ignoreEos && id == eosToken)) id,
  ];
  if (validIds.isEmpty) {
    throw StateError('CosyVoice2 RAS helper received invalid input.');
  }
  var maxLogit = logits[validIds.first].toDouble();
  for (final id in validIds.skip(1)) {
    if (logits[id] > maxLogit) maxLogit = logits[id].toDouble();
  }
  var sum = 0.0;
  final candidates = <_Candidate>[];
  for (final id in validIds) {
    final logit = logits[id].toDouble();
    final prob = exp(logit - maxLogit);
    sum += prob;
    candidates.add(_Candidate(id, logit)..prob = prob);
  }
  for (final candidate in candidates) {
    candidate.prob /= sum;
  }
  return _draw(candidates, randomDraw);
}

int _draw(List<_Candidate> candidates, double randomDraw) {
  final target =
      randomDraw.clamp(0.0, 0.9999999999999999) *
      candidates.fold<double>(0, (sum, candidate) => sum + candidate.prob);
  var cumulative = 0.0;
  for (final candidate in candidates) {
    cumulative += candidate.prob;
    if (target < cumulative) return candidate.id;
  }
  return candidates.last.id;
}

int _bestLogit(
  Float32List logits, {
  required int eosToken,
  required bool ignoreEos,
}) {
  var best = -1;
  var bestLogit = double.negativeInfinity;
  for (var id = 0; id <= eosToken; id += 1) {
    if (ignoreEos && id == eosToken) continue;
    final logit = logits[id].toDouble();
    if (logit > bestLogit || (logit == bestLogit && id < best)) {
      best = id;
      bestLogit = logit;
    }
  }
  if (best < 0) {
    throw StateError('CosyVoice2 RAS helper received invalid input.');
  }
  return best;
}

bool _shouldFallback(
  int candidate,
  List<int> history, {
  required int winSize,
  required double tauR,
}) {
  if (winSize <= 0 || tauR < 0 || !tauR.isFinite) {
    throw StateError('CosyVoice2 RAS helper received invalid input.');
  }
  final start = history.length > winSize ? history.length - winSize : 0;
  var repetitions = 0;
  for (var i = start; i < history.length; i += 1) {
    if (history[i] == candidate) repetitions += 1;
  }
  return repetitions >= winSize * tauR;
}

int _positiveMaxTokens(int maxTokens) {
  if (maxTokens <= 0) {
    throw RangeError.value(maxTokens, 'maxTokens', 'must be positive');
  }
  return maxTokens;
}
