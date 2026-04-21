/// MLX runtime for the ECAPA-TDNN speaker encoder.
///
/// The runtime wraps a loaded [EcapaBundle] and exposes a single entry point:
/// `encode(Float32List samples) -> Float32List embedding` (192-dim, unnormalized).
///
/// Implementation notes
/// --------------------
///
/// * Input audio is passed through [EcapaFbankFrontend], producing a mean
///   normalized `(T, 80)` MLX tensor in NTC layout.
/// * The ECAPA backbone is evaluated in NTC layout via the helpers in
///   [speaker_embedding/nn.dart]. MLX `conv1d` consumes `(N, L, C_in)` with
///   weights `(C_out, kW, C_in)` natively, so the final result matches the
///   SpeechBrain reference implementation after a transpose on input/output
///   against fixtures captured in channel-first form.
/// * We expose an optional [forwardWithFixtureHooks] that returns the full
///   per-stage tensor map consumed by the parity test. Production code uses
///   [encode] which only realises the final embedding.
library;

import 'dart:typed_data';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

import 'bundle.dart';
import 'fbank.dart';
import 'nn.dart';

/// Holds the final encoded output plus metadata useful for downstream gating.
class EcapaEncodeResult {
  const EcapaEncodeResult({required this.embedding, required this.numFrames});

  final Float32List embedding;
  final int numFrames;
}

/// Per-stage snapshots used by the parity test. Each tensor is owned by the
/// caller and must be closed.
class EcapaForwardTrace {
  EcapaForwardTrace({
    required this.fbankRaw,
    required this.fbankNorm,
    required this.block0,
    required this.block1,
    required this.block2,
    required this.block3,
    required this.preMfaConcat,
    required this.mfa,
    required this.asp,
    required this.aspBn,
    required this.fc,
    required this.embedding,
  });

  // All tensors are in NTC / (B,T,C) layout except the pooled stages which
  // retain `T=1` and the final `embedding` which is `(192,)`.
  final MlxArray fbankRaw; // (T, 80)
  final MlxArray fbankNorm; // (T, 80)
  final MlxArray block0; // (1, T, 1024)
  final MlxArray block1;
  final MlxArray block2;
  final MlxArray block3;
  final MlxArray preMfaConcat; // (1, T, 3072)
  final MlxArray mfa; // (1, T, 3072)
  final MlxArray asp; // (1, 1, 6144)
  final MlxArray aspBn; // (1, 1, 6144)
  final MlxArray fc; // (1, 1, 192)
  final MlxArray embedding; // (192,)

  void close() {
    fbankRaw.close();
    fbankNorm.close();
    block0.close();
    block1.close();
    block2.close();
    block3.close();
    preMfaConcat.close();
    mfa.close();
    asp.close();
    aspBn.close();
    fc.close();
    embedding.close();
  }
}

class EcapaRuntime {
  EcapaRuntime(this.bundle) : _frontend = EcapaFbankFrontend(bundle);

  final EcapaBundle bundle;
  final EcapaFbankFrontend _frontend;

  EcapaManifest get manifest => bundle.manifest;

  /// Encode a PCM16 mono waveform (float32 [-1, 1]) into the 192-dim ECAPA
  /// embedding. No L2 normalization is applied.
  EcapaEncodeResult encode(Float32List samples) {
    final trace = forwardWithFixtureHooks(samples);
    try {
      final flat = trace.embedding.toFloat32List();
      return EcapaEncodeResult(
        embedding: Float32List.fromList(flat),
        numFrames: trace.block0.shape[1],
      );
    } finally {
      trace.close();
    }
  }

  /// Forward pass that returns every intermediate tensor the parity test
  /// asserts against. Caller owns the returned trace.
  EcapaForwardTrace forwardWithFixtureHooks(Float32List samples) {
    final feat = _frontend.encode(samples);

    MlxArray? x; // current NTC activation
    MlxArray? block0;
    MlxArray? block1;
    MlxArray? block2;
    MlxArray? block3;
    MlxArray? preMfaConcat;
    MlxArray? mfa;
    MlxArray? asp;
    MlxArray? aspBn;
    MlxArray? fc;
    MlxArray? embedding;

    try {
      // Add batch dim: (T, 80) -> (1, T, 80).
      x = feat.norm.expandDims(0);

      // Block 0: TDNNBlock(in=80, out=1024, k=5, d=1).
      block0 = tdnnBlock(
        x,
        weight: bundle.require('blocks.0.w'),
        bias: bundle.require('blocks.0.b'),
        bn: FusedBn(
          scale: bundle.require('blocks.0.bn.scale'),
          bias: bundle.require('blocks.0.bn.bias'),
        ),
        kernelSize: 5,
        dilation: 1,
      );
      x.close();
      x = null;

      // Blocks 1..3: SERes2NetBlock.
      block1 = _seRes2NetFromBundle(block0, index: 1, kernelSize: 3, dilation: 2);
      block2 = _seRes2NetFromBundle(block1, index: 2, kernelSize: 3, dilation: 3);
      block3 = _seRes2NetFromBundle(block2, index: 3, kernelSize: 3, dilation: 4);

      // Concatenate block1..3 along channel axis (axis=2 in NTC).
      preMfaConcat = mx.concatenate([block1, block2, block3], axis: 2);

      // MFA TDNNBlock(3072 -> 3072, k=1).
      mfa = tdnnBlock(
        preMfaConcat,
        weight: bundle.require('mfa.w'),
        bias: bundle.require('mfa.b'),
        bn: FusedBn(
          scale: bundle.require('mfa.bn.scale'),
          bias: bundle.require('mfa.bn.bias'),
        ),
        kernelSize: 1,
        dilation: 1,
      );

      // Attentive statistics pooling.
      asp = attentiveStatisticsPool(
        mfa,
        tdnnW: bundle.require('asp.tdnn.w'),
        tdnnB: bundle.require('asp.tdnn.b'),
        tdnnBn: FusedBn(
          scale: bundle.require('asp.tdnn.bn.scale'),
          bias: bundle.require('asp.tdnn.bn.bias'),
        ),
        convW: bundle.require('asp.conv.w'),
        convB: bundle.require('asp.conv.b'),
        eps: manifest.aspEps,
      );

      // Final BN on pooled (B, 1, 6144).
      aspBn = FusedBn(
        scale: bundle.require('asp_bn.scale'),
        bias: bundle.require('asp_bn.bias'),
      ).apply(asp);

      // FC 1x1 conv 6144 -> 192.
      fc = speechBrainConv1d(
        aspBn,
        weight: bundle.require('fc.w'),
        bias: bundle.require('fc.b'),
        kernelSize: 1,
        dilation: 1,
        padSame: false,
      );

      // Extract (192,) embedding.
      embedding = fc.reshape([manifest.embeddingDim]);

      MlxRuntime.evalAll([
        feat.raw,
        feat.norm,
        block0,
        block1,
        block2,
        block3,
        preMfaConcat,
        mfa,
        asp,
        aspBn,
        fc,
        embedding,
      ]);

      final trace = EcapaForwardTrace(
        fbankRaw: feat.raw,
        fbankNorm: feat.norm,
        block0: block0,
        block1: block1,
        block2: block2,
        block3: block3,
        preMfaConcat: preMfaConcat,
        mfa: mfa,
        asp: asp,
        aspBn: aspBn,
        fc: fc,
        embedding: embedding,
      );

      // Null out locals so finally doesn't double-close them.
      block0 = null;
      block1 = null;
      block2 = null;
      block3 = null;
      preMfaConcat = null;
      mfa = null;
      asp = null;
      aspBn = null;
      fc = null;
      embedding = null;
      return trace;
    } catch (_) {
      feat.close();
      rethrow;
    } finally {
      x?.close();
      block0?.close();
      block1?.close();
      block2?.close();
      block3?.close();
      preMfaConcat?.close();
      mfa?.close();
      asp?.close();
      aspBn?.close();
      fc?.close();
      embedding?.close();
    }
  }

  MlxArray _seRes2NetFromBundle(
    MlxArray input, {
    required int index,
    required int kernelSize,
    required int dilation,
  }) {
    final scale = manifest.res2netScale;
    final resWeights = <MlxArray>[];
    final resBiases = <MlxArray>[];
    final resBns = <FusedBn>[];
    for (var j = 0; j < scale - 1; j++) {
      resWeights.add(bundle.require('blocks.$index.res2net.$j.w'));
      resBiases.add(bundle.require('blocks.$index.res2net.$j.b'));
      resBns.add(FusedBn(
        scale: bundle.require('blocks.$index.res2net.$j.bn.scale'),
        bias: bundle.require('blocks.$index.res2net.$j.bn.bias'),
      ));
    }
    return seRes2NetBlock(
      input,
      tdnn1W: bundle.require('blocks.$index.tdnn1.w'),
      tdnn1B: bundle.require('blocks.$index.tdnn1.b'),
      tdnn1Bn: FusedBn(
        scale: bundle.require('blocks.$index.tdnn1.bn.scale'),
        bias: bundle.require('blocks.$index.tdnn1.bn.bias'),
      ),
      resWeights: resWeights,
      resBiases: resBiases,
      resBns: resBns,
      tdnn2W: bundle.require('blocks.$index.tdnn2.w'),
      tdnn2B: bundle.require('blocks.$index.tdnn2.b'),
      tdnn2Bn: FusedBn(
        scale: bundle.require('blocks.$index.tdnn2.bn.scale'),
        bias: bundle.require('blocks.$index.tdnn2.bn.bias'),
      ),
      seConv1W: bundle.require('blocks.$index.se.conv1.w'),
      seConv1B: bundle.require('blocks.$index.se.conv1.b'),
      seConv2W: bundle.require('blocks.$index.se.conv2.w'),
      seConv2B: bundle.require('blocks.$index.se.conv2.b'),
      res2netScale: scale,
      kernelSize: kernelSize,
      dilation: dilation,
    );
  }

  void close() {
    bundle.close();
  }
}

/// L2-normalize an embedding in place. Safe for use prior to cosine dot-product
/// comparisons in profile code.
Float32List l2NormalizeInPlace(Float32List embedding) {
  var sumSq = 0.0;
  for (var i = 0; i < embedding.length; i++) {
    sumSq += embedding[i] * embedding[i];
  }
  if (sumSq <= 0) return embedding;
  final inv = 1.0 / _sqrt(sumSq);
  for (var i = 0; i < embedding.length; i++) {
    embedding[i] = embedding[i] * inv;
  }
  return embedding;
}

double cosineSimilarity(Float32List a, Float32List b) {
  if (a.length != b.length) {
    throw ArgumentError('embedding lengths must match');
  }
  var dot = 0.0;
  var aSq = 0.0;
  var bSq = 0.0;
  for (var i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    aSq += a[i] * a[i];
    bSq += b[i] * b[i];
  }
  if (aSq == 0 || bSq == 0) return 0.0;
  return dot / (_sqrt(aSq) * _sqrt(bSq));
}

double _sqrt(double x) {
  if (x <= 0) return 0.0;
  var guess = x;
  for (var i = 0; i < 10; i++) {
    guess = 0.5 * (guess + x / guess);
  }
  return guess;
}
