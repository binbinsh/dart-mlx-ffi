library;

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

import 'albert.dart';
import 'args.dart';
import 'core.dart';
import 'loader.dart';
import 'prosody.dart';
import 'text.dart';

final class KittenFrontResult {
  const KittenFrontResult({
    required this.asr,
    required this.f0Pred,
    required this.nPred,
    required this.style,
    required this.predDur,
  });

  final MlxArray asr;
  final MlxArray f0Pred;
  final MlxArray nPred;
  final MlxArray style;
  final MlxArray predDur;

  void close() {
    predDur.close();
    style.close();
    nPred.close();
    f0Pred.close();
    asr.close();
  }
}

final class KittenFrontRunner {
  KittenFrontRunner._({
    required this.config,
    required this.tensors,
    required this.bert,
    required this.bertEncoder,
    required this.predictor,
    required this.textEncoder,
  });

  factory KittenFrontRunner.load(String snapshotPath) {
    final config = ModelConfig.fromSnapshot(snapshotPath);
    final tensors = loadKittenTensors(snapshotPath);
    return KittenFrontRunner._(
      config: config,
      tensors: tensors,
      bert: KittenAlbert.load(
        tensors,
        prefix: 'bert',
        config: config.plbert,
        quant: config.quantization,
      ),
      bertEncoder: Linear.load(
        tensors,
        'bert_encoder',
        quant: config.quantization,
      ),
      predictor: ProsodyPredictor.load(
        tensors,
        prefix: 'predictor',
        config: config,
      ),
      textEncoder: TextEncoder.load(
        tensors,
        prefix: 'text_encoder',
        config: config,
      ),
    );
  }

  final ModelConfig config;
  final Map<String, MlxArray> tensors;
  final KittenAlbert bert;
  final Linear bertEncoder;
  final ProsodyPredictor predictor;
  final TextEncoder textEncoder;

  KittenFrontResult run(MlxArray inputIds, MlxArray refS, {double speed = 1.0}) {
    if (inputIds.shape[0] != 1 || refS.shape[0] != 1) {
      throw ArgumentError('KittenFrontRunner currently expects batch size 1.');
    }

    final textMask = MlxArray.fromBoolList(
      List<bool>.filled(inputIds.shape[1], false),
      shape: [1, inputIds.shape[1]],
    );
    final attentionMask = MlxArray.ones(
      [1, inputIds.shape[1]],
      dtype: MlxDType.MLX_INT32,
    );

    final bertOutput = bert(inputIds, attentionMask: attentionMask);
    attentionMask.close();

    final dEn = mx.transposeAxes(bertEncoder(bertOutput.sequenceOutput), [0, 2, 1]);
    bertOutput.sequenceOutput.close();
    bertOutput.pooledOutput.close();

    final prosodyStyle = refS.slice(start: [0, 128], stop: [1, 256]);
    final decoderStyle = refS.slice(start: [0, 0], stop: [1, 128]);

    final d = predictor.textEncoder(dEn, prosodyStyle, textMask);
    dEn.close();
    final x = predictor.lstm(d).output;
    final durationLogits = predictor.durationProj(x);
    x.close();
    final speedScalar = scalar(speed);
    final duration = durationLogits.sigmoid().sum(axis: -1) / speedScalar;
    speedScalar.close();
    durationLogits.close();
    final rounded = duration.round().reshape([inputIds.shape[1]]);
    final predDurValues = rounded.toList().cast<double>();
    final predDurList = predDurValues
        .map((value) => value < 1.0 ? 1 : value.toInt())
        .toList(growable: false);
    final predDur = MlxArray.fromInt32List(
      predDurList,
      shape: [inputIds.shape[1]],
    );
    rounded.close();
    duration.close();

    final alignment = _buildAlignment(predDurList, inputIds.shape[1]);
    final transposedD = mx.transposeAxes(d, [0, 2, 1]);
    final en = mx.matmul(transposedD, alignment);
    transposedD.close();
    final prosody = predictor.f0Ntrain(en, prosodyStyle);
    en.close();

    final encodedText = textEncoder(inputIds, textMask);
    textMask.close();
    final asr = mx.matmul(encodedText, alignment);
    encodedText.close();
    alignment.close();
    d.close();
    prosodyStyle.close();
    return KittenFrontResult(
      asr: asr,
      f0Pred: prosody.f0,
      nPred: prosody.noise,
      style: decoderStyle,
      predDur: predDur,
    );
  }
}

MlxArray _buildAlignment(List<int> counts, int tokenCount) {
  final frames = counts.fold<int>(0, (sum, value) => sum + value);
  final values = List<double>.filled(tokenCount * frames, 0.0);
  var frame = 0;
  for (var token = 0; token < counts.length; token++) {
    final count = counts[token];
    for (var step = 0; step < count; step++) {
      values[(token * frames) + frame] = 1.0;
      frame++;
    }
  }
  return MlxArray.fromFloat32List(values, shape: [1, tokenCount, frames]);
}
