import 'dart:convert';
import 'dart:io';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

import 'mlx_audio/models/kitten_tts/kitten_tts.dart';

void main(List<String> args) {
  if (args.length != 4) {
    stderr.writeln(
      'usage: dart run benchmark/kitten_tts/front_trace.dart <snapshot_path> <input_ids_json> <ref_s_json> <output_path>',
    );
    exitCode = 64;
    return;
  }

  final snapshotPath = args[0];
  final inputIds = _intMatrixToArray(args[1]);
  final refS = _floatMatrixToArray(args[2]);
  final outputPath = args[3];

  final runner = KittenFrontRunner.load(snapshotPath);
  try {
    final textMask = MlxArray.fromBoolList(
      List<bool>.filled(inputIds.shape[1], false),
      shape: [1, inputIds.shape[1]],
    );
    final attentionMask = MlxArray.ones(
      [1, inputIds.shape[1]],
      dtype: MlxDType.MLX_INT32,
    );
    final bertOut = runner.bert(inputIds, attentionMask: attentionMask);
    attentionMask.close();
    final dEn = mx.transposeAxes(
      runner.bertEncoder(bertOut.sequenceOutput),
      [0, 2, 1],
    );
    final prosodyStyle = refS.slice(start: [0, 128], stop: [1, 256]);
    final decoderStyle = refS.slice(start: [0, 0], stop: [1, 128]);

    final d = runner.predictor.textEncoder(dEn, prosodyStyle, textMask);
    final durationLstm = runner.predictor.lstm(d).output;
    final durationLogits = runner.predictor.durationProj(durationLstm);
    final duration = durationLogits.sigmoid().sum(axis: -1);
    final rounded = duration.round().reshape([inputIds.shape[1]]);
    final predDurList = rounded
        .toList()
        .cast<double>()
        .map((value) => value < 1.0 ? 1 : value.toInt())
        .toList(growable: false);
    final predDur = MlxArray.fromInt32List(
      predDurList,
      shape: [predDurList.length],
    );
    final alignment = _buildAlignment(predDurList, inputIds.shape[1]);
    final dT = mx.transposeAxes(d, [0, 2, 1]);
    final en = mx.matmul(dT, alignment);
    final sharedIn = mx.transposeAxes(en, [0, 2, 1]);
    final sharedOut = runner.predictor.shared(sharedIn).output;

    var f0 = mx.transposeAxes(sharedOut, [0, 2, 1]);
    final f0Blocks = <MlxArray>[];
    for (final block in runner.predictor.f0Blocks) {
      final next = block(f0, prosodyStyle);
      f0Blocks.add(next);
      f0 = next;
    }

    final block1 = runner.predictor.f0Blocks[1];
    final block1Input = f0Blocks[0];
    final block1Shortcut = _shortcutTrace(block1, block1Input);
    final block1Residual = _residualTrace(block1, block1Input, prosodyStyle);

    final f0ProjIn = mx.transposeAxes(f0, [0, 2, 1]);
    final f0Projected = runner.predictor.f0Proj(f0ProjIn);
    final f0Back = mx.transposeAxes(f0Projected, [0, 2, 1]);
    final f0Pred = f0Back.reshape([f0Back.shape[0], f0Back.shape[2]]);

    final encodedText = runner.textEncoder(inputIds, textMask);
    final asr = mx.matmul(encodedText, alignment);

    final tensors = <String, MlxArray>{
      'd_en': dEn,
      'd': d,
      'duration_lstm': durationLstm,
      'duration_logits': durationLogits,
      'pred_dur': predDur,
      'alignment': alignment,
      'en': en,
      'shared_out': sharedOut,
      'f0_block0': f0Blocks[0],
      'f0_block1_shortcut': block1Shortcut,
      'f0_block1_residual': block1Residual,
      'f0_block1': f0Blocks[1],
      'f0_block2': f0Blocks[2],
      'f0_pred': f0Pred,
      'decoder_style': decoderStyle,
      'asr': asr,
    };
    mx.io.saveSafetensors(outputPath, tensors);

    bertOut.sequenceOutput.close();
    bertOut.pooledOutput.close();
    textMask.close();
    rounded.close();
    dT.close();
    sharedIn.close();
    f0ProjIn.close();
    f0Projected.close();
    f0Back.close();
    prosodyStyle.close();
    encodedText.close();
    stdout.writeln(outputPath);
  } finally {
    inputIds.close();
    refS.close();
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

MlxArray _shortcutTrace(AdainResBlk1d block, MlxArray input) {
  var x = mx.transposeAxes(input, [0, 2, 1]);
  if (block.upsample) {
    final repeated = x.repeat(2, axis: 1);
    x.close();
    x = repeated;
  }
  final back = mx.transposeAxes(x, [0, 2, 1]);
  x.close();
  if (!block.learnedShortcut) {
    return back;
  }
  final convIn = mx.transposeAxes(back, [0, 2, 1]);
  back.close();
  final convOut = block.conv1x1!.call(convIn);
  convIn.close();
  final restored = mx.transposeAxes(convOut, [0, 2, 1]);
  convOut.close();
  return restored;
}

MlxArray _residualTrace(AdainResBlk1d block, MlxArray input, MlxArray style) {
  var x = block.norm1(input, style);
  final act1 = leakyRelu(x);
  x.close();
  x = act1;

  var work = mx.transposeAxes(x, [0, 2, 1]);
  x.close();
  if (block.upsample) {
    final pooled = block.pool!.call(work, transpose: true);
    work.close();
    final zero = scalar(0.0, dtype: pooled.dtype);
    final padded = mx.pad(
      pooled,
      axes: [1],
      lowPads: [0],
      highPads: [1],
      padValue: zero,
    );
    zero.close();
    pooled.close();
    work = padded;
  }
  x = mx.transposeAxes(work, [0, 2, 1]);
  work.close();

  work = mx.transposeAxes(x, [0, 2, 1]);
  x.close();
  final conv1Out = block.conv1.call(work);
  work.close();
  x = mx.transposeAxes(conv1Out, [0, 2, 1]);
  conv1Out.close();

  final normed = block.norm2(x, style);
  x.close();
  x = leakyRelu(normed);
  normed.close();

  work = mx.transposeAxes(x, [0, 2, 1]);
  x.close();
  final conv2Out = block.conv2.call(work);
  work.close();
  final out = mx.transposeAxes(conv2Out, [0, 2, 1]);
  conv2Out.close();
  return out;
}

MlxArray _intMatrixToArray(String jsonText) {
  final matrix = (jsonDecode(jsonText) as List<Object?>)
      .map(
        (row) => (row as List<Object?>).cast<num>().map((v) => v.toInt()).toList(),
      )
      .toList(growable: false);
  final rows = matrix.length;
  final cols = matrix.first.length;
  final flat = <int>[
    for (final row in matrix) ...row,
  ];
  return MlxArray.fromInt32List(flat, shape: [rows, cols]);
}

MlxArray _floatMatrixToArray(String jsonText) {
  final matrix = (jsonDecode(jsonText) as List<Object?>)
      .map(
        (row) => (row as List<Object?>)
            .cast<num>()
            .map((v) => v.toDouble())
            .toList(),
      )
      .toList(growable: false);
  final rows = matrix.length;
  final cols = matrix.first.length;
  final flat = <double>[
    for (final row in matrix) ...row,
  ];
  return MlxArray.fromFloat32List(flat, shape: [rows, cols]);
}
