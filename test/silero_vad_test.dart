@TestOn('mac-os')
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:test/test.dart';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';
import 'package:dart_mlx_ffi/models.dart';

void main() {
  test('loads a synthetic Silero VAD bundle and processes one frame', () async {
    final dir = Directory.systemTemp.createTempSync('dart_mlx_ffi_silero_');
    try {
      final manifest = <String, Object?>{
        'format': 'cmdspace-mlx-silero-vad/v1',
        'model_id': 'synthetic/silero-vad',
        'weights': 'model.safetensors',
        'sample_rate': 16000,
        'window_samples': 512,
        'context_samples': 64,
        'hidden_size': 128,
        'n_fft': 256,
        'hop_length': 128,
      };
      File(
        '${dir.path}/cmdspace_mlx_silero_vad.json',
      ).writeAsStringSync(jsonEncode(manifest));

      final tensors = <String, MlxArray>{
        'stft_conv.weight': _zeros([258, 1, 256]),
        'conv1.weight': _zeros([128, 129, 3]),
        'conv1.bias': _zeros([128]),
        'conv2.weight': _zeros([64, 128, 3]),
        'conv2.bias': _zeros([64]),
        'conv3.weight': _zeros([64, 64, 3]),
        'conv3.bias': _zeros([64]),
        'conv4.weight': _zeros([128, 64, 3]),
        'conv4.bias': _zeros([128]),
        'lstm_cell.weight_ih': _zeros([512, 128]),
        'lstm_cell.weight_hh': _zeros([512, 128]),
        'lstm_cell.bias_ih': _zeros([512]),
        'lstm_cell.bias_hh': _zeros([512]),
        'final_conv.weight': _zeros([1, 128, 1]),
        'final_conv.bias': _zeros([1]),
      };

      try {
        mx.io.saveSafetensors('${dir.path}/model.safetensors', tensors);
      } finally {
        for (final tensor in tensors.values) {
          tensor.close();
        }
      }

      final bundle = await loadSileroVadBundle(dir.path);
      try {
        final runtime = SileroVadRuntime(bundle);
        final state = runtime.createState();
        try {
          final samples = Float32List(runtime.frameSamples);
          final result = runtime.processFrame(samples: samples, state: state);
          try {
            expect(runtime.sampleRate, 16000);
            expect(runtime.frameSamples, 576);
            expect(result.probability, inInclusiveRange(0.0, 1.0));
            expect(result.probability.isFinite, isTrue);
          } finally {
            result.state.close();
          }
        } finally {
          state.close();
        }
      } finally {
        bundle.close();
      }
    } finally {
      dir.deleteSync(recursive: true);
    }
  });
}

MlxArray _zeros(List<int> shape) =>
    MlxArray.full(shape, 0.0, dtype: MlxDType.MLX_FLOAT32);
