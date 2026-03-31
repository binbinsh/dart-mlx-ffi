// ignore_for_file: unused_import

@TestOn('mac-os')
library;

import 'dart:convert';
import 'dart:io';

import 'package:test/test.dart';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';
import 'package:dart_mlx_ffi/models.dart';

void main() {
  test('loads and runs a synthetic dense qwen3.5 snapshot', () {
    final dir = Directory.systemTemp.createTempSync('dart_mlx_ffi_qwen35_');
    try {
      final config = <String, Object?>{
        'model_type': 'qwen3_5',
        'tie_word_embeddings': false,
        'quantization': <String, Object?>{
          'group_size': 64,
          'bits': 4,
          'mode': 'affine',
        },
        'text_config': <String, Object?>{
          'hidden_size': 4,
          'intermediate_size': 8,
          'num_hidden_layers': 1,
          'num_attention_heads': 1,
          'num_key_value_heads': 1,
          'head_dim': 4,
          'rms_norm_eps': 1e-6,
          'vocab_size': 16,
          'full_attention_interval': 1,
          'linear_num_value_heads': 1,
          'linear_num_key_heads': 1,
          'linear_key_head_dim': 4,
          'linear_value_head_dim': 4,
          'linear_conv_kernel_dim': 4,
          'rope_parameters': <String, Object?>{
            'mrope_section': <int>[2, 1, 1],
            'rope_theta': 1000000.0,
            'partial_rotary_factor': 1.0,
          },
        },
      };
      File('${dir.path}/config.json').writeAsStringSync(jsonEncode(config));

      final tensors = <String, MlxArray>{
        'model.embed_tokens.weight': _f32(
          List<double>.generate(16 * 4, (i) => ((i % 9) - 4) / 8),
          [16, 4],
        ),
        'model.norm.weight': _f32([1, 1, 1, 1], [4]),
        'model.layers.0.input_layernorm.weight': _f32([1, 1, 1, 1], [4]),
        'model.layers.0.post_attention_layernorm.weight': _f32(
          [1, 1, 1, 1],
          [4],
        ),
        'model.layers.0.self_attn.q_proj.weight': _f32(
          List<double>.generate(8 * 4, (i) => ((i % 7) - 3) / 10),
          [8, 4],
        ),
        'model.layers.0.self_attn.k_proj.weight': _f32(
          List<double>.generate(4 * 4, (i) => ((i % 5) - 2) / 9),
          [4, 4],
        ),
        'model.layers.0.self_attn.v_proj.weight': _f32(
          List<double>.generate(4 * 4, (i) => ((i % 6) - 2) / 9),
          [4, 4],
        ),
        'model.layers.0.self_attn.o_proj.weight': _f32(
          List<double>.generate(4 * 4, (i) => ((i % 4) - 1) / 8),
          [4, 4],
        ),
        'model.layers.0.self_attn.q_norm.weight': _f32([1, 1, 1, 1], [4]),
        'model.layers.0.self_attn.k_norm.weight': _f32([1, 1, 1, 1], [4]),
        'model.layers.0.mlp.gate_proj.weight': _f32(
          List<double>.generate(8 * 4, (i) => ((i % 8) - 3) / 10),
          [8, 4],
        ),
        'model.layers.0.mlp.up_proj.weight': _f32(
          List<double>.generate(8 * 4, (i) => ((i % 10) - 5) / 12),
          [8, 4],
        ),
        'model.layers.0.mlp.down_proj.weight': _f32(
          List<double>.generate(4 * 8, (i) => ((i % 7) - 3) / 11),
          [4, 8],
        ),
        'lm_head.weight': _f32(
          List<double>.generate(16 * 4, (i) => ((i % 11) - 5) / 10),
          [16, 4],
        ),
      };

      try {
        mx.io.saveSafetensors('${dir.path}/model.safetensors', tensors);
      } finally {
        for (final value in tensors.values) {
          value.close();
        }
      }

      final runner = Qwen3_5Runner.load(dir.path);
      try {
        final out = runner.run(<int>[1, 2, 3]).astype(MlxDType.MLX_FLOAT32);
        try {
          expect(out.shape, <int>[1, 16]);
          final values = out.toList().cast<double>();
          expect(values, hasLength(16));
          expect(values.every((v) => v.isFinite), isTrue);
        } finally {
          out.close();
        }

        final full = runner
            .runFullLogits(<int>[1, 2, 3])
            .astype(MlxDType.MLX_FLOAT32);
        try {
          expect(full.shape, <int>[1, 16]);
          final values = full.toList().cast<double>();
          expect(values, hasLength(16));
          expect(values.every((v) => v.isFinite), isTrue);
        } finally {
          full.close();
        }

        final next = runner.nextTokenId(<int>[1, 2, 3]);
        expect(next, inInclusiveRange(0, 15));

        final generated = runner.generateGreedy(<int>[1, 2, 3], 2);
        expect(generated, _generateGreedyReference(runner, <int>[1, 2, 3], 2));
      } finally {
        runner.close();
      }
    } finally {
      dir.deleteSync(recursive: true);
    }
  });

  test('loads and runs a synthetic mixed-bit qwen3.5 snapshot', () {
    final dir = Directory.systemTemp.createTempSync(
      'dart_mlx_ffi_qwen35_mixed_',
    );
    try {
      final config = <String, Object?>{
        'model_type': 'qwen3_5',
        'tie_word_embeddings': false,
        'quantization': <String, Object?>{
          'group_size': 32,
          'bits': 4,
          'mode': 'affine',
          'model.embed_tokens': <String, Object?>{
            'group_size': 32,
            'bits': 8,
            'mode': 'affine',
          },
          'model.layers.0.self_attn.q_proj': <String, Object?>{
            'group_size': 32,
            'bits': 8,
            'mode': 'affine',
          },
          'model.layers.0.mlp.down_proj': <String, Object?>{
            'group_size': 32,
            'bits': 8,
            'mode': 'affine',
          },
          'lm_head': <String, Object?>{
            'group_size': 32,
            'bits': 8,
            'mode': 'affine',
          },
        },
        'text_config': <String, Object?>{
          'hidden_size': 32,
          'intermediate_size': 64,
          'num_hidden_layers': 1,
          'num_attention_heads': 1,
          'num_key_value_heads': 1,
          'head_dim': 32,
          'rms_norm_eps': 1e-6,
          'vocab_size': 32,
          'full_attention_interval': 1,
          'linear_num_value_heads': 4,
          'linear_num_key_heads': 2,
          'linear_key_head_dim': 16,
          'linear_value_head_dim': 8,
          'linear_conv_kernel_dim': 4,
          'rope_parameters': <String, Object?>{
            'mrope_section': <int>[16, 8, 8],
            'rope_theta': 1000000.0,
            'partial_rotary_factor': 1.0,
          },
        },
      };
      File('${dir.path}/config.json').writeAsStringSync(jsonEncode(config));

      final tensors = <String, MlxArray>{
        'model.norm.weight': _f32(List<double>.filled(32, 1.0), [32]),
        'model.layers.0.input_layernorm.weight': _f32(
          List<double>.filled(32, 1.0),
          [32],
        ),
        'model.layers.0.post_attention_layernorm.weight': _f32(
          List<double>.filled(32, 1.0),
          [32],
        ),
        'model.layers.0.self_attn.q_norm.weight': _f32(
          List<double>.filled(32, 1.0),
          [32],
        ),
        'model.layers.0.self_attn.k_norm.weight': _f32(
          List<double>.filled(32, 1.0),
          [32],
        ),
        'model.layers.0.self_attn.o_proj.weight': _f32(
          List<double>.generate(32 * 32, (i) => ((i % 17) - 8) / 16),
          [32, 32],
        ),
      };

      _addQuantized(
        tensors,
        'model.embed_tokens',
        List<double>.generate(32 * 32, (i) => ((i % 23) - 11) / 16),
        [32, 32],
        groupSize: 32,
        bits: 8,
      );
      _addQuantized(
        tensors,
        'model.layers.0.self_attn.q_proj',
        List<double>.generate(64 * 32, (i) => ((i % 19) - 9) / 20),
        [64, 32],
        groupSize: 32,
        bits: 8,
      );
      _addQuantized(
        tensors,
        'model.layers.0.self_attn.k_proj',
        List<double>.generate(32 * 32, (i) => ((i % 13) - 6) / 16),
        [32, 32],
        groupSize: 32,
        bits: 4,
      );
      _addQuantized(
        tensors,
        'model.layers.0.self_attn.v_proj',
        List<double>.generate(32 * 32, (i) => ((i % 15) - 7) / 18),
        [32, 32],
        groupSize: 32,
        bits: 4,
      );
      _addQuantized(
        tensors,
        'model.layers.0.mlp.gate_proj',
        List<double>.generate(64 * 32, (i) => ((i % 21) - 10) / 18),
        [64, 32],
        groupSize: 32,
        bits: 4,
      );
      _addQuantized(
        tensors,
        'model.layers.0.mlp.up_proj',
        List<double>.generate(64 * 32, (i) => ((i % 25) - 12) / 20),
        [64, 32],
        groupSize: 32,
        bits: 4,
      );
      _addQuantized(
        tensors,
        'model.layers.0.mlp.down_proj',
        List<double>.generate(32 * 64, (i) => ((i % 27) - 13) / 22),
        [32, 64],
        groupSize: 32,
        bits: 8,
      );
      _addQuantized(
        tensors,
        'lm_head',
        List<double>.generate(32 * 32, (i) => ((i % 29) - 14) / 24),
        [32, 32],
        groupSize: 32,
        bits: 8,
      );

      try {
        mx.io.saveSafetensors('${dir.path}/model.safetensors', tensors);
      } finally {
        for (final value in tensors.values) {
          value.close();
        }
      }

      final runner = Qwen3_5Runner.load(dir.path);
      try {
        final out = runner.run(<int>[1, 2, 3]).astype(MlxDType.MLX_FLOAT32);
        try {
          expect(out.shape, <int>[1, 16]);
          final values = out.toList().cast<double>();
          expect(values, hasLength(16));
          expect(values.every((v) => v.isFinite), isTrue);
        } finally {
          out.close();
        }

        final full = runner
            .runFullLogits(<int>[1, 2, 3])
            .astype(MlxDType.MLX_FLOAT32);
        try {
          expect(full.shape, <int>[1, 32]);
          final values = full.toList().cast<double>();
          expect(values, hasLength(32));
          expect(values.every((v) => v.isFinite), isTrue);
        } finally {
          full.close();
        }

        final generated = runner.generateGreedy(<int>[1, 2, 3], 2);
        expect(generated, _generateGreedyReference(runner, <int>[1, 2, 3], 2));
      } finally {
        runner.close();
      }
    } finally {
      dir.deleteSync(recursive: true);
    }
  });
}

MlxArray _f32(List<double> values, List<int> shape) =>
    MlxArray.fromFloat32List(values, shape: shape);

void _addQuantized(
  Map<String, MlxArray> tensors,
  String prefix,
  List<double> values,
  List<int> shape, {
  required int groupSize,
  required int bits,
}) {
  final weight = _f32(values, shape);
  try {
    final quantized = mx.quant.quantize(
      weight,
      groupSize: groupSize,
      bits: bits,
      mode: 'affine',
    );
    tensors['$prefix.weight'] = quantized.weights;
    tensors['$prefix.scales'] = quantized.scales;
    tensors['$prefix.biases'] = quantized.biases!;
  } finally {
    weight.close();
  }
}

List<int> _generateGreedyReference(
  Qwen3_5Runner runner,
  List<int> promptIds,
  int maxNewTokens, {
  int? eosTokenId,
}) {
  final tokens = List<int>.from(promptIds);
  for (var index = 0; index < maxNewTokens; index++) {
    final next = runner.nextTokenId(tokens);
    tokens.add(next);
    if (eosTokenId != null && next == eosTokenId) {
      break;
    }
  }
  return tokens;
}
