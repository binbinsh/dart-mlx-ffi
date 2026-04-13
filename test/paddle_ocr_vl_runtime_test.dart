@TestOn('mac-os')
library;

import 'dart:io';

import 'package:test/test.dart';

import 'package:dart_mlx_ffi/src/models/paddle_ocr_vl/paddle_ocr_vl.dart';

final _snapshotPath = () {
  final home = Platform.environment['HOME']!;
  return '$home/.cache/huggingface/hub/'
      'models--mlx-community--PaddleOCR-VL-1.5-8bit/'
      'snapshots/37d4c85284434b6e6fd4c03f8b719b1aefaa013c';
}();

void main() {
  final snapshotExists = Directory(_snapshotPath).existsSync();

  group('PaddleOCR-VL runtime defaults', () {
    late PaddleOcrVlConfig config;

    setUpAll(() {
      if (!snapshotExists) {
        return;
      }
      config = PaddleOcrVlConfig.fromSnapshot(_snapshotPath);
    });

    tearDown(() {
      PaddleOcrVlDebugOverrides.reset();
    });

    test('defaults to uniform KV cache scheme', () {
      if (!snapshotExists) {
        return;
      }
      expect(config.kvCacheQuantSchemeForCurrentPlatform, 'uniform');
      expect(config.turboQuantBitsForCurrentPlatform, isNull);
      expect(config.turboDensePrefillForCurrentPlatform, isFalse);
    });

    test('debug override can force turboquant KV cache scheme', () {
      if (!snapshotExists) {
        return;
      }
      PaddleOcrVlDebugOverrides.kvQuantScheme = 'turboquant';
      expect(config.kvCacheQuantSchemeForCurrentPlatform, 'turboquant');
      expect(config.turboQuantBitsForCurrentPlatform, isNotNull);
      expect(config.turboDensePrefillForCurrentPlatform, isTrue);
    });

    test('debug override can force uniform KV cache bits', () {
      if (!snapshotExists) {
        return;
      }
      PaddleOcrVlDebugOverrides.kvBits = 4;
      expect(config.kvCacheQuantSchemeForCurrentPlatform, 'uniform');
      expect(config.kvCacheQuantBitsForCurrentPlatform, 4);
    });
  });
}
