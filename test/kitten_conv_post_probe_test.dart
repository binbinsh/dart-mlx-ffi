import 'dart:convert';
import 'dart:io';

import 'package:test/test.dart';

void main() {
  test(
    'kitten conv_post probe runs and reports MLX and ANE results',
    () async {
      final result = await Process.run('uv', [
        'run',
        'python',
        'benchmark/kitten_tts/conv_post_probe.py',
      ]);

      expect(result.exitCode, 0, reason: result.stderr.toString());
      final report = jsonDecode(result.stdout as String) as Map<String, Object?>;
      expect(report['runtime'], 'kitten_tts_conv_post_probe');
      expect(report['mlx_per_iter_ms'], isA<num>());
      final ane = report['ane'] as Map<String, Object?>;
      expect(ane['returncode'], isA<num>());
    },
    timeout: const Timeout(Duration(minutes: 4)),
  );
}
