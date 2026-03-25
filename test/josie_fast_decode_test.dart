import 'dart:convert';
import 'dart:io';

import 'package:test/test.dart';

void main() {
  test(
    'josie fused fast decode private ane benchmark runs',
    () async {
      final result = await Process.run('uv', [
        'run',
        'python',
        'private_ane/models/josie/python/fast_decode_compare.py',
        '--json',
        '--warmup',
        '1',
        '--iters',
        '1',
        '--token-limit',
        '4',
      ]);

      expect(result.exitCode, 0, reason: result.stderr.toString());
      final report = jsonDecode(result.stdout as String) as Map<String, Object?>;
      expect(report['runtime'], 'josie_fast_decode_fused_private_vs_mlx');
      expect(report['token_count'], 4);
      expect(report['ane_total_ms'], isA<num>());
      expect(report['mlx_total_ms'], isA<num>());
      expect(report['max_abs_diff'], isA<num>());
      expect(report['mean_abs_diff'], isA<num>());
    },
    timeout: const Timeout(Duration(minutes: 2)),
  );
}
