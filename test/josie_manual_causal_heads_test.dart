import 'dart:convert';
import 'dart:io';

import 'package:test/test.dart';

void main() {
  test(
    'josie split-head manual causal private ane matches mlx argmax on a short prompt',
    () async {
      final result = await Process.run('uv', [
        'run',
        'python',
        'private_ane/models/josie/python/manual_causal_heads_compare.py',
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
      expect(report['token_count'], 4);
      expect(report['argmax_matches_mlx'], true);
      expect((report['max_abs_diff'] as num) < 0.5, isTrue);
      expect((report['mean_abs_diff'] as num) < 0.05, isTrue);
    },
    timeout: const Timeout(Duration(minutes: 2)),
  );
}
