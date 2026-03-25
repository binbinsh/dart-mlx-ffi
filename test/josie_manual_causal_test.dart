import 'dart:convert';
import 'dart:io';

import 'package:test/test.dart';

void main() {
  test(
    'josie manual causal private ane path runs on single-token input',
    () async {
      final result = await Process.run('uv', [
        'run',
        'python',
        'private_ane/models/josie/python/manual_causal_compare.py',
        '--json',
        '--warmup',
        '1',
        '--iters',
        '1',
        '--prompt',
        'Hello',
        '--token-limit',
        '1',
      ]);

      expect(result.exitCode, 0, reason: result.stderr.toString());
      final report = jsonDecode(result.stdout as String) as Map<String, Object?>;
      expect(report['token_count'], 1);
      expect((report['ane_total_ms'] as num) > 0, isTrue);
      expect((report['mlx_total_ms'] as num) > 0, isTrue);
      expect(report['argmax_matches_mlx'], true);
    },
    timeout: const Timeout(Duration(minutes: 2)),
  );
}
