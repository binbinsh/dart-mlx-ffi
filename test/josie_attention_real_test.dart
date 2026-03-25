import 'dart:convert';
import 'dart:io';

import 'package:test/test.dart';

void main() {
  test('josie real attention hybrid private ane stays close to mlx', () async {
    final result = await Process.run('uv', [
      'run',
      'python',
      'private_ane/models/josie/python/attention_real_compare.py',
      '--json',
      '--warmup',
      '2',
      '--iters',
      '5',
    ]);

    expect(result.exitCode, 0, reason: result.stderr.toString());
    final report = jsonDecode(result.stdout as String) as Map<String, Object?>;
    expect((report['ane_sdpa_per_iter_ms'] as num) > 0, isTrue);
    expect((report['mlx_attention_per_iter_ms'] as num) > 0, isTrue);
    expect((report['max_abs_diff'] as num) < 0.4, isTrue);
    expect((report['mean_abs_diff'] as num) < 0.03, isTrue);
  });
}
