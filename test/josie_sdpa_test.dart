import 'dart:convert';
import 'dart:io';

import 'package:test/test.dart';

void main() {
  test('josie sdpa core private ane matches mlx within tolerance', () async {
    final result = await Process.run('uv', [
      'run',
      'python',
      'private_ane/models/josie/python/sdpa_compare.py',
      '--json',
      '--seq-len',
      '8',
      '--warmup',
      '2',
      '--iters',
      '5',
    ]);

    expect(result.exitCode, 0, reason: result.stderr.toString());
    final report = jsonDecode(result.stdout as String) as Map<String, Object?>;
    expect((report['ane_per_iter_ms'] as num) > 0, isTrue);
    expect((report['mlx_per_iter_ms'] as num) > 0, isTrue);
    expect((report['max_abs_diff'] as num) < 0.01, isTrue);
    expect((report['mean_abs_diff'] as num) < 0.001, isTrue);
  });
}
