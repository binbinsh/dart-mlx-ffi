import 'dart:convert';
import 'dart:io';

import 'package:test/test.dart';

void main() {
  test('josie full model hybrid private ane matches fp16 reference on single-token input', () async {
    final result = await Process.run('uv', [
      'run',
      'python',
      'private_ane/models/josie/python/full_model_compare.py',
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
    expect(report['last_token_argmax_match_vs_fp16_ref'], true);
    expect((report['max_abs_diff_vs_fp16_ref'] as num) < 0.01, isTrue);
    expect((report['mean_abs_diff_vs_fp16_ref'] as num) < 0.001, isTrue);
  });
}
