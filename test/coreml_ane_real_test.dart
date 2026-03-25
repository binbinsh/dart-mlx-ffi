import 'dart:convert';
import 'dart:io';

import 'package:test/test.dart';

void main() {
  test(
    'coreml ane benchmark runs two real models under latency budget',
    () async {
      final result = await Process.run('uv', [
        'run',
        'python',
        'tool/coreml_ane_real_bench.py',
        '--json',
        '--warmup',
        '2',
        '--iters',
        '20',
        '--max-ms',
        '5.0',
      ]);

      expect(result.exitCode, 0, reason: result.stderr.toString());
      final report =
          jsonDecode(result.stdout as String) as Map<String, Object?>;
      final models = report['models'] as List<Object?>;
      expect(models, hasLength(2));

      final first = Map<String, Object?>.from(models[0]! as Map);
      final second = Map<String, Object?>.from(models[1]! as Map);
      expect(first['name'], 'iris-3class');
      expect(second['name'], 'iris-binary');
      expect((first['per_iter_ms'] as num) < 5.0, isTrue);
      expect((second['per_iter_ms'] as num) < 5.0, isTrue);
    },
  );
}
