import 'dart:convert';
import 'dart:io';

import 'package:test/test.dart';

void main() {
  test(
    'josie layer ops probe reports real-weight block statuses',
    () async {
      final result = await Process.run('uv', [
        'run',
        'python',
        'private_ane/models/josie/python/layer_ops_probe.py',
        '--json',
        '--layer',
        '0',
        '--spatial',
        '1',
      ]);

      expect(result.exitCode, 0, reason: result.stderr.toString());
      final report = jsonDecode(result.stdout as String) as Map<String, Object?>;
      expect(report['runtime'], 'josie_layer_ops_probe');
      final models = (report['models'] as List).cast<Map<String, Object?>>();
      final names = models.map((model) => model['name']).toSet();
      expect(names.contains('q_proj'), isTrue);
      expect(names.contains('k_proj'), isTrue);
      expect(names.contains('v_proj'), isTrue);
      expect(names.contains('o_proj'), isTrue);
      expect(names.contains('ffn'), isTrue);
    },
    timeout: const Timeout(Duration(minutes: 4)),
  );
}
