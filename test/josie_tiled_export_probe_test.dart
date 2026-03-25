import 'dart:convert';
import 'dart:io';

import 'package:test/test.dart';

void main() {
  test(
    'josie tiled export probe reports conv and linear variants',
    () async {
      final result = await Process.run('uv', [
        'run',
        'python',
        'private_ane/models/josie/python/tiled_export_probe.py',
        '--json',
        '--layer',
        '0',
        '--spatial',
        '1',
        '--tile-size',
        '1024',
      ]);

      expect(result.exitCode, 0, reason: result.stderr.toString());
      final report = jsonDecode(result.stdout as String) as Map<String, Object?>;
      expect(report['runtime'], 'josie_tiled_export_probe');
      final models = (report['models'] as List).cast<Map<String, Object?>>();
      final names = models.map((model) => model['name']).toSet();
      expect(names.contains('q_proj_conv'), isTrue);
      expect(names.contains('q_proj_linear'), isTrue);
      expect(names.contains('q_proj_tiled_conv_1024'), isTrue);
      expect(names.contains('q_proj_tiled_linear_1024'), isTrue);
      expect(names.contains('o_proj_conv'), isTrue);
      expect(names.contains('o_proj_linear'), isTrue);
    },
    timeout: const Timeout(Duration(minutes: 4)),
  );
}
