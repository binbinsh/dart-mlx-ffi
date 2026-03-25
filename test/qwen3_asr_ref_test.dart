import 'dart:convert';
import 'dart:io';

import 'package:test/test.dart';

const _modelDir = 'tmp/Qwen3-ASR-1.7B';
const _text = 'Hello from the speech test. This is a simple speech test.';

void main() {
  test(
    'qwen3-asr reference transcription matches expected text when local model is available',
    () async {
      final modelFile = File('$_modelDir/model-00001-of-00002.safetensors');
      if (!modelFile.existsSync()) {
        return;
      }
      final tempDir = await Directory.systemTemp.createTemp('qwen3_asr_ref_');
      addTearDown(() => tempDir.delete(recursive: true));
      final audioPath = '${tempDir.path}/qwen3_asr_ref_test.aiff';

      final result = await Process.run('uv', [
        'run',
        '--isolated',
        '--with',
        'qwen-asr',
        'python',
        'private_ane/models/qwen3_asr/python/ref_transcribe.py',
        '--model',
        _modelDir,
        '--audio-out',
        audioPath,
        '--text',
        _text,
      ]);

      expect(result.exitCode, 0, reason: result.stderr.toString());
      final report =
          jsonDecode(result.stdout as String) as Map<String, Object?>;
      expect(report['match'], true, reason: report.toString());
    },
    timeout: const Timeout(Duration(minutes: 4)),
  );
}
