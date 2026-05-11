import 'dart:io';

import 'package:dart_inference/models.dart';

Future<void> main(List<String> args) async {
  final prompt = args.isEmpty
      ? 'Answer in one short sentence: what is 2 + 2?'
      : args.join(' ');
  final env = Platform.environment;
  final quant = env['DS4_QUANT']?.trim().toLowerCase() == 'q4'
      ? Ds4Quant.q4
      : Ds4Quant.q2;
  final contextLength = int.tryParse(env['DS4_CONTEXT'] ?? '') ?? 4096;
  final maxTokens = int.tryParse(env['DS4_MAX_TOKENS'] ?? '') ?? 32;
  final thinking = switch (env['DS4_THINKING']?.trim().toLowerCase()) {
    'max' => Ds4Thinking.max,
    'high' => Ds4Thinking.high,
    _ => Ds4Thinking.disabled,
  };

  stderr.writeln(
    'ds4 FFI inference starting: quant=${quant.cliName}, '
    'context=$contextLength, maxTokens=$maxTokens',
  );
  final backend = Ds4FfiBackend(
    Ds4FfiConfig(
      libraryPath: env['DS4_LIBRARY'] ?? '',
      sourceDir: env['DS4_SOURCE_DIR'] ?? '',
      modelPath: env['DS4_MODEL'] ?? '',
      quant: quant,
      contextLength: contextLength,
      token: _token(env),
    ),
  );
  try {
    await for (final chunk in backend.stream(
      Ds4ChatRequest(
        messages: [
          <String, Object?>{'role': 'user', 'content': prompt},
        ],
        maxTokens: maxTokens,
        temperature: 0,
        thinking: thinking,
      ),
    )) {
      if (chunk.kind == Ds4ChatChunkKind.reasoning && chunk.text.isNotEmpty) {
        stderr.write(chunk.text);
      } else if (chunk.kind == Ds4ChatChunkKind.content &&
          chunk.text.isNotEmpty) {
        stdout.write(chunk.text);
      }
    }
    stdout.writeln();
  } finally {
    await backend.close();
  }
}

String? _token(Map<String, String> env) {
  final value = (env['DS4_TOKEN'] ?? env['HF_TOKEN'] ?? '').trim();
  return value.isEmpty ? null : value;
}
