import 'dart:convert';
import 'dart:io';
import 'dart:math';

import 'package:dart_inference/models.dart';

Future<void> main(List<String> args) async {
  late final _Opts opts;
  try {
    opts = _parseArgs(args);
  } catch (error, stack) {
    if (args.contains('--json')) {
      stdout.writeln(
        jsonEncode({'ok': false, 'error': '$error', 'stack': '$stack'}),
      );
    } else {
      stderr.writeln('SMOKE FAILED: $error');
      stderr.writeln(stack);
    }
    exitCode = 64;
    return;
  }
  final paths = Sarashina2TtsPaths(modelDir: opts.modelDir);
  final cosyPaths = CosyVoice2Paths(modelDir: opts.modelDir);
  void log(String message) {
    if (opts.json) {
      stderr.writeln(message);
    } else {
      stdout.writeln(message);
    }
  }

  log('==> loading Sarashina2 split LLM from ${paths.modelDir}');
  final bundle = CosyVoice2PartialOnnxBundle.load(
    paths: cosyPaths,
    provider: opts.provider,
    deviceId: opts.deviceId,
    requireProvider: false,
    numThreads: opts.numThreads,
    componentNames: const {'llm_prefill', 'llm_decode', 'llm_decoder_head'},
  );

  for (final status in bundle.statuses) {
    if (status.file.name == 'llm_prefill' ||
        status.file.name == 'llm_decode' ||
        status.file.name == 'llm_decoder_head') {
      log(
        '   ${status.file.name}: exists=${status.exists}, '
        'loaded=${status.loaded}, provider=${status.selectedProvider}, '
        'loadMs=${(status.loadElapsedMicroseconds ?? 0) / 1000.0}'
        '${status.error == null ? "" : ", error=${status.error}"}',
      );
    }
  }

  Sarashina2LlmDriver? driver;
  Sarashina2BaseTokenizer? tokenizer;
  Sarashina2LlmState? state;
  Sarashina2SemanticSamplerState? samplerState;
  Map<String, Object?>? payload;
  try {
    driver = await Sarashina2LlmDriver.load(bundle: bundle, paths: paths);
    tokenizer = Sarashina2BaseTokenizer.fromFile(
      paths.tokenizerSidecar,
      tokenMap: driver.tokenMap,
    );

    final prompt = buildSarashina2Prompt(
      text: opts.text,
      promptText: opts.promptText,
      promptTokens: opts.promptSemanticTokens,
    );
    log('==> prompt: ${jsonEncode(prompt)}');
    final promptIds = tokenizer
        .encodePromptTokenIds(
          text: opts.text,
          promptText: opts.promptText,
          promptTokens: opts.promptSemanticTokens,
        )
        .toList(growable: false);
    log('   prompt token ids (${promptIds.length}): $promptIds');

    final prefillStart = DateTime.now().microsecondsSinceEpoch;
    state = driver.prefillTokenIds(promptIds);
    final prefillMs =
        (DateTime.now().microsecondsSinceEpoch - prefillStart) / 1000.0;
    log(
      '==> prefill: ${prefillMs.toStringAsFixed(2)} ms '
      '(seq=${promptIds.length}, totalSeq=${state.totalSeq})',
    );

    samplerState = Sarashina2SemanticSamplerState(tokenMap: driver.tokenMap);
    final rng = Random(opts.seed);
    final generated = <int>[];
    final stepLatencies = <double>[];
    var stoppedOnEos = false;
    for (var step = 0; step < opts.steps; step += 1) {
      final tokenId = driver.sampleNextSemanticTokenizerId(
        state: state,
        generatedSemanticTokens: generated,
        samplerState: samplerState,
        eosId: sarashina2EosTokenId,
        temperature: opts.temperature,
        topP: opts.topP,
        frequencyPenalty: opts.frequencyPenalty,
        randomDraw: rng.nextDouble(),
      );
      if (tokenId == sarashina2EosTokenId) {
        stoppedOnEos = true;
        log('   step $step sampled EOS');
        break;
      }
      final semanticId = driver.tokenMap.semanticIdForTokenizerId(tokenId);
      if (semanticId == null) {
        throw StateError('sampled non-semantic tokenizer id $tokenId');
      }
      generated.add(semanticId);
      samplerState.recordSemanticId(semanticId);
      final decodeStart = DateTime.now().microsecondsSinceEpoch;
      driver.decodeTokenId(state: state, tokenId: tokenId);
      final decodeMs =
          (DateTime.now().microsecondsSinceEpoch - decodeStart) / 1000.0;
      stepLatencies.add(decodeMs);
      log(
        '   step $step tokenizerId=$tokenId semantic=$semanticId '
        'decodeMs=${decodeMs.toStringAsFixed(2)}',
      );
    }

    final semanticText = sarashina2SemanticTokensToText(generated);
    log(
      '==> decode latencies (ms): '
      '${stepLatencies.map((value) => value.toStringAsFixed(2)).toList()}',
    );
    log('==> semantic tokens: $generated');
    log('==> semantic text: $semanticText');
    payload = {
      'ok': true,
      'modelDir': paths.modelDir,
      'provider': opts.provider,
      'deviceId': opts.deviceId,
      'numThreads': opts.numThreads,
      'text': opts.text,
      'promptText': opts.promptText,
      'promptSemanticTokens': opts.promptSemanticTokens,
      'prompt': prompt,
      'promptTokenIds': promptIds,
      'promptTokenCount': promptIds.length,
      'stepsRequested': opts.steps,
      'stoppedOnEos': stoppedOnEos,
      'semanticTokens': generated,
      'semanticTokenText': semanticText,
      'semanticTokenCount': generated.length,
      'prefillMs': prefillMs,
      'decodeStepMs': stepLatencies,
      'sampler': {
        'temperature': opts.temperature,
        'topP': opts.topP,
        'frequencyPenalty': opts.frequencyPenalty,
        'seed': opts.seed,
      },
      'components': [
        for (final status in bundle.statuses)
          if (status.file.name == 'llm_prefill' ||
              status.file.name == 'llm_decode' ||
              status.file.name == 'llm_decoder_head')
            {
              'name': status.file.name,
              'exists': status.exists,
              'loaded': status.loaded,
              'selectedProvider': status.selectedProvider,
              'loadElapsedMs': (status.loadElapsedMicroseconds ?? 0) / 1000.0,
              if (status.error != null) 'error': status.error,
            },
      ],
    };
    if (opts.json) {
      stdout.writeln(jsonEncode(payload));
    } else {
      stdout.writeln('==> SMOKE OK');
    }
  } catch (error, stack) {
    stderr.writeln('SMOKE FAILED: $error');
    stderr.writeln(stack);
    exitCode = 1;
    if (opts.json) {
      stdout.writeln(
        jsonEncode({
          'ok': false,
          'error': '$error',
          'stack': '$stack',
          'partial': ?payload,
        }),
      );
    }
  } finally {
    samplerState?.close();
    state?.close();
    tokenizer?.close();
    driver?.close();
    bundle.close();
  }
}

final class _Opts {
  const _Opts({
    required this.modelDir,
    required this.text,
    required this.promptText,
    required this.promptSemanticTokens,
    required this.steps,
    required this.provider,
    required this.deviceId,
    required this.numThreads,
    required this.temperature,
    required this.topP,
    required this.frequencyPenalty,
    required this.seed,
    required this.json,
  });

  final String modelDir;
  final String text;
  final String promptText;
  final List<int> promptSemanticTokens;
  final int steps;
  final String provider;
  final int deviceId;
  final int numThreads;
  final double temperature;
  final double topP;
  final double frequencyPenalty;
  final int seed;
  final bool json;
}

_Opts _parseArgs(List<String> args) {
  String? modelDir;
  var text = 'hello';
  var promptText = '';
  var promptSemanticTokens = const <int>[];
  var steps = 8;
  var provider = 'cuda';
  var deviceId = 0;
  var numThreads = 4;
  var temperature = 0.0;
  var topP = 0.95;
  var frequencyPenalty = 0.0;
  var seed = 0;
  var json = false;

  for (var i = 0; i < args.length; i += 1) {
    final arg = args[i];
    String next() {
      if (i + 1 >= args.length) {
        throw ArgumentError('Missing value for $arg');
      }
      i += 1;
      return args[i];
    }

    switch (arg) {
      case '--model-dir':
        modelDir = next();
      case '--text':
        text = next();
      case '--prompt-text':
        promptText = next();
      case '--prompt-semantic-tokens':
        promptSemanticTokens = _parseIntList(next());
      case '--steps':
        steps = int.parse(next());
      case '--provider':
        provider = next();
      case '--device-id':
        deviceId = int.parse(next());
      case '--num-threads':
        numThreads = int.parse(next());
      case '--temperature':
        temperature = double.parse(next());
      case '--top-p':
        topP = double.parse(next());
      case '--frequency-penalty':
        frequencyPenalty = double.parse(next());
      case '--seed':
        seed = int.parse(next());
      case '--json':
        json = true;
      case '-h':
      case '--help':
        stdout.writeln(
          'Usage: dart run bin/sarashina2_llm_smoke.dart '
          '--model-dir <sarashina2.2-tts> [--text TEXT] [--steps N] '
          '[--prompt-text TEXT --prompt-semantic-tokens IDS] '
          '[--provider cuda|cpu] [--temperature N] [--top-p N] '
          '[--frequency-penalty N] [--seed N] [--json]',
        );
        exit(0);
      default:
        throw ArgumentError('Unknown flag: $arg');
    }
  }

  if (modelDir == null || modelDir.isEmpty) {
    throw ArgumentError('--model-dir is required');
  }
  if (text.trim().isEmpty) {
    throw ArgumentError('--text must not be empty');
  }
  if (steps < 1) {
    throw ArgumentError('--steps must be positive');
  }
  if (!temperature.isFinite || temperature < 0) {
    throw ArgumentError('--temperature must be non-negative');
  }
  if (!topP.isFinite || topP <= 0) {
    throw ArgumentError('--top-p must be positive');
  }
  if (!frequencyPenalty.isFinite) {
    throw ArgumentError('--frequency-penalty must be finite');
  }
  final effectivePromptText = promptText.trim();
  if ((effectivePromptText.isEmpty) != (promptSemanticTokens.isEmpty)) {
    throw ArgumentError(
      '--prompt-text and --prompt-semantic-tokens must either both be empty '
      'or both be set',
    );
  }
  validateSarashina2SemanticTokens(promptSemanticTokens);
  return _Opts(
    modelDir: modelDir,
    text: text,
    promptText: effectivePromptText,
    promptSemanticTokens: promptSemanticTokens,
    steps: steps,
    provider: provider,
    deviceId: deviceId,
    numThreads: numThreads,
    temperature: temperature,
    topP: topP,
    frequencyPenalty: frequencyPenalty,
    seed: seed,
    json: json,
  );
}

List<int> _parseIntList(String value) {
  if (value.trim().isEmpty) {
    return const [];
  }
  return value
      .split(RegExp(r'[\s,]+'))
      .where((part) => part.isNotEmpty)
      .map(int.parse)
      .toList(growable: false);
}
