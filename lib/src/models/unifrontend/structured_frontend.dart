import 'dart:convert';
import 'dart:ffi' as ffi;
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_inference/runtime.dart';
import 'package:ffi/ffi.dart';

import '../../runtime/native_bindings.dart' as native;

part 'structured_input.dart';
part 'structured_ssml.dart';
part 'structured_targets.dart';

final class StructuredFrontendResult {
  StructuredFrontendResult({
    required this.input,
    required this.ir,
    required this.ssml,
    required this.ttsText,
    required this.elapsedMicroseconds,
    required this.provider,
  });

  final String input;
  final FrontendIr ir;
  final String ssml;
  final String ttsText;
  final int elapsedMicroseconds;
  final String provider;

  Map<String, Object?> toJson() => {
    'input': input,
    'ir': ir.toJson(),
    'ssml': ssml,
    'ttsText': ttsText,
    'elapsedMicroseconds': elapsedMicroseconds,
    'provider': provider,
  };
}

final class DartStructuredFrontendRuntime {
  DartStructuredFrontendRuntime._({
    required this.session,
    required this.config,
    required this.inputBuilder,
    required this.decoder,
    required this.selectedProvider,
  });

  final DartOnnxSession session;
  final StructuredFrontendConfig config;
  final StructuredInputBuilder inputBuilder;
  final StructuredDecoder decoder;
  final String selectedProvider;

  static Future<DartStructuredFrontendRuntime> load({
    required String modelPath,
    required String exportConfigPath,
    required String structuredConfigPath,
    required String tokenizerJsonPath,
    required String charVocabPath,
    required String labelSpacePath,
    String? englishTnLexiconPath,
    required String provider,
    required int deviceId,
    required bool requireProvider,
    required int numThreads,
    Map<String, Object?> backendOptions = const {},
  }) async {
    final config = await StructuredFrontendConfig.load(
      exportConfigPath: exportConfigPath,
      structuredConfigPath: structuredConfigPath,
    );
    final tokenizer = await MmBertBpeTokenizer.load(tokenizerJsonPath);
    final charVocab = await CharVocab.load(charVocabPath);
    final decoder = await StructuredDecoder.load(
      labelSpacePath: labelSpacePath,
      englishTnLexiconPath: englishTnLexiconPath,
      emphasisThreshold: config.emphasisThreshold,
    );
    final session = DartOnnxSession.load(
      DartOnnxConfig(
        modelPath: modelPath,
        id: 'structured_unifrontend_onnx_dart',
        family: 'structured_unifrontend',
        provider: provider,
        deviceId: deviceId,
        requireProvider: requireProvider,
        numThreads: numThreads,
        backendOptions: backendOptions,
      ),
    );
    return DartStructuredFrontendRuntime._(
      session: session,
      config: config,
      inputBuilder: StructuredInputBuilder(
        tokenizer: tokenizer,
        charVocab: charVocab,
        config: config,
        targetResolver: decoder.targetResolver,
      ),
      decoder: decoder,
      selectedProvider: session.selectedProvider,
    );
  }

  StructuredFrontendResult process(String text) {
    return processBatch([text]).first;
  }

  List<StructuredFrontendResult> processBatch(List<String> texts) {
    if (texts.isEmpty) {
      return const [];
    }
    final results = <StructuredFrontendResult>[];
    final sw = Stopwatch()..start();
    for (var start = 0; start < texts.length; start += config.batchSize) {
      final chunk = texts
          .sublist(start, math.min(start + config.batchSize, texts.length))
          .toList(growable: false);
      final encoded = inputBuilder.encodeBatch(chunk);
      try {
        final outputs = session.run(
          encoded.toModelInputs(
            batchSize: config.batchSize,
            tokenLength: config.tokenLength,
            charLength: config.charLength,
            homographTargets: config.homographTargets,
            polyphoneTargets: config.polyphoneTargets,
            numHomographClasses: config.numHomographClasses,
            numPolyphoneClasses: config.numPolyphoneClasses,
          ),
        );
        try {
          final provider = outputs.providerOr(selectedProvider);
          for (var row = 0; row < encoded.activeRows; row++) {
            final text = encoded.texts[row];
            final ir = decoder.decode(
              text: text,
              numChars: encoded.numChars[row],
              outputs: outputs.outputs,
              homographTargets: encoded.homographTargets[row],
              polyphoneTargets: encoded.polyphoneTargets[row],
              rowIndex: row,
            );
            final ssml = composeSsml(text, ir);
            results.add(
              StructuredFrontendResult(
                input: text,
                ir: ir,
                ssml: ssml,
                ttsText: stripSsmlForTts(ssml),
                elapsedMicroseconds: 0,
                provider: provider,
              ),
            );
          }
        } finally {
          outputs.close();
        }
      } finally {
        encoded.close();
      }
    }
    sw.stop();
    if (results.isEmpty) {
      return const [];
    }
    final elapsedPerItem = sw.elapsedMicroseconds ~/ results.length;
    return [
      for (final result in results)
        StructuredFrontendResult(
          input: result.input,
          ir: result.ir,
          ssml: result.ssml,
          ttsText: result.ttsText,
          elapsedMicroseconds: elapsedPerItem,
          provider: result.provider,
        ),
    ];
  }

  void close() {
    inputBuilder.close();
    decoder.close();
    session.close();
  }
}

final class FrontendIr {
  final List<String> emotionLabels = [];
  final List<SpanLabel> emphasisSpans = [];
  final List<PronunciationItem> homographItems = [];
  final List<PronunciationItem> polyphoneItems = [];
  final List<TnItem> tnEnItems = [];
  final List<TnItem> tnZhItems = [];

  Map<String, Object?> toJson() => {
    'emotionLabels': emotionLabels,
    'emphasisSpans': [for (final item in emphasisSpans) item.toJson()],
    'homographItems': [for (final item in homographItems) item.toJson()],
    'polyphoneItems': [for (final item in polyphoneItems) item.toJson()],
    'tnEnItems': [for (final item in tnEnItems) item.toJson()],
    'tnZhItems': [for (final item in tnZhItems) item.toJson()],
  };
}

final class SpanLabel {
  SpanLabel(this.start, this.end, this.label);
  final int start;
  final int end;
  final String label;

  Map<String, Object?> toJson() => {'start': start, 'end': end, 'label': label};
}

final class TnItem {
  TnItem({
    required this.start,
    required this.end,
    required this.surface,
    required this.tnType,
    required this.spoken,
  });
  final int start;
  final int end;
  final String surface;
  final String tnType;
  final String spoken;

  Map<String, Object?> toJson() => {
    'start': start,
    'end': end,
    'surface': surface,
    'tnType': tnType,
    'spoken': spoken,
  };
}

final class StructuredDecoder {
  StructuredDecoder({
    required this.emotionLabels,
    required this.tnEnTypes,
    required this.tnZhTypes,
    required this.targetResolver,
    required this.englishTnLexicon,
    required this.emphasisThreshold,
  });

  final List<String> emotionLabels;
  final List<String> tnEnTypes;
  final List<String> tnZhTypes;
  final PronunciationTargetResolver targetResolver;
  final Map<String, Map<String, String>> englishTnLexicon;
  final double emphasisThreshold;

  static Future<StructuredDecoder> load({
    required String labelSpacePath,
    String? englishTnLexiconPath,
    required double emphasisThreshold,
  }) async {
    final payload =
        jsonDecode(await File(labelSpacePath).readAsString()) as Map;
    final englishTnLexicon = <String, Map<String, String>>{};
    if (englishTnLexiconPath != null &&
        await File(englishTnLexiconPath).exists()) {
      final lexiconRaw =
          jsonDecode(await File(englishTnLexiconPath).readAsString()) as Map;
      for (final entry in lexiconRaw.entries) {
        final bySurface = <String, String>{};
        final value = entry.value;
        if (value is Map) {
          for (final item in value.entries) {
            bySurface[item.key.toString()] = item.value.toString();
          }
        }
        englishTnLexicon[entry.key.toString()] = bySurface;
      }
    }
    final homographPronunciations = List<String>.from(
      payload['homograph_pronunciations'] as List? ?? const [],
    );
    final polyphonePronunciations = List<String>.from(
      payload['polyphone_pronunciations'] as List? ?? const [],
    );
    return StructuredDecoder(
      emotionLabels: List<String>.from(
        payload['emotion_labels'] as List? ?? const [],
      ),
      tnEnTypes: List<String>.from(payload['tn_en_types'] as List? ?? const []),
      tnZhTypes: List<String>.from(payload['tn_zh_types'] as List? ?? const []),
      targetResolver: PronunciationTargetResolver.fromLabelSpace(
        homographPronunciations: homographPronunciations,
        polyphonePronunciations: polyphonePronunciations,
        homographSurfaceCandidates: payload['homograph_surface_candidates'],
        polyphoneSurfaceCandidates: payload['polyphone_surface_candidates'],
      ),
      englishTnLexicon: englishTnLexicon,
      emphasisThreshold: emphasisThreshold,
    );
  }

  void close() {
    targetResolver.close();
  }

  FrontendIr decode({
    required String text,
    required int numChars,
    required Map<String, Object?> outputs,
    required List<PronunciationItem> homographTargets,
    required List<PronunciationItem> polyphoneTargets,
    int rowIndex = 0,
  }) {
    final ir = FrontendIr();
    final emotion = _floatTensor(outputs['emotion_logits']);
    if (emotion != null &&
        emotion.data.isNotEmpty &&
        emotionLabels.isNotEmpty) {
      var best = 0;
      final emotionCount = emotion.shape.isNotEmpty
          ? emotion.shape.last
          : emotion.data.length;
      final emotionOffset = emotion.shape.length >= 2
          ? rowIndex * emotionCount
          : 0;
      final limit = math.min(emotionLabels.length, emotionCount);
      if (limit > 0) {
        final ids = _activeIdsFromTensor(
          emotion,
          offset: emotionOffset,
          count: limit,
          threshold: 0.5,
        );
        for (final id in ids.active) {
          ir.emotionLabels.add(emotionLabels[id]);
        }
        best = ids.best;
        if (ir.emotionLabels.isEmpty) {
          ir.emotionLabels.add(emotionLabels[best]);
        }
      }
    }
    final emph = _floatTensor(outputs['emphasis_char_logits']);
    if (emph != null) {
      if (emph.shape.length >= 3) {
        final labelCount = emph.shape.last;
        final charLength = emph.shape[emph.shape.length - 2];
        final rowOffset = rowIndex * charLength * labelCount;
        ir.emphasisSpans.addAll(
          _decodeBioesFromTensor(
            tensor: emph,
            base: rowOffset,
            itemCount: numChars,
            stride: labelCount,
            classCount: labelCount,
            label: 'EMPHASIS',
          ),
        );
      } else {
        ir.emphasisSpans.addAll(
          _decodeBinarySpans(
            emph,
            numChars,
            'EMPHASIS',
            emphasisThreshold,
            offset: emph.shape.length >= 2 ? rowIndex * emph.shape.last : 0,
          ),
        );
      }
      final normalizedEmphasis = _normalizeEmphasisSpans(
        text,
        ir.emphasisSpans,
      );
      ir.emphasisSpans
        ..clear()
        ..addAll(normalizedEmphasis);
    }
    ir.homographItems.addAll(
      _decodePronunciationItems(
        targets: homographTargets,
        tensor: _floatTensor(outputs['homograph_pron_logits_multi']),
        labels: targetResolver.homographPronunciations,
        rowIndex: rowIndex,
      ),
    );
    ir.polyphoneItems.addAll(
      _decodePronunciationItems(
        targets: polyphoneTargets,
        tensor: _floatTensor(outputs['polyphone_pron_logits_multi']),
        labels: targetResolver.polyphonePronunciations,
        rowIndex: rowIndex,
      ),
    );
    final enSpan = _floatTensor(outputs['tn_en_char_span_logits']);
    final enType = _floatTensor(outputs['tn_en_char_type_logits']);
    if (enSpan != null && enType != null) {
      ir.tnEnItems.addAll(
        _decodeTnItems(
          text: text,
          spanTensor: enSpan,
          typeTensor: enType,
          numChars: numChars,
          rowIndex: rowIndex,
          typeLabels: tnEnTypes,
          englishTnLexicon: englishTnLexicon,
          chinese: false,
        ),
      );
    }
    final zhSpan = _floatTensor(outputs['tn_zh_char_span_logits']);
    final zhType = _floatTensor(outputs['tn_zh_char_type_logits']);
    if (zhSpan != null && zhType != null) {
      ir.tnZhItems.addAll(
        _decodeTnItems(
          text: text,
          spanTensor: zhSpan,
          typeTensor: zhType,
          numChars: numChars,
          rowIndex: rowIndex,
          typeLabels: tnZhTypes,
          englishTnLexicon: const {},
          chinese: true,
        ),
      );
    }
    return ir;
  }
}

final class _FloatTensor {
  _FloatTensor(this.data, this.shape, this.nativeData);
  final Float32List data;
  final List<int> shape;
  final ffi.Pointer<ffi.Float> nativeData;
}

_FloatTensor? _floatTensor(Object? value) {
  if (value is! RuntimeTensor || value.dtype != RuntimeTensorDataType.float32) {
    return null;
  }
  return _FloatTensor(
    Float32List.view(
      value.bytes.buffer,
      value.bytes.offsetInBytes,
      value.bytes.lengthInBytes ~/ 4,
    ),
    value.shape,
    value.nativeData?.cast<ffi.Float>() ?? ffi.nullptr,
  );
}

List<SpanLabel> _decodeBinarySpans(
  _FloatTensor tensor,
  int numChars,
  String label,
  double threshold, {
  int offset = 0,
}) {
  final spans = <SpanLabel>[];
  final logits = tensor.data;
  final limit = math.min(numChars, math.max(0, logits.length - offset));
  final pointer = tensor.nativeData;
  if (limit <= 0) {
    return const [];
  }
  _requireNative(pointer);
  final starts = calloc<ffi.Int32>(limit);
  final ends = calloc<ffi.Int32>(limit);
  final count = calloc<ffi.IntPtr>();
  final error = calloc<ffi.Pointer<ffi.Char>>();
  try {
    final status = native.decSpans(
      pointer,
      tensor.data.length,
      offset,
      limit,
      numChars,
      threshold,
      starts,
      ends,
      count,
      error,
    );
    if (status != 0) {
      throw StateError(_takeDecodeError(error));
    }
    for (var i = 0; i < count.value; i++) {
      spans.add(SpanLabel(starts[i], ends[i], label));
    }
    return spans;
  } finally {
    calloc
      ..free(starts)
      ..free(ends)
      ..free(count)
      ..free(error);
  }
}

List<int> _argmaxIds({
  required _FloatTensor tensor,
  required int base,
  required int itemCount,
  required int stride,
  required int classCount,
}) {
  if (itemCount <= 0 || classCount <= 0) {
    return const [];
  }
  final pointer = tensor.nativeData;
  _requireNative(pointer);
  final out = calloc<ffi.Int32>(itemCount);
  final error = calloc<ffi.Pointer<ffi.Char>>();
  try {
    final status = native.decArgmax(
      pointer,
      tensor.data.length,
      base,
      itemCount,
      stride,
      classCount,
      out,
      error,
    );
    if (status != 0) {
      throw StateError(_takeDecodeError(error));
    }
    return out.asTypedList(itemCount).toList(growable: false);
  } finally {
    calloc
      ..free(out)
      ..free(error);
  }
}

final class _ActiveIds {
  _ActiveIds({required this.active, required this.best});

  final List<int> active;
  final int best;
}

_ActiveIds _activeIdsFromTensor(
  _FloatTensor tensor, {
  required int offset,
  required int count,
  required double threshold,
}) {
  if (count <= 0) {
    return _ActiveIds(active: const [], best: 0);
  }
  final pointer = tensor.nativeData;
  _requireNative(pointer);
  final out = calloc<ffi.Int32>(count);
  final activeCount = calloc<ffi.IntPtr>();
  final best = calloc<ffi.Int32>();
  final error = calloc<ffi.Pointer<ffi.Char>>();
  try {
    final status = native.decActive(
      pointer,
      tensor.data.length,
      offset,
      count,
      threshold,
      out,
      activeCount,
      best,
      error,
    );
    if (status != 0) {
      throw StateError(_takeDecodeError(error));
    }
    return _ActiveIds(
      active: out.asTypedList(activeCount.value).toList(growable: false),
      best: best.value,
    );
  } finally {
    calloc
      ..free(out)
      ..free(activeCount)
      ..free(best)
      ..free(error);
  }
}

void _requireNative(ffi.Pointer<ffi.Float> pointer) {
  if (pointer == ffi.nullptr) {
    throw StateError(
      'Structured decoder requires native-backed float32 tensors.',
    );
  }
}

String _takeDecodeError(ffi.Pointer<ffi.Pointer<ffi.Char>> error) {
  final value = error.value;
  if (value == ffi.nullptr) return 'Native decoder call failed.';
  try {
    return value.cast<Utf8>().toDartString();
  } finally {
    native.freeStr(value);
    error.value = ffi.nullptr;
  }
}

List<TnItem> _decodeTnItems({
  required String text,
  required _FloatTensor spanTensor,
  required _FloatTensor typeTensor,
  required int numChars,
  required int rowIndex,
  required List<String> typeLabels,
  required Map<String, Map<String, String>> englishTnLexicon,
  required bool chinese,
}) {
  if (spanTensor.shape.length < 3) {
    return const [];
  }
  final labelCount = spanTensor.shape.last;
  final spanCharLength = spanTensor.shape[spanTensor.shape.length - 2];
  final spanRowOffset = rowIndex * spanCharLength * labelCount;
  final spans = _decodeBioesFromTensor(
    tensor: spanTensor,
    base: spanRowOffset,
    itemCount: numChars,
    stride: labelCount,
    classCount: labelCount,
    label: 'TN',
  );
  final items = <TnItem>[];
  final typeCount = typeTensor.shape.last;
  final typeCharLength = typeTensor.shape[typeTensor.shape.length - 2];
  final typeRowOffset = rowIndex * typeCharLength * typeCount;
  final spanTypeIds = _spanTypeIdsFromTensor(
    tensor: typeTensor,
    base: typeRowOffset,
    itemCount: numChars,
    stride: typeCount,
    classCount: typeCount,
    spans: spans,
  );
  for (var spanIndex = 0; spanIndex < spans.length; spanIndex++) {
    final span = spans[spanIndex];
    final typeId = spanTypeIds[spanIndex];
    final surface = text.substring(span.start, span.end);
    final type = typeId < typeLabels.length ? typeLabels[typeId] : 'UNKNOWN';
    final spoken = chinese
        ? verbalizeChinese(surface)
        : _verbalizeEnglishWithLexicon(surface, type, englishTnLexicon);
    items.add(
      TnItem(
        start: span.start,
        end: span.end,
        surface: surface,
        tnType: type,
        spoken: spoken,
      ),
    );
  }
  return items;
}

List<int> _spanTypeIdsFromTensor({
  required _FloatTensor tensor,
  required int base,
  required int itemCount,
  required int stride,
  required int classCount,
  required List<SpanLabel> spans,
}) {
  if (spans.isEmpty) {
    return const [];
  }
  if (classCount <= 0) {
    return List<int>.filled(spans.length, 0, growable: false);
  }
  final pointer = tensor.nativeData;
  _requireNative(pointer);
  final starts = calloc<ffi.Int32>(spans.length);
  final ends = calloc<ffi.Int32>(spans.length);
  final counts = calloc<ffi.Int32>(classCount);
  final out = calloc<ffi.Int32>(spans.length);
  final error = calloc<ffi.Pointer<ffi.Char>>();
  try {
    for (var i = 0; i < spans.length; i++) {
      starts[i] = spans[i].start;
      ends[i] = spans[i].end;
    }
    final status = native.decSpanTypes(
      pointer,
      tensor.data.length,
      base,
      itemCount,
      stride,
      classCount,
      starts,
      ends,
      spans.length,
      counts,
      out,
      error,
    );
    if (status != 0) {
      throw StateError(_takeDecodeError(error));
    }
    return out.asTypedList(spans.length).toList(growable: false);
  } finally {
    calloc
      ..free(starts)
      ..free(ends)
      ..free(counts)
      ..free(out)
      ..free(error);
  }
}

List<SpanLabel> _decodeBioesFromTensor({
  required _FloatTensor tensor,
  required int base,
  required int itemCount,
  required int stride,
  required int classCount,
  required String label,
}) {
  if (itemCount <= 0 || classCount <= 0) {
    return const [];
  }
  final pointer = tensor.nativeData;
  _requireNative(pointer);
  final starts = calloc<ffi.Int32>(itemCount);
  final ends = calloc<ffi.Int32>(itemCount);
  final count = calloc<ffi.IntPtr>();
  final error = calloc<ffi.Pointer<ffi.Char>>();
  try {
    final status = native.decBioes(
      pointer,
      tensor.data.length,
      base,
      itemCount,
      stride,
      classCount,
      starts,
      ends,
      count,
      error,
    );
    if (status != 0) {
      throw StateError(_takeDecodeError(error));
    }
    return [
      for (var i = 0; i < count.value; i++)
        SpanLabel(starts[i], ends[i], label),
    ];
  } finally {
    calloc
      ..free(starts)
      ..free(ends)
      ..free(count)
      ..free(error);
  }
}
