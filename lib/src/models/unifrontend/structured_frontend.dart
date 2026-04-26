import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_inference/runtime.dart';

part 'structured_input.dart';
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
      for (var i = 0; i < limit; i++) {
        final value = emotion.data[emotionOffset + i];
        final prob = 1.0 / (1.0 + math.exp(-value));
        if (prob >= 0.5) {
          ir.emotionLabels.add(emotionLabels[i]);
        }
        if (value > emotion.data[emotionOffset + best]) {
          best = i;
        }
      }
      if (ir.emotionLabels.isEmpty) {
        ir.emotionLabels.add(emotionLabels[best]);
      }
    }
    final emph = _floatTensor(outputs['emphasis_char_logits']);
    if (emph != null) {
      if (emph.shape.length >= 3) {
        final labelCount = emph.shape.last;
        final charLength = emph.shape[emph.shape.length - 2];
        final rowOffset = rowIndex * charLength * labelCount;
        final ids = <int>[];
        for (var c = 0; c < numChars; c++) {
          var best = 0;
          var bestVal = -double.infinity;
          for (var k = 0; k < labelCount; k++) {
            final v = emph.data[rowOffset + c * labelCount + k];
            if (v > bestVal) {
              bestVal = v;
              best = k;
            }
          }
          ids.add(best);
        }
        ir.emphasisSpans.addAll(_decodeBioes(ids, numChars, 'EMPHASIS'));
      } else {
        ir.emphasisSpans.addAll(
          _decodeBinarySpans(
            emph.data,
            numChars,
            'EMPHASIS',
            emphasisThreshold,
            offset: emph.shape.length >= 2 ? rowIndex * emph.shape.last : 0,
          ),
        );
      }
      ir.emphasisSpans
        ..clear()
        ..addAll(_normalizeEmphasisSpans(text, ir.emphasisSpans));
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
  _FloatTensor(this.data, this.shape);
  final Float32List data;
  final List<int> shape;
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
  );
}

List<SpanLabel> _decodeBinarySpans(
  Float32List logits,
  int numChars,
  String label,
  double threshold, {
  int offset = 0,
}) {
  final spans = <SpanLabel>[];
  int? start;
  final limit = math.min(numChars, math.max(0, logits.length - offset));
  for (var i = 0; i < limit; i++) {
    final prob = 1.0 / (1.0 + math.exp(-logits[offset + i]));
    final active = prob >= threshold;
    if (active && start == null) {
      start = i;
    } else if (!active && start != null) {
      spans.add(SpanLabel(start, i, label));
      start = null;
    }
  }
  if (start != null) {
    spans.add(SpanLabel(start, numChars, label));
  }
  return spans;
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
  final spanIds = <int>[];
  for (var c = 0; c < numChars; c++) {
    var best = 0;
    var bestVal = -double.infinity;
    for (var k = 0; k < labelCount; k++) {
      final v = spanTensor.data[spanRowOffset + c * labelCount + k];
      if (v > bestVal) {
        bestVal = v;
        best = k;
      }
    }
    spanIds.add(best);
  }
  final spans = _decodeBioes(spanIds, numChars, 'TN');
  final items = <TnItem>[];
  final typeCount = typeTensor.shape.last;
  final typeCharLength = typeTensor.shape[typeTensor.shape.length - 2];
  final typeRowOffset = rowIndex * typeCharLength * typeCount;
  for (final span in spans) {
    final counts = List<int>.filled(math.max(typeCount, 1), 0);
    for (var c = span.start; c < span.end; c++) {
      var best = 0;
      var bestVal = -double.infinity;
      for (var k = 0; k < typeCount; k++) {
        final v = typeTensor.data[typeRowOffset + c * typeCount + k];
        if (v > bestVal) {
          bestVal = v;
          best = k;
        }
      }
      counts[best] += 1;
    }
    var typeId = 0;
    for (var i = 1; i < counts.length; i++) {
      if (counts[i] > counts[typeId]) {
        typeId = i;
      }
    }
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

List<SpanLabel> _decodeBioes(List<int> ids, int numChars, String label) {
  final out = <SpanLabel>[];
  int? start;
  for (var i = 0; i < math.min(numChars, ids.length); i++) {
    final id = ids[i];
    if (id == 4) {
      out.add(SpanLabel(i, i + 1, label));
      start = null;
    } else if (id == 1) {
      start = i;
    } else if (id == 3 && start != null) {
      out.add(SpanLabel(start, i + 1, label));
      start = null;
    } else if (id == 0) {
      start = null;
    }
  }
  return out;
}

const _emphasisTrimCharacters = ' \t\r\n"\\\'. ,;:!?()[]{}';

List<SpanLabel> _normalizeEmphasisSpans(String text, List<SpanLabel> spans) {
  final trimmed = <SpanLabel>[];
  for (final span in spans) {
    var start = span.start;
    var end = span.end;
    while (start < end && _emphasisTrimCharacters.contains(text[start])) {
      start += 1;
    }
    while (end > start && _emphasisTrimCharacters.contains(text[end - 1])) {
      end -= 1;
    }
    if (end > start) {
      trimmed.add(SpanLabel(start, end, span.label));
    }
  }
  if (trimmed.isEmpty) {
    return const [];
  }
  final merged = <SpanLabel>[trimmed.first];
  for (final span in trimmed.skip(1)) {
    final previous = merged.last;
    final gap = text.substring(previous.end, span.start);
    if (gap.isNotEmpty && gap.trim().isEmpty) {
      merged[merged.length - 1] = SpanLabel(
        previous.start,
        span.end,
        previous.label,
      );
    } else {
      merged.add(span);
    }
  }
  return merged;
}

String composeSsml(String text, FrontendIr ir) {
  final tags = <_Tag>[];
  for (final span in ir.emphasisSpans) {
    tags.add(_Tag(span.start, span.end, 'emphasis', const {}));
  }
  for (final item in [...ir.homographItems, ...ir.polyphoneItems]) {
    if (item.pronunciation.isNotEmpty) {
      tags.add(
        _Tag(item.start, item.end, 'phoneme', {'ph': item.pronunciation}),
      );
    }
  }
  for (final item in _selectTnItems(text, ir)) {
    if (item.spoken.isNotEmpty && item.spoken != item.surface) {
      tags.add(_Tag(item.start, item.end, 'sub', {'alias': item.spoken}));
    }
  }
  tags.sort(
    (a, b) => a.start == b.start
        ? a.end.compareTo(b.end)
        : a.start.compareTo(b.start),
  );
  final inner = _applyTags(text, tags);
  if (ir.emotionLabels.isNotEmpty) {
    return '<speak><emotion type="${_xml(ir.emotionLabels.first)}">$inner</emotion></speak>';
  }
  return '<speak>$inner</speak>';
}

List<TnItem> _selectTnItems(String text, FrontendIr ir) {
  if (ir.tnEnItems.isEmpty) {
    return ir.tnZhItems;
  }
  if (ir.tnZhItems.isEmpty) {
    return ir.tnEnItems;
  }
  final primary = _preferChineseTn(text) ? ir.tnZhItems : ir.tnEnItems;
  final secondary = identical(primary, ir.tnZhItems)
      ? ir.tnEnItems
      : ir.tnZhItems;
  final selected = <TnItem>[];
  void addNonOverlapping(Iterable<TnItem> items) {
    for (final item in items) {
      final overlaps = selected.any(
        (taken) => !(item.end <= taken.start || item.start >= taken.end),
      );
      if (!overlaps) {
        selected.add(item);
      }
    }
  }

  addNonOverlapping(primary);
  addNonOverlapping(secondary);
  selected.sort((a, b) {
    final startCompare = a.start.compareTo(b.start);
    if (startCompare != 0) {
      return startCompare;
    }
    return a.end.compareTo(b.end);
  });
  return selected;
}

bool _preferChineseTn(String text) {
  var cjk = 0;
  var latin = 0;
  for (final rune in text.runes) {
    if (rune >= 0x4e00 && rune <= 0x9fff) {
      cjk += 1;
    } else if ((rune >= 0x41 && rune <= 0x5a) ||
        (rune >= 0x61 && rune <= 0x7a)) {
      latin += 1;
    }
  }
  return cjk > 0 && cjk >= latin;
}

final class _Tag {
  _Tag(this.start, this.end, this.name, this.attrs);
  final int start;
  final int end;
  final String name;
  final Map<String, String> attrs;
}

String _applyTags(String text, List<_Tag> tags) {
  final out = StringBuffer();
  var cursor = 0;
  for (final tag in tags) {
    if (tag.start < cursor || tag.end <= tag.start || tag.end > text.length) {
      continue;
    }
    out.write(_xml(text.substring(cursor, tag.start)));
    final attrs = tag.attrs.entries
        .map((e) => ' ${e.key}="${_xml(e.value)}"')
        .join();
    out.write('<${tag.name}$attrs>');
    out.write(_xml(text.substring(tag.start, tag.end)));
    out.write('</${tag.name}>');
    cursor = tag.end;
  }
  out.write(_xml(text.substring(cursor)));
  return out.toString();
}

String _xml(String value) => value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');

String stripSsmlForTts(String ssml) {
  return ssml
      .replaceAllMapped(
        RegExp(
          r'<sub\b[^>]*alias="([^"]*)"[^>]*>.*?</sub>',
          caseSensitive: false,
        ),
        (m) => m.group(1) ?? '',
      )
      .replaceAll(RegExp(r'<[^>]+>'), '');
}

String _verbalizeEnglishWithLexicon(
  String surface,
  String type,
  Map<String, Map<String, String>> lexicon,
) {
  final typed = lexicon[type];
  final trimmed = surface.trim();
  return typed?[surface] ??
      typed?[trimmed] ??
      lexicon['UNKNOWN']?[surface] ??
      lexicon['UNKNOWN']?[trimmed] ??
      verbalizeEnglish(surface);
}

String verbalizeEnglish(String surface) {
  final s = surface.trim();
  final money = RegExp(r'^\$([0-9]+)(?:\.([0-9]{1,2}))?$').firstMatch(s);
  if (money != null) {
    final dollars = _englishInt(int.tryParse(money.group(1)!) ?? 0);
    final centsRaw = money.group(2);
    if (centsRaw == null) {
      return '$dollars dollars';
    }
    final cents = _englishInt(int.tryParse(centsRaw.padRight(2, '0')) ?? 0);
    return '$dollars dollars and $cents cents';
  }
  final n = int.tryParse(s.replaceAll(',', ''));
  if (n != null) {
    return _englishInt(n);
  }
  return s;
}

String verbalizeChinese(String surface) {
  const digits = {
    '0': '零',
    '1': '一',
    '2': '二',
    '3': '三',
    '4': '四',
    '5': '五',
    '6': '六',
    '7': '七',
    '8': '八',
    '9': '九',
  };
  return surface.split('').map((ch) => digits[ch] ?? ch).join();
}

String _englishInt(int n) {
  const small = [
    'zero',
    'one',
    'two',
    'three',
    'four',
    'five',
    'six',
    'seven',
    'eight',
    'nine',
    'ten',
    'eleven',
    'twelve',
    'thirteen',
    'fourteen',
    'fifteen',
    'sixteen',
    'seventeen',
    'eighteen',
    'nineteen',
  ];
  const tens = [
    '',
    '',
    'twenty',
    'thirty',
    'forty',
    'fifty',
    'sixty',
    'seventy',
    'eighty',
    'ninety',
  ];
  if (n < 0) {
    return 'minus ${_englishInt(-n)}';
  }
  if (n < 20) {
    return small[n];
  }
  if (n < 100) {
    final r = n % 10;
    return r == 0 ? tens[n ~/ 10] : '${tens[n ~/ 10]} ${small[r]}';
  }
  if (n < 1000) {
    final r = n % 100;
    return r == 0
        ? '${small[n ~/ 100]} hundred'
        : '${small[n ~/ 100]} hundred ${_englishInt(r)}';
  }
  if (n < 1000000) {
    final r = n % 1000;
    return r == 0
        ? '${_englishInt(n ~/ 1000)} thousand'
        : '${_englishInt(n ~/ 1000)} thousand ${_englishInt(r)}';
  }
  return n.toString();
}
