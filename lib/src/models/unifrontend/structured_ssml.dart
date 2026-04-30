part of 'structured_frontend.dart';

const _ssmlEmphasis = 1;
const _ssmlPhoneme = 2;
const _ssmlSub = 3;

List<SpanLabel> _normalizeEmphasisSpans(String text, List<SpanLabel> spans) {
  if (spans.isEmpty) {
    return const [];
  }
  final out = <SpanLabel>[];
  for (final span in spans) {
    var start = span.start.clamp(0, text.length).toInt();
    var end = span.end.clamp(start, text.length).toInt();
    while (start < end && text[start].trim().isEmpty) {
      start += 1;
    }
    while (end > start && text[end - 1].trim().isEmpty) {
      end -= 1;
    }
    if (start < end) {
      out.add(SpanLabel(start, end, span.label));
    }
  }
  return out;
}

String composeSsml(String text, FrontendIr ir) {
  final tags = <_SsmlTag>[];
  for (final span in ir.emphasisSpans) {
    tags.add(_SsmlTag(span.start, span.end, _ssmlEmphasis));
  }
  for (final item in [...ir.homographItems, ...ir.polyphoneItems]) {
    if (item.pronunciation.isNotEmpty) {
      tags.add(
        _SsmlTag(item.start, item.end, _ssmlPhoneme, item.pronunciation),
      );
    }
  }
  for (final item in _selectTnItems(text, ir)) {
    if (item.spoken.isNotEmpty && item.spoken != item.surface) {
      tags.add(_SsmlTag(item.start, item.end, _ssmlSub, item.spoken));
    }
  }
  return _composeSsml(
    text,
    tags,
    ir.emotionLabels.isEmpty ? null : ir.emotionLabels.first,
  );
}

List<TnItem> _selectTnItems(String text, FrontendIr ir) {
  final en = ir.tnEnItems;
  final zh = ir.tnZhItems;
  if (en.isEmpty) {
    return zh;
  }
  if (zh.isEmpty) {
    return en;
  }
  final preferZh = looksChinese(text);
  final primary = preferZh ? zh : en;
  final secondary = preferZh ? en : zh;
  final out = <TnItem>[];
  for (final item in primary) {
    out.add(item);
  }
  for (final item in secondary) {
    if (!out.any((selected) => _overlaps(selected, item))) {
      out.add(item);
    }
  }
  out.sort((a, b) => a.start.compareTo(b.start));
  return out;
}

bool _overlaps(TnItem left, TnItem right) =>
    left.start < right.end && right.start < left.end;

bool looksChinese(String text) {
  return text.runes.any(_isChineseRune);
}

final class _SsmlTag {
  _SsmlTag(this.start, this.end, this.kind, [this.value]);
  final int start;
  final int end;
  final int kind;
  final String? value;
}

String _composeSsml(String text, List<_SsmlTag> tags, String? emotion) {
  final ordered = tags.toList()
    ..sort((a, b) {
      final byStart = a.start.compareTo(b.start);
      if (byStart != 0) return byStart;
      return b.end.compareTo(a.end);
    });
  final out = StringBuffer('<speak>');
  var cursor = 0;
  for (final tag in ordered) {
    final start = tag.start.clamp(cursor, text.length).toInt();
    final end = tag.end.clamp(start, text.length).toInt();
    out.write(_xmlEscape(text.substring(cursor, start)));
    final surface = _xmlEscape(text.substring(start, end));
    final value = _xmlEscape(tag.value ?? '');
    switch (tag.kind) {
      case _ssmlEmphasis:
        out.write('<emphasis>$surface</emphasis>');
      case _ssmlPhoneme:
        out.write('<phoneme ph="$value">$surface</phoneme>');
      case _ssmlSub:
        out.write('<sub alias="$value">$surface</sub>');
      default:
        out.write(surface);
    }
    cursor = end;
  }
  out.write(_xmlEscape(text.substring(cursor)));
  out.write('</speak>');
  return out.toString();
}

String stripSsmlForTts(String ssml) {
  final withAliases = ssml.replaceAllMapped(
    RegExp(
      r'<sub\b[^>]*\balias="([^"]*)"[^>]*>.*?</sub>',
      caseSensitive: false,
      dotAll: true,
    ),
    (match) => _xmlUnescape(match.group(1)!),
  );
  return _collapseText(
    _xmlUnescape(withAliases.replaceAll(RegExp(r'<[^>]+>', dotAll: true), '')),
  );
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
  final trimmed = surface.trim();
  final money = RegExp(r'^\$(\d+)$').firstMatch(trimmed);
  if (money != null) {
    return '${_englishNumber(int.parse(money.group(1)!))} dollars';
  }
  final number = int.tryParse(trimmed);
  if (number != null) {
    return _englishNumber(number);
  }
  return trimmed;
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
  return surface.runes
      .map(
        (rune) =>
            digits[String.fromCharCode(rune)] ?? String.fromCharCode(rune),
      )
      .join();
}

String _englishNumber(int value) {
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
  if (value < 20) return small[value];
  if (value < 100) {
    final ten = value ~/ 10;
    final rest = value % 10;
    return rest == 0 ? tens[ten] : '${tens[ten]} ${small[rest]}';
  }
  if (value < 1000) {
    final hundred = value ~/ 100;
    final rest = value % 100;
    return rest == 0
        ? '${small[hundred]} hundred'
        : '${small[hundred]} hundred ${_englishNumber(rest)}';
  }
  return value.toString();
}

bool _isChineseRune(int rune) =>
    (rune >= 0x3400 && rune <= 0x4dbf) ||
    (rune >= 0x4e00 && rune <= 0x9fff) ||
    (rune >= 0xf900 && rune <= 0xfaff);

String _xmlEscape(String value) => value
    .replaceAll('&', '&amp;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&apos;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');

String _xmlUnescape(String value) => value
    .replaceAll('&quot;', '"')
    .replaceAll('&apos;', "'")
    .replaceAll('&lt;', '<')
    .replaceAll('&gt;', '>')
    .replaceAll('&amp;', '&');

String _collapseText(String value) =>
    value.replaceAll(RegExp(r'\s+'), ' ').trim();
