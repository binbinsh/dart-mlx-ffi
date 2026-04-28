part of 'structured_frontend.dart';

const _ssmlEmphasis = 1;
const _ssmlPhoneme = 2;
const _ssmlSub = 3;
const _tnSourceEn = 1;
const _tnSourceZh = 2;

List<SpanLabel> _normalizeEmphasisSpans(String text, List<SpanLabel> spans) {
  if (spans.isEmpty) {
    return const [];
  }
  final input = text.toNativeUtf8(allocator: calloc).cast<ffi.Char>();
  final starts = calloc<ffi.Int32>(spans.length);
  final ends = calloc<ffi.Int32>(spans.length);
  final outStarts = calloc<ffi.Int32>(spans.length);
  final outEnds = calloc<ffi.Int32>(spans.length);
  final outCount = calloc<ffi.IntPtr>();
  final error = calloc<ffi.Pointer<ffi.Char>>();
  try {
    for (var i = 0; i < spans.length; i += 1) {
      starts[i] = spans[i].start;
      ends[i] = spans[i].end;
    }
    final status = native.textNormSpans(
      input,
      starts,
      ends,
      spans.length,
      outStarts,
      outEnds,
      outCount,
      error,
    );
    if (status != 0) {
      throw StateError(_takeTextError(error));
    }
    return [
      for (var i = 0; i < outCount.value; i += 1)
        SpanLabel(outStarts[i], outEnds[i], spans.first.label),
    ];
  } finally {
    calloc
      ..free(input)
      ..free(starts)
      ..free(ends)
      ..free(outStarts)
      ..free(outEnds)
      ..free(outCount)
      ..free(error);
  }
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
  return _composeSsmlNative(
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
  final input = text.toNativeUtf8(allocator: calloc).cast<ffi.Char>();
  final enRanges = _TnRangeArray(en);
  final zhRanges = _TnRangeArray(zh);
  final maxCount = en.length + zh.length;
  final sources = calloc<ffi.Int32>(maxCount);
  final indices = calloc<ffi.Int32>(maxCount);
  final count = calloc<ffi.IntPtr>();
  final error = calloc<ffi.Pointer<ffi.Char>>();
  try {
    final status = native.textSelectTn(
      input,
      enRanges.starts,
      enRanges.ends,
      enRanges.length,
      zhRanges.starts,
      zhRanges.ends,
      zhRanges.length,
      sources,
      indices,
      count,
      error,
    );
    if (status != 0) {
      throw StateError(_takeTextError(error));
    }
    return [
      for (var i = 0; i < count.value; i += 1)
        _tnItemForSource(sources[i], indices[i], en, zh),
    ];
  } finally {
    enRanges.close();
    zhRanges.close();
    calloc
      ..free(input)
      ..free(sources)
      ..free(indices)
      ..free(count)
      ..free(error);
  }
}

TnItem _tnItemForSource(
  int source,
  int index,
  List<TnItem> en,
  List<TnItem> zh,
) {
  if (source == _tnSourceEn) {
    return en[index];
  }
  if (source == _tnSourceZh) {
    return zh[index];
  }
  throw StateError('Native TN selector returned unknown source $source.');
}

bool looksChinese(String text) {
  final input = text.toNativeUtf8(allocator: calloc).cast<ffi.Char>();
  try {
    return native.textHasZh(input) != 0;
  } finally {
    calloc.free(input);
  }
}

final class _TnRangeArray {
  _TnRangeArray(List<TnItem> items)
    : length = items.length,
      starts = items.isEmpty ? ffi.nullptr : calloc<ffi.Int32>(items.length),
      ends = items.isEmpty ? ffi.nullptr : calloc<ffi.Int32>(items.length) {
    for (var i = 0; i < items.length; i += 1) {
      starts[i] = items[i].start;
      ends[i] = items[i].end;
    }
  }

  final int length;
  final ffi.Pointer<ffi.Int32> starts;
  final ffi.Pointer<ffi.Int32> ends;

  void close() {
    if (starts != ffi.nullptr) {
      calloc.free(starts);
    }
    if (ends != ffi.nullptr) {
      calloc.free(ends);
    }
  }
}

final class _SsmlTag {
  _SsmlTag(this.start, this.end, this.kind, [this.value]);
  final int start;
  final int end;
  final int kind;
  final String? value;
}

String _composeSsmlNative(String text, List<_SsmlTag> tags, String? emotion) {
  final input = text.toNativeUtf8(allocator: calloc).cast<ffi.Char>();
  final nativeTags = _TextTagArray(tags);
  final emotionPtr = emotion == null
      ? ffi.nullptr
      : emotion.toNativeUtf8(allocator: calloc).cast<ffi.Char>();
  final error = calloc<ffi.Pointer<ffi.Char>>();
  ffi.Pointer<ffi.Char> out = ffi.nullptr;
  try {
    out = native.textSsml(
      input,
      nativeTags.pointer,
      nativeTags.length,
      emotionPtr,
      error,
    );
    if (out == ffi.nullptr) {
      throw StateError(_takeTextError(error));
    }
    return out.cast<Utf8>().toDartString();
  } finally {
    if (out != ffi.nullptr) {
      native.freeStr(out);
    }
    nativeTags.close();
    calloc.free(input);
    if (emotionPtr != ffi.nullptr) {
      calloc.free(emotionPtr);
    }
    calloc.free(error);
  }
}

final class _TextTagArray {
  _TextTagArray(List<_SsmlTag> tags)
    : length = tags.length,
      pointer = tags.isEmpty
          ? ffi.nullptr
          : calloc<native.TextTagAbi>(tags.length) {
    try {
      for (var i = 0; i < tags.length; i += 1) {
        final tag = tags[i];
        pointer[i]
          ..start = tag.start
          ..end = tag.end
          ..kind = tag.kind
          ..value = ffi.nullptr;
        final value = tag.value;
        if (value != null) {
          final ptr = value.toNativeUtf8(allocator: calloc).cast<ffi.Char>();
          _strings.add(ptr);
          pointer[i].value = ptr;
        }
      }
    } catch (_) {
      close();
      rethrow;
    }
  }

  final int length;
  final ffi.Pointer<native.TextTagAbi> pointer;
  final List<ffi.Pointer<ffi.Char>> _strings = [];

  void close() {
    for (final value in _strings) {
      calloc.free(value);
    }
    _strings.clear();
    if (pointer != ffi.nullptr) {
      calloc.free(pointer);
    }
  }
}

String stripSsmlForTts(String ssml) {
  return _textCall(ssml, native.textStripSsml);
}

String _takeTextError(ffi.Pointer<ffi.Pointer<ffi.Char>> error) {
  final value = error.value;
  if (value == ffi.nullptr) return 'Native text call failed.';
  try {
    return value.cast<Utf8>().toDartString();
  } finally {
    native.freeStr(value);
    error.value = ffi.nullptr;
  }
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
  return _textCall(surface, native.textTnEn);
}

String verbalizeChinese(String surface) {
  return _textCall(surface, native.textTnZh);
}

String _textCall(
  String value,
  ffi.Pointer<ffi.Char> Function(
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
  call,
) {
  final input = value.toNativeUtf8(allocator: calloc).cast<ffi.Char>();
  final error = calloc<ffi.Pointer<ffi.Char>>();
  ffi.Pointer<ffi.Char> out = ffi.nullptr;
  try {
    out = call(input, error);
    if (out == ffi.nullptr) {
      throw StateError(_takeTextError(error));
    }
    return out.cast<Utf8>().toDartString();
  } finally {
    if (out != ffi.nullptr) {
      native.freeStr(out);
    }
    calloc
      ..free(input)
      ..free(error);
  }
}
