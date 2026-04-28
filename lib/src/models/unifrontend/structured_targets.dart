part of 'structured_frontend.dart';

final class PronunciationItem {
  const PronunciationItem({
    required this.start,
    required this.end,
    required this.surface,
    required this.pronunciation,
    this.candidates = const [],
    this.candidateIds = const [],
  });

  final int start;
  final int end;
  final String surface;
  final String pronunciation;
  final List<String> candidates;
  final List<int> candidateIds;

  PronunciationItem withPronunciation(String value) => PronunciationItem(
    start: start,
    end: end,
    surface: surface,
    pronunciation: value,
    candidates: candidates,
    candidateIds: candidateIds,
  );

  Map<String, Object?> toJson() => {
    'start': start,
    'end': end,
    'surface': surface,
    'pronunciation': pronunciation,
    'candidates': candidates,
  };
}

final _targetFinalizer = Finalizer<ffi.Pointer<ffi.Void>>((handle) {
  if (handle != ffi.nullptr) {
    native.targetFree(handle);
  }
});

final class PronunciationTargetResolver {
  factory PronunciationTargetResolver({
    required List<String> homographPronunciations,
    required List<String> polyphonePronunciations,
    required Map<String, List<String>> homographSurfaceCandidates,
    required Map<String, List<String>> polyphoneSurfaceCandidates,
  }) {
    final homographPronToId = {
      for (var i = 0; i < homographPronunciations.length; i++)
        homographPronunciations[i]: i,
    };
    final polyphonePronToId = {
      for (var i = 0; i < polyphonePronunciations.length; i++)
        polyphonePronunciations[i]: i,
    };
    final homographByLowerSurface = {
      for (final entry in homographSurfaceCandidates.entries)
        entry.key.toLowerCase(): entry.value,
    };
    final homographSurfaces = <String>[];
    final homographCandidates = <List<String>>[];
    final homographCandidateIds = <List<int>>[];
    for (final entry in homographByLowerSurface.entries) {
      if (entry.key.isNotEmpty && entry.value.length >= 2) {
        homographSurfaces.add(entry.key);
        final labels = <String>{...entry.value};
        for (final original in homographSurfaceCandidates.entries) {
          if (original.key.toLowerCase() == entry.key) {
            labels.addAll(original.value);
          }
        }
        homographCandidates.add(List<String>.unmodifiable(labels));
        homographCandidateIds.add(_candidateIds(labels, homographPronToId));
      }
    }
    final polyphoneSurfaces = <String>[];
    final polyphoneCandidates = <List<String>>[];
    final polyphoneCandidateIds = <List<int>>[];
    for (final entry in polyphoneSurfaceCandidates.entries) {
      if (entry.key.isNotEmpty && entry.value.length >= 2) {
        polyphoneSurfaces.add(entry.key);
        polyphoneCandidates.add(List<String>.unmodifiable(entry.value));
        polyphoneCandidateIds.add(
          _candidateIds(entry.value, polyphonePronToId),
        );
      }
    }
    final handle = _createTargetMatcher(
      homographSurfaces: homographSurfaces,
      homographCandidateIds: homographCandidateIds,
      polyphoneSurfaces: polyphoneSurfaces,
      polyphoneCandidateIds: polyphoneCandidateIds,
    );
    final resolver = PronunciationTargetResolver._(
      homographPronunciations: homographPronunciations,
      polyphonePronunciations: polyphonePronunciations,
      homographSurfaceCandidates: homographSurfaceCandidates,
      polyphoneSurfaceCandidates: polyphoneSurfaceCandidates,
      homographByLowerSurface: homographByLowerSurface,
      homographNativeCandidates: List<List<String>>.unmodifiable(
        homographCandidates,
      ),
      polyphoneSurfaces: List<String>.unmodifiable(polyphoneSurfaces),
      polyphoneNativeCandidates: List<List<String>>.unmodifiable(
        polyphoneCandidates,
      ),
      homographPronToId: homographPronToId,
      polyphonePronToId: polyphonePronToId,
      handle: handle,
    );
    _targetFinalizer.attach(resolver, handle, detach: resolver);
    return resolver;
  }

  PronunciationTargetResolver._({
    required this.homographPronunciations,
    required this.polyphonePronunciations,
    required this.homographSurfaceCandidates,
    required this.polyphoneSurfaceCandidates,
    required Map<String, List<String>> homographByLowerSurface,
    required List<List<String>> homographNativeCandidates,
    required List<String> polyphoneSurfaces,
    required List<List<String>> polyphoneNativeCandidates,
    required Map<String, int> homographPronToId,
    required Map<String, int> polyphonePronToId,
    required ffi.Pointer<ffi.Void> handle,
  }) : _homographByLowerSurface = homographByLowerSurface,
       _homographNativeCandidates = homographNativeCandidates,
       _polyphoneSurfaces = polyphoneSurfaces,
       _polyphoneNativeCandidates = polyphoneNativeCandidates,
       _handle = handle,
       _homographPronToId = homographPronToId,
       _polyphonePronToId = polyphonePronToId;

  final List<String> homographPronunciations;
  final List<String> polyphonePronunciations;
  final Map<String, List<String>> homographSurfaceCandidates;
  final Map<String, List<String>> polyphoneSurfaceCandidates;
  final Map<String, List<String>> _homographByLowerSurface;
  final List<List<String>> _homographNativeCandidates;
  final List<String> _polyphoneSurfaces;
  final List<List<String>> _polyphoneNativeCandidates;
  final ffi.Pointer<ffi.Void> _handle;
  final Map<String, int> _homographPronToId;
  final Map<String, int> _polyphonePronToId;
  bool _closed = false;

  static PronunciationTargetResolver fromLabelSpace({
    required List<String> homographPronunciations,
    required List<String> polyphonePronunciations,
    required Object? homographSurfaceCandidates,
    required Object? polyphoneSurfaceCandidates,
  }) {
    return PronunciationTargetResolver(
      homographPronunciations: homographPronunciations,
      polyphonePronunciations: polyphonePronunciations,
      homographSurfaceCandidates: _candidateMap(homographSurfaceCandidates),
      polyphoneSurfaceCandidates: _candidateMap(polyphoneSurfaceCandidates),
    );
  }

  List<PronunciationItem> proposeHomographs(String text) {
    final matches = _targetMatches(text, homographs: true);
    try {
      return _homographItems(text, matches, matches.count);
    } finally {
      matches.close();
    }
  }

  List<PronunciationItem> proposePolyphones(String text) {
    final matches = _targetMatches(text, homographs: false);
    try {
      return _polyphoneItems(matches, matches.count);
    } finally {
      matches.close();
    }
  }

  List<PronunciationItem> _homographItems(
    String text,
    _NativeTargetMatches matches,
    int count, {
    bool includeCandidateIds = true,
  }) {
    return [
      for (var i = 0; i < count; i += 1)
        PronunciationItem(
          start: matches.items[i].start,
          end: matches.items[i].end,
          surface: text.substring(matches.items[i].start, matches.items[i].end),
          pronunciation: '',
          candidates: _homographNativeCandidates[matches.items[i].index],
          candidateIds: includeCandidateIds
              ? _matchIds(matches.items[i])
              : const [],
        ),
    ];
  }

  List<PronunciationItem> _polyphoneItems(
    _NativeTargetMatches matches,
    int count, {
    bool includeCandidateIds = true,
  }) {
    return [
      for (var i = 0; i < count; i += 1)
        PronunciationItem(
          start: matches.items[i].start,
          end: matches.items[i].end,
          surface: _polyphoneSurfaces[matches.items[i].index],
          pronunciation: '',
          candidates: _polyphoneNativeCandidates[matches.items[i].index],
          candidateIds: includeCandidateIds
              ? _matchIds(matches.items[i])
              : const [],
        ),
    ];
  }

  List<int> homographCandidateIds(PronunciationItem item) {
    if (item.candidateIds.isNotEmpty && item.pronunciation.isEmpty) {
      return item.candidateIds;
    }
    final labels = <String>{
      ...?homographSurfaceCandidates[item.surface],
      ...?_homographByLowerSurface[item.surface.toLowerCase()],
      ...item.candidates,
      if (item.pronunciation.isNotEmpty) item.pronunciation,
    };
    final ids = <int>[];
    for (final label in labels.toList()..sort()) {
      final id = _homographPronToId[label];
      if (id != null) {
        ids.add(id);
      }
    }
    return ids;
  }

  List<int> polyphoneCandidateIds(PronunciationItem item) {
    if (item.candidateIds.isNotEmpty && item.pronunciation.isEmpty) {
      return item.candidateIds;
    }
    final labels = <String>{
      ...?polyphoneSurfaceCandidates[item.surface],
      ...item.candidates,
      if (item.pronunciation.isNotEmpty) item.pronunciation,
    };
    final ids = <int>[];
    for (final label in labels.toList()..sort()) {
      final id = _polyphonePronToId[label];
      if (id != null) {
        ids.add(id);
      }
    }
    return ids;
  }

  _NativeTargetMatches _targetMatches(String text, {required bool homographs}) {
    if (_closed) {
      throw StateError('pronunciation target resolver is closed.');
    }
    final input = text.toNativeUtf8(allocator: calloc).cast<ffi.Char>();
    final matches = calloc<ffi.Pointer<native.TargetMatchAbi>>();
    final count = calloc<ffi.IntPtr>();
    final error = calloc<ffi.Pointer<ffi.Char>>();
    try {
      final status = homographs
          ? native.targetHomographs(_handle, input, matches, count, error)
          : native.targetPolyphones(_handle, input, matches, count, error);
      if (status != 0) {
        throw StateError(_takeFillError(error));
      }
      return _NativeTargetMatches(matches.value, count.value);
    } finally {
      calloc
        ..free(input)
        ..free(matches)
        ..free(count)
        ..free(error);
    }
  }

  void close() {
    if (_closed) {
      return;
    }
    _closed = true;
    _targetFinalizer.detach(this);
    native.targetFree(_handle);
  }
}

final class _NativeTargetMatches {
  _NativeTargetMatches(this.items, this.count);

  final ffi.Pointer<native.TargetMatchAbi> items;
  final int count;

  void close() {
    if (items != ffi.nullptr) {
      native.targetFreeMatches(items, count);
    }
  }
}

List<int> _matchIds(native.TargetMatchAbi match) {
  if (match.idCount <= 0 || match.ids == ffi.nullptr) {
    return const [];
  }
  return List<int>.generate(
    match.idCount,
    (index) => match.ids[index],
    growable: false,
  );
}

Map<String, List<String>> _candidateMap(Object? raw) {
  final out = <String, List<String>>{};
  if (raw is! Map) {
    return out;
  }
  for (final entry in raw.entries) {
    final value = entry.value;
    if (value is Iterable) {
      out[entry.key.toString()] = [
        for (final candidate in value) candidate.toString(),
      ];
    }
  }
  return out;
}

List<int> _candidateIds(Iterable<String> labels, Map<String, int> idsByLabel) {
  final ids = <int>[];
  for (final label in (labels.toList()..sort())) {
    final id = idsByLabel[label];
    if (id != null) {
      ids.add(id);
    }
  }
  return List<int>.unmodifiable(ids);
}

final class _Int32Rows {
  _Int32Rows(List<List<int>> rows)
    : rowCount = rows.length,
      offsets = rows.isEmpty
          ? ffi.nullptr
          : calloc<ffi.IntPtr>(rows.length + 1) {
    try {
      var total = 0;
      for (var i = 0; i < rows.length; i += 1) {
        offsets[i] = total;
        total += rows[i].length;
      }
      if (rows.isNotEmpty) {
        offsets[rows.length] = total;
      }
      valueCount = total;
      values = total == 0 ? ffi.nullptr : calloc<ffi.Int32>(total);
      var cursor = 0;
      for (final row in rows) {
        for (final value in row) {
          values[cursor] = value;
          cursor += 1;
        }
      }
    } catch (_) {
      close();
      rethrow;
    }
  }

  final int rowCount;
  final ffi.Pointer<ffi.IntPtr> offsets;
  int valueCount = 0;
  ffi.Pointer<ffi.Int32> values = ffi.nullptr;

  void close() {
    if (offsets != ffi.nullptr) {
      calloc.free(offsets);
    }
    if (values != ffi.nullptr) {
      calloc.free(values);
    }
  }
}

ffi.Pointer<ffi.Void> _createTargetMatcher({
  required List<String> homographSurfaces,
  required List<List<int>> homographCandidateIds,
  required List<String> polyphoneSurfaces,
  required List<List<int>> polyphoneCandidateIds,
}) {
  final homographs = _CStringArray(homographSurfaces);
  final homographIds = _Int32Rows(homographCandidateIds);
  final polyphones = _CStringArray(polyphoneSurfaces);
  final polyphoneIds = _Int32Rows(polyphoneCandidateIds);
  final error = calloc<ffi.Pointer<ffi.Char>>();
  try {
    final handle = native.targetNew(
      homographs.pointer,
      homographs.length,
      homographIds.offsets,
      homographIds.values,
      homographIds.valueCount,
      polyphones.pointer,
      polyphones.length,
      polyphoneIds.offsets,
      polyphoneIds.values,
      polyphoneIds.valueCount,
      error,
    );
    if (handle == ffi.nullptr) {
      throw StateError(_takeFillError(error));
    }
    return handle;
  } finally {
    homographs.close();
    homographIds.close();
    polyphones.close();
    polyphoneIds.close();
    calloc.free(error);
  }
}

List<PronunciationItem> _decodePronunciationItems({
  required List<PronunciationItem> targets,
  required _FloatTensor? tensor,
  required List<String> labels,
  int rowIndex = 0,
}) {
  if (targets.isEmpty ||
      tensor == null ||
      labels.isEmpty ||
      tensor.shape.length < 2) {
    return const [];
  }
  final hasBatch = tensor.shape.length >= 3;
  final targetsDim = hasBatch ? tensor.shape[1] : tensor.shape[0];
  final classesDim = hasBatch ? tensor.shape[2] : tensor.shape[1];
  final rowOffset = hasBatch ? rowIndex * targetsDim * classesDim : 0;
  final targetCount = math.min(targets.length, targetsDim);
  final classCount = math.min(labels.length, classesDim);
  if (targetCount <= 0 || classCount <= 0) {
    return const [];
  }
  final out = <PronunciationItem>[];
  final ids = _argmaxIds(
    tensor: tensor,
    base: rowOffset,
    itemCount: targetCount,
    stride: classesDim,
    classCount: classCount,
  );
  for (var targetIdx = 0; targetIdx < targetCount; targetIdx++) {
    final best = ids[targetIdx];
    out.add(targets[targetIdx].withPronunciation(labels[best]));
  }
  return out;
}
