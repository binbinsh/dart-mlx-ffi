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
    return PronunciationTargetResolver._(
      homographPronunciations: homographPronunciations,
      polyphonePronunciations: polyphonePronunciations,
      homographSurfaceCandidates: homographSurfaceCandidates,
      polyphoneSurfaceCandidates: polyphoneSurfaceCandidates,
      homographByLowerSurface: homographByLowerSurface,
      homographSurfaces: List<String>.unmodifiable(homographSurfaces),
      homographNativeCandidates: List<List<String>>.unmodifiable(
        homographCandidates,
      ),
      homographNativeCandidateIds: List<List<int>>.unmodifiable(
        homographCandidateIds,
      ),
      polyphoneSurfaces: List<String>.unmodifiable(polyphoneSurfaces),
      polyphoneNativeCandidates: List<List<String>>.unmodifiable(
        polyphoneCandidates,
      ),
      polyphoneNativeCandidateIds: List<List<int>>.unmodifiable(
        polyphoneCandidateIds,
      ),
      homographPronToId: homographPronToId,
      polyphonePronToId: polyphonePronToId,
    );
  }

  PronunciationTargetResolver._({
    required this.homographPronunciations,
    required this.polyphonePronunciations,
    required this.homographSurfaceCandidates,
    required this.polyphoneSurfaceCandidates,
    required Map<String, List<String>> homographByLowerSurface,
    required List<String> homographSurfaces,
    required List<List<String>> homographNativeCandidates,
    required List<List<int>> homographNativeCandidateIds,
    required List<String> polyphoneSurfaces,
    required List<List<String>> polyphoneNativeCandidates,
    required List<List<int>> polyphoneNativeCandidateIds,
    required Map<String, int> homographPronToId,
    required Map<String, int> polyphonePronToId,
  }) : _homographByLowerSurface = homographByLowerSurface,
       _homographSurfaces = homographSurfaces,
       _homographNativeCandidates = homographNativeCandidates,
       _homographNativeCandidateIds = homographNativeCandidateIds,
       _polyphoneSurfaces = polyphoneSurfaces,
       _polyphoneNativeCandidates = polyphoneNativeCandidates,
       _polyphoneNativeCandidateIds = polyphoneNativeCandidateIds,
       _homographPronToId = homographPronToId,
       _polyphonePronToId = polyphonePronToId;

  final List<String> homographPronunciations;
  final List<String> polyphonePronunciations;
  final Map<String, List<String>> homographSurfaceCandidates;
  final Map<String, List<String>> polyphoneSurfaceCandidates;
  final Map<String, List<String>> _homographByLowerSurface;
  final List<String> _homographSurfaces;
  final List<List<String>> _homographNativeCandidates;
  final List<List<int>> _homographNativeCandidateIds;
  final List<String> _polyphoneSurfaces;
  final List<List<String>> _polyphoneNativeCandidates;
  final List<List<int>> _polyphoneNativeCandidateIds;
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
          candidateIds: includeCandidateIds ? matches.items[i].ids : const [],
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
          candidateIds: includeCandidateIds ? matches.items[i].ids : const [],
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
    final matches = <_TargetMatch>[];
    if (homographs) {
      final lower = text.toLowerCase();
      for (var index = 0; index < _homographSurfaces.length; index += 1) {
        final surface = _homographSurfaces[index];
        var start = 0;
        while (true) {
          final found = lower.indexOf(surface, start);
          if (found < 0) break;
          final end = found + surface.length;
          if (_isBoundary(lower, found - 1) && _isBoundary(lower, end)) {
            matches.add(
              _TargetMatch(
                start: found,
                end: end,
                index: index,
                ids: _homographNativeCandidateIds[index],
              ),
            );
          }
          start = math.max(end, found + 1);
        }
      }
    } else {
      for (var index = 0; index < _polyphoneSurfaces.length; index += 1) {
        final surface = _polyphoneSurfaces[index];
        var start = 0;
        while (true) {
          final found = text.indexOf(surface, start);
          if (found < 0) break;
          matches.add(
            _TargetMatch(
              start: found,
              end: found + surface.length,
              index: index,
              ids: _polyphoneNativeCandidateIds[index],
            ),
          );
          start = found + surface.length;
        }
      }
    }
    matches.sort(
      (a, b) => a.start == b.start
          ? a.end.compareTo(b.end)
          : a.start.compareTo(b.start),
    );
    return _NativeTargetMatches(List<_TargetMatch>.unmodifiable(matches));
  }

  void close() {
    if (_closed) {
      return;
    }
    _closed = true;
  }
}

final class _NativeTargetMatches {
  _NativeTargetMatches(this.items);

  final List<_TargetMatch> items;
  int get count => items.length;

  void close() {}
}

final class _TargetMatch {
  const _TargetMatch({
    required this.start,
    required this.end,
    required this.index,
    required this.ids,
  });

  final int start;
  final int end;
  final int index;
  final List<int> ids;
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

bool _isBoundary(String text, int index) {
  if (index < 0 || index >= text.length) {
    return true;
  }
  final code = text.codeUnitAt(index);
  return !((code >= 0x30 && code <= 0x39) ||
      (code >= 0x41 && code <= 0x5a) ||
      (code >= 0x61 && code <= 0x7a) ||
      code == 0x5f);
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
