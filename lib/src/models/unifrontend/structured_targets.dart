part of 'structured_frontend.dart';

final class PronunciationItem {
  const PronunciationItem({
    required this.start,
    required this.end,
    required this.surface,
    required this.pronunciation,
    this.candidates = const [],
  });

  final int start;
  final int end;
  final String surface;
  final String pronunciation;
  final List<String> candidates;

  PronunciationItem withPronunciation(String value) => PronunciationItem(
    start: start,
    end: end,
    surface: surface,
    pronunciation: value,
    candidates: candidates,
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
  PronunciationTargetResolver({
    required this.homographPronunciations,
    required this.polyphonePronunciations,
    required this.homographSurfaceCandidates,
    required this.polyphoneSurfaceCandidates,
  }) : _homographByLowerSurface = {
         for (final entry in homographSurfaceCandidates.entries)
           entry.key.toLowerCase(): entry.value,
       },
       _homographPronToId = {
         for (var i = 0; i < homographPronunciations.length; i++)
           homographPronunciations[i]: i,
       },
       _polyphonePronToId = {
         for (var i = 0; i < polyphonePronunciations.length; i++)
           polyphonePronunciations[i]: i,
       };

  final List<String> homographPronunciations;
  final List<String> polyphonePronunciations;
  final Map<String, List<String>> homographSurfaceCandidates;
  final Map<String, List<String>> polyphoneSurfaceCandidates;
  final Map<String, List<String>> _homographByLowerSurface;
  final Map<String, int> _homographPronToId;
  final Map<String, int> _polyphonePronToId;

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
    final matches = <_TargetMatch>[];
    for (final match in RegExp(
      r"[A-Za-z]+(?:['-][A-Za-z]+)*",
    ).allMatches(text)) {
      final surface = match.group(0) ?? '';
      final candidates = _homographByLowerSurface[surface.toLowerCase()];
      if (surface.isEmpty || candidates == null || candidates.length < 2) {
        continue;
      }
      matches.add(_TargetMatch(match.start, match.end, surface, candidates));
    }
    return [
      for (final match in _longestNonOverlappingTargets(matches))
        PronunciationItem(
          start: match.start,
          end: match.end,
          surface: match.surface,
          pronunciation: '',
          candidates: match.candidates,
        ),
    ];
  }

  List<PronunciationItem> proposePolyphones(String text) {
    final matches = <_TargetMatch>[];
    for (final entry in polyphoneSurfaceCandidates.entries) {
      final surface = entry.key;
      final candidates = entry.value;
      if (surface.isEmpty || candidates.length < 2) {
        continue;
      }
      var start = 0;
      while (start < text.length) {
        final idx = text.indexOf(surface, start);
        if (idx < 0) {
          break;
        }
        matches.add(
          _TargetMatch(idx, idx + surface.length, surface, candidates),
        );
        start = idx + 1;
      }
    }
    return [
      for (final match in _longestNonOverlappingTargets(matches))
        PronunciationItem(
          start: match.start,
          end: match.end,
          surface: match.surface,
          pronunciation: '',
          candidates: match.candidates,
        ),
    ];
  }

  List<int> homographCandidateIds(PronunciationItem item) {
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
}

final class _TargetMatch {
  const _TargetMatch(this.start, this.end, this.surface, this.candidates);

  final int start;
  final int end;
  final String surface;
  final List<String> candidates;
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

List<_TargetMatch> _longestNonOverlappingTargets(List<_TargetMatch> matches) {
  final sorted = matches.toList()
    ..sort((a, b) {
      final lengthCompare = (b.end - b.start).compareTo(a.end - a.start);
      if (lengthCompare != 0) {
        return lengthCompare;
      }
      final startCompare = a.start.compareTo(b.start);
      if (startCompare != 0) {
        return startCompare;
      }
      return a.end.compareTo(b.end);
    });
  final taken = <(int, int)>[];
  final selected = <_TargetMatch>[];
  for (final match in sorted) {
    final overlaps = taken.any(
      (span) => !(match.end <= span.$1 || match.start >= span.$2),
    );
    if (overlaps) {
      continue;
    }
    taken.add((match.start, match.end));
    selected.add(match);
  }
  selected.sort((a, b) {
    final startCompare = a.start.compareTo(b.start);
    if (startCompare != 0) {
      return startCompare;
    }
    final endCompare = a.end.compareTo(b.end);
    if (endCompare != 0) {
      return endCompare;
    }
    return a.surface.compareTo(b.surface);
  });
  return selected;
}

List<int> tokenPositionsForSpan(
  List<(int, int)> offsets, {
  required int start,
  required int end,
}) {
  final positions = <int>[];
  for (var idx = 0; idx < offsets.length; idx++) {
    final (tokStart, tokEnd) = offsets[idx];
    if (tokEnd <= tokStart) {
      continue;
    }
    if (tokEnd <= start || tokStart >= end) {
      continue;
    }
    positions.add(idx);
  }
  return positions;
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
  for (var targetIdx = 0; targetIdx < targetCount; targetIdx++) {
    final offset = rowOffset + targetIdx * classesDim;
    var best = 0;
    var bestVal = -double.infinity;
    for (var classIdx = 0; classIdx < classCount; classIdx++) {
      final value = tensor.data[offset + classIdx];
      if (value > bestVal) {
        bestVal = value;
        best = classIdx;
      }
    }
    out.add(targets[targetIdx].withPronunciation(labels[best]));
  }
  return out;
}
