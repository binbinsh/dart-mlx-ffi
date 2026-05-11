part of 'qwen3_tts.dart';

final class Qwen3TtsPreparedReference {
  const Qwen3TtsPreparedReference({
    required this.refText,
    required this.speakerEmbedding,
    required this.refCodes,
  });

  factory Qwen3TtsPreparedReference.load(String path) {
    final file = File(path);
    if (!file.existsSync()) {
      throw StateError('Prepared Qwen3-TTS reference not found: $path');
    }
    final decoded = jsonDecode(file.readAsStringSync());
    if (decoded is! Map<String, Object?>) {
      throw StateError('Invalid prepared Qwen3-TTS reference: $path');
    }
    final rawText = decoded['ref_text']?.toString() ?? '';
    final rawSpeaker = decoded['speaker_embedding'];
    final rawCodes = decoded['ref_codes'];
    if (rawText.trim().isEmpty || rawSpeaker is! List || rawCodes is! List) {
      throw StateError('Prepared Qwen3-TTS reference is incomplete: $path');
    }
    final speakerEmbedding = Float32List(rawSpeaker.length);
    for (var i = 0; i < rawSpeaker.length; i++) {
      final value = rawSpeaker[i];
      if (value is! num) {
        throw StateError('Invalid speaker embedding value at index $i in $path');
      }
      speakerEmbedding[i] = value.toDouble();
    }
    final groups = <Int32List>[];
    for (var g = 0; g < rawCodes.length; g++) {
      final row = rawCodes[g];
      if (row is! List) {
        throw StateError('Invalid ref_codes group at index $g in $path');
      }
      final ints = Int32List(row.length);
      for (var i = 0; i < row.length; i++) {
        final value = row[i];
        if (value is! num) {
          throw StateError('Invalid ref_codes[$g][$i] value in $path');
        }
        ints[i] = value.toInt();
      }
      groups.add(ints);
    }
    return Qwen3TtsPreparedReference(
      refText: rawText,
      speakerEmbedding: speakerEmbedding,
      refCodes: groups,
    );
  }

  final String refText;
  final Float32List speakerEmbedding;
  final List<Int32List> refCodes;

  int get numCodeGroups => refCodes.length;
  int get refTime => refCodes.isEmpty ? 0 : refCodes.first.length;

  MlxArray createSpeakerEmbeddingArray() {
    return MlxArray.fromFloat32List(
      speakerEmbedding,
      shape: [1, 1, speakerEmbedding.length],
    );
  }

  MlxArray createRefCodesArray() {
    if (refCodes.isEmpty) {
      throw StateError('Prepared Qwen3-TTS reference has no ref_codes.');
    }
    final time = refCodes.first.length;
    final flat = Int32List(numCodeGroups * time);
    var cursor = 0;
    for (final group in refCodes) {
      if (group.length != time) {
        throw StateError('Prepared Qwen3-TTS ref_codes groups must share the same length.');
      }
      flat.setRange(cursor, cursor + time, group);
      cursor += time;
    }
    return MlxArray.fromInt32List(flat, shape: [1, numCodeGroups, time]);
  }
}
