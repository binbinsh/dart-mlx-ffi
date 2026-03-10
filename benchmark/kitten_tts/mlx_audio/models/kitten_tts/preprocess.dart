library;

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

export '../../preprocess.dart' show basicEnglishTokenize;
export '../../text_cleaner.dart' show TextCleaner;

import '../../preprocess.dart';
import '../../text_cleaner.dart';

List<int> buildInputIdsFromPhonemes(
  String phonemeText, {
  TextCleaner? cleaner,
}) {
  final resolvedCleaner = cleaner ?? TextCleaner();
  final phonemes = basicEnglishTokenize(phonemeText).join(' ');
  final ids = resolvedCleaner(phonemes);
  return <int>[0, ...ids, 0];
}

MlxArray buildInputArrayFromPhonemes(
  String phonemeText, {
  TextCleaner? cleaner,
}) {
  final ids = buildInputIdsFromPhonemes(phonemeText, cleaner: cleaner);
  return MlxArray.fromInt32List(ids, shape: [1, ids.length]);
}
