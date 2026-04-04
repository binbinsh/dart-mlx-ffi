library;

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

export 'preprocess_basic.dart' show basicEnglishTokenize;
export 'text_cleaner.dart' show TextCleaner;

import 'espeak_g2p.dart';
import 'preprocess_basic.dart';
import 'text_cleaner.dart';

/// Converts an IPA phoneme string to padded token IDs.
///
/// The input [phonemeText] should already be phonemised IPA text.
/// Output: `[0, ...token_ids..., 0]` (pad tokens at start and end).
List<int> buildInputIdsFromPhonemes(
  String phonemeText, {
  TextCleaner? cleaner,
}) {
  final resolvedCleaner = cleaner ?? TextCleaner();
  final phonemes = basicEnglishTokenize(phonemeText).join(' ');
  final ids = resolvedCleaner(phonemes);
  return <int>[0, ...ids, 0];
}

/// Converts an IPA phoneme string to a padded `[1, seqLen]` MlxArray.
MlxArray buildInputArrayFromPhonemes(
  String phonemeText, {
  TextCleaner? cleaner,
}) {
  final ids = buildInputIdsFromPhonemes(phonemeText, cleaner: cleaner);
  return MlxArray.fromInt32List(ids, shape: [1, ids.length]);
}

/// Converts raw English text to padded token IDs via eSpeak-NG G2P.
///
/// Performs: text → eSpeak IPA phonemes → tokenize → pad.
List<int> buildInputIdsFromText(
  String text, {
  required EspeakG2p g2p,
  TextCleaner? cleaner,
}) {
  final phonemes = g2p.textToPhonemes(text);
  return buildInputIdsFromPhonemes(phonemes, cleaner: cleaner);
}

/// Converts raw English text to a padded `[1, seqLen]` MlxArray via G2P.
MlxArray buildInputArrayFromText(
  String text, {
  required EspeakG2p g2p,
  TextCleaner? cleaner,
}) {
  final ids = buildInputIdsFromText(text, g2p: g2p, cleaner: cleaner);
  return MlxArray.fromInt32List(ids, shape: [1, ids.length]);
}
