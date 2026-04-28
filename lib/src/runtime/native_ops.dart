part of 'native_bindings.dart';

@ffi.Native<
  ffi.Pointer<ffi.Void> Function(
    ffi.Pointer<ffi.Float>,
    ffi.IntPtr,
    ffi.Int32,
    ffi.Pointer<ffi.IntPtr>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_audio_wav_pcm16')
external ffi.Pointer<ffi.Void> audioWavPcm16(
  ffi.Pointer<ffi.Float> samples,
  int sampleCount,
  int sampleRate,
  ffi.Pointer<ffi.IntPtr> byteLength,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<
  ffi.Pointer<ffi.Void> Function(
    ffi.Pointer<ffi.Pointer<ffi.Float>>,
    ffi.Pointer<ffi.IntPtr>,
    ffi.IntPtr,
    ffi.Int32,
    ffi.Pointer<ffi.IntPtr>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_audio_wav_pcm16_chunks')
external ffi.Pointer<ffi.Void> audioWavPcm16Chunks(
  ffi.Pointer<ffi.Pointer<ffi.Float>> sampleChunks,
  ffi.Pointer<ffi.IntPtr> sampleCounts,
  int chunkCount,
  int sampleRate,
  ffi.Pointer<ffi.IntPtr> byteLength,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<
  ffi.Pointer<ffi.Void> Function(
    ffi.Pointer<ffi.Pointer<ffi.Float>>,
    ffi.Pointer<ffi.IntPtr>,
    ffi.IntPtr,
    ffi.Pointer<ffi.IntPtr>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_audio_concat_f32')
external ffi.Pointer<ffi.Void> audioConcatF32(
  ffi.Pointer<ffi.Pointer<ffi.Float>> sampleChunks,
  ffi.Pointer<ffi.IntPtr> sampleCounts,
  int chunkCount,
  ffi.Pointer<ffi.IntPtr> sampleCount,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<
  ffi.Pointer<ffi.Char> Function(
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<TextTagAbi>,
    ffi.IntPtr,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_text_ssml')
external ffi.Pointer<ffi.Char> textSsml(
  ffi.Pointer<ffi.Char> input,
  ffi.Pointer<TextTagAbi> tags,
  int tagCount,
  ffi.Pointer<ffi.Char> emotion,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<
  ffi.Pointer<ffi.Char> Function(
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_text_strip_ssml')
external ffi.Pointer<ffi.Char> textStripSsml(
  ffi.Pointer<ffi.Char> input,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<
  ffi.Pointer<ffi.Char> Function(
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_text_tn_en')
external ffi.Pointer<ffi.Char> textTnEn(
  ffi.Pointer<ffi.Char> input,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<
  ffi.Pointer<ffi.Char> Function(
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_text_tn_zh')
external ffi.Pointer<ffi.Char> textTnZh(
  ffi.Pointer<ffi.Char> input,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<ffi.Int32 Function(ffi.Pointer<ffi.Char>)>(
  symbol: 'dinf_text_has_zh',
)
external int textHasZh(ffi.Pointer<ffi.Char> input);

@ffi.Native<
  ffi.Pointer<ffi.Void> Function(
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Char>,
    ffi.Int32,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_esp_new')
external ffi.Pointer<ffi.Void> espNew(
  ffi.Pointer<ffi.Char> libraryPath,
  ffi.Pointer<ffi.Char> dataPath,
  ffi.Pointer<ffi.Char> voice,
  int phonemeMode,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Void>)>(symbol: 'dinf_esp_free')
external void espFree(ffi.Pointer<ffi.Void> handle);

@ffi.Native<
  ffi.Pointer<ffi.Char> Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_esp_text')
external ffi.Pointer<ffi.Char> espText(
  ffi.Pointer<ffi.Void> handle,
  ffi.Pointer<ffi.Char> text,
  ffi.Pointer<ffi.Char> voice,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<
  ffi.Pointer<ffi.Char> Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_esp_kok_text')
external ffi.Pointer<ffi.Char> espKokText(
  ffi.Pointer<ffi.Void> handle,
  ffi.Pointer<ffi.Char> text,
  ffi.Pointer<ffi.Char> language,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<
  ffi.Pointer<ffi.Char> Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_esp_kok_ssml')
external ffi.Pointer<ffi.Char> espKokSsml(
  ffi.Pointer<ffi.Void> handle,
  ffi.Pointer<ffi.Char> ssml,
  ffi.Pointer<ffi.Char> language,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<
  ffi.Int32 Function(
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Int32>,
    ffi.Pointer<ffi.Int32>,
    ffi.IntPtr,
    ffi.Pointer<ffi.Int32>,
    ffi.Pointer<ffi.Int32>,
    ffi.Pointer<ffi.IntPtr>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_text_norm_spans')
external int textNormSpans(
  ffi.Pointer<ffi.Char> input,
  ffi.Pointer<ffi.Int32> starts,
  ffi.Pointer<ffi.Int32> ends,
  int count,
  ffi.Pointer<ffi.Int32> outStarts,
  ffi.Pointer<ffi.Int32> outEnds,
  ffi.Pointer<ffi.IntPtr> outCount,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<
  ffi.Int32 Function(
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Int32>,
    ffi.Pointer<ffi.Int32>,
    ffi.IntPtr,
    ffi.Pointer<ffi.Int32>,
    ffi.Pointer<ffi.Int32>,
    ffi.IntPtr,
    ffi.Pointer<ffi.Int32>,
    ffi.Pointer<ffi.Int32>,
    ffi.Pointer<ffi.IntPtr>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_text_select_tn')
external int textSelectTn(
  ffi.Pointer<ffi.Char> input,
  ffi.Pointer<ffi.Int32> enStarts,
  ffi.Pointer<ffi.Int32> enEnds,
  int enCount,
  ffi.Pointer<ffi.Int32> zhStarts,
  ffi.Pointer<ffi.Int32> zhEnds,
  int zhCount,
  ffi.Pointer<ffi.Int32> outSources,
  ffi.Pointer<ffi.Int32> outIndices,
  ffi.Pointer<ffi.IntPtr> outCount,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<
  ffi.Int32 Function(
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<StructConfigAbi>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_struct_config')
external int structConfig(
  ffi.Pointer<ffi.Char> exportPath,
  ffi.Pointer<ffi.Char> structuredPath,
  ffi.Pointer<StructConfigAbi> out,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<
  ffi.Pointer<ffi.Void> Function(
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
    ffi.Pointer<ffi.Int64>,
    ffi.IntPtr,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
    ffi.IntPtr,
    ffi.Int64,
    ffi.Int64,
    ffi.Int64,
    ffi.Int64,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_bpe_new')
external ffi.Pointer<ffi.Void> bpeNew(
  ffi.Pointer<ffi.Pointer<ffi.Char>> vocabKeys,
  ffi.Pointer<ffi.Int64> vocabIds,
  int vocabCount,
  ffi.Pointer<ffi.Pointer<ffi.Char>> mergeKeys,
  int mergeCount,
  int bosId,
  int eosId,
  int padId,
  int unkId,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Void>)>(symbol: 'dinf_bpe_free')
external void bpeFree(ffi.Pointer<ffi.Void> handle);

@ffi.Native<
  ffi.Int32 Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Char>,
    ffi.IntPtr,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Int32>,
    ffi.Pointer<ffi.Int32>,
    ffi.Pointer<ffi.IntPtr>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_bpe_encode')
external int bpeEncode(
  ffi.Pointer<ffi.Void> handle,
  ffi.Pointer<ffi.Char> text,
  int maxLength,
  ffi.Pointer<ffi.Int64> ids,
  ffi.Pointer<ffi.Int32> starts,
  ffi.Pointer<ffi.Int32> ends,
  ffi.Pointer<ffi.IntPtr> count,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<
  ffi.Pointer<ffi.Void> Function(
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
    ffi.Pointer<ffi.Int64>,
    ffi.IntPtr,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
    ffi.IntPtr,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
    ffi.Pointer<ffi.Int64>,
    ffi.IntPtr,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_qwen2_bpe_new')
external ffi.Pointer<ffi.Void> qwen2BpeNew(
  ffi.Pointer<ffi.Pointer<ffi.Char>> vocabKeys,
  ffi.Pointer<ffi.Int64> vocabIds,
  int vocabCount,
  ffi.Pointer<ffi.Pointer<ffi.Char>> mergeKeys,
  int mergeCount,
  ffi.Pointer<ffi.Pointer<ffi.Char>> specialTexts,
  ffi.Pointer<ffi.Int64> specialIds,
  int specialCount,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Void>)>(
  symbol: 'dinf_qwen2_bpe_free',
)
external void qwen2BpeFree(ffi.Pointer<ffi.Void> handle);

@ffi.Native<
  ffi.Int32 Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Char>,
    ffi.IntPtr,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.IntPtr>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_qwen2_bpe_encode')
external int qwen2BpeEncode(
  ffi.Pointer<ffi.Void> handle,
  ffi.Pointer<ffi.Char> text,
  int maxLength,
  ffi.Pointer<ffi.Int64> ids,
  ffi.Pointer<ffi.IntPtr> count,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<
  ffi.Pointer<ffi.Char> Function(
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_kok_norm')
external ffi.Pointer<ffi.Char> kokNorm(
  ffi.Pointer<ffi.Char> text,
  ffi.Pointer<ffi.Char> language,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<
  ffi.Pointer<ffi.Char> Function(
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_kok_post')
external ffi.Pointer<ffi.Char> kokPost(
  ffi.Pointer<ffi.Char> text,
  ffi.Pointer<ffi.Char> language,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<
  ffi.Pointer<ffi.Char> Function(
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_kok_plain')
external ffi.Pointer<ffi.Char> kokPlain(
  ffi.Pointer<ffi.Char> text,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<
  ffi.Pointer<ffi.Char> Function(
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_kok_exp')
external ffi.Pointer<ffi.Char> kokExplicit(
  ffi.Pointer<ffi.Char> text,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<
  ffi.Pointer<ffi.Char> Function(
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_kok_pin_norm')
external ffi.Pointer<ffi.Char> kokPinyinNorm(
  ffi.Pointer<ffi.Char> text,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<ffi.Int32 Function(ffi.Pointer<ffi.Char>)>(symbol: 'dinf_kok_pin')
external int kokLooksPinyin(ffi.Pointer<ffi.Char> text);

@ffi.Native<
  ffi.Int32 Function(
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Pointer<KokoroRunAbi>>,
    ffi.Pointer<ffi.IntPtr>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_kok_runs')
external int kokRuns(
  ffi.Pointer<ffi.Char> text,
  ffi.Pointer<ffi.Char> defaultLanguage,
  ffi.Pointer<ffi.Pointer<KokoroRunAbi>> out,
  ffi.Pointer<ffi.IntPtr> count,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<
  ffi.Pointer<ffi.Char> Function(
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Int32>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_kok_lang')
external ffi.Pointer<ffi.Char> kokLanguage(
  ffi.Pointer<ffi.Char> text,
  ffi.Pointer<ffi.Char> requested,
  ffi.Pointer<ffi.Int32> mixed,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<ffi.Void Function(ffi.Pointer<KokoroRunAbi>, ffi.IntPtr)>(
  symbol: 'dinf_kok_free_runs',
)
external void kokFreeRuns(ffi.Pointer<KokoroRunAbi> runs, int count);

@ffi.Native<
  ffi.Int32 Function(
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Pointer<KokoroSsmlAbi>>,
    ffi.Pointer<ffi.IntPtr>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_kok_ssml')
external int kokSsml(
  ffi.Pointer<ffi.Char> text,
  ffi.Pointer<ffi.Pointer<KokoroSsmlAbi>> out,
  ffi.Pointer<ffi.IntPtr> count,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<ffi.Void Function(ffi.Pointer<KokoroSsmlAbi>, ffi.IntPtr)>(
  symbol: 'dinf_kok_free_ssml',
)
external void kokFreeSsml(ffi.Pointer<KokoroSsmlAbi> items, int count);

@ffi.Native<
  ffi.Int32 Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Int64>,
    ffi.IntPtr,
    ffi.IntPtr,
    ffi.IntPtr,
    ffi.Pointer<ffi.Int32>,
    ffi.Pointer<ffi.Int32>,
    ffi.Pointer<ffi.IntPtr>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_bpe_fill')
external int bpeFill(
  ffi.Pointer<ffi.Void> handle,
  ffi.Pointer<ffi.Char> text,
  ffi.Pointer<ffi.Int64> values,
  ffi.Pointer<ffi.Int64> mask,
  int length,
  int offset,
  int width,
  ffi.Pointer<ffi.Int32> starts,
  ffi.Pointer<ffi.Int32> ends,
  ffi.Pointer<ffi.IntPtr> count,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<
  ffi.Pointer<ffi.Void> Function(
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
    ffi.IntPtr,
    ffi.Pointer<ffi.IntPtr>,
    ffi.Pointer<ffi.Int32>,
    ffi.IntPtr,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
    ffi.IntPtr,
    ffi.Pointer<ffi.IntPtr>,
    ffi.Pointer<ffi.Int32>,
    ffi.IntPtr,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_tgt_new')
external ffi.Pointer<ffi.Void> targetNew(
  ffi.Pointer<ffi.Pointer<ffi.Char>> homographs,
  int homographCount,
  ffi.Pointer<ffi.IntPtr> homographOffsets,
  ffi.Pointer<ffi.Int32> homographIds,
  int homographIdCount,
  ffi.Pointer<ffi.Pointer<ffi.Char>> polyphones,
  int polyphoneCount,
  ffi.Pointer<ffi.IntPtr> polyphoneOffsets,
  ffi.Pointer<ffi.Int32> polyphoneIds,
  int polyphoneIdCount,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Void>)>(symbol: 'dinf_tgt_free')
external void targetFree(ffi.Pointer<ffi.Void> handle);

@ffi.Native<
  ffi.Int32 Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Pointer<TargetMatchAbi>>,
    ffi.Pointer<ffi.IntPtr>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_tgt_homographs')
external int targetHomographs(
  ffi.Pointer<ffi.Void> handle,
  ffi.Pointer<ffi.Char> text,
  ffi.Pointer<ffi.Pointer<TargetMatchAbi>> matches,
  ffi.Pointer<ffi.IntPtr> count,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<
  ffi.Int32 Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Pointer<TargetMatchAbi>>,
    ffi.Pointer<ffi.IntPtr>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_tgt_polyphones')
external int targetPolyphones(
  ffi.Pointer<ffi.Void> handle,
  ffi.Pointer<ffi.Char> text,
  ffi.Pointer<ffi.Pointer<TargetMatchAbi>> matches,
  ffi.Pointer<ffi.IntPtr> count,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<ffi.Void Function(ffi.Pointer<TargetMatchAbi>, ffi.IntPtr)>(
  symbol: 'dinf_tgt_free_matches',
)
external void targetFreeMatches(ffi.Pointer<TargetMatchAbi> matches, int count);

@ffi.Native<
  ffi.Pointer<ffi.Char> Function(
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Int32>,
    ffi.IntPtr,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_kok_filter')
external ffi.Pointer<ffi.Char> kokFilter(
  ffi.Pointer<ffi.Char> phonemes,
  ffi.Pointer<ffi.Int32> codes,
  int codeCount,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<
  ffi.Pointer<ffi.Char> Function(
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_kok_clean')
external ffi.Pointer<ffi.Char> kokClean(
  ffi.Pointer<ffi.Char> phonemes,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<
  ffi.Int32 Function(
    ffi.Pointer<ffi.Float>,
    ffi.IntPtr,
    ffi.Pointer<ffi.Float>,
    ffi.IntPtr,
    ffi.IntPtr,
    ffi.IntPtr,
    ffi.IntPtr,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_kok_row')
external int kokRow(
  ffi.Pointer<ffi.Float> style,
  int styleLength,
  ffi.Pointer<ffi.Float> voice,
  int voiceLength,
  int voiceRows,
  int voiceRowLength,
  int index,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<
  ffi.Int32 Function(
    ffi.Pointer<ffi.Int64>,
    ffi.IntPtr,
    ffi.Pointer<ffi.Int64>,
    ffi.IntPtr,
    ffi.Pointer<ffi.Float>,
    ffi.IntPtr,
    ffi.Pointer<ffi.Float>,
    ffi.IntPtr,
    ffi.IntPtr,
    ffi.IntPtr,
    ffi.Pointer<ffi.Float>,
    ffi.IntPtr,
    ffi.Float,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_kok_inputs')
external int kokInputs(
  ffi.Pointer<ffi.Int64> inputIds,
  int inputLength,
  ffi.Pointer<ffi.Int64> tokenIds,
  int tokenCount,
  ffi.Pointer<ffi.Float> style,
  int styleLength,
  ffi.Pointer<ffi.Float> voice,
  int voiceLength,
  int voiceRows,
  int voiceRowLength,
  ffi.Pointer<ffi.Float> speedOut,
  int speedLength,
  double speed,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<
  ffi.Int32 Function(
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Int32>,
    ffi.Pointer<ffi.Int64>,
    ffi.IntPtr,
    ffi.IntPtr,
    ffi.Int32,
    ffi.Pointer<KokoroPlanAbi>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_kok_plan')
external int kokPlan(
  ffi.Pointer<ffi.Char> phonemes,
  ffi.Pointer<ffi.Int32> codes,
  ffi.Pointer<ffi.Int64> ids,
  int vocabCount,
  int maxTokens,
  int includeText,
  ffi.Pointer<KokoroPlanAbi> out,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<ffi.Void Function(ffi.Pointer<KokoroPlanAbi>)>(
  symbol: 'dinf_kok_free_plan',
)
external void kokFreePlan(ffi.Pointer<KokoroPlanAbi> plan);

@ffi.Native<
  ffi.Int32 Function(
    ffi.Pointer<ffi.Uint8>,
    ffi.IntPtr,
    ffi.Pointer<NpyAbi>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_kok_npy')
external int kokNpy(
  ffi.Pointer<ffi.Uint8> bytes,
  int byteCount,
  ffi.Pointer<NpyAbi> out,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<
  ffi.Int32 Function(
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Pointer<NpyAbi>>,
    ffi.Pointer<ffi.IntPtr>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_kok_npz')
external int kokNpz(
  ffi.Pointer<ffi.Char> path,
  ffi.Pointer<ffi.Pointer<NpyAbi>> out,
  ffi.Pointer<ffi.IntPtr> count,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<
  ffi.Int32 Function(
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<KokoroVocabAbi>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_kok_vocab')
external int kokVocab(
  ffi.Pointer<ffi.Char> path,
  ffi.Pointer<KokoroVocabAbi> out,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<ffi.Void Function(ffi.Pointer<KokoroVocabAbi>)>(
  symbol: 'dinf_kok_free_vocab',
)
external void kokFreeVocab(ffi.Pointer<KokoroVocabAbi> value);

@ffi.Native<ffi.Void Function(ffi.Pointer<NpyAbi>)>(symbol: 'dinf_kok_free_npy')
external void kokFreeNpy(ffi.Pointer<NpyAbi> value);

@ffi.Native<ffi.Void Function(ffi.Pointer<NpyAbi>, ffi.IntPtr)>(
  symbol: 'dinf_kok_free_npz',
)
external void kokFreeNpz(ffi.Pointer<NpyAbi> items, int count);

@ffi.Native<
  ffi.Int32 Function(
    ffi.Pointer<ffi.Float>,
    ffi.IntPtr,
    ffi.IntPtr,
    ffi.IntPtr,
    ffi.IntPtr,
    ffi.IntPtr,
    ffi.Pointer<ffi.Int32>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_dec_argmax')
external int decArgmax(
  ffi.Pointer<ffi.Float> data,
  int dataLength,
  int base,
  int itemCount,
  int stride,
  int classCount,
  ffi.Pointer<ffi.Int32> out,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<
  ffi.Int32 Function(
    ffi.Pointer<ffi.Float>,
    ffi.IntPtr,
    ffi.IntPtr,
    ffi.IntPtr,
    ffi.IntPtr,
    ffi.IntPtr,
    ffi.Pointer<ffi.Int32>,
    ffi.Pointer<ffi.Int32>,
    ffi.Pointer<ffi.IntPtr>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_dec_bioes')
external int decBioes(
  ffi.Pointer<ffi.Float> data,
  int dataLength,
  int base,
  int itemCount,
  int stride,
  int classCount,
  ffi.Pointer<ffi.Int32> starts,
  ffi.Pointer<ffi.Int32> ends,
  ffi.Pointer<ffi.IntPtr> spanCount,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<
  ffi.Int32 Function(
    ffi.Pointer<ffi.Float>,
    ffi.IntPtr,
    ffi.IntPtr,
    ffi.IntPtr,
    ffi.IntPtr,
    ffi.IntPtr,
    ffi.Pointer<ffi.Int32>,
    ffi.Pointer<ffi.Int32>,
    ffi.IntPtr,
    ffi.Pointer<ffi.Int32>,
    ffi.Pointer<ffi.Int32>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_dec_span_types')
external int decSpanTypes(
  ffi.Pointer<ffi.Float> data,
  int dataLength,
  int base,
  int itemCount,
  int stride,
  int classCount,
  ffi.Pointer<ffi.Int32> starts,
  ffi.Pointer<ffi.Int32> ends,
  int spanCount,
  ffi.Pointer<ffi.Int32> counts,
  ffi.Pointer<ffi.Int32> out,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<
  ffi.Int32 Function(
    ffi.Pointer<ffi.Float>,
    ffi.IntPtr,
    ffi.IntPtr,
    ffi.IntPtr,
    ffi.Double,
    ffi.Pointer<ffi.Int32>,
    ffi.Pointer<ffi.IntPtr>,
    ffi.Pointer<ffi.Int32>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_dec_active')
external int decActive(
  ffi.Pointer<ffi.Float> data,
  int dataLength,
  int offset,
  int count,
  double threshold,
  ffi.Pointer<ffi.Int32> out,
  ffi.Pointer<ffi.IntPtr> activeCount,
  ffi.Pointer<ffi.Int32> best,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<
  ffi.Int32 Function(
    ffi.Pointer<ffi.Float>,
    ffi.IntPtr,
    ffi.IntPtr,
    ffi.IntPtr,
    ffi.IntPtr,
    ffi.Double,
    ffi.Pointer<ffi.Int32>,
    ffi.Pointer<ffi.Int32>,
    ffi.Pointer<ffi.IntPtr>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_dec_spans')
external int decSpans(
  ffi.Pointer<ffi.Float> data,
  int dataLength,
  int offset,
  int count,
  int finalEnd,
  double threshold,
  ffi.Pointer<ffi.Int32> starts,
  ffi.Pointer<ffi.Int32> ends,
  ffi.Pointer<ffi.IntPtr> spanCount,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<
  ffi.Int32 Function(
    ffi.Pointer<ffi.Int64>,
    ffi.IntPtr,
    ffi.Int64,
    ffi.Pointer<ffi.Int64>,
    ffi.IntPtr,
    ffi.Pointer<ffi.Int64>,
    ffi.IntPtr,
    ffi.Int64,
    ffi.Pointer<ffi.Int64>,
    ffi.IntPtr,
    ffi.Pointer<ffi.Uint8>,
    ffi.IntPtr,
    ffi.Pointer<ffi.Uint8>,
    ffi.IntPtr,
    ffi.Pointer<ffi.Uint8>,
    ffi.IntPtr,
    ffi.Pointer<ffi.Uint8>,
    ffi.IntPtr,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_struct_reset')
external int structReset(
  ffi.Pointer<ffi.Int64> inputIds,
  int inputLength,
  int tokenPadId,
  ffi.Pointer<ffi.Int64> attention,
  int attentionLength,
  ffi.Pointer<ffi.Int64> charIds,
  int charLength,
  int charPadId,
  ffi.Pointer<ffi.Int64> charMask,
  int charMaskLength,
  ffi.Pointer<ffi.Uint8> homographTargets,
  int homographTargetLength,
  ffi.Pointer<ffi.Uint8> homographCandidates,
  int homographCandidateLength,
  ffi.Pointer<ffi.Uint8> polyphoneTargets,
  int polyphoneTargetLength,
  ffi.Pointer<ffi.Uint8> polyphoneCandidates,
  int polyphoneCandidateLength,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<
  ffi.Int32 Function(
    ffi.Pointer<ffi.Uint8>,
    ffi.IntPtr,
    ffi.IntPtr,
    ffi.IntPtr,
    ffi.Pointer<ffi.Uint8>,
    ffi.IntPtr,
    ffi.IntPtr,
    ffi.IntPtr,
    ffi.Pointer<TargetMatchAbi>,
    ffi.IntPtr,
    ffi.Pointer<ffi.Int32>,
    ffi.Pointer<ffi.Int32>,
    ffi.IntPtr,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_struct_matches')
external int structMatches(
  ffi.Pointer<ffi.Uint8> targetValues,
  int targetLength,
  int targetOffset,
  int targetWidth,
  ffi.Pointer<ffi.Uint8> candidateValues,
  int candidateLength,
  int candidateOffset,
  int candidateWidth,
  ffi.Pointer<TargetMatchAbi> matches,
  int matchCount,
  ffi.Pointer<ffi.Int32> tokenStarts,
  ffi.Pointer<ffi.Int32> tokenEnds,
  int tokenCount,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<
  ffi.Int32 Function(
    ffi.Pointer<ffi.Int64>,
    ffi.Pointer<ffi.Int64>,
    ffi.IntPtr,
    ffi.IntPtr,
    ffi.IntPtr,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Int32>,
    ffi.Pointer<ffi.Int64>,
    ffi.IntPtr,
    ffi.Int64,
    ffi.Int64,
    ffi.Pointer<ffi.IntPtr>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_fill_chars_i64')
external int fillCharsI64(
  ffi.Pointer<ffi.Int64> values,
  ffi.Pointer<ffi.Int64> mask,
  int length,
  int offset,
  int width,
  ffi.Pointer<ffi.Char> text,
  ffi.Pointer<ffi.Int32> codes,
  ffi.Pointer<ffi.Int64> ids,
  int vocabCount,
  int padId,
  int unkId,
  ffi.Pointer<ffi.IntPtr> count,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<ffi.Pointer<ffi.Void> Function(ffi.IntPtr)>(symbol: 'dinf_vec_new')
external ffi.Pointer<ffi.Void> vecNew(int dimension);

@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Void>)>(symbol: 'dinf_vec_free')
external void vecFree(ffi.Pointer<ffi.Void> handle);

@ffi.Native<ffi.IntPtr Function(ffi.Pointer<ffi.Void>)>(symbol: 'dinf_vec_len')
external int vecLen(ffi.Pointer<ffi.Void> handle);

@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Void>)>(symbol: 'dinf_vec_clear')
external void vecClear(ffi.Pointer<ffi.Void> handle);

@ffi.Native<
  ffi.Int32 Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Char>,
    ffi.Pointer<ffi.Double>,
    ffi.IntPtr,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_vec_put')
external int vecPut(
  ffi.Pointer<ffi.Void> handle,
  ffi.Pointer<ffi.Char> id,
  ffi.Pointer<ffi.Double> values,
  int length,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<
  ffi.Int32 Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
    ffi.Pointer<ffi.Double>,
    ffi.IntPtr,
    ffi.IntPtr,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_vec_put_many')
external int vecPutMany(
  ffi.Pointer<ffi.Void> handle,
  ffi.Pointer<ffi.Pointer<ffi.Char>> ids,
  ffi.Pointer<ffi.Double> values,
  int count,
  int dimension,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<ffi.Int32 Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Char>)>(
  symbol: 'dinf_vec_remove',
)
external int vecRemove(ffi.Pointer<ffi.Void> handle, ffi.Pointer<ffi.Char> id);

@ffi.Native<
  ffi.Int32 Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Double>,
    ffi.IntPtr,
    ffi.IntPtr,
    ffi.Double,
    ffi.Pointer<ffi.Pointer<VecResultAbi>>,
    ffi.Pointer<ffi.IntPtr>,
    ffi.Pointer<ffi.Pointer<ffi.Char>>,
  )
>(symbol: 'dinf_vec_search')
external int vecSearch(
  ffi.Pointer<ffi.Void> handle,
  ffi.Pointer<ffi.Double> query,
  int length,
  int topK,
  double minScore,
  ffi.Pointer<ffi.Pointer<VecResultAbi>> results,
  ffi.Pointer<ffi.IntPtr> count,
  ffi.Pointer<ffi.Pointer<ffi.Char>> error,
);

@ffi.Native<ffi.Void Function(ffi.Pointer<VecResultAbi>, ffi.IntPtr)>(
  symbol: 'dinf_vec_free_results',
)
external void vecFreeResults(ffi.Pointer<VecResultAbi> results, int count);

@ffi.Native<ffi.Int32 Function(ffi.Pointer<ffi.Double>, ffi.IntPtr)>(
  symbol: 'dinf_vec_l2_norm',
)
external int vecL2Norm(ffi.Pointer<ffi.Double> values, int length);
