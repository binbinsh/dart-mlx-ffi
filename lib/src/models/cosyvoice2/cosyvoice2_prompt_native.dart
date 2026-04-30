import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:dart_inference/runtime.dart';

import 'cosyvoice2_kaldi_fbank.dart' as kaldi;
import 'cosyvoice2_mel.dart';

enum CosyPromptMelKind { matcha80, whisper128 }

final class CosyPromptNativePlan {
  CosyPromptNativePlan();

  bool _closed = false;

  ffi.Pointer<ffi.Void> get nativeHandle {
    if (_closed) {
      throw StateError('CosyVoice2 prompt plan is closed.');
    }
    return ffi.nullptr;
  }

  void close() {
    _closed = true;
  }
}

final class CosyPromptFeatureBuffer {
  CosyPromptFeatureBuffer._({
    required this.data,
    required this.frames,
    required this.bins,
  });

  final NativeTensorBuffer data;
  final int frames;
  final int bins;

  void close() {
    data.close();
  }
}

CosyPromptFeatureBuffer cosyPromptMelSpectrogramBuffer(
  Object audio, {
  required CosyPromptMelKind kind,
  CosyPromptNativePlan? plan,
}) {
  final audioValues = _float32Values(audio);
  final cfg = switch (kind) {
    CosyPromptMelKind.matcha80 => MelConfig.matcha80,
    CosyPromptMelKind.whisper128 => MelConfig.whisper128,
  };
  final mel = computeMelSpectrogram(audioValues, cfg);
  final out = NativeTensorBuffer.float32([mel.numMels, mel.nFrames]);
  try {
    out.asFloat32List().setAll(0, mel.data);
    return CosyPromptFeatureBuffer._(
      data: out,
      frames: mel.nFrames,
      bins: mel.numMels,
    );
  } catch (_) {
    out.close();
    rethrow;
  }
}

CosyPromptFeatureBuffer cosyPromptKaldiFbankBuffer(
  Object audio, {
  CosyPromptNativePlan? plan,
}) {
  final fb = kaldi.computeKaldiFbank(
    _float32Values(audio),
    const kaldi.KaldiFbankConfig(),
  );
  final out = NativeTensorBuffer.float32([fb.nFrames, fb.numMelBins]);
  try {
    out.asFloat32List().setAll(0, fb.data);
    return CosyPromptFeatureBuffer._(
      data: out,
      frames: fb.nFrames,
      bins: fb.numMelBins,
    );
  } catch (_) {
    out.close();
    rethrow;
  }
}

void cosyPromptCepstralMeanNormalizeInPlace(
  NativeTensorBuffer feat, {
  required int frames,
  required int bins,
}) {
  if (feat.dtype != RuntimeTensorDataType.float32) {
    throw StateError('Expected float32 fbank, got ${feat.dtype.name}.');
  }
  if (frames < 0) {
    throw RangeError.value(frames, 'frames', 'must be non-negative');
  }
  if (bins <= 0) {
    throw RangeError.value(bins, 'bins', 'must be positive');
  }
  final expectedLength = frames * bins;
  final values = feat.asFloat32List();
  if (values.length != expectedLength) {
    throw StateError(
      'CMN input length is ${values.length}, expected $expectedLength.',
    );
  }
  kaldi.cepstralMeanNormalize(values, frames, bins);
}

int cosyPromptMelFrameCount(int audioLength, CosyPromptMelKind kind) {
  final nFft = switch (kind) {
    CosyPromptMelKind.matcha80 => 1920,
    CosyPromptMelKind.whisper128 => 400,
  };
  final hop = switch (kind) {
    CosyPromptMelKind.matcha80 => 480,
    CosyPromptMelKind.whisper128 => 160,
  };
  final center = kind == CosyPromptMelKind.whisper128;
  final dropLast = kind == CosyPromptMelKind.whisper128;
  final pad = center ? nFft ~/ 2 : (nFft - hop) ~/ 2;
  if (audioLength <= pad) {
    throw ArgumentError(
      'reflect-pad of $pad requires audio length > $pad '
      '(got $audioLength)',
    );
  }
  final paddedLength = audioLength + 2 * pad;
  final rawFrames = ((paddedLength - nFft) ~/ hop) + 1;
  if (rawFrames <= 0) {
    throw ArgumentError('audio too short for CosyVoice2 mel kind $kind');
  }
  if (!dropLast) {
    return rawFrames;
  }
  final frames = rawFrames - 1;
  if (frames < 1) {
    throw ArgumentError('drop-last-frame leaves no frames');
  }
  return frames;
}

int cosyPromptKaldiFbankFrameCount(int audioLength) {
  const frameLength = 400;
  const frameShift = 160;
  if (audioLength < frameLength) {
    return 0;
  }
  return ((audioLength - frameLength) ~/ frameShift) + 1;
}

Float32List _float32Values(Object source) {
  return withNativeFloat32Source(source, (pointer, length) {
    if (length == 0 || pointer == ffi.nullptr) return Float32List(0);
    return Float32List.fromList(pointer.asTypedList(length));
  });
}
