import 'dart:convert';
import 'dart:ffi';
import 'dart:math';
import 'dart:typed_data';

import '../../runtime/native_float32_source.dart';
import '../../runtime/runtime.dart' show RuntimeTensor, RuntimeTensorDataType;

final class PcmAudio {
  const PcmAudio({required this.samples, required this.sampleRate});

  final Float32List samples;
  final int sampleRate;
}

PcmAudio decodeWav(Uint8List bytes) {
  final data = ByteData.sublistView(bytes);
  if (bytes.length < 44 ||
      _ascii(bytes, 0, 4) != 'RIFF' ||
      _ascii(bytes, 8, 4) != 'WAVE') {
    throw const FormatException('expected RIFF/WAVE audio');
  }

  int? format;
  int? channels;
  int? sampleRate;
  int? bitsPerSample;
  int? dataOffset;
  int? dataLength;
  var offset = 12;
  while (offset + 8 <= bytes.length) {
    final id = _ascii(bytes, offset, 4);
    final size = data.getUint32(offset + 4, Endian.little);
    final body = offset + 8;
    if (body + size > bytes.length) {
      throw const FormatException('truncated WAV chunk');
    }
    if (id == 'fmt ') {
      if (size < 16) {
        throw const FormatException('invalid WAV fmt chunk');
      }
      format = data.getUint16(body, Endian.little);
      channels = data.getUint16(body + 2, Endian.little);
      sampleRate = data.getUint32(body + 4, Endian.little);
      bitsPerSample = data.getUint16(body + 14, Endian.little);
    } else if (id == 'data') {
      dataOffset = body;
      dataLength = size;
    }
    offset = body + size + (size.isOdd ? 1 : 0);
  }

  if (format == null ||
      channels == null ||
      sampleRate == null ||
      bitsPerSample == null ||
      dataOffset == null ||
      dataLength == null) {
    throw const FormatException('WAV is missing fmt or data chunk');
  }
  if (channels < 1) {
    throw const FormatException('WAV channel count must be positive');
  }

  final bytesPerSample = bitsPerSample ~/ 8;
  if (bytesPerSample * 8 != bitsPerSample || bytesPerSample == 0) {
    throw FormatException('unsupported WAV sample depth: $bitsPerSample');
  }
  final frameCount = dataLength ~/ (bytesPerSample * channels);
  final out = Float32List(frameCount);
  var p = dataOffset;
  for (var frame = 0; frame < frameCount; frame += 1) {
    var mixed = 0.0;
    for (var ch = 0; ch < channels; ch += 1) {
      mixed += _readSample(data, p, format, bitsPerSample);
      p += bytesPerSample;
    }
    out[frame] = (mixed / channels).clamp(-1.0, 1.0).toDouble();
  }
  return PcmAudio(samples: out, sampleRate: sampleRate);
}

Uint8List encodeWavPcm16(Float32List samples, {required int sampleRate}) {
  return encodeWavPcm16Source(samples, sampleRate: sampleRate);
}

Uint8List encodeWavPcm16Tensor(
  RuntimeTensor tensor, {
  required int sampleRate,
}) {
  return encodeWavPcm16Source(tensor, sampleRate: sampleRate);
}

Uint8List encodeWavPcm16Source(
  Object samples, {
  required int sampleRate,
  int? sampleCount,
}) {
  final available = nativeFloat32SourceLength(samples);
  final count = sampleCount ?? available;
  if (count < 0 || count > available) {
    throw RangeError.range(count, 0, available, 'sampleCount');
  }
  if (samples is RuntimeTensor) {
    _checkFloat32Tensor(samples);
  }
  return withNativeFloat32Source(samples, (pointer, length) {
    if (count > length) {
      throw RangeError.range(count, 0, length, 'sampleCount');
    }
    if (count == 0) {
      return _encodeWavPcm16(Float32List(0), sampleRate);
    }
    return _encodeWavPcm16(
      Float32List.fromList(pointer.asTypedList(count)),
      sampleRate,
    );
  });
}

Uint8List encodeWavPcm16Sources(
  List<Object> chunks, {
  required int sampleRate,
}) {
  return _encodeWavPcm16(concatFloat32Sources(chunks), sampleRate);
}

Float32List concatFloat32Sources(List<Object> chunks) {
  if (chunks.isEmpty) {
    return Float32List(0);
  }
  final copied = [for (final chunk in chunks) _copyFloat32(chunk)];
  final total = copied.fold<int>(0, (sum, chunk) => sum + chunk.length);
  final out = Float32List(total);
  var offset = 0;
  for (final chunk in copied) {
    out.setAll(offset, chunk);
    offset += chunk.length;
  }
  return out;
}

Float32List copyFloat32Prefix(Object samples, int sampleCount) {
  final available = nativeFloat32SourceLength(samples);
  if (sampleCount < 0 || sampleCount > available) {
    throw RangeError.range(sampleCount, 0, available, 'sampleCount');
  }
  if (sampleCount == 0) {
    return Float32List(0);
  }
  if (samples is Float32List) {
    return Float32List.fromList(
      Float32List.sublistView(samples, 0, sampleCount),
    );
  }
  if (samples is RuntimeTensor) {
    _checkFloat32Tensor(samples);
    return Float32List.fromList(
      Float32List.sublistView(samples.asFloat32List(), 0, sampleCount),
    );
  }
  return withNativeFloat32Source(samples, (pointer, length) {
    if (sampleCount > length) {
      throw RangeError.range(sampleCount, 0, length, 'sampleCount');
    }
    return Float32List.fromList(pointer.asTypedList(sampleCount));
  });
}

int audioFloat32SampleCount(Object samples) {
  return nativeFloat32SourceLength(samples);
}

void _checkFloat32Tensor(RuntimeTensor tensor) {
  if (tensor.dtype != RuntimeTensorDataType.float32) {
    throw StateError(
      'audio tensor has dtype ${tensor.dtype.name}; expected float32.',
    );
  }
}

Uint8List encodeWavPcm16Pointer(
  dynamic samples, {
  required int sampleCount,
  required int sampleRate,
}) {
  if (sampleCount < 0) {
    throw RangeError.range(sampleCount, 0, null, 'sampleCount');
  }
  if (sampleCount == 0) return _encodeWavPcm16(Float32List(0), sampleRate);
  final values = Float32List.fromList(samples.asTypedList(sampleCount));
  return _encodeWavPcm16(values, sampleRate);
}

Uint8List? decodeAudioDataUrl(Object? value) {
  if (value == null) {
    return null;
  }
  final text = value.toString();
  final comma = text.indexOf(',');
  final payload = comma >= 0 ? text.substring(comma + 1) : text;
  if (payload.isEmpty) {
    return null;
  }
  return base64Decode(payload);
}

String _ascii(Uint8List bytes, int offset, int length) =>
    ascii.decode(bytes.sublist(offset, offset + length));

Float32List _copyFloat32(Object source) {
  return withNativeFloat32Source(source, (pointer, length) {
    if (length == 0) return Float32List(0);
    return Float32List.fromList(pointer.asTypedList(length));
  });
}

Uint8List _encodeWavPcm16(Float32List samples, int sampleRate) {
  final dataLength = samples.length * 2;
  final out = Uint8List(44 + dataLength);
  final data = ByteData.sublistView(out);
  void text(int offset, String value) {
    out.setAll(offset, ascii.encode(value));
  }

  text(0, 'RIFF');
  data.setUint32(4, 36 + dataLength, Endian.little);
  text(8, 'WAVE');
  text(12, 'fmt ');
  data.setUint32(16, 16, Endian.little);
  data.setUint16(20, 1, Endian.little);
  data.setUint16(22, 1, Endian.little);
  data.setUint32(24, sampleRate, Endian.little);
  data.setUint32(28, sampleRate * 2, Endian.little);
  data.setUint16(32, 2, Endian.little);
  data.setUint16(34, 16, Endian.little);
  text(36, 'data');
  data.setUint32(40, dataLength, Endian.little);
  var offset = 44;
  for (final sample in samples) {
    final finite = sample.isFinite ? sample : 0.0;
    final pcm = (finite.clamp(-1.0, 1.0) * 32767.0).round();
    data.setInt16(offset, pcm, Endian.little);
    offset += 2;
  }
  return out;
}

double _readSample(ByteData data, int offset, int format, int bitsPerSample) {
  if (format == 1) {
    switch (bitsPerSample) {
      case 8:
        return (data.getUint8(offset) - 128) / 128.0;
      case 16:
        return data.getInt16(offset, Endian.little) / 32768.0;
      case 24:
        var value =
            data.getUint8(offset) |
            (data.getUint8(offset + 1) << 8) |
            (data.getUint8(offset + 2) << 16);
        if ((value & 0x800000) != 0) {
          value |= ~0xffffff;
        }
        return value / 8388608.0;
      case 32:
        return data.getInt32(offset, Endian.little) / 2147483648.0;
    }
  }
  if (format == 3 && bitsPerSample == 32) {
    final value = data.getFloat32(offset, Endian.little);
    if (value.isNaN || value.isInfinite) {
      return 0.0;
    }
    return max(-1.0, min(1.0, value));
  }
  throw FormatException(
    'unsupported WAV format=$format bitsPerSample=$bitsPerSample',
  );
}
