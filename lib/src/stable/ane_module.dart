part of '../stable_api.dart';

/// Private Apple Neural Engine bridge helpers.
abstract final class MlxAnePrivate {
  /// Whether the current binary was compiled with the private ANE bridge.
  static bool isCompiled() => shim.dart_mlx_ane_private_is_compiled();

  /// Whether the private ANE bridge is active in the current binary.
  static bool isEnabled() =>
      _aneRuntimeEnabledOverride() && shim.dart_mlx_ane_private_is_enabled();

  /// Returns a capability probe for the private ANE runtime bridge.
  static MlxAnePrivateInfo probe() {
    _clearAnePrivateError();
    final json = _copyOwnedString(shim.dart_mlx_ane_private_probe_json_copy());
    final values = Map<String, Object?>.from(jsonDecode(json) as Map);
    values['enabled'] =
        values['enabled'] == true && _aneRuntimeEnabledOverride();
    return MlxAnePrivateInfo._(values);
  }

  /// Creates a private ANE in-memory model from MIL text.
  static MlxAnePrivateModel modelFromMil(
    String milText, {
    List<MlxAneWeight> weights = const [],
  }) {
    _ensureAneRuntimeEnabled();
    return MlxAnePrivateModel.fromMil(milText, weights: weights);
  }

  /// Creates a private ANE in-memory model from MIL text with weight offsets.
  static MlxAnePrivateModel modelFromMilWithOffsets(
    String milText, {
    List<MlxAneWeightWithOffset> weights = const [],
  }) {
    _ensureAneRuntimeEnabled();
    return MlxAnePrivateModel.fromMilWithOffsets(milText, weights: weights);
  }

  /// Encodes fp32 values into fp16 byte storage suitable for ANE buffers.
  static Uint8List encodeFp16Bytes(Float32List values) {
    _ensureAneRuntimeEnabled();
    return _encodeFp16Bytes(values);
  }

  /// Decodes fp16 bytes from ANE buffers into fp32 values.
  static Float32List decodeFp16Bytes(Uint8List bytes) {
    _ensureAneRuntimeEnabled();
    return _decodeFp16Bytes(bytes);
  }

  /// Encodes float32 values as raw little-endian bytes without fp16 conversion.
  static Uint8List encodeRawFloat32Bytes(Float32List values) {
    _ensureAneRuntimeEnabled();
    return _encodeRawFloat32Bytes(values);
  }

  /// Decodes raw little-endian float32 bytes without fp16 conversion.
  static Float32List decodeRawFloat32Bytes(Uint8List bytes) {
    _ensureAneRuntimeEnabled();
    return _decodeRawFloat32Bytes(bytes);
  }
}

/// Module-style private ANE namespace.
final class MlxAnePrivateModule {
  const MlxAnePrivateModule._();

  /// Whether the current binary was compiled with the private ANE bridge.
  bool isCompiled() => MlxAnePrivate.isCompiled();

  /// Whether the private ANE bridge is active in the current binary.
  bool isEnabled() => MlxAnePrivate.isEnabled();

  /// Returns a capability probe for the private ANE runtime bridge.
  MlxAnePrivateInfo probe() => MlxAnePrivate.probe();

  /// Creates a private ANE in-memory model from MIL text.
  MlxAnePrivateModel modelFromMil(
    String milText, {
    List<MlxAneWeight> weights = const [],
  }) => MlxAnePrivate.modelFromMil(milText, weights: weights);

  /// Creates a private ANE in-memory model from MIL text with weight offsets.
  MlxAnePrivateModel modelFromMilWithOffsets(
    String milText, {
    List<MlxAneWeightWithOffset> weights = const [],
  }) => MlxAnePrivate.modelFromMilWithOffsets(milText, weights: weights);

  /// Encodes fp32 values into fp16 byte storage suitable for ANE buffers.
  Uint8List encodeFp16Bytes(Float32List values) =>
      MlxAnePrivate.encodeFp16Bytes(values);

  /// Decodes fp16 bytes from ANE buffers into fp32 values.
  Float32List decodeFp16Bytes(Uint8List bytes) =>
      MlxAnePrivate.decodeFp16Bytes(bytes);

  /// Encodes float32 values as raw little-endian bytes without fp16 conversion.
  Uint8List encodeRawFloat32Bytes(Float32List values) =>
      MlxAnePrivate.encodeRawFloat32Bytes(values);

  /// Decodes raw little-endian float32 bytes without fp16 conversion.
  Float32List decodeRawFloat32Bytes(Uint8List bytes) =>
      MlxAnePrivate.decodeRawFloat32Bytes(bytes);
}
