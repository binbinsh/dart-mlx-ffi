library;

import 'args.dart';

export 'args.dart';

final class KittenIstftNetConfig {
  const KittenIstftNetConfig({
    required this.resblockKernelSizes,
    required this.upsampleRates,
    required this.upsampleInitialChannel,
    required this.resblockDilationSizes,
    required this.upsampleKernelSizes,
    required this.genIstftNfft,
    required this.genIstftHopSize,
  });

  factory KittenIstftNetConfig.fromJson(Map<String, Object?> json) {
    return KittenIstftNetConfig(
      resblockKernelSizes:
          (json['resblock_kernel_sizes'] as List<Object?>)
              .map((e) => (e as num).toInt())
              .toList(growable: false),
      upsampleRates:
          (json['upsample_rates'] as List<Object?>)
              .map((e) => (e as num).toInt())
              .toList(growable: false),
      upsampleInitialChannel:
          (json['upsample_initial_channel'] as num).toInt(),
      resblockDilationSizes:
          (json['resblock_dilation_sizes'] as List<Object?>)
              .map(
                (row) =>
                    (row as List<Object?>)
                        .map((e) => (e as num).toInt())
                        .toList(growable: false),
              )
              .toList(growable: false),
      upsampleKernelSizes:
          (json['upsample_kernel_sizes'] as List<Object?>)
              .map((e) => (e as num).toInt())
              .toList(growable: false),
      genIstftNfft: (json['gen_istft_n_fft'] as num).toInt(),
      genIstftHopSize: (json['gen_istft_hop_size'] as num).toInt(),
    );
  }

  final List<int> resblockKernelSizes;
  final List<int> upsampleRates;
  final int upsampleInitialChannel;
  final List<List<int>> resblockDilationSizes;
  final List<int> upsampleKernelSizes;
  final int genIstftNfft;
  final int genIstftHopSize;
}

extension ModelConfigDecoderExt on ModelConfig {
  KittenIstftNetConfig get istftnetConfig =>
      KittenIstftNetConfig.fromJson(istftnet);

  int get resolvedDecoderOutDim => decoderOutDim ?? maxConvDim;
}
