library;

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

import 'config.dart';
import 'core.dart';
import 'vocoder_core.dart';

final class Generator {
  Generator({
    required this.config,
    required this.mSource,
    required this.noiseConvs,
    required this.noiseRes,
    required this.ups,
    required this.resblocks,
    required this.convPost,
  }) : reflectionPad = const ReflectionPad1d(left: 1, right: 0);

  factory Generator.load(
    Map<String, MlxArray> tensors, {
    required String prefix,
    required ModelConfig config,
    bool activationQuant = false,
  }) {
    final istft = config.istftnetConfig;
    final noiseConvs = <KittenConv1dLayer>[];
    final noiseRes = <AdaINResBlock1>[];
    final ups = <ConvWeighted>[];
    final resblocks = <AdaINResBlock1>[];
    for (var i = 0; i < istft.upsampleRates.length; i++) {
      final u = istft.upsampleRates[i];
      final k = istft.upsampleKernelSizes[i];
      ups.add(
        ConvWeighted.load(
          tensors,
          '$prefix.ups.$i',
          stride: u,
          padding: (k - u) ~/ 2,
          activationQuant: activationQuant,
        ),
      );
      final channel = istft.upsampleInitialChannel ~/ (1 << (i + 1));
      for (var j = 0; j < istft.resblockKernelSizes.length; j++) {
        resblocks.add(
          AdaINResBlock1.load(
            tensors,
            '$prefix.resblocks.${i * istft.resblockKernelSizes.length + j}',
            channels: channel,
            quant: config.quantization,
            kernelSize: istft.resblockKernelSizes[j],
            dilation: istft.resblockDilationSizes[j],
            activationQuant: activationQuant,
          ),
        );
      }
      if (i + 1 < istft.upsampleRates.length) {
        final strideF0 = istft.upsampleRates
            .sublist(i + 1)
            .fold(1, (a, b) => a * b);
        noiseConvs.add(
          KittenConv1dLayer.load(
            tensors,
            '$prefix.noise_convs.$i',
            stride: strideF0,
            padding: (strideF0 + 1) ~/ 2,
            activationQuant: activationQuant,
          ),
        );
        noiseRes.add(
          AdaINResBlock1.load(
            tensors,
            '$prefix.noise_res.$i',
            channels: channel,
            quant: config.quantization,
            kernelSize: 7,
            dilation: const [1, 3, 5],
            activationQuant: activationQuant,
          ),
        );
      } else {
        noiseConvs.add(
          KittenConv1dLayer.load(
            tensors,
            '$prefix.noise_convs.$i',
            activationQuant: activationQuant,
          ),
        );
        noiseRes.add(
          AdaINResBlock1.load(
            tensors,
            '$prefix.noise_res.$i',
            channels: channel,
            quant: config.quantization,
            kernelSize: 11,
            dilation: const [1, 3, 5],
            activationQuant: activationQuant,
          ),
        );
      }
    }
    return Generator(
      config: config,
      mSource: SourceModuleHnNSF.load(
        tensors,
        '$prefix.m_source',
        config: config,
        activationQuant: activationQuant,
      ),
      noiseConvs: noiseConvs,
      noiseRes: noiseRes,
      ups: ups,
      resblocks: resblocks,
      convPost: ConvWeighted.load(
        tensors,
        '$prefix.conv_post',
        padding: 3,
        activationQuant: activationQuant,
      ),
    );
  }

  final ModelConfig config;
  final SourceModuleHnNSF mSource;
  final List<KittenConv1dLayer> noiseConvs;
  final List<AdaINResBlock1> noiseRes;
  final List<ConvWeighted> ups;
  final List<AdaINResBlock1> resblocks;
  final ConvWeighted convPost;
  final ReflectionPad1d reflectionPad;

  int get numKernels => config.istftnetConfig.resblockKernelSizes.length;
  int get numUpsamples => config.istftnetConfig.upsampleRates.length;
  int get f0UpsampleScale =>
      config.istftnetConfig.upsampleRates.fold(1, (a, b) => a * b) *
      config.istftnetConfig.genIstftHopSize;

  MlxArray upsampleF0(MlxArray f0) {
    final shaped =
        f0.ndim == 2 ? mx.transposeAxes(f0.expandDims(1), [0, 2, 1]) : f0;
    final upsampled = shaped.repeat(f0UpsampleScale, axis: 1);
    if (f0.ndim == 2) {
      shaped.close();
    }
    return upsampled;
  }

  KittenSourceOutput sourceFromF0(MlxArray f0) {
    final upsampled = upsampleF0(f0);
    try {
      return mSource(upsampled);
    } finally {
      upsampled.close();
    }
  }

  KittenGeneratorProjection forwardWithHar(
    MlxArray input,
    MlxArray style,
    MlxArray har,
  ) {
    var current = input;
    for (var i = 0; i < numUpsamples; i++) {
      final activated = leakyRelu(current, negativeSlope: 0.1);
      if (!identical(current, input)) {
        current.close();
      }

      final xSource0 = noiseConvs[i](har);
      final xSource1 = mx.transposeAxes(xSource0, [0, 2, 1]);
      final xSource = noiseRes[i](xSource1, style);
      xSource0.close();
      xSource1.close();

      final upIn = mx.transposeAxes(activated, [0, 2, 1]);
      final up0 = ups[i](upIn, transpose: true);
      final up1 = mx.transposeAxes(up0, [0, 2, 1]);
      activated.close();
      upIn.close();
      up0.close();

      final mergedBase =
          i == numUpsamples - 1 && up1.shape[2] < xSource.shape[2]
              ? reflectionPad(up1)
              : up1;
      if (!identical(mergedBase, up1)) {
        up1.close();
      }
      final merged = mergedBase + xSource;
      mergedBase.close();
      xSource.close();

      MlxArray? acc;
      for (var j = 0; j < numKernels; j++) {
        final branch = resblocks[i * numKernels + j](merged, style);
        if (acc == null) {
          acc = branch;
        } else {
          final sum = acc + branch;
          acc.close();
          branch.close();
          acc = sum;
        }
      }
      merged.close();
      final scale = scalar(1.0 / numKernels, dtype: acc!.dtype);
      current = acc * scale;
      scale.close();
      acc.close();
    }

    final postAct = leakyRelu(current, negativeSlope: 0.01);
    current.close();
    final postIn = mx.transposeAxes(postAct, [0, 2, 1]);
    final projected0 = convPost(postIn);
    final projected = mx.transposeAxes(projected0, [0, 2, 1]);
    postAct.close();
    postIn.close();
    projected0.close();

    final nfft = config.istftnetConfig.genIstftNfft;
    final specSlice = projected.slice(
      start: [0, 0, 0],
      stop: [projected.shape[0], (nfft ~/ 2) + 1, projected.shape[2]],
    );
    final phaseSlice = projected.slice(
      start: [0, (nfft ~/ 2) + 1, 0],
      stop: [projected.shape[0], projected.shape[1], projected.shape[2]],
    );
    final spec = mx.exp(specSlice);
    final phase = mx.sin(phaseSlice);
    projected.close();
    specSlice.close();
    phaseSlice.close();
    return KittenGeneratorProjection(spec: spec, phase: phase);
  }
}

final class KittenDecoder {
  const KittenDecoder({
    required this.config,
    required this.encode,
    required this.decode,
    required this.f0Conv,
    required this.nConv,
    required this.asrRes,
    required this.generator,
  });

  factory KittenDecoder.load(
    Map<String, MlxArray> tensors, {
    String prefix = 'decoder',
    required ModelConfig config,
    bool activationQuant = false,
  }) {
    return KittenDecoder(
      config: config,
      encode: AdainResBlk1d.load(
        tensors,
        '$prefix.encode',
        dimIn: config.hiddenDim + 2,
        dimOut: config.maxConvDim,
        styleDim: config.styleDim,
        quant: config.quantization,
        activationQuant: activationQuant,
      ),
      decode: List<AdainResBlk1d>.generate(
        4,
        (index) => AdainResBlk1d.load(
          tensors,
          '$prefix.decode.$index',
          dimIn: config.maxConvDim + 2 + config.asrResDim,
          dimOut:
              index == 3 ? config.resolvedDecoderOutDim : config.maxConvDim,
          styleDim: config.styleDim,
          upsample: index == 3,
          quant: config.quantization,
          activationQuant: activationQuant,
        ),
        growable: false,
      ),
      f0Conv: ConvWeighted.load(
        tensors,
        '$prefix.F0_conv',
        stride: 2,
        padding: 1,
        groups: 1,
        activationQuant: activationQuant,
      ),
      nConv: ConvWeighted.load(
        tensors,
        '$prefix.N_conv',
        stride: 2,
        padding: 1,
        groups: 1,
        activationQuant: activationQuant,
      ),
      asrRes: ConvWeighted.load(
        tensors,
        '$prefix.asr_res.0',
        padding: 0,
        activationQuant: activationQuant,
      ),
      generator: Generator.load(
        tensors,
        prefix: '$prefix.generator',
        config: config,
        activationQuant: activationQuant,
      ),
    );
  }

  final ModelConfig config;
  final AdainResBlk1d encode;
  final List<AdainResBlk1d> decode;
  final ConvWeighted f0Conv;
  final ConvWeighted nConv;
  final ConvWeighted asrRes;
  final Generator generator;

  KittenGeneratorProjection forwardProjection({
    required MlxArray asr,
    required MlxArray f0Curve,
    required MlxArray noise,
    required MlxArray style,
    required MlxArray har,
  }) {
    final style2d =
        style.ndim == 1 ? style.reshape([1, style.shape[0]]) : style;
    final f0In = mx.transposeAxes(f0Curve.expandDims(1), [0, 2, 1]);
    final nIn = mx.transposeAxes(noise.expandDims(1), [0, 2, 1]);
    final f0Down0 = f0Conv(f0In);
    final f0Down = mx.transposeAxes(f0Down0, [0, 2, 1]);
    final nDown0 = nConv(nIn);
    final nDown = mx.transposeAxes(nDown0, [0, 2, 1]);
    f0In.close();
    nIn.close();
    f0Down0.close();
    nDown0.close();

    final merged0 = mx.concatenate([asr, f0Down, nDown], axis: 1);
    var hidden = encode(merged0, style2d);
    merged0.close();
    final asrIn = mx.transposeAxes(asr, [0, 2, 1]);
    final asrRes0 = asrRes(asrIn);
    final asrSkip = mx.transposeAxes(asrRes0, [0, 2, 1]);
    asrIn.close();
    asrRes0.close();
    var useResidual = true;
    for (final block in decode) {
      if (useResidual) {
        final cat = mx.concatenate([hidden, asrSkip, f0Down, nDown], axis: 1);
        hidden.close();
        hidden = cat;
      }
      final next = block(hidden, style2d);
      hidden.close();
      hidden = next;
      if (block.upsample) {
        useResidual = false;
      }
    }
    asrSkip.close();
    f0Down.close();
    nDown.close();
    if (style.ndim == 1) {
      style2d.close();
    }
    final projection = generator.forwardWithHar(hidden, style, har);
    hidden.close();
    return projection;
  }
}
