import 'package:flutter/material.dart';

import 'package:dart_mlx_ffi/dart_mlx_ffi.dart';

void main() {
  runApp(const DemoApp());
}

class DemoApp extends StatelessWidget {
  const DemoApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(home: DemoScreen());
  }
}

class DemoScreen extends StatefulWidget {
  const DemoScreen({super.key});

  @override
  State<DemoScreen> createState() => _DemoScreenState();
}

class _DemoScreenState extends State<DemoScreen> {
  late final Future<_DemoState> _demo = _loadDemo();

  Future<_DemoState> _loadDemo() async {
    final version = MlxVersion.current();
    final device = MlxDevice.defaultDevice();
    final info = device.info;
    final stream = MlxStream.defaultFor(device);
    final distributedAvailable = mx.distributed.isAvailable();
    var distributed = 'unavailable';
    if (distributedAvailable) {
      MlxDistributedGroup? group;
      try {
        group = mx.distributed.init(strict: false);
        distributed = 'rank=${group.rank} size=${group.size}';
      } on MlxException catch (error) {
        distributed = error.toString();
      } finally {
        group?.close();
      }
    }
    final a = MlxArray.fromFloat32List([1, 2, 3, 4], shape: [2, 2]);
    final b = MlxArray.fromFloat32List([5, 6, 7, 8], shape: [2, 2]);
    final zeros = MlxArray.zeros([2, 2]);
    final range = MlxArray.arange(0, 4, 1).reshape([2, 2]);
    final added = MlxOps.add(a, b);
    final multiplied = MlxOps.matmul(a, b);
    final reduced = MlxOps.sum(multiplied);
    final random = MlxRandom.uniform([2, 2]);
    final features = MlxArray.fromFloat32List([0.2, -0.1, 0.4], shape: [1, 3]);
    final weights = MlxArray.fromFloat32List(
      [0.5, -0.2, 0.3, -0.4, 0.8, 0.1, 0.2, 0.2, 0.6],
      shape: [3, 3],
    );
    final bias = MlxArray.fromFloat32List([0.1, -0.1, 0.05], shape: [1, 3]);
    final logits = mx.add(mx.matmul(features, weights), bias);
    final probs = mx.softmax(logits, axis: 1);
    final topk = mx.topK(probs, 2, axis: 1);
    final benchmark = _runBench();
    final weightsBytes = mx.io.saveBytes(weights);
    final reloadedWeights = mx.io.loadBytes(weightsBytes, stream: stream);
    final byteReader = MlxBytesReader(weightsBytes);
    final rereadWeights = mx.io.loadReader(byteReader, stream: stream);

    try {
      return _DemoState(
        version: version,
        device: device.toString(),
        deviceInfo: info.values.toString(),
        stream: stream.toString(),
        distributed: distributed,
        zeros: zeros.toString(),
        range: range.toString(),
        added: added.toString(),
        multiplied: multiplied.toString(),
        reduced: reduced.toString(),
        random: random.toString(),
        modelLogits: logits.toString(),
        modelProbabilities: probs.toString(),
        modelTopK: topk.toString(),
        benchmark: benchmark,
        bytesIo:
            'bytes=${weightsBytes.length}, reload=${reloadedWeights.toString()}',
        readerIo: rereadWeights.toString(),
      );
    } finally {
      rereadWeights.close();
      byteReader.close();
      reloadedWeights.close();
      topk.close();
      probs.close();
      logits.close();
      bias.close();
      weights.close();
      features.close();
      random.close();
      reduced.close();
      multiplied.close();
      added.close();
      range.close();
      zeros.close();
      stream.close();
      b.close();
      a.close();
      device.close();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('dart_mlx_ffi')),
      body: FutureBuilder<_DemoState>(
        future: _demo,
        builder: (context, snapshot) {
          if (snapshot.hasError) {
            return Padding(
              padding: const EdgeInsets.all(24),
              child: Text(snapshot.error.toString()),
            );
          }
          if (!snapshot.hasData) {
            return const Center(child: CircularProgressIndicator());
          }
          final data = snapshot.data!;
          return ListView(
            padding: const EdgeInsets.all(24),
            children: [
              _Section(title: 'MLX Version', body: data.version),
              _Section(title: 'Default Device', body: data.device),
              _Section(title: 'Device Info', body: data.deviceInfo),
              _Section(title: 'Default Stream', body: data.stream),
              _Section(title: 'Distributed', body: data.distributed),
              _Section(title: 'Zeros', body: data.zeros),
              _Section(title: 'Arange + Reshape', body: data.range),
              _Section(title: 'A + B', body: data.added),
              _Section(title: 'A @ B', body: data.multiplied),
              _Section(title: 'sum(A @ B)', body: data.reduced),
              _Section(title: 'Uniform Random', body: data.random),
              _Section(title: 'Tiny Model Logits', body: data.modelLogits),
              _Section(
                title: 'Tiny Model Probabilities',
                body: data.modelProbabilities,
              ),
              _Section(title: 'Tiny Model TopK', body: data.modelTopK),
              _Section(title: 'Tiny Benchmark', body: data.benchmark),
              _Section(title: 'Bytes IO', body: data.bytesIo),
              _Section(title: 'Reader IO', body: data.readerIo),
            ],
          );
        },
      ),
    );
  }
}

String _runBench() {
  final features = MlxArray.fromFloat32List(
    [0.2, -0.1, 0.4, 0.8, 0.3, -0.5, 0.7, 0.1, -0.2, 0.9, -0.4, 0.6],
    shape: [4, 3],
  );
  final weights = MlxArray.fromFloat32List(
    [0.5, -0.2, 0.3, -0.4, 0.8, 0.1, 0.2, 0.2, 0.6],
    shape: [3, 3],
  );
  final bias = MlxArray.fromFloat32List([0.1, -0.1, 0.05], shape: [1, 3]);

  try {
    for (var i = 0; i < 5; i++) {
      final topk = _runStep(features, weights, bias);
      topk.close();
    }
    MlxMemory.resetPeak();
    final beforePeak = MlxMemory.peakBytes();
    final stopwatch = Stopwatch()..start();
    MlxArray? lastTopk;
    try {
      for (var i = 0; i < 20; i++) {
        lastTopk?.close();
        lastTopk = _runStep(features, weights, bias);
      }
      stopwatch.stop();
      final totalMs = stopwatch.elapsedMicroseconds / 1000.0;
      final afterPeak = MlxMemory.peakBytes();
      return '20 iters, ${(totalMs / 20).toStringAsFixed(3)} ms/iter, '
          'peak delta=${afterPeak - beforePeak} bytes, last=${lastTopk.toString()}';
    } finally {
      lastTopk?.close();
    }
  } finally {
    bias.close();
    weights.close();
    features.close();
  }
}

MlxArray _runStep(MlxArray features, MlxArray weights, MlxArray bias) {
  final logits = mx.add(mx.matmul(features, weights), bias);
  final probs = mx.softmax(logits, axis: 1);
  final topk = mx.topK(probs, 2, axis: 1);
  MlxRuntime.evalAll([topk]);
  probs.close();
  logits.close();
  return topk;
}

class _Section extends StatelessWidget {
  const _Section({required this.title, required this.body});

  final String title;
  final String body;

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.only(bottom: 16),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(title, style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            Text(body),
          ],
        ),
      ),
    );
  }
}

class _DemoState {
  const _DemoState({
    required this.version,
    required this.device,
    required this.deviceInfo,
    required this.stream,
    required this.distributed,
    required this.zeros,
    required this.range,
    required this.added,
    required this.multiplied,
    required this.reduced,
    required this.random,
    required this.modelLogits,
    required this.modelProbabilities,
    required this.modelTopK,
    required this.benchmark,
    required this.bytesIo,
    required this.readerIo,
  });

  final String version;
  final String device;
  final String deviceInfo;
  final String stream;
  final String distributed;
  final String zeros;
  final String range;
  final String added;
  final String multiplied;
  final String reduced;
  final String random;
  final String modelLogits;
  final String modelProbabilities;
  final String modelTopK;
  final String benchmark;
  final String bytesIo;
  final String readerIo;
}
