part of 'qwen3_5.dart';

MlxMetalKernel? _gatedDeltaKernel;
bool _gatedDeltaKernelTried = false;
MlxFunction? _swiGluCompiled;
MlxFunction? _computeGCompiled;
MlxFunction? _gatedDeltaStepCompiled;
bool _gatedDeltaKernelWarmed = false;
final Map<String, MlxMetalConfig> _gatedDeltaKernelConfigCache = {};
final Map<int, MlxArray> _gatedDeltaKernelTCache = {};

String _gatedDeltaKernelMode() =>
    Platform.environment['QWEN35_USE_GDELTA_KERNEL'] ?? '1';

int _gatedDeltaGridX() =>
    int.tryParse(Platform.environment['QWEN35_GDELTA_GRID_X'] ?? '') ?? 32;

int _gatedDeltaThreadgroupY() =>
    int.tryParse(Platform.environment['QWEN35_GDELTA_TG_Y'] ?? '') ?? 4;

bool _supportsGatedDeltaKernel({
  required int numValueHeads,
  required int numKeyHeads,
  required int keyHeadDim,
}) =>
    numKeyHeads > 0 &&
    numValueHeads > 0 &&
    numValueHeads % numKeyHeads == 0 &&
    keyHeadDim >= 32 &&
    keyHeadDim % 32 == 0;

MlxFunction _getSwiGluCompiled() {
  final existing = _swiGluCompiled;
  if (existing != null) {
    return existing;
  }
  final fn = MlxFunction.fromCallback((args) {
    final gate = args[0];
    final x = args[1];
    final sig = gate.sigmoid();
    final out = (gate * sig) * x;
    sig.close();
    return [out];
  });
  final compiled = fn.compile();
  fn.close();
  _swiGluCompiled = compiled;
  return compiled;
}

MlxFunction _getComputeGCompiled() {
  final existing = _computeGCompiled;
  if (existing != null) {
    return existing;
  }
  final fn = MlxFunction.fromCallback((args) {
    final aLog = args[0];
    final a = args[1];
    final dtBias = args[2];
    final expA = aLog.astype(MlxDType.MLX_FLOAT32).exp();
    final dt = a + dtBias;
    final zeros = MlxArray.zeros(dt.shape, dtype: dt.dtype);
    final softplus = mx.logaddexp(dt, zeros);
    zeros.close();
    dt.close();
    final scaled = expA * softplus.astype(MlxDType.MLX_FLOAT32);
    expA.close();
    softplus.close();
    final negative = scaled.negative();
    scaled.close();
    final out = negative.exp().astype(a.dtype);
    negative.close();
    return [out];
  });
  final compiled = fn.compile();
  fn.close();
  _computeGCompiled = compiled;
  return compiled;
}

MlxArray _computeGEager(MlxArray aLog, MlxArray a, MlxArray dtBias) {
  final expA = aLog.astype(MlxDType.MLX_FLOAT32).exp();
  final dt = a + dtBias;
  final zeros = MlxArray.zeros(dt.shape, dtype: dt.dtype);
  final softplus = mx.logaddexp(dt, zeros);
  zeros.close();
  dt.close();
  final scaled = expA * softplus.astype(MlxDType.MLX_FLOAT32);
  expA.close();
  softplus.close();
  final negative = scaled.negative();
  scaled.close();
  final out = negative.exp().astype(a.dtype);
  negative.close();
  return out;
}

void _warmQwen35CompiledHelpers(Qwen3_5Config config) {
  if (Platform.environment['QWEN35_DISABLE_COMPILED_HELPERS'] == '1') {
    return;
  }
  for (final seqLen in const [1, 8, 16, 24, 32]) {
    final computeG = _getComputeGCompiled();
    final swiglu = _getSwiGluCompiled();
    final aLog = MlxArray.zeros([
      1,
      1,
      config.linearNumValueHeads,
    ], dtype: config.computeDType);
    final a = MlxArray.zeros([
      1,
      seqLen,
      config.linearNumValueHeads,
    ], dtype: config.computeDType);
    final dtBias = MlxArray.zeros([
      1,
      1,
      config.linearNumValueHeads,
    ], dtype: config.computeDType);
    final zGate = MlxArray.zeros([
      1,
      seqLen,
      config.linearNumValueHeads,
      config.linearValueHeadDim,
    ], dtype: config.computeDType);
    final zX = MlxArray.zeros([
      1,
      seqLen,
      config.linearNumValueHeads,
      config.linearValueHeadDim,
    ], dtype: config.computeDType);
    final mlpGate = MlxArray.zeros([
      seqLen,
      config.intermediateSize,
    ], dtype: config.computeDType);
    final mlpX = MlxArray.zeros([
      seqLen,
      config.intermediateSize,
    ], dtype: config.computeDType);
    try {
      final gOut = computeG([aLog, a, dtBias]);
      final zOut = swiglu([zGate, zX]);
      final mlpOut = swiglu([mlpGate, mlpX]);
      try {
        MlxRuntime.evalAll([...gOut, ...zOut, ...mlpOut]);
      } finally {
        for (final value in [...gOut, ...zOut, ...mlpOut]) {
          value.close();
        }
      }
    } finally {
      aLog.close();
      a.close();
      dtBias.close();
      zGate.close();
      zX.close();
      mlpGate.close();
      mlpX.close();
    }
  }
}

MlxFunction _getGatedDeltaStepCompiled() {
  final existing = _gatedDeltaStepCompiled;
  if (existing != null) {
    return existing;
  }
  final fn = MlxFunction.fromCallback((args) {
    final q = args[0];
    final k = args[1];
    final v = args[2];
    final g = args[3];
    final beta = args[4];
    final state = args[5];
    final oldState = state;
    final decay = g.reshape([g.shape[0], g.shape[1], 1, 1]);
    final decayed = state * decay;
    decay.close();
    final kvMem = (decayed * k.reshape([k.shape[0], k.shape[1], 1, k.shape[2]]))
        .sum(axis: 3);
    final delta = (v - kvMem) * beta.reshape([beta.shape[0], beta.shape[1], 1]);
    kvMem.close();
    final newState =
        decayed +
        k.reshape([k.shape[0], k.shape[1], 1, k.shape[2]]) *
            delta.reshape([delta.shape[0], delta.shape[1], delta.shape[2], 1]);
    decayed.close();
    delta.close();
    final y = (newState * q.reshape([q.shape[0], q.shape[1], 1, q.shape[2]]))
        .sum(axis: 3);
    oldState.close();
    return [y, newState];
  });
  _gatedDeltaStepCompiled = fn.compile();
  fn.close();
  return _gatedDeltaStepCompiled!;
}

({MlxArray output, MlxArray state}) _runGatedDeltaFallback(
  MlxArray q,
  MlxArray k,
  MlxArray v,
  MlxArray g,
  MlxArray beta,
  MlxArray state,
) => _runGatedDeltaFallbackFixed(
  q.shape[1],
  numValueHeads: v.shape[2],
  numKeyHeads: q.shape[2],
  keyHeadDim: q.shape[3],
  valueHeadDim: v.shape[3],
  q: q,
  k: k,
  v: v,
  g: g,
  beta: beta,
  state: state,
);

({MlxArray output, MlxArray state}) _runGatedDeltaFallbackFixed(
  int seqLen, {
  required int numValueHeads,
  required int numKeyHeads,
  required int keyHeadDim,
  required int valueHeadDim,
  required MlxArray q,
  required MlxArray k,
  required MlxArray v,
  required MlxArray g,
  required MlxArray beta,
  required MlxArray state,
}) {
  final repeatFactor = numValueHeads ~/ numKeyHeads;
  if (seqLen == 1) {
    return _runGatedDeltaSingleStep(
      repeatFactor: repeatFactor,
      numValueHeads: numValueHeads,
      numKeyHeads: numKeyHeads,
      keyHeadDim: keyHeadDim,
      valueHeadDim: valueHeadDim,
      q: q,
      k: k,
      v: v,
      g: g,
      beta: beta,
      state: state,
    );
  }
  MlxArray qHeads = q;
  MlxArray kHeads = k;
  if (repeatFactor > 1) {
    qHeads = q.repeat(repeatFactor, axis: 2);
    kHeads = k.repeat(repeatFactor, axis: 2);
  }

  final outputs = <MlxArray>[];
  for (var index = 0; index < seqLen; index++) {
    final qStep = _sliceGatedDeltaStep4(
      qHeads,
      index,
      dim2: numValueHeads,
      dim3: keyHeadDim,
      shape: [1, numValueHeads, keyHeadDim],
    );
    final kStep = _sliceGatedDeltaStep4(
      kHeads,
      index,
      dim2: numValueHeads,
      dim3: keyHeadDim,
      shape: [1, numValueHeads, keyHeadDim],
    );
    final vStep = _sliceGatedDeltaStep4(
      v,
      index,
      dim2: numValueHeads,
      dim3: valueHeadDim,
      shape: [1, numValueHeads, valueHeadDim],
    );
    final gStep = _sliceGatedDeltaStep3(
      g,
      index,
      dim2: numValueHeads,
      shape: [1, numValueHeads],
    );
    final betaStep = _sliceGatedDeltaStep3(
      beta,
      index,
      dim2: numValueHeads,
      shape: [1, numValueHeads],
    );
    final useCompiled =
        Platform.environment['QWEN35_USE_GDELTA_STEP_COMPILED'] != '0';
    late MlxArray y;
    late MlxArray newState;
    if (useCompiled) {
      final step = _getGatedDeltaStepCompiled()([
        qStep,
        kStep,
        vStep,
        gStep,
        betaStep,
        state,
      ]);
      y = step[0].reshape([1, 1, numValueHeads, valueHeadDim]);
      newState = step[1];
    } else {
      final oldState = state;
      final decay = gStep.reshape([1, numValueHeads, 1, 1]);
      final decayed = state * decay;
      decay.close();
      final kvMem = (decayed * kStep.reshape([1, numValueHeads, 1, keyHeadDim]))
          .sum(axis: 3);
      final delta = (vStep - kvMem) * betaStep.reshape([1, numValueHeads, 1]);
      kvMem.close();
      final steppedState =
          decayed +
          kStep.reshape([1, numValueHeads, 1, keyHeadDim]) *
              delta.reshape([1, numValueHeads, valueHeadDim, 1]);
      decayed.close();
      delta.close();
      y = (steppedState * qStep.reshape([1, numValueHeads, 1, keyHeadDim]))
          .sum(axis: 3)
          .reshape([1, 1, numValueHeads, valueHeadDim]);
      oldState.close();
      newState = steppedState;
    }
    betaStep.close();
    outputs.add(y);
    qStep.close();
    kStep.close();
    vStep.close();
    gStep.close();
    state.close();
    state = newState;
  }
  if (!identical(qHeads, q)) {
    qHeads.close();
  }
  if (!identical(kHeads, k)) {
    kHeads.close();
  }
  g.close();
  beta.close();
  final output = mx.concatenate(outputs, axis: 1);
  for (final value in outputs) {
    value.close();
  }
  return (output: output, state: state);
}

({MlxArray output, MlxArray state}) _runGatedDeltaSingleStep({
  required int repeatFactor,
  required int numValueHeads,
  required int numKeyHeads,
  required int keyHeadDim,
  required int valueHeadDim,
  required MlxArray q,
  required MlxArray k,
  required MlxArray v,
  required MlxArray g,
  required MlxArray beta,
  required MlxArray state,
}) {
  MlxArray qStep = q.reshape([1, numKeyHeads, keyHeadDim]);
  MlxArray kStep = k.reshape([1, numKeyHeads, keyHeadDim]);
  if (repeatFactor > 1) {
    final repeatedQ = qStep.repeat(repeatFactor, axis: 1);
    final repeatedK = kStep.repeat(repeatFactor, axis: 1);
    qStep.close();
    kStep.close();
    qStep = repeatedQ;
    kStep = repeatedK;
  }
  final vStep = v.reshape([1, numValueHeads, valueHeadDim]);
  final gStep = g.reshape([1, numValueHeads]);
  final betaStep = beta.reshape([1, numValueHeads]);
  try {
    final useCompiled =
        Platform.environment['QWEN35_USE_GDELTA_STEP_COMPILED'] != '0';
    if (useCompiled) {
      final step = _getGatedDeltaStepCompiled()([
        qStep,
        kStep,
        vStep,
        gStep,
        betaStep,
        state,
      ]);
      state.close();
      return (
        output: step[0].reshape([1, 1, numValueHeads, valueHeadDim]),
        state: step[1],
      );
    }
    final oldState = state;
    final decay = gStep.reshape([1, numValueHeads, 1, 1]);
    final decayed = state * decay;
    decay.close();
    final kvMem = (decayed * kStep.reshape([1, numValueHeads, 1, keyHeadDim]))
        .sum(axis: 3);
    final delta = (vStep - kvMem) * betaStep.reshape([1, numValueHeads, 1]);
    kvMem.close();
    final steppedState =
        decayed +
        kStep.reshape([1, numValueHeads, 1, keyHeadDim]) *
            delta.reshape([1, numValueHeads, valueHeadDim, 1]);
    decayed.close();
    delta.close();
    final y = (steppedState * qStep.reshape([1, numValueHeads, 1, keyHeadDim]))
        .sum(axis: 3)
        .reshape([1, 1, numValueHeads, valueHeadDim]);
    oldState.close();
    return (output: y, state: steppedState);
  } finally {
    qStep.close();
    kStep.close();
    vStep.close();
    gStep.close();
    betaStep.close();
  }
}

MlxArray _sliceGatedDeltaStep4(
  MlxArray input,
  int index, {
  required int dim2,
  required int dim3,
  required List<int> shape,
}) => input
    .slice(start: [0, index, 0, 0], stop: [1, index + 1, dim2, dim3])
    .reshape(shape);

MlxArray _sliceGatedDeltaStep3(
  MlxArray input,
  int index, {
  required int dim2,
  required List<int> shape,
}) => input
    .slice(start: [0, index, 0], stop: [1, index + 1, dim2])
    .reshape(shape);

void _warmQwen35GDeltaStep({
  required int numValueHeads,
  required int keyHeadDim,
  required int valueHeadDim,
  required MlxDType dtype,
}) {
  if (Platform.environment['QWEN35_USE_GDELTA_STEP_COMPILED'] == '0') {
    return;
  }
  final q = MlxArray.zeros([1, numValueHeads, keyHeadDim], dtype: dtype);
  final k = MlxArray.zeros([1, numValueHeads, keyHeadDim], dtype: dtype);
  final v = MlxArray.zeros([1, numValueHeads, valueHeadDim], dtype: dtype);
  final g = MlxArray.zeros([1, numValueHeads], dtype: dtype);
  final beta = MlxArray.zeros([1, numValueHeads], dtype: dtype);
  final state = MlxArray.zeros([
    1,
    numValueHeads,
    valueHeadDim,
    keyHeadDim,
  ], dtype: dtype);
  try {
    final outputs = _getGatedDeltaStepCompiled()([q, k, v, g, beta, state]);
    try {
      MlxRuntime.evalAll(outputs);
    } finally {
      for (final output in outputs) {
        output.close();
      }
    }
  } finally {
    q.close();
    k.close();
    v.close();
    g.close();
    beta.close();
    state.close();
  }
}

MlxMetalKernel? _getGatedDeltaKernel() {
  if (_gatedDeltaKernelTried) {
    return _gatedDeltaKernel;
  }
  _gatedDeltaKernelTried = true;
  if (!MlxMetal.isAvailable()) {
    return null;
  }
  try {
    _gatedDeltaKernel = mx.fast.metalKernel(
      'gated_delta_step',
      ['q', 'k', 'v', 'g', 'beta', 'state_in', 'T'],
      ['y', 'state_out'],
      '''
auto n = thread_position_in_grid.z;
auto b_idx = n / Hv;
auto hv_idx = n % Hv;
auto hk_idx = hv_idx / (Hv / Hk);
constexpr int n_per_t = Dk / 32;
auto t_steps = T[0];

auto q_ = q + b_idx * t_steps * Hk * Dk + hk_idx * Dk;
auto k_ = k + b_idx * t_steps * Hk * Dk + hk_idx * Dk;
auto v_ = v + b_idx * t_steps * Hv * Dv + hv_idx * Dv;
y += b_idx * t_steps * Hv * Dv + hv_idx * Dv;

auto dk_idx = thread_position_in_threadgroup.x;
auto dv_idx = thread_position_in_grid.y;

auto i_state = state_in + (n * Dv + dv_idx) * Dk;
auto o_state = state_out + (n * Dv + dv_idx) * Dk;

float state[n_per_t];
for (int i = 0; i < n_per_t; ++i) {
  auto s_idx = n_per_t * dk_idx + i;
  state[i] = static_cast<float>(i_state[s_idx]);
}

auto g_ = g + b_idx * t_steps * Hv;
auto beta_ = beta + b_idx * t_steps * Hv;

for (int t = 0; t < t_steps; ++t) {
  float kv_mem = 0.0f;
  for (int i = 0; i < n_per_t; ++i) {
    auto s_idx = n_per_t * dk_idx + i;
    state[i] = state[i] * g_[hv_idx];
    kv_mem += state[i] * k_[s_idx];
  }
  kv_mem = simd_sum(kv_mem);

  auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

  float out = 0.0f;
  for (int i = 0; i < n_per_t; ++i) {
    auto s_idx = n_per_t * dk_idx + i;
    state[i] = state[i] + k_[s_idx] * delta;
    out += state[i] * q_[s_idx];
  }
  out = simd_sum(out);
  if (thread_index_in_simdgroup == 0) {
    y[dv_idx] = static_cast<InT>(out);
  }

  q_ += Hk * Dk;
  k_ += Hk * Dk;
  v_ += Hv * Dv;
  y += Hv * Dv;
  g_ += Hv;
  beta_ += Hv;
}

for (int i = 0; i < n_per_t; ++i) {
  auto s_idx = n_per_t * dk_idx + i;
  o_state[s_idx] = static_cast<InT>(state[i]);
}
''',
    );
  } on MlxException {
    if (Platform.environment['QWEN35_KERNEL_DEBUG'] == '1') {
      stderr.writeln('qwen35_run: gated-delta kernel creation failed');
    }
    _gatedDeltaKernel = null;
  }
  return _gatedDeltaKernel;
}

({MlxArray output, MlxArray state})? _tryGatedDeltaKernel(
  MlxArray q,
  MlxArray k,
  MlxArray v,
  MlxArray g,
  MlxArray beta,
  MlxArray state,
) {
  final kernelMode = _gatedDeltaKernelMode();
  if (kernelMode != '1' && kernelMode != 'decode') {
    return null;
  }
  if (kernelMode == 'decode' && q.shape[1] != 1) {
    return null;
  }
  if (!_supportsGatedDeltaKernel(
    numValueHeads: v.shape[2],
    numKeyHeads: q.shape[2],
    keyHeadDim: q.shape[3],
  )) {
    return null;
  }
  final kernel = _getGatedDeltaKernel();
  if (kernel == null) {
    return null;
  }
  try {
    return _runGatedDeltaKernel(kernel, q, k, v, g, beta, state);
  } on MlxException catch (error) {
    if (Platform.environment['QWEN35_KERNEL_DEBUG'] == '1') {
      stderr.writeln('qwen35_run: gated-delta kernel apply failed: $error');
    }
    return null;
  }
}

({MlxArray output, MlxArray state}) _runGatedDeltaKernel(
  MlxMetalKernel kernel,
  MlxArray q,
  MlxArray k,
  MlxArray v,
  MlxArray g,
  MlxArray beta,
  MlxArray state,
) {
  final b = q.shape[0];
  final t = q.shape[1];
  final hk = q.shape[2];
  final dk = q.shape[3];
  final hv = v.shape[2];
  final dv = v.shape[3];
  final config = _gatedDeltaKernelConfig(
    batch: b,
    t: t,
    hk: hk,
    hv: hv,
    dk: dk,
    dv: dv,
    dtype: q.dtype,
    stateDType: q.dtype,
  );
  final tArray = _gatedDeltaKernelT(t);
  final outputs = kernel.apply([q, k, v, g, beta, state, tArray], config);
  if (Platform.environment['QWEN35_KERNEL_DEBUG'] == '1') {
    stderr.writeln('qwen35_run: gated-delta kernel applied');
  }
  return (output: outputs[0], state: outputs[1]);
}

MlxMetalConfig _gatedDeltaKernelConfig({
  required int batch,
  required int t,
  required int hk,
  required int hv,
  required int dk,
  required int dv,
  required MlxDType dtype,
  required MlxDType stateDType,
}) {
  final key = '$batch:$t:$hk:$hv:$dk:$dv:${dtype.value}:${stateDType.value}';
  final cached = _gatedDeltaKernelConfigCache[key];
  if (cached != null) {
    return cached;
  }
  final config = mx.fast.metalConfig();
  final gridX = _gatedDeltaGridX();
  final tgY = _gatedDeltaThreadgroupY();
  config.addOutputArg([batch, t, hv, dv], dtype);
  config.addOutputArg([batch, hv, dv, dk], stateDType);
  config.setGrid(gridX, dv, batch * hv);
  config.setThreadGroup(32, tgY, 1);
  config.setVerbose(Platform.environment['QWEN35_KERNEL_DEBUG'] == '1');
  config.addTemplateDtype('InT', dtype);
  config.addTemplateInt('Hk', hk);
  config.addTemplateInt('Hv', hv);
  config.addTemplateInt('Dk', dk);
  config.addTemplateInt('Dv', dv);
  _gatedDeltaKernelConfigCache[key] = config;
  return config;
}

MlxArray _gatedDeltaKernelT(int t) {
  final cached = _gatedDeltaKernelTCache[t];
  if (cached != null) {
    return cached;
  }
  final out = MlxArray.fromInt32List([t], shape: [1]);
  _gatedDeltaKernelTCache[t] = out;
  return out;
}

void _warmQwen35GDeltaKernel({
  required int numValueHeads,
  required int numKeyHeads,
  required int keyHeadDim,
  required int valueHeadDim,
  required MlxDType dtype,
}) {
  if (_gatedDeltaKernelWarmed) {
    return;
  }
  final kernelMode = _gatedDeltaKernelMode();
  if (kernelMode != '1' && kernelMode != 'decode') {
    return;
  }
  if (!_supportsGatedDeltaKernel(
    numValueHeads: numValueHeads,
    numKeyHeads: numKeyHeads,
    keyHeadDim: keyHeadDim,
  )) {
    return;
  }
  final q = MlxArray.zeros([1, 1, numKeyHeads, keyHeadDim], dtype: dtype);
  final k = MlxArray.zeros([1, 1, numKeyHeads, keyHeadDim], dtype: dtype);
  final v = MlxArray.zeros([1, 1, numValueHeads, valueHeadDim], dtype: dtype);
  final g = MlxArray.zeros([1, 1, numValueHeads], dtype: dtype);
  final beta = MlxArray.zeros([1, 1, numValueHeads], dtype: dtype);
  final state = MlxArray.zeros([
    1,
    numValueHeads,
    valueHeadDim,
    keyHeadDim,
  ], dtype: dtype);
  try {
    try {
      final warmed = _tryGatedDeltaKernel(q, k, v, g, beta, state);
      if (warmed != null) {
        try {
          MlxRuntime.evalAll([warmed.output, warmed.state]);
          _gatedDeltaKernelWarmed = true;
        } finally {
          warmed.output.close();
          warmed.state.close();
        }
      }
    } on MlxException {
      if (Platform.environment['QWEN35_KERNEL_DEBUG'] == '1') {
        stderr.writeln('qwen35_run: gated-delta kernel warmup failed');
      }
    }
  } finally {
    q.close();
    k.close();
    v.close();
    g.close();
    beta.close();
    state.close();
  }
}
