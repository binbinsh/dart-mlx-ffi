part of 'qwen3_5.dart';

MlxMetalKernel? _gatedDeltaKernel;
bool _gatedDeltaKernelTried = false;
MlxFunction? _gatedDeltaStepCompiled;
MlxFunction? _swiGluCompiled;
MlxFunction? _computeGCompiled;

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
    final decay = g.reshape([g.shape[0], g.shape[1], 1, 1]);
    final decayed = state * decay;
    decay.close();
    final kvMem = (decayed * k.reshape([k.shape[0], k.shape[1], 1, k.shape[2]])).sum(axis: 3);
    final delta = (v - kvMem) * beta.reshape([beta.shape[0], beta.shape[1], 1]);
    kvMem.close();
    final newState =
        decayed +
        k.reshape([k.shape[0], k.shape[1], 1, k.shape[2]]) *
            delta.reshape([delta.shape[0], delta.shape[1], delta.shape[2], 1]);
    decayed.close();
    final y =
        (newState * q.reshape([q.shape[0], q.shape[1], 1, q.shape[2]])).sum(axis: 3);
    return [y, newState];
  });
  _gatedDeltaStepCompiled = fn.compile(shapeless: true);
  fn.close();
  return _gatedDeltaStepCompiled!;
}

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
  _swiGluCompiled = fn.compile(shapeless: true);
  fn.close();
  return _swiGluCompiled!;
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
  _computeGCompiled = fn.compile(shapeless: true);
  fn.close();
  return _computeGCompiled!;
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
      ['q', 'k', 'v', 'g', 'beta', 'state_in'],
      ['y', 'state_out'],
      '''
auto n = thread_position_in_grid.z;
auto b_idx = n / Hv;
auto hv_idx = n % Hv;
auto hk_idx = hv_idx / (Hv / Hk);
constexpr int n_per_t = Dk / 32;

auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
y += b_idx * T * Hv * Dv + hv_idx * Dv;

auto dk_idx = thread_position_in_threadgroup.x;
auto dv_idx = thread_position_in_grid.y;

auto i_state = state_in + (n * Dv + dv_idx) * Dk;
auto o_state = state_out + (n * Dv + dv_idx) * Dk;

float state[n_per_t];
for (int i = 0; i < n_per_t; ++i) {
  auto s_idx = n_per_t * dk_idx + i;
  state[i] = static_cast<float>(i_state[s_idx]);
}

auto g_ = g + b_idx * T * Hv;
auto beta_ = beta + b_idx * T * Hv;

for (int t = 0; t < T; ++t) {
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
  if (Platform.environment['QWEN35_USE_GDELTA_KERNEL'] == '0') {
    return null;
  }
  final kernel = _getGatedDeltaKernel();
  if (kernel == null) {
    return null;
  }
  MlxRuntime.evalAll([q, k, v, g, beta, state]);
  final config = mx.fast.metalConfig();
  try {
    final b = q.shape[0];
    final t = q.shape[1];
    final hk = q.shape[2];
    final dk = q.shape[3];
    final hv = v.shape[2];
    final dv = v.shape[3];
    config.addOutputArg([b, t, hv, dv], q.dtype);
    config.addOutputArg(state.shape, q.dtype);
    config.setGrid(32, dv, b * hv);
    config.setThreadGroup(32, 4, 1);
    config.setVerbose(Platform.environment['QWEN35_KERNEL_DEBUG'] == '1');
    config.addTemplateDtype('InT', q.dtype);
    config.addTemplateInt('T', t);
    config.addTemplateInt('Hk', hk);
    config.addTemplateInt('Hv', hv);
    config.addTemplateInt('Dk', dk);
    config.addTemplateInt('Dv', dv);
    final outputs = kernel.apply([q, k, v, g, beta, state], config);
    if (Platform.environment['QWEN35_KERNEL_DEBUG'] == '1') {
      stderr.writeln('qwen35_run: gated-delta kernel applied');
    }
    return (output: outputs[0], state: outputs[1]);
  } on MlxException catch (error) {
    if (Platform.environment['QWEN35_KERNEL_DEBUG'] == '1') {
      stderr.writeln('qwen35_run: gated-delta kernel apply failed: $error');
    }
    return null;
  } finally {
    config.close();
  }
}
