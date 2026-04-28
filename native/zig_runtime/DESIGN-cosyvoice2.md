# CosyVoice2 Pure-Dart Runtime Design

Status: design only. No code yet.

## Goal

End-to-end CosyVoice2-0.5B inference (zero-shot voice clone TTS) on the
`dart_inference` runtime, with no Python on the runtime path.

Primary deliverables:

1. Dart class `CosyVoice2DartRuntime` in `lib/src/models/cosyvoice2/`.
2. Optional Zig helper `dinf_cosy_*` exports for hot kernels (RAS sampling,
   token-id -> embedding gather), if benchmarks justify them.
3. Wired entry in `bin/tts_server.dart` so the existing TTS service can route
   to cosyvoice2 once readiness flips to `ready`.

## Non-Goals

- No vLLM-style continuous batching. Single request, single stream.
- No bistream (text-streaming) inference. Static text only in v1.
- No Apple `coreml` / `mlx-c` backends in v1. ONNX only.
- No quantization in v1. fp32 graphs as exported.

## Architectural Choice: Dart-Driven AR Loop

Two valid placements for the autoregressive decode loop:

| Layer | Pros | Cons |
| --- | --- | --- |
| Dart | Matches kokoro pattern. Reuses existing `InferenceSession`, `RuntimeTensor`, `NativeTensorBuffer`. No new C ABI surface. Easy to debug. | Extra Dart -> C FFI per decode step (49 inputs / 49 outputs * up to ~1500 steps). |
| Zig | One FFI call for the whole synthesis. KV-cache lives in Zig arena, no Dart-visible per-step churn. | Requires new ORT-from-Zig path (does not exist today; Dart owns ORT). Significant new ABI. |

**Decision: Dart-driven loop**. Rationale:

- Kokoro proves the Dart-driven pattern is fast enough (~30 RTF on CPU).
- The `dart_inference` ORT bindings are already in Dart, not Zig. Moving ORT
  ownership into Zig is a much larger project than this one.
- The hot per-step cost is dominated by the ORT decode call (~1.4 GB graph),
  not by FFI dispatch. Even at 1500 steps the FFI overhead is sub-millisecond.
- Zig stays the home for **stateless preprocessing/sampling kernels**, matching
  current convention.

The Zig contribution is therefore narrow: a `dinf_cosy_ras_sample` C ABI
function that takes a logits buffer + recent-token window + RNG seed and
returns the chosen token id.

## Asset Surface (already exported)

All assets live under `models/CosyVoice2-0.5B/` per the catalog entry:

| File | Size | Role |
| --- | --- | --- |
| `campplus.onnx` | small | Speaker embedding from prompt mel |
| `speech_tokenizer_v2.onnx` | small | Prompt waveform -> speech tokens |
| `flow.encoder.fp32.onnx` | 165 MB | Speech tokens -> 512-d encoder out |
| `flow.decoder.estimator.fp32.onnx` | medium | Diffusion estimator (CFG, batch=2) |
| `llm_prefill.onnx` | 1.4 GB | Qwen2 prefill, no past KV |
| `llm_decode.onnx` | 1.4 GB | Qwen2 single-step with past KV |
| `llm_decoder_head.onnx` | 23 MB | Linear 896 -> 6564 (speech vocab) |
| `hift.onnx` | 80 MB | Vocoder, no cross-chunk source cache |
| `hift_streaming.onnx` | 80 MB | Vocoder with non-empty source cache |
| `llm_embeddings.npz` | 542 MB | text(151936x896), llm(2x896), speech(6564x896) |

LLM-graph KV cache: 24 layers x {key, value} = 48 tensors, each shape
`[batch, 2, seq, 64]` (GQA: 2 KV heads, head_dim=64).

## ONNX I/O Signatures

### llm_prefill.onnx
- Inputs: `inputs_embeds[B,S,896]`, `attention_mask[B,S]`
- Outputs: `hidden[B,S,896]`, `present_key_{0..23}[B,2,S,64]`, `present_value_{0..23}[B,2,S,64]`

### llm_decode.onnx
- Inputs: `inputs_embeds[B,1,896]`, `attention_mask[B,total_seq]`,
  `past_key_{0..23}[B,2,past_seq,64]`, `past_value_{0..23}[B,2,past_seq,64]`
- Outputs: `hidden[B,1,896]`, `present_key_{0..23}[B,2,total_seq,64]`,
  `present_value_{0..23}[B,2,total_seq,64]`

### llm_decoder_head.onnx
- Inputs: `hidden[B,S,896]`
- Outputs: `logits[B,S,6564]`

### speech_tokenizer_v2.onnx
- Inputs: `feats[1,128,T]`, `feats_length[1]` (int32)
- Outputs: `indices[B,T_tok]` (int32)

### campplus.onnx
- Inputs: `input[B,S,80]`
- Outputs: `output[B,192]`

### flow.encoder.fp32.onnx
- Inputs: `xs[B,S,512]`, `xs_lens[B]` (int32)
- Outputs: `encoder_out[B,S_out,512]`, `encoder_mask[B,1,S_out]` (bool)

### flow.decoder.estimator.fp32.onnx
- Inputs: all batch-2 (CFG): `x`, `mask`, `mu`, `t`, `spks`, `cond`
- Outputs: `estimator_out[2,80,S]`

### hift.onnx / hift_streaming.onnx
- Inputs: `speech_feat[B,80,T_mel]`, `cache_source[B,1,T_cache]`
- Outputs: `audio[B,T_audio]`, `source_cache[B,1,T_source]`
- `hift_streaming.onnx` is the only variant where `T_cache > 0` is honored;
  `hift.onnx` was traced with `T_cache=0` and the cache branch was constant-
  folded out.

## Synthesis Pipeline

```
prompt_audio
  -> mel_extract (CPU; matches CosyVoice2 frontend)
  -> [campplus] -> spk_embed [192]
  -> [speech_tokenizer_v2] -> prompt_speech_tokens [T_p]

text -> tokenize (Qwen2 BPE) -> text_ids [T_t]
prompt_text -> tokenize -> prompt_text_ids [T_pt]

embed_lookup:
  text_emb       = gather(text_embedding, [prompt_text_ids; text_ids])
  sos_eos_emb    = llm_embedding[0]   # llm_special row 0
  task_id_emb    = llm_embedding[1]   # llm_special row 1
  prompt_sp_emb  = gather(speech_embedding, prompt_speech_tokens)
  lm_input       = cat([sos_eos_emb, text_emb, task_id_emb, prompt_sp_emb], dim=1)

# AR loop:
hidden, kv = llm_prefill.run(lm_input, attn_mask=ones)
total_seq = lm_input.length
out_tokens = []
for step in range(max_len):
  logits = llm_decoder_head.run(hidden[:, -1:])     # [B,1,6564]
  log_probs = log_softmax(logits[0,0])
  ignore_eos = step < min_len
  top_id = ras_sample(log_probs, out_tokens, sampling=25,
                      ignore_eos=ignore_eos)
  if top_id == 6561: break                          # speech_token_size = EOS
  if top_id  > 6561: continue                       # special, skip
  out_tokens.append(top_id)
  next_emb = speech_embedding[top_id].reshape(1,1,896)
  total_seq += 1
  hidden, kv = llm_decode.run(next_emb, attn_mask=ones[total_seq], past=kv)

# Vocoder pipeline (chunked / streaming):
flow_in = lookup(speech_token_embedding_table, out_tokens)  # 512-d table
encoder_out, mask = flow_encoder.run(flow_in, lens)
mel = flow_decoder_diffusion(encoder_out, mask, spk_embed, cond)
                              # iterative; calls flow.decoder.estimator
                              # with batch=2 (CFG) per step
audio_chunk, source_cache = hift_streaming.run(mel, source_cache_prev)
emit(audio_chunk)
```

## State Lifecycle

`CosyVoice2DartRuntime` owns:

- 8 `InferenceSession` (one per ONNX file)
- Decoded `npz` weight maps as `NativeTensorBuffer` (interned, never copied
  onto Dart heap; gather lookups read native memory directly)
- Voice-prompt cache: per voice, `(spk_embed, prompt_speech_tokens, prompt_text_ids)`
- Per-request scratch:
  - prefill `inputs_embeds` arena (max-text-tokens-shaped `NativeTensorBuffer`)
  - decode `inputs_embeds` arena (shape `[1,1,896]`, reused every step)
  - 48 KV tensor handles (Dart-side `RuntimeTensor` references; ORT owns memory)
  - decode `attention_mask` arena (i64, max-decode-length capacity)

KV handoff: `llm_prefill.run(...)` returns 49 outputs as a `ModelOutputs`. Dart
extracts the 48 `present_*` `RuntimeTensor`s and passes them as the 48
`past_*` inputs to the first `llm_decode.run(...)`. Each subsequent decode
step consumes the previous step's `present_*` and produces the next `present_*`.
The previous step's KV `RuntimeTensor`s must be `close()`d after the next
step's run completes (ORT-allocated buffers).

## Tokenizer

Qwen2 BPE tokenizer is required for `text_ids`. CosyVoice2-0.5B uses the
upstream Qwen tokenizer; vocab is 151936. Two paths:

1. Reuse an existing pure-Dart Qwen2 BPE tokenizer if one exists in
   `dart-inference` (search confirms there is no `qwen` tokenizer module).
2. Port `tokenizer.json` parsing using HuggingFace tokenizer JSON format.
   `bpe.zig` already exists in the runtime — check whether it can load the
   merges/vocab from `tokenizer.json` directly. If not, add a thin Dart
   wrapper that uses `bpe.zig` for the encode hot path.

Decision: reuse `bpe.zig` if its merges-table API matches `tokenizer.json`
format; otherwise add a Dart-side adapter. Defer concrete choice until
implementation phase.

## Mel Extractor

Required by speech_tokenizer_v2 (128-mel) and campplus (80-mel from the same
audio). Two extractors with different mel banks are needed.

CosyVoice2 frontend uses Whisper-style mel for the tokenizer (n_mels=128,
n_fft=1024, hop=160 at 16kHz) and a 80-mel for campplus. Implement both as
pure-Dart functions that accept a `NativeTensorBuffer` audio input and emit
`NativeTensorBuffer` mels. STFT can reuse the manual cos/sin DFT pattern from
the hift export (no `dart:complex`) or call into `audio.zig`.

Defer audio resampling. Assume callers provide 16 kHz mono for the prompt.

## RAS Sampler (Zig kernel)

```c
// dinf_cosy_ras_sample
//
// Samples one token id from `logits` using nucleus-sampling with repetition
// penalty (RAS, VALL-E 2 style). Pure function with caller-supplied RNG seed.
//
// Inputs:
//   logits:        [vocab_size] float32, log-softmaxed already
//   recent_tokens: [recent_count] int32, last decoded tokens (for RAS check)
//   recent_count:  isize, length of recent_tokens, may be 0
//   vocab_size:    isize, expected 6564
//   sampling:      i32, fallback random sample top_k (e.g. 25)
//   top_p:         f32, nucleus threshold (default 0.8)
//   top_k:         i32, max nucleus count (default 25)
//   win_size:      i32, RAS recent-window (default 10)
//   tau_r:         f32, repetition threshold (default 0.1)
//   ignore_eos:    i32 (bool), if non-zero retry until non-EOS
//   max_trials:    i32, retry budget (default 100)
//   eos_token:     i32, EOS id (6561 = speech_token_size)
//   rng_state:     *mut u64, splitmix64 state, mutated in-place
//   error_out:     *mut [*c]u8
//
// Returns: i32 token id, or -1 on error.
```

Internal implementation mirrors `cosyvoice/utils/common.py:ras_sampling`:

1. `nucleus_sampling(logits, top_p, top_k)` -> tentative top_id
2. Count occurrences in `recent_tokens[-win_size:]`; if rep >= win_size*tau_r,
   fall back to `random_sampling`.
3. If `ignore_eos` and result == EOS, retry up to `max_trials`.

Both `nucleus_sampling` and `random_sampling` need a softmax + cumulative
probability scan + multinomial draw. Use a Ziggurat-free path: linear scan
through sorted probs is plenty fast at vocab=6564.

RNG: splitmix64 fed by a session-owned u64 state. Caller seed for
reproducibility. No std-lib `Random` because Zig's std.Random API is overkill
here and we want bit-exact reproducibility across platforms.

## Diffusion Decoder Loop

`flow.decoder.estimator.fp32.onnx` is one denoising step. CosyVoice2 uses
~10 steps with classifier-free guidance (batch=2: cond + uncond).

The loop is small (10 ORT calls) and trivially Dart-driven. No Zig helper
needed. Loop body:

```dart
for (var step = 0; step < numSteps; step++) {
  final t = scheduler.timestepAt(step);   // scalar -> [2] tensor
  final eps = estimator.run({
    'x': xCondAndUncond,        // [2, 80, S]
    'mask': maskCondAndUncond,  // [2, 1, S]
    'mu': muCondAndUncond,
    't': tTensor,
    'spks': spksCondAndUncond,
    'cond': condCondAndUncond,
  });
  // CFG combine and EulerStep
  ...
}
```

Concrete scheduler / number of steps to mirror upstream
`cosyvoice/flow/flow_matching.py:CausalConditionalCFM.solve_euler`. Verify
the ODE solver and step count from that file during implementation.

## Streaming

CosyVoice2 streams at the speech-token level: every N decoded tokens (default
~25 tokens = ~1s of audio at 25 tokens/sec), the partial token stream is run
through flow + hift_streaming and a chunk of audio is emitted. The
`source_cache` from each hift call feeds into the next call to avoid
glitches across chunks.

Dart API surface:

```dart
Stream<Uint8List> synthesize({
  required String text,
  required String voiceName,
  int chunkTokens = 25,
  double speed = 1.0,
});
```

Backpressure: backed by a `StreamController<Uint8List>` with a `pause()`/
`resume()` callback that suspends the AR loop between iterations.

## C ABI Additions

Only one new export expected:

```
i32 dinf_cosy_ras_sample(
  const f32* logits, isize vocab_size,
  const i32* recent_tokens, isize recent_count,
  i32 sampling, f32 top_p, i32 top_k,
  i32 win_size, f32 tau_r,
  i32 ignore_eos, i32 max_trials, i32 eos_token,
  u64* rng_state,
  [*c]u8* error_out,
);  // returns chosen token id, or -1 on error
```

Optional, only if profiling justifies:

```
i32 dinf_cosy_gather_embed_f32(
  const f32* table, isize rows, isize cols,
  const i32* ids, isize id_count,
  f32* out,        // caller-allocated [id_count, cols]
  [*c]u8* error_out,
);
```

(Dart can do the gather with `Float32List` views over `NativeTensorBuffer`s
directly. Add the Zig version only if the Dart loop shows up in profiles.)

No `dinf_cosy_open` / `dinf_cosy_run` / `dinf_cosy_close` style session API.
That belongs in Dart.

## Catalog Update on Completion

Promote `runtime: 'dart_onnx_partial'` -> `'dart_onnx'`,
`readiness: 'partial'` -> `'ready'`, clear `blockers`. Add cosyvoice2 to the
default-prewarm set in `bin/tts_server.dart` if appropriate.

## Open Questions

1. Mel parameters for `speech_tokenizer_v2`. Confirm from
   `cosyvoice/dataset/processor.py` mel-spectrogram config.
2. Number of diffusion steps + CFG scale to use at inference. Current upstream
   default is 10 steps, scale=0.7 based on `flow_matching.py`. Verify.
3. Whether the existing `bpe.zig` can directly load a HuggingFace
   `tokenizer.json` for Qwen2, or whether a thin Dart-side tokenizer adapter
   is required.
4. Memory: 1.4 GB prefill + 1.4 GB decode = 2.8 GB of weights. Confirm the
   ORT shared-weights option (`add_session_config_entry("session.use_env_allocators", "1")`)
   actually shares weights between the two graphs at the storage level.
   Otherwise prefill weights can be released after the prefill call to halve
   peak RSS.

## Implementation Order

1. Tokenizer: pure-Dart Qwen2 BPE adapter (use `bpe.zig` if compatible).
2. Mel extractors: 80-mel and 128-mel.
3. Voice prompt path: campplus + speech_tokenizer_v2 + spk_embed cache.
4. RAS sampler: `dinf_cosy_ras_sample` in Zig + Dart binding.
5. AR loop: prefill -> head -> sample -> embed -> decode chain.
6. Flow encoder + diffusion decoder.
7. Vocoder + chunked streaming using hift_streaming.
8. End-to-end smoke test against reference Python CosyVoice2 output (waveform
   correlation > 0.99 on a fixed seed).
9. Catalog promote to `ready`.
10. Wire into `bin/tts_server.dart`.

Each step is testable in isolation against the corresponding Python reference.

---

## Handoff Snapshot (as of this session)

### Done
1. **Step 1 — Tokenizer.** Qwen2 byte-level BPE in Zig
   (`qwen2_bpe.zig`, `qwen2_unicode.zig`, C ABI in
   `qwen2_bpe_api.zig`). 20/20 HF parity at the Zig level
   (`qwen2_parity_test.zig`).
2. **Step 4 — Dart bindings + AR loop scaffold.**
   - FFI bindings: `dinf_qwen2_bpe_{new,free,encode}` in
     `lib/src/runtime/native_ops.dart`.
   - High-level wrapper: `Qwen2BpeTokenizer` in
     `lib/src/models/cosyvoice2/qwen2_tokenizer.dart` (loads
     `vocab.json` + `merges.txt`, registers the 4 base + 16
     CosyVoice2 specials).
   - Dart parity test: `test/qwen2_tokenizer_parity_test.dart`
     (gated on `QWEN2_TOKENIZER_DIR`, 20/20 passing through the
     full Dart -> C ABI -> Zig path).
   - LLM driver: `CosyVoice2LlmDriver`
     (`lib/src/models/cosyvoice2/cosyvoice2_llm_driver.dart`),
     owning `llm_prefill` + `llm_decode` + `llm_decoder_head`
     sessions, the embedding NPZ tables, and the tokenizer.
     Exposes `prefill / decodeStep / headLogits / embedTextTokens
     / embedSpeechToken` primitives keyed off the actual ONNX I/O
     names (`inputs_embeds`, `attention_mask`, `past_key_*`,
     `past_value_*`, `present_key_*`, `present_value_*`, `hidden`,
     `logits`).
   - `CosyVoice2Paths` refreshed: `llm_prefill / llm_decode /
     llm_decoder_head / hift_streaming` registered as
     `requiredForSynthesis`; legacy monolithic `llm.onnx` kept
     but downgraded to optional.
3. **Step 9 — End-to-end smoke.**
   - `bin/cosyvoice2_llm_smoke.dart` runs the full pipeline:
     load partial bundle -> tokenize -> embed -> prefill -> 5
     greedy decode steps -> head logits.
   - Verified on CPU EP, RTX 5090 host. Sample run for
     "你好，世界" produces token ids
     `[56568, 52801, 3837, 99244, 97120]`, prefill ~63 ms,
     decode steps ~52 ms each, sampled speech tokens within
     `[0, 6564)`.

### Deferred (open work)

- **Step 2 — Mel extractors.** Need 80-mel extractor (for
  `campplus`) and 128-mel extractor (for `speech_tokenizer_v2`).
  Pure-Dart STFT or a Zig FFT helper; `kok_phon` already has
  STFT primitives that can be specialized.
- **Step 3 — Voice prompt path.** Wire `campplus` + a mel
  extractor into a `SpeakerEmbeddingExtractor` (Dart) and
  cache `speech_token_v2` outputs alongside speaker embeddings
  per voice. Requires Step 2.
- **Step 5 — RAS sampler.** `cosyvoice2.utils.common.ras_sampling`
  needs a Zig port (`dinf_cosy_ras_sample`) plus a Dart
  binding. Inputs: logits, win_size, tau, top_p, top_k,
  decoded-history. Output: chosen token id. Today the smoke
  uses greedy argmax — replace it before any quality eval.
- **Step 6 — Flow.** Wire `flow_encoder_fp32` + the `flow.decoder.estimator.fp32`
  diffusion loop. Decoder integrates ODE over a small number
  of steps (default 10, CFG scale 0.7) — needs a Dart-side
  scheduler. KV-style state-passing not needed; each step is
  a clean forward pass over the conditioned mel stream.
- **Step 7 — Vocoder + streaming.** `hift.onnx` works
  position-by-position; `hift_streaming.onnx` carries
  cross-chunk state. Need a Dart-side streamer that buffers
  the upstream mel chunks and emits PCM in a
  `Stream<Float32List>`.
- **Step 8 — Reference parity.** Compare the full Dart
  pipeline waveform against the upstream Python CosyVoice2
  on a fixed seed/voice/text. Gate: correlation > 0.99 in
  the windowed time domain, MSE < 1e-4 in mel space.
- **Step 10 — Catalog & server.** Promote
  `tts_backends_catalog` cosyvoice2 entry from `partial` to
  `ready` (after Step 8 passes). Wire the new driver into
  `bin/tts_server.dart` so existing TTS callers receive
  pure-Dart cosyvoice2 audio.

### Validation Commands

- Whole-package analyze: `dart analyze`
- Tokenizer parity (Dart): `QWEN2_TOKENIZER_DIR=<model-dir>/CosyVoice-BlankEN dart test test/qwen2_tokenizer_parity_test.dart`
- Tokenizer parity (Zig): `QWEN2_TOKENIZER_DIR=<...>/CosyVoice-BlankEN QWEN2_TOKENIZER_CASES=test/fixtures/cosyvoice2/qwen2_tokenizer_cases.json $ZIG test native/zig_runtime/qwen2_parity_test.zig -lc`
- E2E LLM smoke (CPU): `dart run bin/cosyvoice2_llm_smoke.dart --model-dir <model-dir> --provider cpu`
- E2E LLM smoke (CUDA): `dart run bin/cosyvoice2_llm_smoke.dart --model-dir <model-dir> --provider cuda --device-id 0`

### Critical Implementation Notes

- The split-LLM ONNX exports use `inputs_embeds` (not
  `input_ids`); embedding lookup happens Dart-side via
  `llm_embeddings.npz`. Do not reintroduce a graph-level
  embedding op — it would break weight sharing assumptions.
- `present_key_*` shape from `llm_prefill` is `[B, 2, S, 64]`,
  matching `past_key_*` consumed by `llm_decode`. The loop
  must pass them through verbatim each step.
- `attention_mask` grows by 1 per decode step. Today
  `decodeStep` reallocates; for production hot loops this can
  be replaced with a bump-allocated buffer reused across
  steps.
- Greedy argmax in the smoke is intentional. Replace with
  `dinf_cosy_ras_sample` (Step 5) before any quality eval —
  greedy collapses into repeating tokens after ~3 steps,
  which is exactly what the current smoke output shows.
