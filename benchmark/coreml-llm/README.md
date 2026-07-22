# CoreML-LLM Baseline

This directory is reserved for same-model benchmarking against
`john-rocky/CoreML-LLM`.

The baseline runner must emit runtime benchmark JSON with:

- deterministic correctness evidence: greedy token IDs, top-k/logit summaries,
  function-call JSON, VLM text, or embedding cosine/L2 inputs
- speed metrics: prefill tokens/s, decode tokens/s, TTFT, end-to-end latency
- memory metrics: `peak_memory_bytes`, with raw platform fields preserved
- placement diagnostics: Core ML `MLComputePlan` ANE/GPU/CPU counts when
  available

The directory name intentionally uses `coreml-llm` to match the upstream
project name.

The actual SwiftPM package lives in `swift_baseline/` so its package identity
does not collide with the upstream `CoreML-LLM` dependency. The compatibility
shim can be run from the repository root:

```sh
swift benchmark/coreml-llm/baseline_runner.swift \
  --model-id qwen3_5 \
  --artifact /path/to/coreml-llm-model \
  --task text \
  --prompt-file ../../../models/validation/runtime/fixtures/text_prompt.txt \
  --out ../../../models/validation/out/runtime/qwen3_5/ios/coreml_llm.json
```

To validate the package without building every upstream sample executable:

```sh
swift build --product coreml-llm-baseline \
  --package-path benchmark/coreml-llm/swift_baseline
```

The Swift executable supports:

- `--task text`: `CoreMLLLM.stream(...)` greedy text generation
- `--task function`: `FunctionGemma.generate(...)` with optional
  `--tools-file` / `--tools-json`
- `--task embedding`: `EmbeddingGemma.encode(...)` with optional
  `--embedding-query-file` and `--embedding-dim`
- `--task vlm`: image-conditioned `CoreMLLLM.stream(...)` with
  `--image-file`
