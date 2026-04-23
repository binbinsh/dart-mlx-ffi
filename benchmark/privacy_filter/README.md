# OpenAI Privacy Filter MLX Benchmark

This directory contains a benchmark/reference integration for
`openai/privacy-filter`.

It runs the model in Python MLX, exports the same forward graph as an MLX
function, then runs that graph through the Dart MLX importer with identical
`input_ids`.

## Run

```sh
uv run python benchmark/privacy_filter/compare.py \
  --checkpoint tmp/privacy-filter \
  --seq-len 256 \
  --warmup 3 \
  --iters 10 \
  --moe-chunk-size 256 \
  --moe-matmul-dtype bfloat16
```

If `--checkpoint` is omitted, the script resolves a local Hugging Face cache
entry for `openai/privacy-filter` or downloads the snapshot.

## Output

The script writes `benchmark/out/privacy_filter/compare.json` with:

- Python MLX and Dart MLX import timings.
- Python/Dart forward-output diff statistics.
- Viterbi-decoded privacy spans from the Python MLX logits.
- The redacted text assembled from decoded spans.

The generated `.mlxfn` and safetensors inputs are ignored local benchmark
artifacts.

## Performance Knobs

- `--moe-matmul-dtype bfloat16` is the default optimized MLX path. It keeps MoE
  matmuls in bf16 and is substantially faster on Apple GPU.
- `--moe-matmul-dtype float32` keeps the MoE path closer to the upstream
  PyTorch CPU reference logits, but is much slower.
- `--fast-attention` uses MLX fused SDPA. It is experimental because it can
  shift logits more than the default reference attention path.

## Scope

This is not a full Dart redaction API. The tokenizer and Viterbi/span decoding
currently run in Python; Dart is used to validate and time the imported MLX
forward graph.
