# PaddleOCR-VL iOS Runtime Findings

This note records the current iPhone-specific runtime conclusion for
`PaddleOCR-VL-1.5` in the example harness.

## Current Stable Default

The current iPhone default is:

- `uniform` KV cache quantization
- `8-bit` uniform KV
- early vision-weight release before multimodal embedding build

That is now the repo default, not just a harness override.

## Current Validated Baseline

Fresh one-chunk `photo_render_512` on iPhone:

- build marker: `launch-bridge-v72-ios-uniform-default`
- cache after prefill: `dense=0 uniform=18 turbo=0 offset=517`
- peak bytes: `2295298912`
- first 32 generated ids:
  `94036,94527,23,23,8729,94110,9,3,94692,39496,16539,94098,23,23,95268,94110,4,3,94692,94169,70782,93956,6,3,94692,94169,8729,94098,23,23,18355,94746`

This prefix matches the host/reference path for the same real
`photo_render_512` case.

## Why Uniform Became The Default

Real-case validation showed that the previous iPhone default,
`turboquant`, was the main reason the `photo_render_512` decode path drifted
away from the host/reference tokens.

Host-side probe on the same real case showed:

- default host path: `94036,94527,23,23,...`
- iOS-like config with `turboquant`: prefix drifted immediately
- iOS-like config with `uniform`: prefix returned to `94036,94527,23,23,...`

Device-side confirmation matched that result:

- `turboquant` iPhone path produced a different prefix starting at the second
  generated token
- `uniform` iPhone path restored the expected prefix
- `uniform` also reduced total peak from about `2315858888` to `2295298912`

So the important iPhone fix was not a Metal allocator trick. It was selecting
the correct KV-cache scheme for this workload.

## What Is Now Ruled Out

These are no longer attractive default directions for iPhone:

- `turboquant` default KV on this real photo case
  It drifted from the host/reference prefix.
- `uniform 4-bit`
  It only reduced peak from `2295298912` to `2295167840` and started drifting
  at token 4.
- `qmv_fast -> qmv` fallback switching
  It did not reduce peak and was not exact-safe on iPhone.
- decode-only `eval_logits` toggles
  They did not materially change peak.
- command-buffer cadence overrides such as
  `MLX_MAX_OPS_PER_BUFFER` / `MLX_MAX_MB_PER_BUFFER`
  They changed some local peaks but not the final result that mattered.

## Historical Context

Earlier investigation focused on late-token Metal resource growth and
TurboQuant-specific micro-optimizations. That work was still useful for ruling
out dead ends, but it is no longer the best explanation for the current
`photo_render_512` outcome.

For the current repo state, the most important practical conclusion is simpler:

- keep `uniform 8-bit` as the iPhone default
- do not switch back to `turboquant` on iPhone without a real-case parity check
- do not use `uniform 4-bit` as a product default
