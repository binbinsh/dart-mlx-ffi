# Draft Upstream Issue: iOS decoder forward creates persistent Metal resource growth

This note is historical background from the earlier runtime-churn investigation.
It predates the later finding that the real `photo_render_512` iPhone mismatch
was dominated by the KV-cache scheme choice, and that `uniform 8-bit` was both
more accurate and slightly lower-peak than the previous `turboquant` default.

## Summary

On a physical iPhone, `PaddleOCR-VL-1.5` decode shows a stable per-token
runtime growth pattern even after the obvious cache-level explanations are
removed.

The shortest reproducible symptom is:

- fresh one-chunk `photo_render_512`
- `decoderTail step=forward_total`
- roughly `+0.8 MB` active memory per token
- roughly `+340` Metal allocator resources per token

The same trend remains after ruling out several likely causes.

## Environment

- Project: `dart-inference`
- Model: `mlx-community/PaddleOCR-VL-1.5-8bit`
- Target device: physical iPhone, iOS 26.4
- App bundle: Flutter example harness
- Runtime backend: MLX Metal

## Minimal Repro Shape

1. Install the example app on a physical iPhone.
2. Seed the app `Documents/` with:
   - `paddle_ocr_vl_model/`
   - `paddle_ocr_vl_cases/`
3. Run a fresh one-chunk `photo_render_512` decode.
4. Enable decoder tail tracing and inspect:
   - `forward_total`
   - `sample_token`
   - allocator stats

The host-side workflow in this repo is currently:

```sh
tool/ios_pocr.sh --device <iphone-udid> --seed auto --max-launches 1
```

Relevant logs are written under `/tmp/ios_runtime_*`.

## Key Observations

Baseline one-chunk run:

- `forward_total`: offsets `517 -> 643`
- active memory delta: `+95.2 MB`
- allocator resource delta: `+43218`
- cache buffer count stays tiny

The current repo-side summary script emits the same numbers from the raw logs:

```sh
python3 tool/summarize_ios_runtime.py --out /tmp/ios_runtime_summary.md
```

That means the main runtime growth is not explained by logical KV cache bytes.

## Ruled Out

These experiments did **not** materially change the slope:

- forcing `MLX_MAX_OPS_PER_BUFFER=1`
- forcing `MLX_MAX_MB_PER_BUFFER=1`
- setting wired limit to `0`
- forcing per-token GPU synchronize
- TurboQuant state detach attempts
- TurboQuant cache compaction / smaller logical cache bytes
- last-layer TurboQuant special-case removal

The repo also records the current findings in:

- [paddle_ocr_vl_ios_runtime.md](/Users/binbinsh/Projects/Personal/dart-inference/doc/paddle_ocr_vl_ios_runtime.md)

## Why This Looks Runtime-Level

Current traces show:

- `pendingOut` remains small, typically `0..7`
- `cacheCount` remains small, typically low single digits to low tens
- `commits` increase rapidly, so command buffers are being committed
- `sync_per_token` does not cause a meaningful drop in active memory

That combination makes the remaining leading hypothesis:

> decoder forward is creating many small Metal resources per token, and those
> resources remain live long enough to keep process footprint climbing, even
> though cache bytes, cache count, and pending output/fence tracking remain
> small.

## Artifacts

Current repo-side artifacts:

- Summary script:
  [summarize_ios_runtime.py](/Users/binbinsh/Projects/Personal/dart-inference/tool/summarize_ios_runtime.py)
- Latest generated summary:
  `/tmp/ios_runtime_summary.md`
- Example logs:
  - `/tmp/ios_runtime_probe2/launch_1/photo_render_512/live.log`
  - `/tmp/ios_runtime_streamstats/launch_1/photo_render_512/live.log`
  - `/tmp/ios_runtime_cachecount/launch_1/photo_render_512/live.log`
  - `/tmp/ios_runtime_sync_trace/launch_1/photo_render_512/live.log`
  - `/tmp/ios_runtime_wired_override/launch_1/photo_render_512/live.log`

## Practical Mitigation In Product

The only mitigation that has been fully validated on-device so far is:

- checkpoint + resume across launches

That avoids long single-process decode runs, but it is a product-level
workaround, not a runtime fix.
