# Commit #13 — manifest patch for the hybrid PaddleOCR-VL runner

This document captures the edits the user must apply to
`models/shared/dart_contracts/lib/manifest.dart`
**after** the decoder-only MLX 4-bit snapshot produced by
`models/dart/tool/text_lm/convert_paddle_ocr_vl_decoder.sh`
has been uploaded to Hugging Face.

`manifest.dart` is **not edited in this commit** because that file already
carries unrelated WIP changes from earlier in the hybrid OCR refactor
(issue #1).

## Pre-flight checklist

1. Conversion has produced
   `~/snapshots/paddleocr-vl-ernie-mlx-4bit/` with `config.json`,
   `*.safetensors`, and tokenizer files. The script's final block
   verifies via `mlx_lm.load`.
2. The snapshot has been pushed to a Hugging Face repo.
   Recommended namespace: `<USER>/paddleocr-vl-ernie-mlx-4bit` (the
   `mlx-community` mirror should land here too once parity is
   confirmed).

## Edits

The current state of the `paddle_ocr_vl` `ModelSpec` lives at
`models/shared/dart_contracts/lib/manifest.dart` (line numbers may have shifted
under WIP edits).

### 1. `mlxRepo` — point at the new decoder-only snapshot

```diff
       ..._runtimeArtifacts(
         id: 'paddle_ocr_vl',
         sourceModel: 'PaddlePaddle/PaddleOCR-VL-1.5',
-        mlxRepo: 'mlx-community/PaddleOCR-VL-1.5-8bit',
+        mlxRepo: '<USER_HF_NAMESPACE>/paddleocr-vl-ernie-mlx-4bit', // TODO(user)
         mlxArtifact: '.',
         onnxRepo: 'lbm364dl/PaddleOCR-VL-1.5-ONNX',
         onnxArtifact: 'onnx/decoder_model_merged.onnx',
       ),
```

The old `mlx-community/PaddleOCR-VL-1.5-8bit` carried `visual.*`
weights (~385 MB) that the hybrid runner immediately drops on load.
The new snapshot is decoder-only and 4-bit, saving both download and
GPU memory.

### 2. CoreML `runner` metadata — switch to the hybrid runner

```diff
         metadata: const {
           'source': 'superplanner-r2',
           'modelId': 'paddle_ocr_vl',
           'snapshot': 'paddleocr-vl-coreml',
           'manifestUrl': 'https://superplanner.ai/mise/models/manifest.json',
-          'runner': 'PaddleOcrVlCoremlRunner.load',
+          'runner': 'PaddleOcrVlHybridRunner.load',
           'tokenizerRequired': true,
         },
```

`PaddleOcrVlHybridRunner` is in
`models/paddle_ocr_vl/dart/hybrid_runner.dart`.
Note that commit #11 (parallel) deletes the legacy
`PaddleOcrVlCoremlRunner`, so leaving this string at `*.CoremlRunner.load`
will reference a non-existent class.

### 3. `requiredTags` — include both engines

The hybrid runner requires both a CoreML bundle and an MLX snapshot to
be on disk before `load` can succeed. The current single-tag list is
no longer accurate.

```diff
     requiredFiles: const ['pipeline.json', 'tokenizer.json'],
-    requiredTags: const ['coreml'],
+    requiredTags: const ['coreml', 'mlx'],
```

### 4. `requiredFiles` — leave unchanged

**Recommendation: do NOT add `config.json` to `requiredFiles`.**

Evidence (from
`models/paddle_ocr_vl/dart/hybrid_runner.dart`,
`models/paddle_ocr_vl/dart/runner_load.dart`,
and `models/shared/dart/tensor_map.dart`):

| File on disk                          | Read by                                   | Side                  |
|---------------------------------------|-------------------------------------------|-----------------------|
| `pipeline.json`                       | `CoremlPipelineManifest.loadFile`         | CoreML bundle dir     |
| `vision_embed.mlpackage` (and others) | `CoremlSession`                           | CoreML bundle dir     |
| `tokenizer.json`                      | caller (prompt encoding)                  | CoreML bundle dir     |
| `config.json`                         | `PaddleOcrVlConfig.fromSnapshot`          | **MLX snapshot dir**  |
| `*.safetensors`                       | `loadTensorMap` -> `mx.io.loadSafetensors`| **MLX snapshot dir**  |

`requiredFiles` in this `ModelSpec` is documented (see
`docs/manifest_files.md` — if absent, see `model_spec.dart` doc-comments)
to refer to **the primary artifact directory** for the model, which for
PaddleOCR-VL is the CoreML bundle (its `path: 'pipeline.json'`). The
MLX-side `config.json` lives in a *different* artifact directory
(`RuntimeEngine.mlx`) and is implicitly required by virtue of that
artifact existing — it does not belong in this list.

If a future change starts validating per-engine files, add `config.json`
to the MLX-engine artifact's required-files list, not to the top-level
`requiredFiles`.

`tokenizer.json` is currently in `requiredFiles` because the CoreML
side ships a tokenizer, and the hybrid runner still needs it
caller-side for prompt encoding — leave it.

### 5. `mlxArtifact` — leave at `'.'`

The decoder-only snapshot is a directory, and `loadTensorMap` accepts
that directly. No change.

## Verification after applying

1. `dart analyze` — should pass.
2. `dart test test/manifest_test.dart` — should pass (note: this file
   itself has unrelated WIP edits in commit #11/#12; merge ordering may
   matter).
3. End-to-end smoke via the current `models/validation/runtime/` matrix or
   model-specific Dart smoke entry point against the
   newly-published HF repo.

## Architectures patch reference

The conversion script patches the snapshot's `config.json`:

- `architectures`: `["PaddleOCRVLForConditionalGeneration"]` →
  `["Ernie4_5ForCausalLM"]`
- `model_type`: `"paddleocr_vl"` → `"ernie4_5"`
- drops `vision_config`, `image_token_id`, `video_token_id`,
  `vision_start_token_id`, `vision_end_token_id`, `auto_map`.

Evidence for `Ernie4_5ForCausalLM` / `ernie4_5`:
the upstream ERNIE-4.5 LM that PaddleOCR-VL-1.5 is built on
(`baidu/ERNIE-4.5-0.3B-PT`) declares exactly these values in its
`config.json` on the HF Hub. Confirmed live during commit #13:

```text
baidu/ERNIE-4.5-0.3B-PT      -> ['Ernie4_5ForCausalLM']  model_type=ernie4_5
baidu/ERNIE-4.5-0.3B-Base-PT -> ['Ernie4_5ForCausalLM']  model_type=ernie4_5
```

If the HF transformers version installed at conversion time does not
yet expose `Ernie4_5ForCausalLM`, fall back via the wrapper's
`--architecture` argument:

```sh
uv run python models/dart/tool/text_lm/patch_decoder_config.py \
  --config ~/snapshots/paddleocr-vl-ernie-mlx-4bit/config.json \
  --architecture ErnieForCausalLM
```

The Dart `PaddleOcrVlRunner.load` does **not** read the
`architectures` field — it keys on tensor name prefixes
(`language_model.model.*`) — so the patched value only matters for
external HF/MLX loaders used to verify the snapshot.
