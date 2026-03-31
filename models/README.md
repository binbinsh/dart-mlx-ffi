# models

Reusable non-runtime model workflows live here.

Use this directory for artifacts and helpers that are useful outside a single
benchmark report, for example:

- exporting MLX snapshots into reusable `.mlxfn` bundles
- importing and executing exported bundles from Dart
- model-family-specific export helpers that are not tied to one report

Dart model implementations do not live here anymore. They now live under:

- [`../lib/src/models/`](../lib/src/models/)

The current layout is:

- [`common/`](common/) for generic helpers shared by multiple model families
- [`text_lm/`](text_lm/) for text-generation export helpers, including
  Hugging Face -> Unsloth MLX conversion wrappers

`benchmark/` still contains report-generation scripts, Python MLX reference
runners, and benchmark orchestration.
Those scripts should call into `lib/src/models/` for Dart runtime models and
`models/` for reusable export / import tooling.

Private-ANE-specific model code no longer lives in `models/`. It now lives
under [`../private_ane/`](../private_ane/), including:

- [`../private_ane/models/josie/`](../private_ane/models/josie/)
- [`../private_ane/models/qwen3_5_0_8b/`](../private_ane/models/qwen3_5_0_8b/)
- [`../private_ane/models/qwen3_asr/`](../private_ane/models/qwen3_asr/)
