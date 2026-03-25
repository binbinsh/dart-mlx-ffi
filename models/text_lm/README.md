# text lm

Helpers in this directory target text-generation models loaded through
`mlx_lm`.

Current entrypoints:

- [`export_bundle.py`](export_bundle.py): export a local `mlx-lm` snapshot into
  a shapeless `.mlxfn` bundle plus example `input_ids`

Typical flow:

1. Download or prepare a local MLX snapshot.
2. Export a reusable next-token function with [`export_bundle.py`](export_bundle.py).
3. Run the exported artifact from Dart with
   [`../common/import_run.dart`](../common/import_run.dart).
4. Use `benchmark/` scripts only when you want parity or performance reports.
