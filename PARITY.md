# Python Parity

This repository now includes a cross-runtime parity harness for deterministic,
side-effect-free MLX APIs shared by `dart-mlx-ffi` and Python MLX.

## Scope

Covered groups:

- `arith`
- `unary_reduce`
- `tensor`
- `index_extra`
- `misc`
- `scan`
- `conv`
- `linalg`
- `fast`
- `quant`
- `random`

Current coverage:

- `11` groups
- `114` output checks
- result: `0` failures on the current machine

The report is written to:

- [benchmark/out/parity_report.json](/Users/binbinsh/Projects/Personal/dart-mlx-ffi/benchmark/out/parity_report.json)

## Exclusions

This parity harness intentionally excludes:

- file and byte IO
- export/import and closure APIs
- distributed/system/runtime helpers
- custom kernel wrappers
- GPU-unsupported Python linalg ops such as `inv`, `solve`, `pinv`, `qr`,
  `eigh`, `svd`, and `cholesky` in this environment

## Commands

Generate the Python reference:

```sh
.venv/bin/python benchmark/parity_py.py > benchmark/out/parity_python.json
```

Generate the Dart result:

```sh
dart --packages=.dart_tool/package_config.json benchmark/parity_dart.dart > benchmark/out/parity_dart.json
```

Compare both outputs:

```sh
python3 benchmark/parity_compare.py \
  --python-json benchmark/out/parity_python.json \
  --dart-json benchmark/out/parity_dart.json \
  --output benchmark/out/parity_report.json
```

Run a subset of groups:

```sh
.venv/bin/python benchmark/parity_py.py --groups arith,unary_reduce > benchmark/out/parity_python_core.json
dart --packages=.dart_tool/package_config.json benchmark/parity_dart.dart --groups=arith,unary_reduce > benchmark/out/parity_dart_core.json
python3 benchmark/parity_compare.py --python-json benchmark/out/parity_python_core.json --dart-json benchmark/out/parity_dart_core.json --output benchmark/out/parity_report_core.json
```
