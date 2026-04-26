# AGENTS.md

## Versioning

- This package uses the version format `1.yyyy.commit-count`.
- The canonical release form is `1.2026.38` where:
  - `1` is the fixed package major version
  - `2026` is the 4-digit calendar year
  - `38` is the git commit count with no zero padding
- Update [`pubspec.yaml`](pubspec.yaml) and [`CHANGELOG.md`](CHANGELOG.md) together.
- Git tags must match the pubspec version and use the form `v<version>`, Example tag: `v1.2026.38`.

## File Size

- Hand-written source, test, and config files must stay under `1200` lines each. Generated files and vendored third-party files are exempt from the `1200`-line limit. When a file approaches the limit, split by module or responsibility instead of appending more code.
- File names must be short and understandable at a glance.
- Prefer names such as `array.dart`, `ops.dart`, `io.dart`, `bridge_ops.cpp`.
- Avoid meaningless sequence names such as `a.dart`, `tmp.dart`, `bridge_a.cpp`.

## Python Tooling

- Manage Python dependencies with `uv`.
- Keep Python dependency declarations in [`pyproject.toml`](pyproject.toml).
- Prefer `uv sync` to create/update the local environment and `uv run` to execute Python tooling.
- Prefer `uv add` and `uv remove` over `pip install` or ad-hoc virtualenvs.

## Native Runtime Workflow

- Dart must call native providers only through the Zig runtime ABI:
  - `Dart API -> dart_inference_runtime_* -> Zig -> private C/C++/ObjC++ libs`
- Do not reintroduce public raw `mlx-c` bindings or Dart-side per-op MLX FFI.
- For MLX snapshot preparation and benchmark inputs, use the repository's canonical conversion wrapper:
  - [`models/text_lm/convert_unsloth_mlx.py`](models/text_lm/convert_unsloth_mlx.py)
- Treat that wrapper as the default path for:
  - benchmark runs
  - parity checks
  - reproducible local evaluation
- MLX execution belongs behind Zig. Do not introduce alternative MLX conversion flows or ad-hoc manual converter invocations unless there is an explicit reason and that reason is documented in the change.

## Publishing

- Refresh the publish benchmark report before releasing:
  - `uv sync`
  - `uv run --no-project --with mlx-lm --with pillow --with mlx-vlm --with mlx-audio --with parakeet-mlx python benchmark/publish_report.py`
- Validate locally before release:
  - `dart analyze`
  - `dart test`
  - `dart pub publish --dry-run`
- Manual first publish:
  - `dart pub publish`
- GitHub Actions auto-publish can be enabled after the package exists on pub.dev.
  - In pub.dev package admin, enable publishing from GitHub Actions for this repository.
  - The release tag must match the package version format: `v<1.yyyy.commit-count>`.
  - Example: `v1.2026.38`
