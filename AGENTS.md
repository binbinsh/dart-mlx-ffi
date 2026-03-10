# AGENTS.md

## Versioning

- This package uses the canonical pub.dev-friendly version format `YY.MDD.HHMM`.
- The middle segment is `month + day` without unnecessary leading zeroes.
- Example: `26.308.1557` means `2026-03-08 15:57`.
- Update [`pubspec.yaml`](/Users/binbinsh/Projects/Personal/dart-mlx-ffi/pubspec.yaml) and [`CHANGELOG.md`](/Users/binbinsh/Projects/Personal/dart-mlx-ffi/CHANGELOG.md) together.
- Git tags must match the pubspec version and use the form `vYY.MDD.HHMM`.
- Example tag: `v26.308.1557`.

## File Size

- Hand-written source, test, and config files must stay under `800` lines each.
- File names must be short and understandable at a glance.
- Prefer names such as `array.dart`, `ops.dart`, `io.dart`, `bridge_ops.cpp`.
- Avoid meaningless sequence names such as `a.dart`, `tmp.dart`, `bridge_a.cpp`.
- When a file approaches the limit, split by module or responsibility instead of appending more code.
- Generated files and vendored third-party files are exempt from the `800`-line limit:
  - [`lib/src/raw/mlx_bindings.g.dart`](/Users/binbinsh/Projects/Personal/dart-mlx-ffi/lib/src/raw/mlx_bindings.g.dart)
  - anything under [`third_party/`](/Users/binbinsh/Projects/Personal/dart-mlx-ffi/third_party)

## Python Tooling

- Manage Python dependencies with `uv`.
- Keep Python dependency declarations in [`pyproject.toml`](/Users/binbinsh/Projects/Personal/dart-mlx-ffi/pyproject.toml).
- Prefer `uv sync` to create/update the local environment and `uv run` to execute Python tooling.
- Prefer `uv add` and `uv remove` over `pip install` or ad-hoc virtualenvs.

## Publishing

- Refresh the publish benchmark report before releasing:
  - `uv sync`
  - `uv run python benchmark/publish_report.py`
- Validate locally before release:
  - `dart analyze`
  - `dart test`
  - `dart pub publish --dry-run`
- Manual first publish:
  - `dart pub publish`
- GitHub Actions auto-publish can be enabled after the package exists on pub.dev.
  - In pub.dev package admin, enable publishing from GitHub Actions for this repository.
  - The release tag must match the package version format: `vYY.MDD.HHMM`.
  - Example: `v26.310.2016`
