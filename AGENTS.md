# AGENTS.md

## Versioning

- This package uses the version format `1.yyyy.commit-count`.
- The canonical release form is `1.2026.81` where:
  - `1` is the fixed package major version
  - `2026` is the 4-digit calendar year
  - `81` is the git commit count with no zero padding
- Update [`pubspec.yaml`](pubspec.yaml) and [`CHANGELOG.md`](CHANGELOG.md) together.
- Git tags must match the pubspec version and use the form `v<version>`, Example tag: `v1.2026.81`.

## File Size

- Design files and functions deliberately: hand-written source, test, and config files must stay under `1200` lines each, and function names and file names must be concise, clear, and easy to understand. Generated files and vendored third-party files are exempt from the `1200`-line limit. When a file approaches the limit, split by module or responsibility instead of appending more code.
- Prefer names such as `array.dart`, `ops.dart`, `io.dart`, `bridge_ops.cpp`.
- Avoid meaningless sequence names such as `a.dart`, `tmp.dart`, `bridge_a.cpp`.

## Python Tooling

- Manage Python dependencies with `uv`.
- Keep Python dependency declarations in [`pyproject.toml`](pyproject.toml).
- Prefer `uv sync` to create/update the local environment and `uv run` to execute Python tooling.
- Prefer `uv add` and `uv remove` over `pip install` or ad-hoc virtualenvs.

## Native Runtime Workflow

- Dart native providers must use Dart helpers and the Dart FFI/C++ runtime ABI:
  - `Dart API -> Dart FFI/dinf_* -> C/C++/ObjC++ libs`
- Do not add Zig source or Zig build steps back to the package.
- Public MLX stable/raw APIs are part of this package surface. Keep them routed
  through the package-owned Dart FFI/native MLX bridge and avoid ad-hoc external
  converter or runtime paths.
- For MLX snapshot preparation and benchmark inputs, use the repository's canonical conversion wrapper:
  - [`../../../models/dart/tool/text_lm/convert_unsloth_mlx.py`](../../../models/dart/tool/text_lm/convert_unsloth_mlx.py)
- Treat that wrapper as the default path for:
  - benchmark runs
  - parity checks
  - reproducible local evaluation
- Do not introduce alternative MLX conversion flows or ad-hoc manual converter invocations unless there is an explicit reason and that reason is documented in the change.

## Distribution

- This is a private `airuntime-dev` module and must keep `publish_to: none` while
  it depends on sibling model contracts by path.
- Validate changes from this module directory with `dart analyze` and
  `dart test`.
- Keep the standalone repository as the source-history mirror used by the
  `airuntime-dev` submodule; release work happens from its `main` branch only.
