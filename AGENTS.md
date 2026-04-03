# AGENTS.md

## Versioning

- This package uses the version format `YY.MDD.commit_count`.
- The canonical release form is `26.325.7` where:
  - `26` is the 2-digit calendar year
  - `325` is `month + day` with no unnecessary leading zeroes
  - `7` is the git commit count with no zero padding
- Update [`pubspec.yaml`](pubspec.yaml) and [`CHANGELOG.md`](CHANGELOG.md) together.
- Git tags must match the pubspec version and use the form `v<version>`.
- Example tag: `v26.325.7`.

## File Size

- Hand-written source, test, and config files must stay under `800` lines each.
- File names must be short and understandable at a glance.
- Prefer names such as `array.dart`, `ops.dart`, `io.dart`, `bridge_ops.cpp`.
- Avoid meaningless sequence names such as `a.dart`, `tmp.dart`, `bridge_a.cpp`.
- When a file approaches the limit, split by module or responsibility instead of appending more code.
- Generated files and vendored third-party files are exempt from the `800`-line limit:
  - [`lib/src/raw/mlx_bindings.g.dart`](lib/src/raw/mlx_bindings.g.dart)
  - anything under [`third_party/`](third_party/)

## Python Tooling

- Manage Python dependencies with `uv`.
- Keep Python dependency declarations in [`pyproject.toml`](pyproject.toml).
- Prefer `uv sync` to create/update the local environment and `uv run` to execute Python tooling.
- Prefer `uv add` and `uv remove` over `pip install` or ad-hoc virtualenvs.

## MLX Workflow

- For MLX snapshot preparation and benchmark inputs, use the repository's canonical conversion wrapper:
  - [`models/text_lm/convert_unsloth_mlx.py`](models/text_lm/convert_unsloth_mlx.py)
- Treat that wrapper as the default path for:
  - benchmark runs
  - parity checks
  - reproducible local evaluation
- Do not introduce alternative MLX conversion flows or ad-hoc manual converter invocations unless there is an explicit reason and that reason is documented in the change.

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
  - The release tag must match the package version format: `v<YY.MDD.commit_count>`.
  - Example: `v26.325.7`
