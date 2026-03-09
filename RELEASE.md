# Release Checklist

Use this checklist before publishing a tag to pub.dev.

Current status snapshot:

- Stable high-level Dart API is implemented across the common MLX workflow
  surface.
- Local verification currently passes:
  - `dart analyze`
  - `dart test`
  - `flutter build macos --debug`
  - `flutter build ios --simulator --debug`
  - `dart pub publish --dry-run`
- Current dry run reports `0 warnings`.

## Version

- Confirm the version uses `YY.MMDD.HHMM`.
- Update [pubspec.yaml](/Users/binbinsh/Projects/Personal/dart-mlx-ffi/pubspec.yaml).
- Update [CHANGELOG.md](/Users/binbinsh/Projects/Personal/dart-mlx-ffi/CHANGELOG.md).
- Create a matching git tag `vYY.MMDD.HHMM`.

## Validation

- Run `dart analyze`.
- Run `dart test`.
- Run `dart test --coverage=coverage` if you want a fresh coverage snapshot.
- Run `cd example && flutter build macos --debug`.
- Run `cd example && flutter build ios --simulator --debug`.
- Run `dart run benchmark/tiny.dart --warmup=20 --iters=100`.
- Run `dart run tool/dist_smoke.dart` in environments where distributed is expected.
- Run `dart pub publish --dry-run`.

## Packaging

- Confirm [README.md](/Users/binbinsh/Projects/Personal/dart-mlx-ffi/README.md) reflects the current public API.
- Confirm [API_MATRIX.md](/Users/binbinsh/Projects/Personal/dart-mlx-ffi/API_MATRIX.md) matches the current stable surface.
- Confirm [RELEASE_NOTES.md](/Users/binbinsh/Projects/Personal/dart-mlx-ffi/RELEASE_NOTES.md) matches the version being published.
- Confirm [LICENSE](/Users/binbinsh/Projects/Personal/dart-mlx-ffi/LICENSE) is present.
- Confirm [.pubignore](/Users/binbinsh/Projects/Personal/dart-mlx-ffi/.pubignore) excludes only non-package material.
- Confirm generated and vendored files needed for native builds are included.

## GitHub

- Push the release commit to `github.com/binbinsh/dart-mlx-ffi`.
- Push the matching git tag.
- Verify GitHub Actions passed:
  - [ci.yml](/Users/binbinsh/Projects/Personal/dart-mlx-ffi/.github/workflows/ci.yml)
  - [publish.yml](/Users/binbinsh/Projects/Personal/dart-mlx-ffi/.github/workflows/publish.yml)

## pub.dev

- Verify trusted publishing is configured for `github.com/binbinsh/dart-mlx-ffi`.
- Confirm the publish workflow is triggered by the version tag.
- After release, check the rendered pub.dev page:
  - package description
  - README formatting
  - supported platforms
  - example rendering
