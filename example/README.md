# dart_mlx_ffi example

This Flutter example is now a minimal sanity demo for the package.

Run it with:

```sh
flutter run -d macos
```

or:

```sh
flutter run -d ios
```

The example app shows:

- MLX version
- default device info
- Metal availability
- current memory stats
- basic `add`
- basic `matmul`

It is intentionally small and no longer contains the internal iPhone
profiling harness that was used during PaddleOCR-VL runtime tuning.
