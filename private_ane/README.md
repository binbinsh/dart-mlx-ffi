# private_ane

`private_ane/` is the project-local module boundary for all private Apple Neural
Engine work.

It is organized separately from the base `dart_mlx_ffi` MLX FFI surface so the
private runtime, generic probes, and model-specific implementations live
together.

Layout:

- `shared/`: generic private-ANE runtime helpers and reference benchmarks
- `models/`: model-specific private-ANE implementations

Current model directories:

- `models/qwen3_5_0_8b/`
- `models/qwen3_asr/`
- `models/josie/`
