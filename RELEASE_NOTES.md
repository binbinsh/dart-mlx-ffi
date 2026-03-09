# Release Notes

## 26.0308.1557

### Highlights

- Added a full Apple-platform MLX package layout with vendored native sources,
  raw generated bindings, and bundled native build hooks.
- Built a broad stable Dart API over the MLX C surface for arrays, tensor
  algebra, random, FFT, linear algebra, quantization, convolutions, streams,
  distributed wrappers, and in-memory IO.
- Added stable in-memory IO abstractions with `MlxBytesReader` and
  `MlxBytesWriter`.
- Added a tiny model-style Flutter example demo with device, stream,
  distributed, and bytes IO sections.
- Added Flutter example builds for macOS and iOS simulator.

### Stable Dart Surface

- Arrays and constructors: `zeros`, `ones`, `full`, `arange`,
  `zerosLike/onesLike/fullLike`
- Tensor/indexing: `take`, `gather`, `slice`, `einsum`, `tensordot`,
  `meshgrid`, `scatter`, `partition`, `diag`, `kron`
- Math and scans: `softmax`, `logSumExp`, `linspace`, `outer`, `median`,
  `cumsum`, `cumprod`, `cummax`, `cummin`, `trace`, `eye`, `tri`
- Random: `uniform`, `normal`, `bernoulli`, `categorical`, `permutation`,
  `gumbel`, `laplace`, `randint`, `multivariateNormal`
- Linalg: `qr`, `eig`, `eigh`, `svd`, `lu`, `luFactor`, `pinv`,
  `solveTriangular`, `norm`
- Quantization: `quantize`, `dequantize`, `quantizedMatmul`, `qqmm`,
  `gatherQmm`, `toFp8`, `fromFp8`
- Runtime/system: `MlxDevice`, `MlxStream`, distributed group wrappers,
  memory and Metal helpers
- IO: file IO, stream-aware IO, bytes IO, safetensors bytes IO, reader/writer
  object wrappers

### Verification

- `dart analyze`
- `dart test`
- `flutter build macos --debug`
- `flutter build ios --simulator --debug`
- `dart pub publish --dry-run`
- Hand-written `lib/` coverage is around `97.26%`

### Notes

- macOS builds use Metal when the local Metal toolchain is available.
- iOS simulator builds intentionally fall back to `MLX_BUILD_METAL=OFF`.
- The raw layer remains available through `package:dart_mlx_ffi/raw.dart` for
  the full MLX C API.
