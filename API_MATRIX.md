# API Matrix

This matrix describes the current high-level Dart coverage relative to the raw
MLX C binding surface.

## Stable High-Level

| Area | Status | Notes |
| --- | --- | --- |
| Arrays | Stable | Constructors, dtype inspection, shape, copy-out helpers |
| Tensor algebra | Stable | `take`, `gather`, `slice`, `einsum`, `tensordot`, `meshgrid`, `diag`, `kron` |
| Tensor updates | Stable | `scatter*`, `putAlongAxis`, `maskedScatter`, `sliceUpdate*` |
| Shape helpers | Stable | `reshape`, `transpose`, `transposeAxes`, `flatten`, `unflatten`, `moveaxis`, `swapaxes`, `tile`, `pad` |
| Numeric helpers | Stable | `linspace`, `outer`, `isClose`, logical ops, `nanToNum`, `divmod` |
| Scan helpers | Stable | `cumsum`, `cumprod`, `cummax`, `cummin`, `logcumsumexp`, `trace`, `eye`, `tri`, `tril`, `triu`, windows |
| Random | Stable | uniform/normal plus categorical, permutation, Gumbel, Laplace, randint, multivariate normal |
| FFT | Stable | `fft`, `ifft`, `rfft`, `irfft`, `fft2`, `fftn`, `rfft2`, `rfftn`, `irfft2`, `irfftn`, shifts |
| Linear algebra | Stable | `inv`, `solve`, `qr`, `eig`, `eigh`, `svd`, `pinv`, `lu`, `luFactor`, `solveTriangular`, norms, cross |
| Quantization | Stable | `quantize`, `dequantize`, `quantizedMatmul`, `qqmm`, `gatherQmm`, FP8 conversion |
| Convolution | Stable | `conv1d/2d/3d/general` and transpose variants |
| Streams | Stable | default stream lookup, stream creation, synchronization, equality |
| Device info | Stable | default device, type/index, info map, default device switching |
| Distributed | Stable wrapper | group creation, collectives, send/recv wrappers; real multi-rank validation still environment-dependent |
| IO files | Stable | load/save, safetensors load/save, stream-aware file load |
| IO bytes | Stable | bytes roundtrips and reusable reader/writer wrappers |
| Functions/transforms | Stable | callback-backed functions, compile, checkpoint, grad/JVP/VJP, custom VJP/JVP |
| Export/import | Stable | file export/import, kwargs export, incremental exporter |
| Fast kernels | Stable wrapper | fast ops plus custom Metal/CUDA kernel wrappers |

## Raw First

| Area | Status | Notes |
| --- | --- | --- |
| Full C API surface | Raw-only | import `package:dart_mlx_ffi/raw.dart` |
| Backend-specific long tail | Partial high-level | long-tail backend/system APIs still better accessed via raw layer |
| Distributed integration setup | Environment-dependent | launcher/runtime wiring is outside this package |

## Practical Reading

- If you want normal Dart-side MLX usage, use `package:dart_mlx_ffi/dart_mlx_ffi.dart`.
- If you need exact parity with the generated C surface, use
  `package:dart_mlx_ffi/raw.dart`.
