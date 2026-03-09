from __future__ import annotations

import argparse
import json
from typing import Any

import mlx.core as mx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", default=None)
    args = parser.parse_args()
    results = {
        "runtime": "python_mlx",
        "mlx_version": getattr(mx, "__version__", "unknown"),
        "cases": parity_cases(),
    }
    if args.groups:
        selected = set(args.groups.split(","))
        results["cases"] = {
            key: value for key, value in results["cases"].items() if key in selected
        }
    print(json.dumps(results))


def parity_cases() -> dict[str, Any]:
    cases: dict[str, Any] = {}

    a22 = f32([2, 2], 3)
    b22 = f32([2, 2], 5)
    a23 = f32([2, 3], 7)
    b23 = f32([2, 3], 9)
    a32 = f32([3, 2], 11)
    v4 = f32([4], 13)
    v3 = f32([3], 15)
    bool22 = bvec([2, 2], 17)
    idx2 = ivec([2], 19, 3)
    idx22 = ivec([2, 2], 21, 3)

    cases["arith"] = {
        "add": enc(a22 + b22),
        "subtract": enc(a22 - b22),
        "multiply": enc(a22 * b22),
        "divide": enc(a22 / (abs_pos(b22, 23))),
        "matmul": enc(a23 @ a32),
    }

    exp_input = f32([4], 25)
    log_input = pos([4], 27)
    trig_input = f32([4], 29)
    reduce_input = f32([2, 3], 31)
    topk_input = f32([6], 33)
    cases["unary_reduce"] = {
        "abs": enc(mx.abs(a22)),
        "negative": enc(-a22),
        "exp": enc(mx.exp(exp_input)),
        "log": enc(mx.log(log_input)),
        "sin": enc(mx.sin(trig_input)),
        "cos": enc(mx.cos(trig_input)),
        "equal": enc(mx.equal(a22, a22)),
        "where": enc(mx.where(bool22, a22, b22)),
        "sum_all": enc(mx.sum(reduce_input)),
        "sum_axis": enc(mx.sum(reduce_input, axis=1, keepdims=True)),
        "mean_all": enc(mx.mean(reduce_input)),
        "mean_axis": enc(mx.mean(reduce_input, axis=0)),
        "logsumexp": enc(mx.logsumexp(reduce_input, axis=1)),
        "softmax": enc(mx.softmax(reduce_input, axis=1)),
        "topk": enc(mx.topk(topk_input, 3)),
    }

    broadcast_in = f32([2], 35)
    expand_in = f32([2], 37)
    clip_in = f32([6], 39)
    sort_in = f32([6], 41)
    cases["tensor"] = {
        "concatenate": enc(mx.concatenate([a22, b22], axis=0)),
        "stack": enc(mx.stack([a22, b22], axis=0)),
        "broadcast_to": enc(mx.broadcast_to(broadcast_in, [2, 2])),
        "expand_dims": enc(mx.expand_dims(expand_in, 0)),
        "squeeze": enc(mx.squeeze(mx.expand_dims(expand_in, 0))),
        "clip": enc(mx.clip(clip_in, a_min=-0.5, a_max=0.5)),
        "minimum": enc(mx.minimum(a22, b22)),
        "maximum": enc(mx.maximum(a22, b22)),
        "argmax": enc(mx.argmax(a22)),
        "argmin_axis": enc(mx.argmin(a22, axis=1, keepdims=True)),
        "sort": enc(mx.sort(sort_in)),
        "argsort": enc(mx.argsort(sort_in)),
        "flatten": enc(mx.flatten(a22)),
        "moveaxis": enc(mx.moveaxis(mx.array(f32_list(8, 43), dtype=mx.float32).reshape(2, 2, 2), 0, 2)),
        "swapaxes": enc(mx.swapaxes(mx.array(f32_list(8, 45), dtype=mx.float32).reshape(2, 2, 2), 0, 1)),
        "transpose_axes": enc(mx.transpose(mx.array(f32_list(8, 47), dtype=mx.float32).reshape(2, 2, 2), [2, 0, 1])),
        "tile": enc(mx.tile(a22, [2, 1])),
        "unflatten": enc(mx.unflatten(mx.arange(0, 8, 1, dtype=mx.float32), axis=0, shape=[2, 4])),
    }

    take_input = f32([2, 3], 49)
    slice_input = f32([6], 51)
    diag_vec = f32([3], 53)
    diag_mat = f32([3, 3], 55)
    kron_a = f32([2], 57)
    kron_b = f32([3], 59)
    mesh_a = f32([2], 61)
    mesh_b = f32([3], 63)
    partition_in = f32([6], 65)
    cases["index_extra"] = {
        "take": enc(mx.take(slice_input, idx2)),
        "take_axis": enc(mx.take(take_input, idx2, axis=1)),
        "take_along_axis": enc(mx.take_along_axis(take_input, idx22, axis=1)),
        "slice": enc(slice_input[1:6:2]),
        "einsum": enc(mx.einsum("ij,jk->ik", a23, a32)),
        "tensordot": enc(mx.tensordot(a23, a32, axes=1)),
        "diag": enc(mx.diag(diag_vec)),
        "diagonal": enc(mx.diagonal(diag_mat)),
        "kron": enc(mx.kron(kron_a, kron_b)),
        "meshgrid0": enc(mx.meshgrid(mesh_a, mesh_b, indexing="ij")[0]),
        "meshgrid1": enc(mx.meshgrid(mesh_a, mesh_b, indexing="ij")[1]),
        "partition": enc(mx.partition(partition_in, 2)),
    }

    misc_in = f32([4], 67)
    misc_pos = pos([4], 69)
    round_in = f32([4], 71)
    erfinv_in = mx.array([-0.75, -0.25, 0.25, 0.75], dtype=mx.float32)
    special = mx.array([mx.nan, mx.inf, -mx.inf, 1.0], dtype=mx.float32)
    cases["misc"] = {
        "floor_divide": enc(mx.floor_divide(pos([4], 73), mx.full([4], 1.5))),
        "logaddexp": enc(mx.logaddexp(v4, v4)),
        "inner": enc(mx.inner(v4, v4)),
        "floor": enc(mx.floor(misc_in)),
        "sqrt": enc(mx.sqrt(misc_pos)),
        "rsqrt": enc(mx.rsqrt(misc_pos)),
        "square": enc(mx.square(misc_in)),
        "reciprocal": enc(mx.reciprocal(misc_pos)),
        "sigmoid": enc(mx.sigmoid(misc_in)),
        "degrees": enc(mx.degrees(misc_in)),
        "radians": enc(mx.radians(misc_in)),
        "expm1": enc(mx.expm1(misc_in)),
        "erf": enc(mx.erf(misc_in)),
        "erfinv": enc(mx.erfinv(erfinv_in)),
        "log1p": enc(mx.log1p(misc_pos)),
        "log2": enc(mx.log2(misc_pos)),
        "log10": enc(mx.log10(misc_pos)),
        "round": enc(mx.round(round_in, decimals=2)),
        "linspace": enc(mx.linspace(-1.0, 1.0, 7)),
        "outer": enc(mx.outer(v3, v4[:3])),
        "isclose": enc(mx.isclose(a22, a22 + 1e-6)),
        "repeat": enc(mx.repeat(v3, 2)),
        "roll": enc(mx.roll(v4, 1)),
        "median": enc(mx.median(misc_in)),
        "nan_to_num": enc(mx.nan_to_num(special, nan=0.0, posinf=9.0, neginf=-9.0)),
        "divmod_q": enc(mx.divmod(mx.array([5.0, 7.0], dtype=mx.float32), mx.array([2.0, 2.0], dtype=mx.float32))[0]),
        "divmod_r": enc(mx.divmod(mx.array([5.0, 7.0], dtype=mx.float32), mx.array([2.0, 2.0], dtype=mx.float32))[1]),
    }

    scan_input = f32([6], 75)
    tri_input = f32([3, 3], 77)
    cases["scan"] = {
        "cumsum": enc(mx.cumsum(scan_input, axis=0)),
        "cumprod": enc(mx.cumprod(pos([6], 79), axis=0)),
        "cummax": enc(mx.cummax(scan_input, axis=0)),
        "cummin": enc(mx.cummin(scan_input, axis=0)),
        "logcumsumexp": enc(mx.logcumsumexp(scan_input, axis=0)),
        "eye": enc(mx.eye(3, 3, 0)),
        "identity": enc(mx.identity(3)),
        "tri": enc(mx.tri(3, 3, 0)),
        "tril": enc(mx.tril(tri_input)),
        "triu": enc(mx.triu(tri_input)),
        "trace": enc(mx.trace(tri_input)),
    }

    conv1_in = f32([1, 6, 2], 81)
    conv1_w = f32([4, 3, 2], 83)
    conv2_in = f32([1, 4, 4, 2], 85)
    conv2_w = f32([3, 3, 3, 2], 87)
    conv3_in = f32([1, 3, 3, 3, 2], 89)
    conv3_w = f32([2, 2, 2, 2, 2], 91)
    cases["conv"] = {
        "conv1d": enc(mx.conv1d(conv1_in, conv1_w, stride=1, padding=1)),
        "conv2d": enc(mx.conv2d(conv2_in, conv2_w, stride=(1, 1), padding=(1, 1))),
        "conv3d": enc(mx.conv3d(conv3_in, conv3_w, stride=(1, 1, 1), padding=(1, 1, 1))),
        "conv_transpose1d": enc(mx.conv_transpose1d(conv1_in, conv1_w, stride=1, padding=1)),
        "conv_transpose2d": enc(mx.conv_transpose2d(conv2_in, conv2_w, stride=(1, 1), padding=(1, 1))),
    }

    cases["linalg"] = {
        "norm": enc(mx.linalg.norm(v4)),
        "cross": enc(mx.linalg.cross(f32([3], 99), f32([3], 101))),
    }

    ln_in = f32([1, 4], 103)
    ln_w = mx.ones([4], dtype=mx.float32)
    ln_b = mx.zeros([4], dtype=mx.float32)
    rope_in = f32([1, 1, 4, 64], 105)
    q = f32([1, 2, 4, 64], 107)
    k = f32([1, 2, 4, 64], 109)
    v = f32([1, 2, 4, 64], 111)
    cases["fast"] = {
        "layer_norm": enc(mx.fast.layer_norm(ln_in, ln_w, ln_b, 1e-5)),
        "rms_norm": enc(mx.fast.rms_norm(ln_in, ln_w, 1e-5)),
        "rope": enc(
            mx.fast.rope(
                rope_in,
                64,
                traditional=False,
                base=1000000.0,
                scale=1.0,
                offset=0,
            )
        ),
        "sdpa": enc(mx.fast.scaled_dot_product_attention(q, k, v, scale=1 / (64 ** 0.5))),
    }

    q_weights = f32([4, 32], 113)
    q_input = f32([2, 32], 115)
    qw, qs, qb = mx.quantize(q_weights, group_size=32, bits=8, mode="affine")
    qq_input = f32([1, 16], 117)
    qq_weights = f32([2, 16], 119)
    cases["quant"] = {
        "dequantize": enc(mx.dequantize(qw, scales=qs, biases=qb, group_size=32, bits=8, mode="affine", dtype=mx.float32)),
        "quantized_matmul": enc(mx.quantized_matmul(q_input, qw, scales=qs, biases=qb, transpose=True, group_size=32, bits=8, mode="affine")),
        "gather_qmm": enc(mx.gather_qmm(q_input, qw, qs, qb, transpose=True, group_size=32, bits=8, mode="affine")),
    }

    p = mx.array([0.2, 0.7], dtype=mx.float32)
    logits = mx.array([[1.0, 2.0, 3.0]], dtype=mx.float32)
    perm_input = mx.array([1, 2, 3, 4], dtype=mx.int32)
    key = mx.random.key(123)
    first, second = mx.random.split(key)
    mx.random.seed(1001)
    seed_uniform = mx.random.uniform(shape=(2, 2), low=-1.0, high=1.0)
    mx.random.seed(1002)
    seed_normal = mx.random.normal(shape=(2, 2), loc=0.0, scale=1.0)
    cases["random"] = {
        "key": enc(key),
        "split_first": enc(first),
        "split_second": enc(second),
        "seed_uniform": enc(seed_uniform),
        "seed_normal": enc(seed_normal),
        "bernoulli": enc(mx.random.bernoulli(p, shape=(2,), key=mx.random.key(2001))),
        "categorical": enc(mx.random.categorical(logits, axis=-1, num_samples=4, key=mx.random.key(2002))),
        "permutation": enc(mx.random.permutation(perm_input, axis=0, key=mx.random.key(2003))),
        "permutation_arange": enc(mx.random.permutation(mx.arange(0, 4, 1, dtype=mx.int32), axis=0, key=mx.random.key(2004))),
        "gumbel": enc(mx.random.gumbel(shape=(2, 2), key=mx.random.key(2005))),
        "laplace": enc(mx.random.laplace(shape=(2, 2), loc=1.0, scale=0.5, key=mx.random.key(2006))),
        "randint": enc(mx.random.randint(0, 10, shape=(4,), key=mx.random.key(2007))),
    }

    return cases


def enc(x: Any) -> dict[str, Any]:
    if not isinstance(x, mx.array):
        raise TypeError(f"Expected mlx.array, got {type(x)!r}")
    y = x
    if x.dtype in (mx.float16, mx.bfloat16):
        y = x.astype(mx.float32)
    return {"shape": list(y.shape), "values": flatten(y.tolist())}


def flatten(value: Any) -> list[Any]:
    if isinstance(value, list):
        out: list[Any] = []
        for item in value:
            out.extend(flatten(item))
        return out
    return [value]


def numel(shape: list[int]) -> int:
    n = 1
    for dim in shape:
        n *= dim
    return n


def f32_list(count: int, seed: int, divisor: int = 64) -> list[float]:
    result: list[float] = []
    for index in range(count):
        numerator = ((index * (seed * 2 + 1) + seed * 7 + 13) % 257) - 128
        result.append(numerator / divisor)
    return result


def f32(shape: list[int], seed: int, divisor: int = 64):
    return mx.array(f32_list(numel(shape), seed, divisor), dtype=mx.float32).reshape(shape)


def pos(shape: list[int], seed: int):
    vals = [abs(v) + 0.25 for v in f32_list(numel(shape), seed, 32)]
    return mx.array(vals, dtype=mx.float32).reshape(shape)


def abs_pos(x, seed: int):
    return mx.abs(x) + mx.full(x.shape, 0.25, dtype=mx.float32)


def ivec(shape: list[int], seed: int, mod: int):
    vals = [((index * (seed + 3)) + seed) % mod for index in range(numel(shape))]
    return mx.array(vals, dtype=mx.int32).reshape(shape)


def bvec(shape: list[int], seed: int):
    vals = [(((index * (seed + 5)) + seed) % 2) == 0 for index in range(numel(shape))]
    return mx.array(vals, dtype=mx.bool_).reshape(shape)


if __name__ == "__main__":
    main()
