from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.gated_delta import gated_delta_update

from private_ane_ctypes import PrivateAneModel


PREFILL_BUCKETS = (8, 16, 32, 64)
PREFILL_MIN_STEPS_FOR_UNROLLED = 12
_SHARED_GD_STEP: dict[tuple[int, int], tuple[PrivateAneModel, object]] = {}
_SHARED_GD_PREFILL: dict[tuple[int, int, int], tuple[PrivateAneModel, object]] = {}
_PACK_F32_CACHE: dict[tuple[int, int], np.ndarray] = {}


def pack_f32(values: np.ndarray, *, dim: int, lane: int, seq_len: int) -> np.ndarray:
    key = (dim, lane)
    packed = _PACK_F32_CACHE.get(key)
    if packed is None:
        packed = np.zeros((dim, lane), dtype=np.float32)
        _PACK_F32_CACHE[key] = packed
    else:
        packed.fill(0.0)
    src = values.reshape(seq_len, dim)
    packed[:, :seq_len] = src.T
    return packed


def unpack_f32(blob: bytes, *, dim: int, lane: int, seq_len: int) -> np.ndarray:
    packed = np.frombuffer(blob, dtype=np.float32).reshape(dim, lane)
    return packed[:, :seq_len].T.reshape(1, seq_len, dim)


def blob_fp16(values: np.ndarray) -> bytes:
    payload = np.asarray(values, dtype=np.float16, order="C").tobytes()
    data = bytearray(128 + len(payload))
    data[0] = 1
    data[4] = 2
    data[64:68] = b"\xEF\xBE\xAD\xDE"
    data[68] = 1
    np.ndarray((), dtype=np.uint32, buffer=data, offset=72)[()] = len(payload)
    np.ndarray((), dtype=np.uint32, buffer=data, offset=80)[()] = 128
    data[128:] = payload
    return bytes(data)


def _to_numpy_f32(array) -> np.ndarray:
    cast = array.astype(mx.float32)
    mx.eval(cast)
    mx.synchronize()
    return np.asarray(cast).astype(np.float32, copy=False)


def linear_mil(*, in_dim: int, out_dim: int, lane: int) -> str:
    return f"""program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}, {{"coremlc-version", "3505.4.1"}}, {{"coremltools-component-milinternal", ""}}, {{"coremltools-version", "9.0"}}}})]
{{
    func main<ios18>(tensor<fp32, [1, {in_dim}, 1, {lane}]> x) {{
        string to_fp16 = const()[name = string("to_fp16"), val = string("fp16")];
        tensor<fp16, [1, {in_dim}, 1, {lane}]> x16 = cast(dtype = to_fp16, x = x)[name = string("cast_in")];
        string c_pad_type = const()[name = string("c_pad_type"), val = string("valid")];
        tensor<int32, [2]> c_strides = const()[name = string("c_strides"), val = tensor<int32, [2]>([1, 1])];
        tensor<int32, [4]> c_pad = const()[name = string("c_pad"), val = tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<int32, [2]> c_dilations = const()[name = string("c_dilations"), val = tensor<int32, [2]>([1, 1])];
        int32 c_groups = const()[name = string("c_groups"), val = int32(1)];
        tensor<fp16, [{out_dim}, {in_dim}, 1, 1]> W = const()[name = string("W"), val = tensor<fp16, [{out_dim}, {in_dim}, 1, 1]>(BLOBFILE(path = string("@model_path/weights/w.bin"), offset = uint64(64)))];
        tensor<fp16, [1, {out_dim}, 1, {lane}]> y16 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)[name = string("proj")];
        string to_fp32 = const()[name = string("to_fp32"), val = string("fp32")];
        tensor<fp32, [1, {out_dim}, 1, {lane}]> y = cast(dtype = to_fp32, x = y16)[name = string("cast_out")];
    }} -> (y);
}}
"""


def depthwise_conv1d_mil(*, channels: int, input_lane: int, output_lane: int, kernel: int) -> str:
    return f"""program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}, {{"coremlc-version", "3505.4.1"}}, {{"coremltools-component-milinternal", ""}}, {{"coremltools-version", "9.0"}}}})]
{{
    func main<ios18>(tensor<fp32, [1, {channels}, 1, {input_lane}]> x) {{
        string to_fp16 = const()[name = string("to_fp16"), val = string("fp16")];
        tensor<fp16, [1, {channels}, 1, {input_lane}]> x16 = cast(dtype = to_fp16, x = x)[name = string("cast_in")];
        string c_pad_type = const()[name = string("c_pad_type"), val = string("valid")];
        tensor<int32, [2]> c_strides = const()[name = string("c_strides"), val = tensor<int32, [2]>([1, 1])];
        tensor<int32, [4]> c_pad = const()[name = string("c_pad"), val = tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<int32, [2]> c_dilations = const()[name = string("c_dilations"), val = tensor<int32, [2]>([1, 1])];
        int32 c_groups = const()[name = string("c_groups"), val = int32({channels})];
        tensor<fp16, [{channels}, 1, 1, {kernel}]> W = const()[name = string("W"), val = tensor<fp16, [{channels}, 1, 1, {kernel}]>(BLOBFILE(path = string("@model_path/weights/w.bin"), offset = uint64(64)))];
        tensor<fp16, [1, {channels}, 1, {output_lane}]> c16 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)[name = string("conv")];
        tensor<fp16, [1, {channels}, 1, {output_lane}]> sig = sigmoid(x = c16)[name = string("sigmoid")];
        tensor<fp16, [1, {channels}, 1, {output_lane}]> y16 = mul(x = c16, y = sig)[name = string("silu")];
        string to_fp32 = const()[name = string("to_fp32"), val = string("fp32")];
        tensor<fp32, [1, {channels}, 1, {output_lane}]> y = cast(dtype = to_fp32, x = y16)[name = string("cast_out")];
    }} -> (y);
}}
"""


def gated_delta_step_mil(*, channels: int, d_k: int) -> str:
    return f"""program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}, {{"coremlc-version", "3505.4.1"}}, {{"coremltools-component-milinternal", ""}}, {{"coremltools-version", "9.0"}}}})]
{{
    func main<ios18>(
        tensor<fp32, [1, {channels}, 1, {d_k}]> q,
        tensor<fp32, [1, {channels}, 1, {d_k}]> k,
        tensor<fp32, [1, {channels}, 1, 1]> v,
        tensor<fp32, [1, {channels}, 1, 1]> g,
        tensor<fp32, [1, {channels}, 1, 1]> beta,
        tensor<fp32, [1, {channels}, 1, {d_k}]> state
    ) {{
        string to_fp16 = const()[name = string("to_fp16"), val = string("fp16")];
        tensor<fp16, [1, {channels}, 1, {d_k}]> q16 = cast(dtype = to_fp16, x = q)[name = string("q16")];
        tensor<fp16, [1, {channels}, 1, {d_k}]> k16 = cast(dtype = to_fp16, x = k)[name = string("k16")];
        tensor<fp16, [1, {channels}, 1, 1]> v16 = cast(dtype = to_fp16, x = v)[name = string("v16")];
        tensor<fp16, [1, {channels}, 1, 1]> g16 = cast(dtype = to_fp16, x = g)[name = string("g16")];
        tensor<fp16, [1, {channels}, 1, 1]> b16 = cast(dtype = to_fp16, x = beta)[name = string("b16")];
        tensor<fp16, [1, {channels}, 1, {d_k}]> s16 = cast(dtype = to_fp16, x = state)[name = string("s16")];
        tensor<fp16, [1, {channels}, 1, {d_k}]> dec = mul(x = s16, y = g16)[name = string("dec")];
        tensor<fp16, [1, {channels}, 1, {d_k}]> kv = mul(x = dec, y = k16)[name = string("kv")];
        tensor<int32, [1]> ax = const()[name = string("ax"), val = tensor<int32, [1]>([3])];
        bool kd = const()[name = string("kd"), val = bool(true)];
        tensor<fp16, [1, {channels}, 1, 1]> mem = reduce_sum(x = kv, axes = ax, keep_dims = kd)[name = string("mem")];
        tensor<fp16, [1, {channels}, 1, 1]> delta = mul(x = sub(x = v16, y = mem), y = b16)[name = string("delta")];
        tensor<fp16, [1, {channels}, 1, {d_k}]> state_out16 = add(x = dec, y = mul(x = k16, y = delta))[name = string("state_out16")];
        tensor<fp16, [1, {channels}, 1, {d_k}]> yprod = mul(x = state_out16, y = q16)[name = string("yprod")];
        tensor<fp16, [1, {channels}, 1, 1]> ysum = reduce_sum(x = yprod, axes = ax, keep_dims = kd)[name = string("ysum")];
        string to_fp32 = const()[name = string("to_fp32"), val = string("fp32")];
        tensor<fp32, [1, {channels}, 1, 1]> y = cast(dtype = to_fp32, x = ysum)[name = string("y")];
        tensor<fp32, [1, {channels}, 1, {d_k}]> state_out = cast(dtype = to_fp32, x = state_out16)[name = string("state_out")];
    }} -> (y, state_out);
}}
"""


def gated_delta_prefill_mil(*, channels: int, d_k: int, steps: int) -> str:
    lines = [
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];",
        f"        tensor<fp16, [1, {channels}, 1, {steps * d_k}]> q16 = cast(dtype = to_fp16, x = q)[name = string(\"q16\")];",
        f"        tensor<fp16, [1, {channels}, 1, {steps * d_k}]> k16 = cast(dtype = to_fp16, x = k)[name = string(\"k16\")];",
        f"        tensor<fp16, [1, {channels}, 1, {steps}]> v16 = cast(dtype = to_fp16, x = v)[name = string(\"v16\")];",
        f"        tensor<fp16, [1, {channels}, 1, {steps}]> g16 = cast(dtype = to_fp16, x = g)[name = string(\"g16\")];",
        f"        tensor<fp16, [1, {channels}, 1, {steps}]> b16 = cast(dtype = to_fp16, x = beta)[name = string(\"b16\")];",
        f"        tensor<fp16, [1, {channels}, 1, {d_k}]> s00 = cast(dtype = to_fp16, x = state)[name = string(\"s00\")];",
        "        tensor<int32, [1]> ax = const()[name = string(\"ax\"), val = tensor<int32, [1]>([3])];",
        "        bool kd = const()[name = string(\"kd\"), val = bool(true)];",
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];",
    ]
    outputs = []
    for t in range(steps):
        begin = t * d_k
        end = (t + 1) * d_k
        scalar_end = t + 1
        p = f"{t:02d}"
        prev = f"s{t:02d}"
        nxt = f"s{t+1:02d}"
        lines.extend(
            [
                f"        tensor<int32, [4]> b{p} = const()[name = string(\"b{p}\"), val = tensor<int32, [4]>([0,0,0,{begin}])];",
                f"        tensor<int32, [4]> e{p} = const()[name = string(\"e{p}\"), val = tensor<int32, [4]>([1,{channels},1,{end}])];",
                f"        tensor<int32, [4]> sb{p} = const()[name = string(\"sb{p}\"), val = tensor<int32, [4]>([0,0,0,{t}])];",
                f"        tensor<int32, [4]> se{p} = const()[name = string(\"se{p}\"), val = tensor<int32, [4]>([1,{channels},1,{scalar_end}])];",
                f"        tensor<fp16, [1, {channels}, 1, {d_k}]> q{p} = slice_by_index(begin = b{p}, end = e{p}, x = q16)[name = string(\"q{p}\")];",
                f"        tensor<fp16, [1, {channels}, 1, {d_k}]> k{p} = slice_by_index(begin = b{p}, end = e{p}, x = k16)[name = string(\"k{p}\")];",
                f"        tensor<fp16, [1, {channels}, 1, 1]> v{p} = slice_by_index(begin = sb{p}, end = se{p}, x = v16)[name = string(\"v{p}\")];",
                f"        tensor<fp16, [1, {channels}, 1, 1]> g{p} = slice_by_index(begin = sb{p}, end = se{p}, x = g16)[name = string(\"g{p}\")];",
                f"        tensor<fp16, [1, {channels}, 1, 1]> bt{p} = slice_by_index(begin = sb{p}, end = se{p}, x = b16)[name = string(\"bt{p}\")];",
                f"        tensor<fp16, [1, {channels}, 1, {d_k}]> d{p} = mul(x = {prev}, y = g{p})[name = string(\"d{p}\")];",
                f"        tensor<fp16, [1, {channels}, 1, {d_k}]> kv{p} = mul(x = d{p}, y = k{p})[name = string(\"kv{p}\")];",
                f"        tensor<fp16, [1, {channels}, 1, 1]> m{p} = reduce_sum(x = kv{p}, axes = ax, keep_dims = kd)[name = string(\"m{p}\")];",
                f"        tensor<fp16, [1, {channels}, 1, 1]> dl{p} = mul(x = sub(x = v{p}, y = m{p}), y = bt{p})[name = string(\"dl{p}\")];",
                f"        tensor<fp16, [1, {channels}, 1, {d_k}]> {nxt} = add(x = d{p}, y = mul(x = k{p}, y = dl{p}))[name = string(\"{nxt}\")];",
                f"        tensor<fp16, [1, {channels}, 1, {d_k}]> yp{p} = mul(x = {nxt}, y = q{p})[name = string(\"yp{p}\")];",
                f"        tensor<fp16, [1, {channels}, 1, 1]> ys{p} = reduce_sum(x = yp{p}, axes = ax, keep_dims = kd)[name = string(\"ys{p}\")];",
                f"        tensor<fp32, [1, {channels}, 1, 1]> o{p} = cast(dtype = to_fp32, x = ys{p})[name = string(\"o{p}\")];",
            ]
        )
        outputs.append(f"o{p}")
    lines.append(
        f"        tensor<fp32, [1, {channels}, 1, {d_k}]> state_out = cast(dtype = to_fp32, x = s{steps:02d})[name = string(\"state_out\")];"
    )
    outputs.append("state_out")
    body = "\n".join(lines)
    return f"""program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}, {{"coremlc-version", "3505.4.1"}}, {{"coremltools-component-milinternal", ""}}, {{"coremltools-version", "9.0"}}}})]
{{
    func main<ios18>(
        tensor<fp32, [1, {channels}, 1, {steps * d_k}]> q,
        tensor<fp32, [1, {channels}, 1, {steps * d_k}]> k,
        tensor<fp32, [1, {channels}, 1, {steps}]> v,
        tensor<fp32, [1, {channels}, 1, {steps}]> g,
        tensor<fp32, [1, {channels}, 1, {steps}]> beta,
        tensor<fp32, [1, {channels}, 1, {d_k}]> state
    ) {{
{body}
    }} -> ({", ".join(outputs)});
}}
"""


def dequantize_linear_weight(linear) -> np.ndarray:
    weight = mx.dequantize(
        linear.weight,
        scales=linear.scales,
        biases=linear.biases,
        group_size=linear.group_size,
        bits=linear.bits,
        mode=linear.mode,
        dtype=mx.float16,
    )
    mx.eval(weight)
    mx.synchronize()
    return np.array(weight.tolist(), dtype=np.float16)


def conv_weight_to_fp16(conv1d) -> np.ndarray:
    weight = conv1d.weight.astype(mx.float16)
    mx.eval(weight)
    mx.synchronize()
    weight_np = np.array(weight.tolist(), dtype=np.float16)
    return np.transpose(weight_np, (0, 2, 1)).reshape(weight_np.shape[0], 1, 1, weight_np.shape[1])


@dataclass
class PrivateLinearCache:
    conv_state: np.ndarray
    state: np.ndarray


@dataclass
class PrivateLinearLayerRuntime:
    linear_attn: object
    lane: int
    conv_lane: int
    hidden_size: int
    conv_dim: int
    value_dim: int
    kernel: int
    in_proj_model: PrivateAneModel
    in_proj_session: object
    conv_model: Optional[PrivateAneModel]
    conv_session: Optional[object]
    gated_delta_model: Optional[PrivateAneModel]
    gated_delta_session: Optional[object]
    gated_delta_prefill_models: dict[int, tuple[PrivateAneModel, object]]
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    a_log_np: np.ndarray
    dt_bias_np: np.ndarray
    norm_weight_np: np.ndarray
    norm_eps: float
    out_proj_model: PrivateAneModel
    out_proj_session: object

    @classmethod
    def build(cls, linear_attn, *, lane: int) -> "PrivateLinearLayerRuntime":
        hidden_size = int(linear_attn.hidden_size)
        conv_dim = int(linear_attn.conv_dim)
        value_dim = int(linear_attn.value_dim)
        kernel = int(linear_attn.conv_kernel_size)
        conv_lane = lane + kernel - 1

        in_proj_weight = np.concatenate(
            [
                dequantize_linear_weight(linear_attn.in_proj_qkv),
                dequantize_linear_weight(linear_attn.in_proj_z),
                dequantize_linear_weight(linear_attn.in_proj_b),
                dequantize_linear_weight(linear_attn.in_proj_a),
            ],
            axis=0,
        ).reshape(-1, hidden_size, 1, 1)
        in_proj_model = PrivateAneModel.from_mil(
            linear_mil(in_dim=hidden_size, out_dim=in_proj_weight.shape[0], lane=lane),
            weights=[("@model_path/weights/w.bin", blob_fp16(in_proj_weight), 64)],
        )
        in_proj_model.compile()
        in_proj_model.load()
        in_proj_session = in_proj_model.create_session(
            input_bytes=[hidden_size * lane * 4],
            output_bytes=[in_proj_weight.shape[0] * lane * 4],
        )

        conv_weight = conv_weight_to_fp16(linear_attn.conv1d)
        conv_model: Optional[PrivateAneModel] = None
        conv_session: Optional[object] = None
        try:
            conv_model = PrivateAneModel.from_mil(
                depthwise_conv1d_mil(
                    channels=conv_dim,
                    input_lane=conv_lane,
                    output_lane=lane,
                    kernel=kernel,
                ),
                weights=[("@model_path/weights/w.bin", blob_fp16(conv_weight), 64)],
            )
            conv_model.compile()
            conv_model.load()
            conv_session = conv_model.create_session(
                input_bytes=[conv_dim * conv_lane * 4],
                output_bytes=[conv_dim * lane * 4],
            )
            conv_session.run_one(bytes(conv_dim * conv_lane * 4))
        except Exception:
            if conv_session is not None:
                conv_session.close()
            if conv_model is not None:
                conv_model.close()
            conv_model = None
            conv_session = None

        out_proj_weight = dequantize_linear_weight(linear_attn.out_proj).reshape(
            hidden_size, value_dim, 1, 1
        )
        out_proj_model = PrivateAneModel.from_mil(
            linear_mil(in_dim=value_dim, out_dim=hidden_size, lane=lane),
            weights=[("@model_path/weights/w.bin", blob_fp16(out_proj_weight), 64)],
        )
        out_proj_model.compile()
        out_proj_model.load()
        out_proj_session = out_proj_model.create_session(
            input_bytes=[value_dim * lane * 4],
            output_bytes=[hidden_size * lane * 4],
        )

        gd_channels = int(linear_attn.num_v_heads * linear_attn.head_v_dim)
        gd_dk = int(linear_attn.head_k_dim)
        gated_delta_model, gated_delta_session = _get_shared_gated_delta_step(
            channels=gd_channels,
            d_k=gd_dk,
        )

        a_log = linear_attn.A_log.astype(mx.float32)
        dt_bias = linear_attn.dt_bias.astype(mx.float32)
        norm_weight = linear_attn.norm.weight.astype(mx.float32)
        mx.eval(a_log, dt_bias, norm_weight)
        mx.synchronize()

        return cls(
            linear_attn=linear_attn,
            lane=lane,
            conv_lane=conv_lane,
            hidden_size=hidden_size,
            conv_dim=conv_dim,
            value_dim=value_dim,
            kernel=kernel,
            in_proj_model=in_proj_model,
            in_proj_session=in_proj_session,
            conv_model=conv_model,
            conv_session=conv_session,
            gated_delta_model=gated_delta_model,
            gated_delta_session=gated_delta_session,
            gated_delta_prefill_models={},
            num_k_heads=int(linear_attn.num_k_heads),
            num_v_heads=int(linear_attn.num_v_heads),
            head_k_dim=int(linear_attn.head_k_dim),
            head_v_dim=int(linear_attn.head_v_dim),
            a_log_np=np.array(a_log.tolist(), dtype=np.float32),
            dt_bias_np=np.array(dt_bias.tolist(), dtype=np.float32),
            norm_weight_np=np.array(norm_weight.tolist(), dtype=np.float32),
            norm_eps=float(linear_attn.layer_norm_epsilon),
            out_proj_model=out_proj_model,
            out_proj_session=out_proj_session,
        )

    def close(self) -> None:
        self.in_proj_session.close()
        self.in_proj_model.close()
        if self.conv_session is not None:
            self.conv_session.close()
        if self.conv_model is not None:
            self.conv_model.close()
        self.out_proj_session.close()
        self.out_proj_model.close()

    def make_cache(self, *, batch_size: int = 1) -> PrivateLinearCache:
        return PrivateLinearCache(
            conv_state=np.zeros(
                (batch_size, self.kernel - 1, self.conv_dim),
                dtype=np.float16,
            ),
            state=np.zeros(
                (batch_size, self.num_v_heads, self.head_v_dim, self.head_k_dim),
                dtype=np.float16,
            ),
        )

    def get_prefill_session(
        self,
        *,
        steps: int,
        channels: int,
        d_k: int,
    ) -> tuple[PrivateAneModel, object]:
        bucket = _pick_prefill_bucket(steps)
        existing = self.gated_delta_prefill_models.get(bucket)
        if existing is not None:
            return existing
        model, session = _get_shared_gated_delta_prefill(
            channels=channels,
            d_k=d_k,
            steps=bucket,
        )
        self.gated_delta_prefill_models[bucket] = (model, session)
        return model, session

    def run(self, x_norm, cache, mask):
        la = self.linear_attn
        bsz, seq_len, _dim = x_norm.shape
        x_norm_np = _to_numpy_f32(x_norm)
        in_blob = self.in_proj_session.run_one(
            pack_f32(x_norm_np, dim=self.hidden_size, lane=self.lane, seq_len=seq_len)
        )
        proj_dim = la.conv_dim + la.value_dim + 2 * la.num_v_heads
        proj_np = unpack_f32(in_blob, dim=proj_dim, lane=self.lane, seq_len=seq_len)

        mixed_qkv = proj_np[..., : la.conv_dim]
        z_np = proj_np[..., la.conv_dim : la.conv_dim + la.value_dim].reshape(
            bsz, seq_len, la.num_v_heads, la.head_v_dim
        )
        b_np = proj_np[
            ..., la.conv_dim + la.value_dim : la.conv_dim + la.value_dim + la.num_v_heads
        ]
        a_np = proj_np[..., -la.num_v_heads :]

        private_cache = isinstance(cache, PrivateLinearCache)
        if private_cache:
            conv_state_np = cache.conv_state.astype(np.float32, copy=False)
        else:
            conv_state = cache[0] if cache[0] is not None else mx.zeros(
                (bsz, la.conv_kernel_size - 1, la.conv_dim), dtype=x_norm.dtype
            )
            conv_state_np = (
                _to_numpy_f32(conv_state)
                if cache[0] is not None
                else np.zeros((bsz, la.conv_kernel_size - 1, la.conv_dim), dtype=np.float32)
            )
        conv_input_np = np.concatenate([conv_state_np, mixed_qkv], axis=1)
        next_conv_state = conv_input_np[:, -(la.conv_kernel_size - 1) :].astype(
            np.float32, copy=False
        )
        if private_cache:
            cache.conv_state[...] = next_conv_state.astype(np.float16)
        else:
            cache[0] = mx.array(next_conv_state, dtype=x_norm.dtype)

        if self.conv_session is not None:
            conv_blob = self.conv_session.run_one(
                pack_f32(
                    conv_input_np,
                    dim=self.conv_dim,
                    lane=self.conv_lane,
                    seq_len=seq_len + la.conv_kernel_size - 1,
                )
            )
            conv_np = unpack_f32(
                conv_blob,
                dim=self.conv_dim,
                lane=self.lane,
                seq_len=seq_len,
            )
        else:
            conv_mx = mx.array(conv_input_np, dtype=x_norm.dtype)
            conv_out = nn.silu(la.conv1d(conv_mx))
            conv_np = _to_numpy_f32(conv_out)

        q_np = conv_np[..., : la.key_dim].reshape(bsz, seq_len, la.num_k_heads, la.head_k_dim)
        k_np = conv_np[..., la.key_dim : 2 * la.key_dim].reshape(
            bsz, seq_len, la.num_k_heads, la.head_k_dim
        )
        v_np = conv_np[..., 2 * la.key_dim :].reshape(
            bsz, seq_len, la.num_v_heads, la.head_v_dim
        )

        inv_scale = la.head_k_dim ** -0.5
        q_np = (inv_scale**2) * rms_norm_np(q_np, eps=1e-6)
        k_np = inv_scale * rms_norm_np(k_np, eps=1e-6)
        beta_np, g_np = compute_g_beta_np(self, a_np=a_np, b_np=b_np)
        if private_cache:
            state = None
            state_np = cache.state.astype(np.float32, copy=False)
        else:
            state = cache[1] if cache[1] is not None else None
            state_np = None if state is None else _to_numpy_f32(state)
        private_result = run_private_gated_delta_sequence(
            self,
            q_np=q_np.astype(np.float32, copy=False),
            k_np=k_np.astype(np.float32, copy=False),
            v_np=v_np.astype(np.float32, copy=False),
            g_np=g_np.astype(np.float32, copy=False),
            beta_np=beta_np.astype(np.float32, copy=False),
            state_np=state_np,
            num_k_heads=la.num_k_heads,
            num_v_heads=la.num_v_heads,
            head_k_dim=la.head_k_dim,
            head_v_dim=la.head_v_dim,
        )

        if private_result is None:
            q = mx.array(q_np, dtype=x_norm.dtype)
            k = mx.array(k_np, dtype=x_norm.dtype)
            v = mx.array(v_np, dtype=x_norm.dtype)
            a = mx.array(a_np, dtype=x_norm.dtype)
            b = mx.array(b_np, dtype=x_norm.dtype)
            out, state = gated_delta_update(
                q,
                k,
                v,
                a,
                b,
                la.A_log,
                la.dt_bias,
                state,
                mask,
                use_kernel=not la.training,
            )
            if private_cache:
                cache.state[...] = _to_numpy_f32(state).astype(np.float16)
            else:
                cache[1] = state
            out_np = _to_numpy_f32(out)
        else:
            out_np, state_np = private_result
            if private_cache:
                cache.state[...] = state_np.astype(np.float16)
            else:
                cache[1] = mx.array(state_np, dtype=x_norm.dtype)

        out_np = norm_gated_np(self, out_np=out_np, z_np=z_np).reshape(bsz, seq_len, -1)
        out_blob = self.out_proj_session.run_one(
            pack_f32(out_np, dim=self.value_dim, lane=self.lane, seq_len=seq_len)
        )
        proj_np = unpack_f32(out_blob, dim=self.hidden_size, lane=self.lane, seq_len=seq_len)
        return mx.array(proj_np, dtype=mx.float32)


def _pick_prefill_bucket(steps: int) -> int:
    for bucket in PREFILL_BUCKETS:
        if steps <= bucket:
            return bucket
    return steps


def _get_shared_gated_delta_step(*, channels: int, d_k: int):
    key = (channels, d_k)
    existing = _SHARED_GD_STEP.get(key)
    if existing is not None:
        return existing
    try:
        model = PrivateAneModel.from_mil(
            gated_delta_step_mil(channels=channels, d_k=d_k),
            weights=[],
        )
        model.compile()
        model.load()
        state_bytes = channels * d_k * 4
        scalar_bytes = channels * 4
        session = model.create_session(
            input_bytes=[state_bytes, state_bytes, scalar_bytes, scalar_bytes, scalar_bytes, state_bytes],
            output_bytes=[scalar_bytes, state_bytes],
        )
        zero_state = bytes(state_bytes)
        zero_scalar = bytes(scalar_bytes)
        session.run_many([zero_state, zero_state, zero_scalar, zero_scalar, zero_scalar, zero_state])
        _SHARED_GD_STEP[key] = (model, session)
        return model, session
    except Exception:
        return None, None


def _get_shared_gated_delta_prefill(*, channels: int, d_k: int, steps: int):
    key = (channels, d_k, steps)
    existing = _SHARED_GD_PREFILL.get(key)
    if existing is not None:
        return existing
    model = PrivateAneModel.from_mil(
        gated_delta_prefill_mil(channels=channels, d_k=d_k, steps=steps),
        weights=[],
    )
    model.compile()
    model.load()
    state_bytes = channels * d_k * 4
    qk_seq_bytes = channels * d_k * steps * 4
    scalar_seq_bytes = channels * steps * 4
    session = model.create_session(
        input_bytes=[qk_seq_bytes, qk_seq_bytes, scalar_seq_bytes, scalar_seq_bytes, scalar_seq_bytes, state_bytes],
        output_bytes=[channels * 4] * steps + [state_bytes],
    )
    _SHARED_GD_PREFILL[key] = (model, session)
    return model, session


def rms_norm_np(x: np.ndarray, *, eps: float, weight: np.ndarray | None = None) -> np.ndarray:
    rms = np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + eps)
    y = x / rms
    if weight is not None:
        y = y * weight.reshape((1,) * (y.ndim - 1) + (-1,))
    return y.astype(np.float32, copy=False)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softplus_np(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def compute_g_beta_np(
    runtime: PrivateLinearLayerRuntime,
    *,
    a_np: np.ndarray,
    b_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    beta = sigmoid_np(b_np).astype(np.float32, copy=False)
    g = np.exp(
        -np.exp(runtime.a_log_np.reshape(1, 1, -1))
        * softplus_np(a_np + runtime.dt_bias_np.reshape(1, 1, -1))
    ).astype(np.float32, copy=False)
    return g, beta


def norm_gated_np(
    runtime: PrivateLinearLayerRuntime,
    *,
    out_np: np.ndarray,
    z_np: np.ndarray,
) -> np.ndarray:
    norm = rms_norm_np(out_np, eps=runtime.norm_eps, weight=runtime.norm_weight_np)
    gate = z_np.astype(np.float32, copy=False)
    return ((gate * sigmoid_np(gate)) * norm).astype(np.float32, copy=False)


def run_private_gated_delta_sequence(
    runtime: PrivateLinearLayerRuntime,
    *,
    q_np: np.ndarray,
    k_np: np.ndarray,
    v_np: np.ndarray,
    g_np: np.ndarray,
    beta_np: np.ndarray,
    state_np: np.ndarray | None,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    bsz, seq_len = q_np.shape[:2]
    if state_np is None:
        state_np = np.zeros((bsz, num_v_heads, head_v_dim, head_k_dim), dtype=np.float32)

    if seq_len == 1:
        session = runtime.gated_delta_session
        if session is None:
            return None
        if num_v_heads != num_k_heads:
            repeat_factor = num_v_heads // num_k_heads
            q_np = np.repeat(q_np, repeat_factor, axis=2)
            k_np = np.repeat(k_np, repeat_factor, axis=2)
        channels = num_v_heads * head_v_dim
        q_base = q_np[:, 0].astype(np.float32, copy=False)
        k_base = k_np[:, 0].astype(np.float32, copy=False)
        v_base = v_np[:, 0].astype(np.float32, copy=False)
        g_base = g_np[:, 0].astype(np.float32, copy=False)
        beta_base = beta_np[:, 0].astype(np.float32, copy=False)
        q_step = np.broadcast_to(
            q_base[:, :, None, :],
            (bsz, num_v_heads, head_v_dim, head_k_dim),
        ).reshape(bsz, channels, 1, head_k_dim)
        k_step = np.broadcast_to(
            k_base[:, :, None, :],
            (bsz, num_v_heads, head_v_dim, head_k_dim),
        ).reshape(bsz, channels, 1, head_k_dim)
        v_step = v_base.reshape(bsz, channels, 1, 1)
        g_step = np.broadcast_to(
            g_base[:, :, None, None],
            (bsz, num_v_heads, head_v_dim, 1),
        ).reshape(bsz, channels, 1, 1)
        beta_step = np.broadcast_to(
            beta_base[:, :, None, None],
            (bsz, num_v_heads, head_v_dim, 1),
        ).reshape(bsz, channels, 1, 1)
        state_flat = state_np.astype(np.float32, copy=False).reshape(
            bsz, channels, 1, head_k_dim
        )
        y_blob, state_blob = session.run_many(
            [
                q_step,
                k_step,
                v_step,
                g_step,
                beta_step,
                state_flat,
            ]
        )
        y_flat = np.frombuffer(y_blob, dtype=np.float32).reshape(bsz, channels)
        state_cur = np.frombuffer(state_blob, dtype=np.float32).reshape(
            bsz, num_v_heads, head_v_dim, head_k_dim
        )
        return y_flat.reshape(bsz, 1, num_v_heads, head_v_dim), state_cur

    if num_v_heads != num_k_heads:
        repeat_factor = num_v_heads // num_k_heads
        q_np = np.repeat(q_np, repeat_factor, axis=2)
        k_np = np.repeat(k_np, repeat_factor, axis=2)

    channels = num_v_heads * head_v_dim

    def flatten_qk(arr: np.ndarray) -> np.ndarray:
        return np.repeat(arr[:, :, :, None, :], head_v_dim, axis=3).reshape(
            bsz, seq_len, channels, head_k_dim
        )

    q_flat_seq = flatten_qk(q_np)
    k_flat_seq = flatten_qk(k_np)
    v_flat_seq = v_np.reshape(bsz, seq_len, channels, 1)
    g_flat_seq = np.repeat(g_np[:, :, :, None, None], head_v_dim, axis=3).reshape(
        bsz, seq_len, channels, 1
    )
    beta_flat_seq = np.repeat(beta_np[:, :, :, None, None], head_v_dim, axis=3).reshape(
        bsz, seq_len, channels, 1
    )
    state_cur = state_np.astype(np.float32, copy=False)

    # Prefill path: one ANE-native unrolled program for sufficiently long sequences.
    if seq_len >= PREFILL_MIN_STEPS_FOR_UNROLLED:
        try:
            bucket = _pick_prefill_bucket(seq_len)
            _model, session = runtime.get_prefill_session(
                steps=seq_len,
                channels=channels,
                d_k=head_k_dim,
            )
            def pack_seq(arr: np.ndarray, *, width: int, fill_value: float = 0.0) -> np.ndarray:
                packed = np.full((channels, bucket * width), fill_value, dtype=np.float32)
                src = arr[0].reshape(seq_len, channels, width)
                for t in range(seq_len):
                    packed[:, t * width : (t + 1) * width] = src[t]
                return packed.reshape(1, channels, 1, bucket * width)

            state_flat = state_cur.reshape(bsz, channels, 1, head_k_dim)
            blobs = session.run_many(
                [
                    pack_seq(q_flat_seq, width=head_k_dim),
                    pack_seq(k_flat_seq, width=head_k_dim),
                    pack_seq(v_flat_seq, width=1),
                    pack_seq(g_flat_seq, width=1, fill_value=1.0),
                    pack_seq(beta_flat_seq, width=1),
                    state_flat.astype(np.float32, copy=False),
                ]
            )
            y_outs = []
            for t in range(seq_len):
                y_flat = np.frombuffer(blobs[t], dtype=np.float32).reshape(bsz, channels)
                y_outs.append(y_flat.reshape(bsz, 1, num_v_heads, head_v_dim))
            state_cur = np.frombuffer(blobs[-1], dtype=np.float32).reshape(
                bsz, num_v_heads, head_v_dim, head_k_dim
            )
            return np.concatenate(y_outs, axis=1), state_cur
        except Exception:
            pass

    # Decode/slow fallback path: ANE-native single-step loop.
    session = runtime.gated_delta_session
    if session is None:
        return None
    outputs = []
    for t in range(seq_len):
        q_step = q_flat_seq[:, t]
        k_step = k_flat_seq[:, t]
        v_step = v_flat_seq[:, t]
        g_step = g_flat_seq[:, t]
        beta_step = beta_flat_seq[:, t]
        state_flat = state_cur.reshape(bsz, channels, 1, head_k_dim)
        y_blob, state_blob = session.run_many(
            [
                q_step.astype(np.float32).reshape(bsz, channels, 1, head_k_dim),
                k_step.astype(np.float32).reshape(bsz, channels, 1, head_k_dim),
                v_step.astype(np.float32).reshape(bsz, channels, 1, 1),
                g_step.astype(np.float32).reshape(bsz, channels, 1, 1),
                beta_step.astype(np.float32).reshape(bsz, channels, 1, 1),
                state_flat.astype(np.float32, copy=False),
            ]
        )
        y_flat = np.frombuffer(y_blob, dtype=np.float32).reshape(bsz, channels)
        state_cur = np.frombuffer(state_blob, dtype=np.float32).reshape(
            bsz, num_v_heads, head_v_dim, head_k_dim
        )
        outputs.append(y_flat.reshape(bsz, 1, num_v_heads, head_v_dim))

    return np.concatenate(outputs, axis=1), state_cur
