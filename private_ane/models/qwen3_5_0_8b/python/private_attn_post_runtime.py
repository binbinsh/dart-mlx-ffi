from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from private_ane_ctypes import PrivateAneModel
from private_linear_runtime import blob_fp16, dequantize_linear_weight, pack_f32, unpack_f32
from private_probe_cache import get as get_probe, make_key as make_probe_key, set as set_probe


@dataclass
class PrivateAttnPostLayerRuntime:
    layer_index: int
    lane: int
    input_dim: int
    output_dim: int
    model: PrivateAneModel
    session: object

    def run(self, *, ctx_np: np.ndarray, gate_np: np.ndarray) -> mx.array:
        seq_len = gate_np.shape[1]
        ctx_flat = (
            ctx_np.transpose(0, 2, 1, 3)
            .reshape(1, seq_len, self.input_dim)
            .astype(np.float32, copy=False)
        )
        gate_flat = gate_np.astype(np.float32, copy=False)
        out_blob = self.session.run_many(
            [
                pack_f32(ctx_flat, dim=self.input_dim, lane=self.lane, seq_len=seq_len),
                pack_f32(gate_flat, dim=self.input_dim, lane=self.lane, seq_len=seq_len),
            ]
        )[0]
        out_np = unpack_f32(out_blob, dim=self.output_dim, lane=self.lane, seq_len=seq_len)
        return mx.array(out_np, dtype=mx.float32)

    def close(self) -> None:
        self.session.close()
        self.model.close()


def attn_post_mil(*, input_dim: int, output_dim: int, lane: int) -> str:
    return f"""program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}, {{"coremlc-version", "3505.4.1"}}, {{"coremltools-component-milinternal", ""}}, {{"coremltools-version", "9.0"}}}})]
{{
    func main<ios18>(
        tensor<fp32, [1, {input_dim}, 1, {lane}]> ctx,
        tensor<fp32, [1, {input_dim}, 1, {lane}]> gate
    ) {{
        string to_fp16 = const()[name = string("to_fp16"), val = string("fp16")];
        tensor<fp16, [1, {input_dim}, 1, {lane}]> ctx16 = cast(dtype = to_fp16, x = ctx)[name = string("ctx16")];
        tensor<fp16, [1, {input_dim}, 1, {lane}]> gate16 = cast(dtype = to_fp16, x = gate)[name = string("gate16")];
        tensor<fp16, [1, {input_dim}, 1, {lane}]> sig = sigmoid(x = gate16)[name = string("sig")];
        tensor<fp16, [1, {input_dim}, 1, {lane}]> fused = mul(x = ctx16, y = sig)[name = string("fused")];
        string c_pad_type = const()[name = string("c_pad_type"), val = string("valid")];
        tensor<int32, [2]> c_strides = const()[name = string("c_strides"), val = tensor<int32, [2]>([1, 1])];
        tensor<int32, [4]> c_pad = const()[name = string("c_pad"), val = tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<int32, [2]> c_dilations = const()[name = string("c_dilations"), val = tensor<int32, [2]>([1, 1])];
        int32 c_groups = const()[name = string("c_groups"), val = int32(1)];
        tensor<fp16, [{output_dim}, {input_dim}, 1, 1]> W = const()[name = string("W"), val = tensor<fp16, [{output_dim}, {input_dim}, 1, 1]>(BLOBFILE(path = string("@model_path/weights/w.bin"), offset = uint64(64)))];
        tensor<fp16, [1, {output_dim}, 1, {lane}]> out16 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = fused)[name = string("proj")];
        string to_fp32 = const()[name = string("to_fp32"), val = string("fp32")];
        tensor<fp32, [1, {output_dim}, 1, {lane}]> out = cast(dtype = to_fp32, x = out16)[name = string("out")];
    }} -> (out);
}}
"""


def build_attn_post_runtimes(model, *, lane: int, attn_layers: set[int] | None = None) -> dict[int, PrivateAttnPostLayerRuntime]:
    runtimes: dict[int, PrivateAttnPostLayerRuntime] = {}
    base = model.language_model.model
    for layer_index, layer in enumerate(base.layers):
        if layer.is_linear:
            continue
        if attn_layers is not None and layer_index not in attn_layers:
            continue
        sa = layer.self_attn
        input_dim = int(sa.num_attention_heads * sa.head_dim)
        weight_2d = dequantize_linear_weight(sa.o_proj)
        output_dim = int(weight_2d.shape[0])
        probe_key = make_probe_key(
            "attn_post",
            layer_index=layer_index,
            input_dim=input_dim,
            output_dim=output_dim,
            lane=lane,
        )
        cached = get_probe(probe_key)
        if cached is False:
            continue
        weight = weight_2d.reshape(output_dim, input_dim, 1, 1)
        model_handle = PrivateAneModel.from_mil(
            attn_post_mil(input_dim=input_dim, output_dim=output_dim, lane=lane),
            weights=[("@model_path/weights/w.bin", blob_fp16(weight), 64)],
        )
        model_handle.compile()
        model_handle.load()
        session = model_handle.create_session(
            input_bytes=[input_dim * lane * 4, input_dim * lane * 4],
            output_bytes=[output_dim * lane * 4],
        )
        try:
            zero = np.zeros((1, input_dim, 1, lane), dtype=np.float32)
            session.run_many([zero, zero])
            set_probe(probe_key, True)
        except Exception:
            session.close()
            model_handle.close()
            set_probe(probe_key, False)
            continue
        runtimes[layer_index] = PrivateAttnPostLayerRuntime(
            layer_index=layer_index,
            lane=lane,
            input_dim=input_dim,
            output_dim=output_dim,
            model=model_handle,
            session=session,
        )
    return runtimes


def close_attn_post_runtimes(runtimes: dict[int, PrivateAttnPostLayerRuntime]) -> None:
    for runtime in runtimes.values():
        runtime.close()
