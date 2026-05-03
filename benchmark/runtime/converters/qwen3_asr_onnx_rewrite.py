from __future__ import annotations

from pathlib import Path
from typing import Any


def _write_decoder_compatible_onnx(source: Path, destination: Path) -> None:
    import onnx

    model = onnx.load(str(source), load_external_data=True)
    _rewrite_decoder_custom_ops(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data_path = destination.with_suffix(".data")
    if data_path.exists():
        data_path.unlink()
    onnx.save_model(
        model,
        str(destination),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path.name,
        size_threshold=1024,
        convert_attribute=False,
    )


def _rewrite_decoder_custom_ops(model: Any) -> None:
    import onnx

    initializer_map = {tensor.name: tensor for tensor in model.graph.initializer}
    removed_initializers: set[str] = set()
    rewritten = []
    for node in model.graph.node:
        if node.op_type == "SimplifiedLayerNormalization":
            rewritten.extend(_expand_simplified_layer_normalization(onnx, model, node))
        elif node.op_type == "MatMulNBits":
            expanded, removed = _expand_matmulnbits(onnx, model, initializer_map, node)
            rewritten.extend(expanded)
            removed_initializers.update(removed)
        else:
            rewritten.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    if removed_initializers:
        kept = [
            tensor
            for tensor in model.graph.initializer
            if tensor.name not in removed_initializers
        ]
        del model.graph.initializer[:]
        model.graph.initializer.extend(kept)


def _expand_simplified_layer_normalization(
    onnx: Any,
    model: Any,
    node: Any,
) -> list[Any]:
    if len(node.input) < 2 or not node.output:
        return [node]
    attrs = {
        attr.name: onnx.helper.get_attribute_value(attr)
        for attr in node.attribute
    }
    axis = int(attrs.get("axis", -1))
    epsilon = float(attrs.get("epsilon", 1e-5))
    input_name, weight_name = node.input[:2]
    square = f"{node.name}_square"
    axes = f"{node.name}_axes"
    mean = f"{node.name}_mean"
    epsilon_name = f"{node.name}_epsilon"
    variance = f"{node.name}_variance"
    denom = f"{node.name}_denom"
    normalized = f"{node.name}_normalized"
    model.graph.initializer.extend(
        [
            onnx.helper.make_tensor(
                name=axes,
                data_type=onnx.TensorProto.INT64,
                dims=[1],
                vals=[axis],
            ),
            onnx.helper.make_tensor(
                name=epsilon_name,
                data_type=onnx.TensorProto.FLOAT,
                dims=[],
                vals=[epsilon],
            ),
        ]
    )
    return [
        onnx.helper.make_node(
            "Mul",
            [input_name, input_name],
            [square],
            name=f"{node.name}_Square",
        ),
        onnx.helper.make_node(
            "ReduceMean",
            [square, axes],
            [mean],
            name=f"{node.name}_ReduceMean",
            keepdims=1,
        ),
        onnx.helper.make_node(
            "Add",
            [mean, epsilon_name],
            [variance],
            name=f"{node.name}_AddEpsilon",
        ),
        onnx.helper.make_node(
            "Sqrt",
            [variance],
            [denom],
            name=f"{node.name}_Sqrt",
        ),
        onnx.helper.make_node(
            "Div",
            [input_name, denom],
            [normalized],
            name=f"{node.name}_Div",
        ),
        onnx.helper.make_node(
            "Mul",
            [normalized, weight_name],
            [node.output[0]],
            name=f"{node.name}_Scale",
        ),
    ]


def _expand_matmulnbits(
    onnx: Any,
    model: Any,
    initializer_map: dict[str, Any],
    node: Any,
) -> tuple[list[Any], set[str]]:
    if len(node.input) < 3 or not node.output:
        return [node], set()
    attrs = {
        attr.name: onnx.helper.get_attribute_value(attr)
        for attr in node.attribute
    }
    required = [name for name in node.input[1:4] if name]
    if any(name not in initializer_map for name in required):
        return [node], set()

    weight = _dequantize_matmulnbits_weight(
        onnx,
        weight_tensor=initializer_map[node.input[1]],
        scale_tensor=initializer_map[node.input[2]],
        zero_point_tensor=initializer_map[node.input[3]]
        if len(node.input) >= 4 and node.input[3]
        else None,
        attrs=attrs,
    )
    weight_name = f"{node.name}_dequant_weight_f16"
    model.graph.initializer.append(
        onnx.numpy_helper.from_array(weight, name=weight_name)
    )
    input_fp16 = f"{node.name}_input_f16"
    matmul_fp16 = f"{node.name}_matmul_f16"
    nodes = [
        onnx.helper.make_node(
            "Cast",
            [node.input[0]],
            [input_fp16],
            name=f"{node.name}_CastInputFp16",
            to=onnx.TensorProto.FLOAT16,
        ),
        onnx.helper.make_node(
            "MatMul",
            [input_fp16, weight_name],
            [matmul_fp16],
            name=f"{node.name}_MatMul",
        ),
        onnx.helper.make_node(
            "Cast",
            [matmul_fp16],
            [node.output[0]],
            name=f"{node.name}_CastOutputFp32",
            to=onnx.TensorProto.FLOAT,
        ),
    ]
    if len(node.input) >= 6 and node.input[5]:
        bias_out = node.output[0]
        cast_out = f"{node.name}_cast_output"
        nodes[-1].output[0] = cast_out
        nodes.append(
            onnx.helper.make_node(
                "Add",
                [cast_out, node.input[5]],
                [bias_out],
                name=f"{node.name}_BiasAdd",
            )
        )
    return nodes, set(required)


def _dequantize_matmulnbits_weight(
    onnx: Any,
    *,
    weight_tensor: Any,
    scale_tensor: Any,
    zero_point_tensor: Any | None,
    attrs: dict[str, Any],
) -> Any:
    import numpy as np

    bits = int(attrs.get("bits", 4))
    block_size = int(attrs["block_size"])
    k_size = int(attrs["K"])
    n_size = int(attrs["N"])
    k_blocks = (k_size + block_size - 1) // block_size
    packed = onnx.numpy_helper.to_array(weight_tensor)
    scales = onnx.numpy_helper.to_array(scale_tensor).astype(np.float32, copy=False)
    if packed.shape[0] != n_size or packed.shape[1] != k_blocks:
        raise SystemExit(
            "Unsupported MatMulNBits weight layout for "
            f"{weight_tensor.name}: expected N={n_size}, k_blocks={k_blocks}, "
            f"got {list(packed.shape)}."
        )
    if scales.shape != (n_size, k_blocks):
        raise SystemExit(
            "Unsupported MatMulNBits scale layout for "
            f"{scale_tensor.name}: expected [{n_size}, {k_blocks}], "
            f"got {list(scales.shape)}."
        )
    quantized = _unpack_nbits(packed, bits=bits, value_count=block_size)
    zero_points = _matmulnbits_zero_points(
        onnx,
        zero_point_tensor,
        bits=bits,
        n_size=n_size,
        k_blocks=k_blocks,
    )
    dequantized = np.empty((k_size, n_size), dtype=np.float16)
    for block in range(k_blocks):
        start = block * block_size
        end = min(start + block_size, k_size)
        values = quantized[:, block, : end - start].astype(np.float32)
        block_values = (values - zero_points[:, block:block + 1]) * scales[
            :,
            block:block + 1,
        ]
        dequantized[start:end, :] = block_values.T.astype(np.float16)
    return dequantized


def _matmulnbits_zero_points(
    onnx: Any,
    zero_point_tensor: Any | None,
    *,
    bits: int,
    n_size: int,
    k_blocks: int,
) -> Any:
    import numpy as np

    if zero_point_tensor is None:
        return np.full((n_size, k_blocks), 1 << (bits - 1), dtype=np.float32)
    zero_points = onnx.numpy_helper.to_array(zero_point_tensor)
    if zero_points.dtype == np.uint8 and zero_points.shape != (n_size, k_blocks):
        unpacked = _unpack_nbits(
            zero_points[:, None, :],
            bits=bits,
            value_count=k_blocks,
        )
        zero_points = unpacked[:, 0, :]
    if zero_points.shape != (n_size, k_blocks):
        raise SystemExit(
            "Unsupported MatMulNBits zero-point layout for "
            f"{zero_point_tensor.name}: expected [{n_size}, {k_blocks}], "
            f"got {list(zero_points.shape)}."
        )
    return zero_points.astype(np.float32, copy=False)


def _unpack_nbits(packed: Any, *, bits: int, value_count: int) -> Any:
    import numpy as np

    if bits <= 0 or bits > 8 or 8 % bits != 0:
        raise SystemExit(f"Unsupported MatMulNBits bit width: {bits}")
    values_per_byte = 8 // bits
    flat = np.asarray(packed, dtype=np.uint8)
    prefix = flat.shape[:-1]
    out = np.empty((*prefix, flat.shape[-1] * values_per_byte), dtype=np.uint8)
    mask = (1 << bits) - 1
    for index in range(values_per_byte):
        out[..., index::values_per_byte] = (flat >> (index * bits)) & mask
    return out[..., :value_count]
