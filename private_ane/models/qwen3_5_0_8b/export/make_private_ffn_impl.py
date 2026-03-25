from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np


MODEL_ID = "mlx-community/Qwen3.5-0.8B-4bit"
DEFAULT_SNAPSHOT = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--mlx-community--Qwen3.5-0.8B-4bit"
    / "snapshots"
    / "da28692b5f139cb0ec58a356b437486b7dac7462"
)
BUILD_INFO = (
    '[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, '
    '{"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, '
    '{"coremltools-version", "9.0"}})]'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--snapshot-dir", default=str(DEFAULT_SNAPSHOT))
    parser.add_argument("--lane", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=5)
    parser.add_argument("--grouped-shard-size", type=int, default=8)
    parser.add_argument(
        "--layers",
        help="Comma-separated layer indices. Defaults to every dense layer.",
    )
    return parser.parse_args()


def load_snapshot(snapshot_dir: Path) -> tuple[dict[str, object], dict[str, mx.array]]:
    config_path = snapshot_dir / "config.json"
    model_path = snapshot_dir / "model.safetensors"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json under {snapshot_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model.safetensors under {snapshot_dir}")
    return json.loads(config_path.read_text(encoding="utf-8")), mx.load(str(model_path))


def detect_text_prefix(keys: list[str]) -> str:
    candidates = (
        "model.",
        "language_model.model.",
        "text_model.model.",
    )
    for prefix in candidates:
        if f"{prefix}embed_tokens.weight" in keys:
            return prefix
    raise RuntimeError("Unable to detect Qwen3.5 text tensor prefix.")


def parse_layers(raw: str | None, count: int) -> list[int]:
    if raw is None or not raw.strip():
        return list(range(count))
    values = []
    for item in raw.split(","):
        value = int(item.strip())
        if value < 0 or value >= count:
            raise ValueError(f"Layer index {value} is out of range 0..{count - 1}")
        values.append(value)
    return sorted(dict.fromkeys(values))


def chunk_layers(layers: list[int], shard_size: int) -> list[list[int]]:
    if shard_size <= 0:
        raise ValueError(f"shard_size must be positive, got {shard_size}")
    return [layers[index : index + shard_size] for index in range(0, len(layers), shard_size)]


def all_ffn_mil(
    *,
    dim: int,
    hidden: int,
    lane: int,
    layers: list[int],
) -> str:
    inputs = ",\n        ".join(
        f"tensor<fp32, [1, {dim}, 1, {lane}]> x{layer:02d}" for layer in layers
    )
    lines = [
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];",
        "        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];",
        "        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];",
        "        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];",
        "        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];",
        "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];",
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];",
    ]
    returns = []
    for layer in layers:
        lines.extend(
            [
                f'        tensor<fp16, [1, {dim}, 1, {lane}]> x16_{layer:02d} = cast(dtype = to_fp16, x = x{layer:02d})[name = string("cast_in_{layer:02d}")];',
                f'        tensor<fp16, [{hidden}, {dim}, 1, 1]> W1_{layer:02d} = const()[name = string("W1_{layer:02d}"), val = tensor<fp16, [{hidden}, {dim}, 1, 1]>(BLOBFILE(path = string("@model_path/weights/l{layer:02d}_w1.bin"), offset = uint64(64)))];',
                f'        tensor<fp16, [{hidden}, {dim}, 1, 1]> W3_{layer:02d} = const()[name = string("W3_{layer:02d}"), val = tensor<fp16, [{hidden}, {dim}, 1, 1]>(BLOBFILE(path = string("@model_path/weights/l{layer:02d}_w3.bin"), offset = uint64(64)))];',
                f'        tensor<fp16, [{dim}, {hidden}, 1, 1]> W2_{layer:02d} = const()[name = string("W2_{layer:02d}"), val = tensor<fp16, [{dim}, {hidden}, 1, 1]>(BLOBFILE(path = string("@model_path/weights/l{layer:02d}_w2.bin"), offset = uint64(64)))];',
                f'        tensor<fp16, [1, {hidden}, 1, {lane}]> h1_{layer:02d} = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W1_{layer:02d}, x = x16_{layer:02d})[name = string("gate_proj_{layer:02d}")];',
                f'        tensor<fp16, [1, {hidden}, 1, {lane}]> h3_{layer:02d} = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W3_{layer:02d}, x = x16_{layer:02d})[name = string("up_proj_{layer:02d}")];',
                f'        tensor<fp16, [1, {hidden}, 1, {lane}]> sig_{layer:02d} = sigmoid(x = h1_{layer:02d})[name = string("sigmoid_{layer:02d}")];',
                f'        tensor<fp16, [1, {hidden}, 1, {lane}]> silu_{layer:02d} = mul(x = h1_{layer:02d}, y = sig_{layer:02d})[name = string("silu_{layer:02d}")];',
                f'        tensor<fp16, [1, {hidden}, 1, {lane}]> gate_{layer:02d} = mul(x = silu_{layer:02d}, y = h3_{layer:02d})[name = string("gate_{layer:02d}")];',
                f'        tensor<fp16, [1, {dim}, 1, {lane}]> y16_{layer:02d} = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W2_{layer:02d}, x = gate_{layer:02d})[name = string("down_proj_{layer:02d}")];',
                f'        tensor<fp32, [1, {dim}, 1, {lane}]> y{layer:02d} = cast(dtype = to_fp32, x = y16_{layer:02d})[name = string("cast_out_{layer:02d}")];',
            ]
        )
        returns.append(f"y{layer:02d}")
    body = "\n".join(lines)
    return f"""program(1.3)
{BUILD_INFO}
{{
    func main<ios18>(
        {inputs}
    ) {{
{body}
    }} -> ({", ".join(returns)});
}}
"""


def grouped_ffn_mil(
    *,
    dim: int,
    hidden: int,
    lane: int,
    groups: int,
) -> str:
    total_dim = dim * groups
    total_hidden = hidden * groups
    return f"""program(1.3)
{BUILD_INFO}
{{
    func main<ios18>(tensor<fp32, [1, {total_dim}, 1, {lane}]> x) {{
        string to_fp16 = const()[name = string("to_fp16"), val = string("fp16")];
        tensor<fp16, [1, {total_dim}, 1, {lane}]> x16 = cast(dtype = to_fp16, x = x)[name = string("cast_in")];
        string c_pad_type = const()[name = string("c_pad_type"), val = string("valid")];
        tensor<int32, [2]> c_strides = const()[name = string("c_strides"), val = tensor<int32, [2]>([1, 1])];
        tensor<int32, [4]> c_pad = const()[name = string("c_pad"), val = tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<int32, [2]> c_dilations = const()[name = string("c_dilations"), val = tensor<int32, [2]>([1, 1])];
        int32 c_groups = const()[name = string("c_groups"), val = int32({groups})];
        tensor<fp16, [{total_hidden}, {dim}, 1, 1]> W1 = const()[name = string("W1"), val = tensor<fp16, [{total_hidden}, {dim}, 1, 1]>(BLOBFILE(path = string("@model_path/weights/w1.bin"), offset = uint64(64)))];
        tensor<fp16, [{total_hidden}, {dim}, 1, 1]> W3 = const()[name = string("W3"), val = tensor<fp16, [{total_hidden}, {dim}, 1, 1]>(BLOBFILE(path = string("@model_path/weights/w3.bin"), offset = uint64(64)))];
        tensor<fp16, [{total_dim}, {hidden}, 1, 1]> W2 = const()[name = string("W2"), val = tensor<fp16, [{total_dim}, {hidden}, 1, 1]>(BLOBFILE(path = string("@model_path/weights/w2.bin"), offset = uint64(64)))];
        tensor<fp16, [1, {total_hidden}, 1, {lane}]> h1 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W1, x = x16)[name = string("gate_proj")];
        tensor<fp16, [1, {total_hidden}, 1, {lane}]> h3 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W3, x = x16)[name = string("up_proj")];
        tensor<fp16, [1, {total_hidden}, 1, {lane}]> sig = sigmoid(x = h1)[name = string("sigmoid")];
        tensor<fp16, [1, {total_hidden}, 1, {lane}]> silu = mul(x = h1, y = sig)[name = string("silu")];
        tensor<fp16, [1, {total_hidden}, 1, {lane}]> gate = mul(x = silu, y = h3)[name = string("gate")];
        tensor<fp16, [1, {total_dim}, 1, {lane}]> y16 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W2, x = gate)[name = string("down_proj")];
        string to_fp32 = const()[name = string("to_fp32"), val = string("fp32")];
        tensor<fp32, [1, {total_dim}, 1, {lane}]> y = cast(dtype = to_fp32, x = y16)[name = string("cast_out")];
    }} -> (y);
}}
"""


def attn_post_mil(*, input_dim: int, output_dim: int, lane: int) -> str:
    return f"""program(1.3)
{BUILD_INFO}
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


def mil_text(*, dim: int, hidden: int, lane: int) -> str:
    return f"""program(1.3)
{BUILD_INFO}
{{
    func main<ios18>(tensor<fp32, [1, {dim}, 1, {lane}]> x) {{
        string to_fp16 = const()[name = string("to_fp16"), val = string("fp16")];
        tensor<fp16, [1, {dim}, 1, {lane}]> x16 = cast(dtype = to_fp16, x = x)[name = string("cast_in")];
        string c_pad_type = const()[name = string("c_pad_type"), val = string("valid")];
        tensor<int32, [2]> c_strides = const()[name = string("c_strides"), val = tensor<int32, [2]>([1, 1])];
        tensor<int32, [4]> c_pad = const()[name = string("c_pad"), val = tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<int32, [2]> c_dilations = const()[name = string("c_dilations"), val = tensor<int32, [2]>([1, 1])];
        int32 c_groups = const()[name = string("c_groups"), val = int32(1)];
        tensor<fp16, [{hidden}, {dim}, 1, 1]> W1 = const()[name = string("W1"), val = tensor<fp16, [{hidden}, {dim}, 1, 1]>(BLOBFILE(path = string("@model_path/weights/w1.bin"), offset = uint64(64)))];
        tensor<fp16, [{hidden}, {dim}, 1, 1]> W3 = const()[name = string("W3"), val = tensor<fp16, [{hidden}, {dim}, 1, 1]>(BLOBFILE(path = string("@model_path/weights/w3.bin"), offset = uint64(64)))];
        tensor<fp16, [{dim}, {hidden}, 1, 1]> W2 = const()[name = string("W2"), val = tensor<fp16, [{dim}, {hidden}, 1, 1]>(BLOBFILE(path = string("@model_path/weights/w2.bin"), offset = uint64(64)))];
        tensor<fp16, [1, {hidden}, 1, {lane}]> h1 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W1, x = x16)[name = string("gate_proj")];
        tensor<fp16, [1, {hidden}, 1, {lane}]> h3 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W3, x = x16)[name = string("up_proj")];
        tensor<fp16, [1, {hidden}, 1, {lane}]> sig = sigmoid(x = h1)[name = string("sigmoid")];
        tensor<fp16, [1, {hidden}, 1, {lane}]> silu = mul(x = h1, y = sig)[name = string("silu")];
        tensor<fp16, [1, {hidden}, 1, {lane}]> gate = mul(x = silu, y = h3)[name = string("gate")];
        tensor<fp16, [1, {dim}, 1, {lane}]> y16 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W2, x = gate)[name = string("down_proj")];
        string to_fp32 = const()[name = string("to_fp32"), val = string("fp32")];
        tensor<fp32, [1, {dim}, 1, {lane}]> y = cast(dtype = to_fp32, x = y16)[name = string("cast_out")];
    }} -> (y);
}}
"""


def blob_fp16(values: np.ndarray) -> bytes:
    payload = np.asarray(values, dtype=np.float16, order="C").tobytes()
    data = bytearray(128 + len(payload))
    data[0] = 1
    data[4] = 2
    data[64:68] = b"\xEF\xBE\xAD\xDE"
    data[68] = 1
    struct.pack_into("<I", data, 72, len(payload))
    struct.pack_into("<I", data, 80, 128)
    data[128:] = payload
    return bytes(data)


def deterministic_input(*, dim: int, lane: int, layer: int) -> np.ndarray:
    values = np.array(
        [((index + layer * 17) % 257) / 257.0 - 0.5 for index in range(dim * lane)],
        dtype=np.float32,
    )
    return values.reshape(dim, lane)


def compute_expected(
    x: np.ndarray,
    *,
    w1: np.ndarray,
    w3: np.ndarray,
    w2: np.ndarray,
) -> np.ndarray:
    x16 = x.astype(np.float16)
    h1 = (w1[:, :, 0, 0] @ x16).astype(np.float16)
    h3 = (w3[:, :, 0, 0] @ x16).astype(np.float16)
    gate = ((h1 * (1.0 / (1.0 + np.exp(-h1.astype(np.float32))))).astype(np.float16) * h3).astype(
        np.float16
    )
    return (w2[:, :, 0, 0] @ gate).astype(np.float32)


def compute_attn_post_expected(
    ctx: np.ndarray,
    gate: np.ndarray,
    *,
    weight: np.ndarray,
) -> np.ndarray:
    ctx16 = ctx.astype(np.float16)
    gate16 = gate.astype(np.float16)
    fused = (ctx16 * (1.0 / (1.0 + np.exp(-gate16.astype(np.float32))))).astype(
        np.float16
    )
    return (weight[:, :, 0, 0] @ fused).astype(np.float32)


def dequantize_linear(
    tensors: dict[str, mx.array],
    prefix: str,
    *,
    group_size: int,
    bits: int,
    mode: str,
) -> np.ndarray:
    weight = tensors[f"{prefix}.weight"]
    scales = tensors[f"{prefix}.scales"]
    biases = tensors.get(f"{prefix}.biases")
    dequantized = mx.dequantize(
        weight,
        scales,
        biases,
        group_size=group_size,
        bits=bits,
        mode=mode,
        dtype=mx.float16,
    )
    return np.array(dequantized, copy=False).astype(np.float16, copy=False)


def load_layer_weights(
    tensors: dict[str, mx.array],
    *,
    layer_prefix: str,
    group_size: int,
    bits: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w1 = dequantize_linear(
        tensors,
        f"{layer_prefix}.gate_proj",
        group_size=group_size,
        bits=bits,
        mode=mode,
    )
    w3 = dequantize_linear(
        tensors,
        f"{layer_prefix}.up_proj",
        group_size=group_size,
        bits=bits,
        mode=mode,
    )
    w2 = dequantize_linear(
        tensors,
        f"{layer_prefix}.down_proj",
        group_size=group_size,
        bits=bits,
        mode=mode,
    )
    return (
        w1.reshape(w1.shape[0], w1.shape[1], 1, 1),
        w3.reshape(w3.shape[0], w3.shape[1], 1, 1),
        w2.reshape(w2.shape[0], w2.shape[1], 1, 1),
    )


def emit_layer(
    out_dir: Path,
    *,
    text_prefix: str,
    tensors: dict[str, mx.array],
    config: dict[str, object],
    layer: int,
    lane: int,
) -> dict[str, object]:
    text_config = dict(config["text_config"])
    quant = dict(config["quantization"])
    dim = int(text_config["hidden_size"])
    hidden = int(text_config["intermediate_size"])
    group_size = int(quant["group_size"])
    bits = int(quant["bits"])
    mode = str(quant["mode"])
    layer_prefix = f"{text_prefix}layers.{layer}.mlp"
    w1, w3, w2 = load_layer_weights(
        tensors,
        layer_prefix=layer_prefix,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )

    layer_dir = out_dir / f"layer_{layer:02d}"
    weight_dir = layer_dir / "weights"
    weight_dir.mkdir(parents=True, exist_ok=True)
    (layer_dir / "model.mil").write_text(
        mil_text(dim=dim, hidden=hidden, lane=lane),
        encoding="utf-8",
    )
    (weight_dir / "w1.bin").write_bytes(blob_fp16(w1))
    (weight_dir / "w3.bin").write_bytes(blob_fp16(w3))
    (weight_dir / "w2.bin").write_bytes(blob_fp16(w2))

    sample = deterministic_input(dim=dim, lane=lane, layer=layer)
    expected = compute_expected(sample, w1=w1, w3=w3, w2=w2)
    input_path = layer_dir / "input_f32.bin"
    expected_path = layer_dir / "expected_f32.bin"
    input_path.write_bytes(sample.astype(np.float32, copy=False).tobytes())
    expected_path.write_bytes(expected.astype(np.float32, copy=False).tobytes())

    return {
        "layer": layer,
        "name": f"qwen35-0.8b-ffn-layer-{layer:02d}-l{lane}",
        "dir": str(layer_dir),
        "model_mil": str(layer_dir / "model.mil"),
        "weights": [
            {
                "path": "@model_path/weights/w1.bin",
                "file": str(weight_dir / "w1.bin"),
                "offset": 64,
            },
            {
                "path": "@model_path/weights/w3.bin",
                "file": str(weight_dir / "w3.bin"),
                "offset": 64,
            },
            {
                "path": "@model_path/weights/w2.bin",
                "file": str(weight_dir / "w2.bin"),
                "offset": 64,
            },
        ],
        "input_f32": str(input_path),
        "expected_f32": str(expected_path),
        "input_bytes": dim * lane * 4,
        "output_bytes": dim * lane * 4,
        "dim": dim,
        "hidden": hidden,
        "lane": lane,
    }


def emit_all_ffn(
    out_dir: Path,
    *,
    bundle_name: str = "all_ffn",
    text_prefix: str,
    tensors: dict[str, mx.array],
    config: dict[str, object],
    layers: list[int],
    lane: int,
) -> dict[str, object]:
    text_config = dict(config["text_config"])
    quant = dict(config["quantization"])
    dim = int(text_config["hidden_size"])
    hidden = int(text_config["intermediate_size"])
    group_size = int(quant["group_size"])
    bits = int(quant["bits"])
    mode = str(quant["mode"])

    bundle_dir = out_dir / bundle_name
    weight_dir = bundle_dir / "weights"
    weight_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "model.mil").write_text(
        all_ffn_mil(dim=dim, hidden=hidden, lane=lane, layers=layers),
        encoding="utf-8",
    )

    weight_specs = []
    for layer in layers:
        layer_prefix = f"{text_prefix}layers.{layer}.mlp"
        w1, w3, w2 = load_layer_weights(
            tensors,
            layer_prefix=layer_prefix,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        for suffix, weight in (("w1", w1), ("w3", w3), ("w2", w2)):
            filename = f"l{layer:02d}_{suffix}.bin"
            file_path = weight_dir / filename
            file_path.write_bytes(blob_fp16(weight))
            weight_specs.append(
                {
                    "layer": layer,
                    "name": suffix,
                    "path": f"@model_path/weights/{filename}",
                    "file": str(file_path),
                    "offset": 64,
                }
            )

    return {
        "name": f"qwen35-0.8b-{bundle_name}-l{lane}",
        "dir": str(bundle_dir),
        "model_mil": str(bundle_dir / "model.mil"),
        "layers": layers,
        "weights": weight_specs,
        "input_byte_sizes": [dim * lane * 4 for _ in layers],
        "output_byte_sizes": [dim * lane * 4 for _ in layers],
        "dim": dim,
        "hidden": hidden,
        "lane": lane,
    }


def emit_shards(
    out_dir: Path,
    *,
    text_prefix: str,
    tensors: dict[str, mx.array],
    config: dict[str, object],
    layers: list[int],
    lane: int,
    shard_size: int,
) -> list[dict[str, object]]:
    shards = []
    for index, group in enumerate(chunk_layers(layers, shard_size)):
        shards.append(
            emit_all_ffn(
                out_dir,
                bundle_name=f"shard_{index:02d}",
                text_prefix=text_prefix,
                tensors=tensors,
                config=config,
                layers=group,
                lane=lane,
            )
        )
    return shards


def emit_grouped_shards(
    out_dir: Path,
    *,
    text_prefix: str,
    tensors: dict[str, mx.array],
    config: dict[str, object],
    layers: list[int],
    lane: int,
    shard_size: int,
) -> list[dict[str, object]]:
    text_config = dict(config["text_config"])
    quant = dict(config["quantization"])
    dim = int(text_config["hidden_size"])
    hidden = int(text_config["intermediate_size"])
    group_size = int(quant["group_size"])
    bits = int(quant["bits"])
    mode = str(quant["mode"])
    shards = []

    for index, group_layers in enumerate(chunk_layers(layers, shard_size)):
        bundle_dir = out_dir / f"grouped_shard_{index:02d}"
        weight_dir = bundle_dir / "weights"
        weight_dir.mkdir(parents=True, exist_ok=True)
        groups = len(group_layers)

        w1_parts = []
        w3_parts = []
        w2_parts = []
        for layer in group_layers:
            layer_prefix = f"{text_prefix}layers.{layer}.mlp"
            w1, w3, w2 = load_layer_weights(
                tensors,
                layer_prefix=layer_prefix,
                group_size=group_size,
                bits=bits,
                mode=mode,
            )
            w1_parts.append(w1)
            w3_parts.append(w3)
            w2_parts.append(w2)

        w1_all = np.concatenate(w1_parts, axis=0)
        w3_all = np.concatenate(w3_parts, axis=0)
        w2_all = np.concatenate(w2_parts, axis=0)

        (bundle_dir / "model.mil").write_text(
            grouped_ffn_mil(dim=dim, hidden=hidden, lane=lane, groups=groups),
            encoding="utf-8",
        )
        (weight_dir / "w1.bin").write_bytes(blob_fp16(w1_all))
        (weight_dir / "w3.bin").write_bytes(blob_fp16(w3_all))
        (weight_dir / "w2.bin").write_bytes(blob_fp16(w2_all))

        sample_parts = [deterministic_input(dim=dim, lane=lane, layer=layer) for layer in group_layers]
        expected_parts = [
            compute_expected(sample, w1=w1, w3=w3, w2=w2)
            for sample, w1, w3, w2 in zip(sample_parts, w1_parts, w3_parts, w2_parts)
        ]
        sample = np.concatenate(sample_parts, axis=0)
        expected = np.concatenate(expected_parts, axis=0)
        input_path = bundle_dir / "input_f32.bin"
        expected_path = bundle_dir / "expected_f32.bin"
        input_path.write_bytes(sample.astype(np.float32, copy=False).tobytes())
        expected_path.write_bytes(expected.astype(np.float32, copy=False).tobytes())

        shards.append(
            {
                "name": f"qwen35-0.8b-grouped-shard-{index:02d}-l{lane}",
                "dir": str(bundle_dir),
                "model_mil": str(bundle_dir / "model.mil"),
                "layers": group_layers,
                "weights": [
                    {
                        "path": "@model_path/weights/w1.bin",
                        "file": str(weight_dir / "w1.bin"),
                        "offset": 64,
                    },
                    {
                        "path": "@model_path/weights/w3.bin",
                        "file": str(weight_dir / "w3.bin"),
                        "offset": 64,
                    },
                    {
                        "path": "@model_path/weights/w2.bin",
                        "file": str(weight_dir / "w2.bin"),
                        "offset": 64,
                    },
                ],
                "group_count": groups,
                "input_byte_sizes": [groups * dim * lane * 4],
                "output_byte_sizes": [groups * dim * lane * 4],
                "input_f32": str(input_path),
                "expected_f32": str(expected_path),
                "dim": dim,
                "hidden": hidden,
                "lane": lane,
            }
        )
    return shards


def emit_attn_post_layers(
    out_dir: Path,
    *,
    text_prefix: str,
    tensors: dict[str, mx.array],
    config: dict[str, object],
    layers: list[int],
    lane: int,
) -> list[dict[str, object]]:
    text_config = dict(config["text_config"])
    quant = dict(config["quantization"])
    hidden_size = int(text_config["hidden_size"])
    head_dim = int(text_config.get("head_dim") or hidden_size // int(text_config["num_attention_heads"]))
    num_attention_heads = int(text_config["num_attention_heads"])
    input_dim = num_attention_heads * head_dim
    output_dim = hidden_size
    group_size = int(quant["group_size"])
    bits = int(quant["bits"])
    mode = str(quant["mode"])
    full_attention_interval = int(text_config.get("full_attention_interval") or 4)

    specs = []
    for layer in layers:
        if (layer + 1) % full_attention_interval != 0:
            continue
        weight = dequantize_linear(
            tensors,
            f"{text_prefix}layers.{layer}.self_attn.o_proj",
            group_size=group_size,
            bits=bits,
            mode=mode,
        ).reshape(output_dim, input_dim, 1, 1)
        layer_dir = out_dir / f"attn_post_{layer:02d}"
        weight_dir = layer_dir / "weights"
        weight_dir.mkdir(parents=True, exist_ok=True)
        (layer_dir / "model.mil").write_text(
            attn_post_mil(input_dim=input_dim, output_dim=output_dim, lane=lane),
            encoding="utf-8",
        )
        (weight_dir / "w.bin").write_bytes(blob_fp16(weight))

        ctx = deterministic_input(dim=input_dim, lane=lane, layer=layer)
        gate = deterministic_input(dim=input_dim, lane=lane, layer=layer + 101)
        expected = compute_attn_post_expected(ctx, gate, weight=weight)
        ctx_path = layer_dir / "ctx_f32.bin"
        gate_path = layer_dir / "gate_f32.bin"
        expected_path = layer_dir / "expected_f32.bin"
        ctx_path.write_bytes(ctx.astype(np.float32, copy=False).tobytes())
        gate_path.write_bytes(gate.astype(np.float32, copy=False).tobytes())
        expected_path.write_bytes(expected.astype(np.float32, copy=False).tobytes())

        specs.append(
            {
                "layer": layer,
                "name": f"qwen35-0.8b-attn-post-layer-{layer:02d}-l{lane}",
                "dir": str(layer_dir),
                "model_mil": str(layer_dir / "model.mil"),
                "weights": [
                    {
                        "path": "@model_path/weights/w.bin",
                        "file": str(weight_dir / "w.bin"),
                        "offset": 64,
                    }
                ],
                "ctx_f32": str(ctx_path),
                "gate_f32": str(gate_path),
                "expected_f32": str(expected_path),
                "input_byte_sizes": [input_dim * lane * 4, input_dim * lane * 4],
                "output_byte_sizes": [output_dim * lane * 4],
                "input_dim": input_dim,
                "output_dim": output_dim,
                "lane": lane,
            }
        )
    return specs


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = Path(args.snapshot_dir)
    config, tensors = load_snapshot(snapshot_dir)
    text_config = dict(config["text_config"])
    text_prefix = detect_text_prefix(list(tensors.keys()))
    layers = parse_layers(args.layers, int(text_config["num_hidden_layers"]))

    metadata = {
        "runtime": "qwen35_private_ffn",
        "model_id": MODEL_ID,
        "snapshot_dir": str(snapshot_dir),
        "text_prefix": text_prefix,
        "lane": int(args.lane),
        "dim": int(text_config["hidden_size"]),
        "hidden": int(text_config["intermediate_size"]),
        "num_hidden_layers": int(text_config["num_hidden_layers"]),
        "layers": [
            emit_layer(
                out_dir,
                text_prefix=text_prefix,
                tensors=tensors,
                config=config,
                layer=layer,
                lane=int(args.lane),
            )
            for layer in layers
        ],
        "all_ffn": emit_all_ffn(
            out_dir,
            text_prefix=text_prefix,
            tensors=tensors,
            config=config,
            layers=layers,
            lane=int(args.lane),
        ),
        "shards": emit_shards(
            out_dir,
            text_prefix=text_prefix,
            tensors=tensors,
            config=config,
            layers=layers,
            lane=int(args.lane),
            shard_size=int(args.shard_size),
        ),
        "grouped_shards": emit_grouped_shards(
            out_dir,
            text_prefix=text_prefix,
            tensors=tensors,
            config=config,
            layers=layers,
            lane=int(args.lane),
            shard_size=int(args.grouped_shard_size),
        ),
        "attn_post_layers": emit_attn_post_layers(
            out_dir,
            text_prefix=text_prefix,
            tensors=tensors,
            config=config,
            layers=layers,
            lane=int(args.lane),
        ),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
