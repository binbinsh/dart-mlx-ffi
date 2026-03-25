from __future__ import annotations

import argparse
import json
import math
import shutil
import urllib.request
import urllib.error
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types


MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
CONFIG_URL = "https://huggingface.co/Qwen/Qwen3-ASR-1.7B/raw/main/config.json"
API_URL = "https://huggingface.co/api/models/Qwen/Qwen3-ASR-1.7B"
ROOT = Path(__file__).resolve().parents[2]
LOCAL_MODEL_DIR = ROOT / "tmp" / "Qwen3-ASR-1.7B"
LOCAL_CONFIG = LOCAL_MODEL_DIR / "config.json"
TEXT_LANE = 32
AUDIO_LANE = 32


def fetch_json(url: str) -> dict[str, object]:
    if "://" not in url:
        return json.loads(Path(url).read_text(encoding="utf-8"))
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_json_with_fallback(
    primary: str,
    *,
    fallback: str | None = None,
    default: dict[str, object] | None = None,
) -> dict[str, object]:
    try:
        return fetch_json(primary)
    except (
        FileNotFoundError,
        urllib.error.URLError,
        TimeoutError,
    ):
        if fallback is not None:
            return fetch_json(fallback)
        if default is not None:
            return default
        raise


def tanh_gelu(x: np.ndarray) -> np.ndarray:
    coeff = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + np.tanh(coeff * (x + 0.044715 * np.power(x, 3))))


def copy_artifacts(compiled_dir: Path, dst_dir: Path) -> None:
    (dst_dir / "weights").mkdir(parents=True, exist_ok=True)
    shutil.copy2(compiled_dir / "model.mil", dst_dir / "model.mil")
    shutil.copy2(
        compiled_dir / "weights" / "weight.bin",
        dst_dir / "weights" / "weight.bin",
    )


def make_text_weights(spec: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(spec["seed"])
    dim = int(spec["dim"])
    hidden = int(spec["hidden"])
    scale = float(spec["weight_scale"])
    w1 = rng.normal(0.0, scale, size=(hidden, dim, 1, 1)).astype(np.float32)
    w3 = rng.normal(0.0, scale, size=(hidden, dim, 1, 1)).astype(np.float32)
    w2 = rng.normal(0.0, scale, size=(dim, hidden, 1, 1)).astype(np.float32)
    return w1, w3, w2


def make_audio_weights(spec: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(spec["seed"])
    dim = int(spec["dim"])
    hidden = int(spec["hidden"])
    scale = float(spec["weight_scale"])
    w1 = rng.normal(0.0, scale, size=(hidden, dim, 1, 1)).astype(np.float32)
    w2 = rng.normal(0.0, scale, size=(dim, hidden, 1, 1)).astype(np.float32)
    return w1, w2


def make_text_model(
    out_dir: Path,
    *,
    dim: int,
    hidden: int,
    lane: int,
    w1: np.ndarray,
    w3: np.ndarray,
    w2: np.ndarray,
) -> Path:
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, dim, 1, lane), dtype=types.fp32)])
    def prog(x):
        h1 = mb.conv(
            x=x,
            weight=w1,
            pad_type="valid",
            strides=[1, 1],
            dilations=[1, 1],
            groups=1,
            name="gate_proj",
        )
        h3 = mb.conv(
            x=x,
            weight=w3,
            pad_type="valid",
            strides=[1, 1],
            dilations=[1, 1],
            groups=1,
            name="up_proj",
        )
        sig = mb.sigmoid(x=h1, name="sigmoid")
        silu = mb.mul(x=h1, y=sig, name="silu")
        gate = mb.mul(x=silu, y=h3, name="gate")
        y = mb.conv(
            x=gate,
            weight=w2,
            pad_type="valid",
            strides=[1, 1],
            dilations=[1, 1],
            groups=1,
            name="down_proj",
        )
        return mb.add(x=x, y=y, name="residual")

    model = ct.convert(prog, convert_to="mlprogram")
    package = out_dir / f"text_ffn_{dim}_{hidden}_{lane}.mlpackage"
    model.save(str(package))
    return Path(ct.models.utils.compile_model(str(package)))


def make_audio_model(
    out_dir: Path,
    *,
    dim: int,
    hidden: int,
    lane: int,
    w1: np.ndarray,
    w2: np.ndarray,
) -> Path:
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, dim, 1, lane), dtype=types.fp32)])
    def prog(x):
        h = mb.conv(
            x=x,
            weight=w1,
            pad_type="valid",
            strides=[1, 1],
            dilations=[1, 1],
            groups=1,
            name="fc1",
        )
        g = mb.gelu(x=h, mode="TANH_APPROXIMATION", name="gelu")
        y = mb.conv(
            x=g,
            weight=w2,
            pad_type="valid",
            strides=[1, 1],
            dilations=[1, 1],
            groups=1,
            name="fc2",
        )
        return mb.add(x=x, y=y, name="residual")

    model = ct.convert(prog, convert_to="mlprogram")
    package = out_dir / f"audio_mlp_{dim}_{hidden}_{lane}.mlpackage"
    model.save(str(package))
    return Path(ct.models.utils.compile_model(str(package)))


def make_input(*, dim: int, lane: int) -> np.ndarray:
    values = np.array([(index % 97) / 97.0 for index in range(dim * lane)], dtype=np.float32)
    return values.reshape(dim, lane)


def compute_text_expected(
    x: np.ndarray,
    *,
    w1: np.ndarray,
    w3: np.ndarray,
    w2: np.ndarray,
) -> np.ndarray:
    h1 = w1[:, :, 0, 0] @ x
    h3 = w3[:, :, 0, 0] @ x
    sig = 1.0 / (1.0 + np.exp(-h1))
    gate = (h1 * sig) * h3
    y = w2[:, :, 0, 0] @ gate
    return (x + y).astype(np.float32, copy=False)


def compute_audio_expected(
    x: np.ndarray,
    *,
    w1: np.ndarray,
    w2: np.ndarray,
) -> np.ndarray:
    h = w1[:, :, 0, 0] @ x
    g = tanh_gelu(h)
    y = w2[:, :, 0, 0] @ g
    return (x + y).astype(np.float32, copy=False)


def build_specs(config: dict[str, object], sha: str | None) -> list[dict[str, object]]:
    thinker = dict(config["thinker_config"])
    audio = dict(thinker["audio_config"])
    text = dict(thinker["text_config"])
    return [
        {
            "name": "qwen3-asr-1.7b-audio-mlp-l32",
            "block_type": "audio_mlp",
            "model_id": MODEL_ID,
            "model_sha": sha,
            "dim": int(audio["d_model"]),
            "hidden": int(audio["encoder_ffn_dim"]),
            "lane": AUDIO_LANE,
            "seed": 43,
            "weight_scale": float(thinker["initializer_range"]),
            "max_ms": 0.5,
        },
        {
            "name": "qwen3-asr-1.7b-text-ffn-l32",
            "block_type": "text_ffn",
            "model_id": MODEL_ID,
            "model_sha": sha,
            "dim": int(text["hidden_size"]),
            "hidden": int(text["intermediate_size"]),
            "lane": TEXT_LANE,
            "seed": 47,
            "weight_scale": float(thinker["initializer_range"]),
            "max_ms": 1.2,
        },
    ]


def emit_block(out_dir: Path, spec: dict[str, object]) -> dict[str, object]:
    model_dir = out_dir / str(spec["name"])
    if spec["block_type"] == "audio_mlp":
        w1, w2 = make_audio_weights(spec)
        compiled = make_audio_model(
            out_dir,
            dim=int(spec["dim"]),
            hidden=int(spec["hidden"]),
            lane=int(spec["lane"]),
            w1=w1,
            w2=w2,
        )
        sample = make_input(dim=int(spec["dim"]), lane=int(spec["lane"]))
        expected = compute_audio_expected(sample, w1=w1, w2=w2)
    else:
        w1, w3, w2 = make_text_weights(spec)
        compiled = make_text_model(
            out_dir,
            dim=int(spec["dim"]),
            hidden=int(spec["hidden"]),
            lane=int(spec["lane"]),
            w1=w1,
            w3=w3,
            w2=w2,
        )
        sample = make_input(dim=int(spec["dim"]), lane=int(spec["lane"]))
        expected = compute_text_expected(sample, w1=w1, w3=w3, w2=w2)

    copy_artifacts(compiled, model_dir)
    input_path = model_dir / "input_f32.bin"
    expected_path = model_dir / "expected_f32.bin"
    input_path.write_bytes(sample.astype(np.float32, copy=False).tobytes())
    expected_path.write_bytes(expected.astype(np.float32, copy=False).tobytes())

    return {
        **spec,
        "dir": str(model_dir),
        "model_mil": str(model_dir / "model.mil"),
        "weight_bin": str(model_dir / "weights" / "weight.bin"),
        "input_f32": str(input_path),
        "expected_f32": str(expected_path),
        "weight_offset": 64,
        "input_bytes": int(spec["dim"]) * int(spec["lane"]) * 4,
        "output_bytes": int(spec["dim"]) * int(spec["lane"]) * 4,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config-url", default=CONFIG_URL)
    parser.add_argument("--api-url", default=API_URL)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = fetch_json_with_fallback(
        args.config_url,
        fallback=str(LOCAL_CONFIG) if LOCAL_CONFIG.exists() else None,
    )
    api = fetch_json_with_fallback(
        args.api_url,
        default={"sha": None},
    )
    specs = build_specs(config, api.get("sha"))
    report: dict[str, object] = {
        "runtime": "qwen3_asr_private_blocks",
        "model_id": MODEL_ID,
        "model_sha": api.get("sha"),
        "models": [emit_block(out_dir, spec) for spec in specs],
    }
    (out_dir / "metadata.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
