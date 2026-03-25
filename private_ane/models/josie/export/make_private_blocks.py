from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types


MODEL_ID = "mlx-community/JOSIE-1.1-4B-Instruct-4bit"
CONFIG_URL = "https://huggingface.co/mlx-community/JOSIE-1.1-4B-Instruct-4bit/raw/main/config.json"
API_URL = "https://huggingface.co/api/models/mlx-community/JOSIE-1.1-4B-Instruct-4bit"
ROOT = Path(__file__).resolve().parents[2]
LOCAL_MODEL_DIR = ROOT / "tmp" / "JOSIE-1.1-4B-Instruct-4bit"
LOCAL_CONFIG = LOCAL_MODEL_DIR / "config.json"
LANE = 32


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
    except (FileNotFoundError, urllib.error.URLError, TimeoutError):
        if fallback is not None:
            return fetch_json(fallback)
        if default is not None:
            return default
        raise


def make_weights(spec: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(spec["seed"])
    dim = int(spec["dim"])
    hidden = int(spec["hidden"])
    scale = float(spec["weight_scale"])
    w1 = rng.normal(0.0, scale, size=(hidden, dim, 1, 1)).astype(np.float32)
    w3 = rng.normal(0.0, scale, size=(hidden, dim, 1, 1)).astype(np.float32)
    w2 = rng.normal(0.0, scale, size=(dim, hidden, 1, 1)).astype(np.float32)
    return w1, w3, w2


def make_ffn_model(
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
    package = out_dir / f"josie_ffn_{dim}_{hidden}_{lane}.mlpackage"
    model.save(str(package))
    return Path(ct.models.utils.compile_model(str(package)))


def make_input(*, dim: int, lane: int) -> np.ndarray:
    values = np.array([(index % 97) / 97.0 for index in range(dim * lane)], dtype=np.float32)
    return values.reshape(dim, lane)


def compute_expected(
    x: np.ndarray,
    *,
    w1: np.ndarray,
    w3: np.ndarray,
    w2: np.ndarray,
) -> np.ndarray:
    h1 = w1[:, :, 0, 0] @ x
    h3 = w3[:, :, 0, 0] @ x
    gate = (h1 * (1.0 / (1.0 + np.exp(-h1)))) * h3
    y = w2[:, :, 0, 0] @ gate
    return (x + y).astype(np.float32, copy=False)


def copy_artifacts(compiled_dir: Path, dst_dir: Path) -> None:
    (dst_dir / "weights").mkdir(parents=True, exist_ok=True)
    (dst_dir / "model.mil").write_text(
        (compiled_dir / "model.mil").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (dst_dir / "weights" / "weight.bin").write_bytes(
        (compiled_dir / "weights" / "weight.bin").read_bytes()
    )


def build_specs(config: dict[str, object], sha: str | None) -> list[dict[str, object]]:
    return [
        {
            "name": "josie-1.1-4b-text-ffn-l32",
            "block_type": "text_ffn",
            "model_id": MODEL_ID,
            "model_sha": sha,
            "dim": int(config["hidden_size"]),
            "hidden": int(config["intermediate_size"]),
            "lane": LANE,
            "seed": 61,
            "weight_scale": 0.02,
            "max_ms": 2.0,
        }
    ]


def emit_block(out_dir: Path, spec: dict[str, object]) -> dict[str, object]:
    w1, w3, w2 = make_weights(spec)
    compiled = make_ffn_model(
        out_dir,
        dim=int(spec["dim"]),
        hidden=int(spec["hidden"]),
        lane=int(spec["lane"]),
        w1=w1,
        w3=w3,
        w2=w2,
    )
    model_dir = out_dir / str(spec["name"])
    copy_artifacts(compiled, model_dir)

    sample = make_input(dim=int(spec["dim"]), lane=int(spec["lane"]))
    expected = compute_expected(sample, w1=w1, w3=w3, w2=w2)
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
    api = fetch_json_with_fallback(args.api_url, default={"sha": None})
    specs = build_specs(config, api.get("sha"))
    report = {
        "runtime": "josie_private_blocks",
        "model_id": MODEL_ID,
        "model_sha": api.get("sha"),
        "models": [emit_block(out_dir, spec) for spec in specs],
    }
    (out_dir / "metadata.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
