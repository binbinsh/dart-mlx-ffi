from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from private_ane_ctypes import PrivateAneModel
from private_linear_runtime import pack_f32, unpack_f32


def _to_numpy_f32(array) -> np.ndarray:
    cast = array.astype(mx.float32)
    mx.eval(cast)
    mx.synchronize()
    return np.asarray(cast).astype(np.float32, copy=False)


@dataclass
class PrivateFfnLayerRuntime:
    layer_index: int
    lane: int
    dim: int
    model: PrivateAneModel
    session: object

    def run(self, norm2):
        seq_len = norm2.shape[1]
        norm2_np = _to_numpy_f32(norm2)
        out_blob = self.session.run_one(
            pack_f32(norm2_np, dim=self.dim, lane=self.lane, seq_len=seq_len)
        )
        mlp_np = unpack_f32(out_blob, dim=self.dim, lane=self.lane, seq_len=seq_len)
        return mx.array(mlp_np, dtype=mx.float32)

    def close(self) -> None:
        self.session.close()
        self.model.close()


def build_ffn_runtimes(artifacts_dir: Path) -> dict[int, PrivateFfnLayerRuntime]:
    runtimes: dict[int, PrivateFfnLayerRuntime] = {}
    metadata = json.loads((artifacts_dir / "metadata.json").read_text())
    for spec in metadata["layers"]:
        layer = int(spec["layer"])
        mil_text = (Path(spec["dir"]) / "model.mil").read_text(encoding="utf-8")
        weights = [
            (
                str(weight_spec["path"]),
                Path(weight_spec["file"]).read_bytes(),
                int(weight_spec["offset"]),
            )
            for weight_spec in spec["weights"]
        ]
        model = PrivateAneModel.from_mil(mil_text, weights=weights)
        model.compile()
        model.load()
        session = model.create_session(
            input_bytes=[int(spec["input_bytes"])],
            output_bytes=[int(spec["output_bytes"])],
        )
        runtimes[layer] = PrivateFfnLayerRuntime(
            layer_index=layer,
            lane=int(spec["lane"]),
            dim=int(spec["dim"]),
            model=model,
            session=session,
        )
    return runtimes


def close_ffn_runtimes(runtimes: dict[int, PrivateFfnLayerRuntime]) -> None:
    for runtime in runtimes.values():
        runtime.close()
