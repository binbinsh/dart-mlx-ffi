from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def _load_export_bundle():
    module_path = Path(__file__).with_name("export_bundle.py")
    spec = importlib.util.spec_from_file_location("export_bundle", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


export_bundle = _load_export_bundle()


class ExportBundleManifestTest(unittest.TestCase):
    def test_build_bundle_manifest_uses_superplanner_schema(self) -> None:
        manifest = export_bundle.build_bundle_manifest(
            bundle_id="qwen3_6_27b",
            name="Qwen3.6 27B",
            kind="text",
            description="Qwen3.6 local MLX bundle",
            source_model_id="mlx-community/Qwen3.6-27B-4bit",
            context_length=262144,
            entrypoint="function.mlxfn",
            sample_inputs="inputs.safetensors",
            input_names=["input_ids"],
            output_names=["logits"],
            metadata={"schema_version": 1},
        )

        self.assertEqual(
            manifest,
            {
                "id": "qwen3_6_27b",
                "name": "Qwen3.6 27B",
                "kind": "text",
                "entrypoint": "function.mlxfn",
                "sample_inputs": "inputs.safetensors",
                "input_names": ["input_ids"],
                "output_names": ["logits"],
                "description": "Qwen3.6 local MLX bundle",
                "source_model_id": "mlx-community/Qwen3.6-27B-4bit",
                "context_length": 262144,
                "metadata": {"schema_version": 1},
            },
        )

    def test_infer_context_length_prefers_text_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "vision_config": {"max_position_embeddings": 4096},
                        "text_config": {"max_position_embeddings": 262144},
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(export_bundle.infer_context_length(root), 262144)

    def test_load_extra_metadata_rejects_non_object_json(self) -> None:
        with self.assertRaises(ValueError):
            export_bundle.load_extra_metadata(
                metadata_file=None,
                metadata_json='["not", "an", "object"]',
            )


if __name__ == "__main__":
    unittest.main()
