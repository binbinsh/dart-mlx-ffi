from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


RUNTIME_DIR = Path(__file__).resolve().parent
CONVERTERS_DIR = RUNTIME_DIR / "converters"
sys.path.insert(0, str(CONVERTERS_DIR))

from onnx_pipeline_to_litert import _catalog_artifact, _litert_pipeline_spec


class OnnxPipelineToLiteRtConverterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dmf_onnx_pipeline_litert_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_catalog_artifact_reads_component_pipeline(self) -> None:
        catalog = self.tmp / "hf_artifacts.yaml"
        catalog.write_text(
            yaml.safe_dump(
                {
                    "models": {
                        "demo": {
                            "artifacts": {
                                "onnx": {
                                    "repo": "acme/demo",
                                    "component_artifacts": {
                                        "embed": "onnx/embed.onnx",
                                    },
                                    "pipeline": {
                                        "format": "dart_mlx_ffi.onnx_pipeline.v1",
                                        "stages": [],
                                    },
                                }
                            }
                        }
                    }
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        artifact = _catalog_artifact(
            catalog_path=catalog,
            model_id="demo",
            source_engine="onnx",
        )

        self.assertEqual(artifact["repo"], "acme/demo")
        self.assertIn("embed", artifact["component_artifacts"])

    def test_litert_pipeline_expands_components_and_signature_mappings(self) -> None:
        component_dir = self.tmp / "components" / "decoder"
        component_dir.mkdir(parents=True)
        tflite = component_dir / "model.tflite"
        tflite.write_bytes(b"TFL3")
        pipeline = {
            "format": "dart_mlx_ffi.onnx_pipeline.v1",
            "stages": [
                {
                    "name": "decoder",
                    "model": "{component:decoder}",
                    "inputs": {"attention_mask": "mask"},
                }
            ],
            "outputs": ["logits"],
        }
        converted = {
            "decoder": {
                "tflite": tflite,
                "inputs": ["input_ids", "attention_mask"],
                "outputs": ["StatefulPartitionedCall:0"],
            }
        }

        spec = _litert_pipeline_spec(
            pipeline=pipeline,
            converted=converted,
            output_dir=self.tmp,
        )

        self.assertEqual(spec["format"], "dart_mlx_ffi.litert_pipeline.v1")
        stage = spec["stages"][0]
        self.assertEqual(stage["model"], "components/decoder/model.tflite")
        self.assertEqual(stage["inputs"]["input_ids"], "input_ids")
        self.assertEqual(stage["inputs"]["attention_mask"], "mask")
        self.assertEqual(stage["outputs"]["StatefulPartitionedCall:0"], "logits")


if __name__ == "__main__":
    unittest.main()
