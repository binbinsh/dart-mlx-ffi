from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path


RUNTIME_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNTIME_DIR))

import latest_model_audit  # noqa: E402


@dataclass
class FakeInfo:
    modelId: str
    sha: str = "sha"
    downloads: int = 0
    tags: list[str] | None = None


@dataclass
class FakeTreeItem:
    path: str


class FakeApi:
    def __init__(self) -> None:
        self.infos = {
            "google/gemma-4-E4B-it": FakeInfo(
                "google/gemma-4-E4B-it",
                sha="source-sha",
                tags=["gemma4"],
            ),
            "mlx-community/gemma-4-e4b-it-4bit": FakeInfo(
                "mlx-community/gemma-4-e4b-it-4bit",
                sha="mlx-sha",
                downloads=20,
                tags=["base_model:google/gemma-4-E4B-it"],
            ),
            "mlboydaisuke/gemma-4-E4B-coreml": FakeInfo(
                "mlboydaisuke/gemma-4-E4B-coreml",
                sha="coreml-sha",
                downloads=10,
                tags=["base_model:google/gemma-4-E4B-it"],
            ),
        }
        self.trees = {
            "google/gemma-4-E4B-it": ["config.json"],
            "mlx-community/gemma-4-e4b-it-4bit": [
                "config.json",
                "model.safetensors",
            ],
            "mlboydaisuke/gemma-4-E4B-coreml": [
                "chunk1.mlmodelc/model.mil",
                "model_config.json",
            ],
        }

    def model_info(self, repo_id: str, **_kwargs: object) -> FakeInfo:
        return self.infos[repo_id]

    def list_models(self, **kwargs: object) -> list[FakeInfo]:
        query = str(kwargs["search"]).lower()
        if "coreml" in query:
            return [self.infos["mlboydaisuke/gemma-4-E4B-coreml"]]
        if "mlx" in query or "4bit" in query:
            return [self.infos["mlx-community/gemma-4-e4b-it-4bit"]]
        return []

    def list_repo_tree(self, repo_id: str, **_kwargs: object) -> list[FakeTreeItem]:
        return [FakeTreeItem(path) for path in self.trees[repo_id]]


class LatestModelAuditTest(unittest.TestCase):
    def test_artifact_exists_accepts_directory_artifacts(self) -> None:
        paths = ["chunk1.mlmodelc/model.mil", "model_config.json"]

        self.assertTrue(latest_model_audit._artifact_exists(paths, "."))
        self.assertTrue(latest_model_audit._artifact_exists(paths, "chunk1.mlmodelc"))
        self.assertFalse(latest_model_audit._artifact_exists(paths, "missing.mlmodelc"))

    def test_runtime_roots_classifies_artifact_types(self) -> None:
        roots = latest_model_audit._runtime_roots(
            [
                "model.safetensors",
                "chunk1.mlmodelc/model.mil",
                "onnx/model_q4f16.onnx",
                "model.litertlm",
            ]
        )

        self.assertEqual(roots["mlx"], ["model.safetensors"])
        self.assertEqual(roots["coreml"], ["chunk1.mlmodelc"])
        self.assertEqual(roots["onnx"], ["onnx/model_q4f16.onnx"])
        self.assertEqual(roots["litert"], ["model.litertlm"])

    def test_build_report_records_configured_artifacts_and_candidates(self) -> None:
        catalog = {
            "models": {
                "gemma4": {
                    "family": "Gemma 4",
                    "source_model": "google/gemma-4-E4B-it",
                    "artifacts": {
                        "mlx": {
                            "repo": "mlx-community/gemma-4-e4b-it-4bit",
                            "artifact": ".",
                        }
                    },
                }
            }
        }

        report = latest_model_audit.build_report(
            catalog,
            catalog_path=Path("catalog.yaml"),
            api=FakeApi(),
            fallback_api=None,
            model_filter=set(),
            limit=5,
            max_candidates=8,
            tree_limit=100,
        )

        model = report["models"][0]
        self.assertEqual(model["id"], "gemma4")
        self.assertTrue(model["source"]["exists"])
        self.assertTrue(model["configured_artifacts"]["mlx"]["artifact_found"])
        self.assertEqual(
            model["configured_artifacts"]["mlx"]["base_models"],
            ["google/gemma-4-E4B-it"],
        )
        self.assertTrue(
            model["configured_artifacts"]["mlx"]["base_model_matches_source"]
        )
        self.assertEqual(
            model["latest_candidates"]["coreml"][0]["repo"],
            "mlboydaisuke/gemma-4-E4B-coreml",
        )

    def test_base_models_normalizes_quantized_and_finetune_tags(self) -> None:
        self.assertEqual(
            latest_model_audit._base_models(
                [
                    "base_model:Qwen/Qwen3-ASR-1.7B",
                    "base_model:quantized:Qwen/Qwen3-ASR-1.7B",
                    "base_model:finetune:Qwen/Qwen3-ASR-1.7B",
                ]
            ),
            ["Qwen/Qwen3-ASR-1.7B"],
        )


if __name__ == "__main__":
    unittest.main()
