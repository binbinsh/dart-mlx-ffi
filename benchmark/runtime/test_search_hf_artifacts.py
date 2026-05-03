from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

RUNTIME_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNTIME_DIR))

import search_hf_artifacts


class SearchHfArtifactsOutPathTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dmf_search_hf_out_"))
        self.runtime_dir = self.tmp / "benchmark" / "runtime"
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_fallback_out_path_mirrors_out_relative_path(self) -> None:
        requested = self.tmp / "benchmark" / "out" / "runtime" / "report.json"
        with mock.patch.object(search_hf_artifacts, "RUNTIME_DIR", self.runtime_dir):
            fallback = search_hf_artifacts._fallback_out_path(requested)
        self.assertEqual(
            fallback,
            self.tmp / "benchmark" / "out_local" / "runtime" / "report.json",
        )

    def test_prepare_out_path_falls_back_when_out_parent_is_not_directory(self) -> None:
        out_root = self.tmp / "benchmark" / "out"
        out_root.parent.mkdir(parents=True, exist_ok=True)
        out_root.write_text("broken symlink marker", encoding="utf-8")
        requested = out_root / "runtime" / "search.json"
        with mock.patch.object(search_hf_artifacts, "RUNTIME_DIR", self.runtime_dir):
            resolved = search_hf_artifacts._prepare_out_path(requested)
        self.assertEqual(
            resolved,
            self.tmp / "benchmark" / "out_local" / "runtime" / "search.json",
        )
        self.assertTrue(resolved.parent.exists())

    def test_search_splits_runtime_paths_from_component_sidecars(self) -> None:
        api = _FakeHfApi(
            models=["owner/mixed-runtime"],
            trees={
                "owner/mixed-runtime": [
                    "assets/token2wav/campplus.onnx",
                    "onnx/model_q4f16.onnx",
                ]
            },
        )

        record = search_hf_artifacts._search_model(
            api,
            model_id="mixed_model",
            model={
                "family": "Mixed Model",
                "source_model": "owner/mixed-model",
            },
            limit=10,
            tree_limit=100,
        )

        self.assertEqual(len(record["runtime_candidates"]), 1)
        self.assertEqual(len(record["component_candidates"]), 0)
        candidate = record["runtime_candidates"][0]
        self.assertEqual(candidate["paths"], ["onnx/model_q4f16.onnx"])
        self.assertEqual(
            candidate["component_paths"],
            ["assets/token2wav/campplus.onnx"],
        )

    def test_search_records_query_errors_and_keeps_partial_results(self) -> None:
        api = _FakeHfApi(
            models=["owner/runtime"],
            trees={"owner/runtime": ["onnx/model.onnx"]},
            failing_queries={"Mixed Model ONNX"},
        )

        record = search_hf_artifacts._search_model(
            api,
            model_id="mixed_model",
            model={
                "family": "Mixed Model",
                "source_model": "owner/mixed-model",
            },
            limit=10,
            tree_limit=100,
        )

        self.assertEqual(len(record["runtime_candidates"]), 1)
        self.assertTrue(record["search_errors"])
        self.assertEqual(record["search_errors"][0]["query"], "Mixed Model ONNX")

    def test_search_retries_rate_limited_queries_on_fallback_endpoint(self) -> None:
        api = _FakeHfApi(
            models=[],
            trees={},
            failing_queries={"Mixed Model"},
        )
        fallback_api = _FakeHfApi(
            models=["owner/fallback-runtime"],
            trees={"owner/fallback-runtime": ["onnx/model.onnx"]},
        )

        record = search_hf_artifacts._search_model(
            api,
            model_id="mixed_model",
            model={
                "family": "Mixed Model",
                "source_model": "owner/mixed-model",
                "search_terms": ["Mixed Model"],
            },
            limit=10,
            tree_limit=100,
            fallback_api=fallback_api,
            fallback_endpoint="https://hf-mirror.com",
        )

        self.assertEqual(record["search_errors"], [])
        self.assertEqual(
            record["search_fallbacks"],
            [{"query": "Mixed Model", "endpoint": "https://hf-mirror.com"}],
        )
        self.assertEqual(len(record["runtime_candidates"]), 1)

    def test_search_catalog_records_generation_time(self) -> None:
        api = _FakeHfApi(models=[], trees={})
        with (
            mock.patch("search_hf_artifacts._hf_api", return_value=api),
            mock.patch("search_hf_artifacts._fallback_apis", return_value=[]),
        ):
            report = search_hf_artifacts.search_catalog(
                {
                    "models": {
                        "mixed_model": {
                            "artifact_coverage": "partial",
                            "family": "Mixed Model",
                        }
                    }
                },
                catalog_path=self.tmp / "hf_artifacts.yaml",
                model_filter=set(),
                include_full=False,
                limit=10,
                tree_limit=100,
            )

        self.assertRegex(report["generated_at"], r"^\d{4}-\d{2}-\d{2}T.*Z$")


class _FakeModel:
    def __init__(self, model_id: str) -> None:
        self.modelId = model_id


class _FakeTreeItem:
    def __init__(self, path: str) -> None:
        self.path = path


class _FakeHfApi:
    def __init__(
        self,
        *,
        models: list[str],
        trees: dict[str, list[str]],
        failing_queries: set[str] | None = None,
    ) -> None:
        self.models = models
        self.trees = trees
        self.failing_queries = failing_queries or set()

    def list_models(self, *, search: str, limit: int) -> list[_FakeModel]:
        if search in self.failing_queries:
            raise RuntimeError("429 Too Many Requests: rate limited")
        return [_FakeModel(model_id) for model_id in self.models[:limit]]

    def list_repo_tree(self, repo: str, *, recursive: bool) -> list[_FakeTreeItem]:
        del recursive
        return [_FakeTreeItem(path) for path in self.trees.get(repo, [])]


if __name__ == "__main__":
    unittest.main()
