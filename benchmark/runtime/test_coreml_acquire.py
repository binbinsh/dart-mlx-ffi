from __future__ import annotations

import sys
import unittest
from pathlib import Path


RUNTIME_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNTIME_DIR))

import coreml_acquire


class CoreMlAcquireTest(unittest.TestCase):
    def test_blocks_without_source_or_gguf(self) -> None:
        plan = coreml_acquire.build_acquisition_plan(
            source_model=None,
            gguf=None,
            output_dir=None,
            tools_root=Path("/tmp/tools"),
            context_length="2048",
            quantize="int4",
            trust_remote_code=False,
            limit=1,
            tree_limit=1,
        )

        self.assertEqual(plan["recommended_action"], "provide_source_model")

    def test_prefers_existing_hf_coreml_candidate(self) -> None:
        original = coreml_acquire.search_existing_coreml
        try:
            coreml_acquire.search_existing_coreml = lambda *_args, **_kwargs: [
                {"repo": "org/model-CoreML", "paths": ["model.mlmodelc"]}
            ]

            plan = coreml_acquire.build_acquisition_plan(
                source_model="org/model",
                gguf=None,
                output_dir=None,
                tools_root=Path("/tmp/tools"),
                context_length="2048",
                quantize="int4",
                trust_remote_code=False,
                limit=1,
                tree_limit=1,
            )
        finally:
            coreml_acquire.search_existing_coreml = original

        self.assertEqual(plan["recommended_action"], "use_existing_hf_coreml")
        self.assertEqual(plan["coreml_candidates"][0]["repo"], "org/model-CoreML")

    def test_falls_back_to_coreml_llm_conversion(self) -> None:
        original = coreml_acquire.search_existing_coreml
        try:
            coreml_acquire.search_existing_coreml = lambda *_args, **_kwargs: []

            plan = coreml_acquire.build_acquisition_plan(
                source_model="org/model",
                gguf=None,
                output_dir=Path("/tmp/out"),
                tools_root=Path("/tmp/tools"),
                context_length="1024",
                quantize="int4",
                trust_remote_code=True,
                limit=1,
                tree_limit=1,
            )
        finally:
            coreml_acquire.search_existing_coreml = original

        self.assertEqual(plan["recommended_action"], "convert_coreml_llm")
        self.assertIn("/tmp/tools/coreml-llm/conversion/convert.py", plan["conversion_command"])
        self.assertIn("--trust-remote-code", plan["conversion_command"])

    def test_compacts_gguf_plan_for_acquisition_report(self) -> None:
        compact = coreml_acquire._compact_gguf_plan(
            {
                "gguf": "model.gguf",
                "source_model": "org/model",
                "architecture": "llama",
                "metadata": {"tokenizer.ggml.tokens": list(range(100))},
                "tensors": {"count": 1},
                "direct_gguf_to_coreml": False,
                "state": "planned",
            }
        )

        self.assertNotIn("metadata", compact)
        self.assertEqual(compact["source_model"], "org/model")

    def test_coreml_search_retries_rate_limit_on_fallback_endpoint(self) -> None:
        created = []

        class FakeModel:
            modelId = "org/model-coreml"

        class FakeTreeItem:
            path = "model.mlmodelc/Manifest.json"

        class FakeApi:
            def __init__(self, endpoint=None) -> None:
                self.endpoint = endpoint or "https://huggingface.co"
                created.append(self.endpoint)

            def list_models(self, *, search, limit):
                del search, limit
                if self.endpoint == "https://huggingface.co":
                    raise RuntimeError("429 Too Many Requests")
                return [FakeModel()]

            def list_repo_tree(self, repo, *, recursive):
                del repo, recursive
                return [FakeTreeItem()]

        original = coreml_acquire.HfApi
        try:
            coreml_acquire.HfApi = FakeApi
            candidates = coreml_acquire.search_existing_coreml(
                "org/model",
                limit=1,
                tree_limit=10,
                fallback_endpoint="https://hf-mirror.com",
            )
        finally:
            coreml_acquire.HfApi = original

        self.assertIn("https://hf-mirror.com", created)
        self.assertEqual(candidates[0]["repo"], "org/model-coreml")
        self.assertEqual(candidates[0]["endpoint"], "https://hf-mirror.com")


if __name__ == "__main__":
    unittest.main()
