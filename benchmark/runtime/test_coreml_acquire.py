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


if __name__ == "__main__":
    unittest.main()
