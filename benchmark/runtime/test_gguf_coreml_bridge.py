from __future__ import annotations

import sys
import unittest
from pathlib import Path


RUNTIME_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNTIME_DIR))

from gguf_coreml_bridge import (
    build_plan,
    coreml_llm_command,
    _field_value,
    infer_source_model,
    summarize_metadata,
)


class GgufCoreMlBridgeTest(unittest.TestCase):
    def test_field_value_prefers_reader_contents_api(self) -> None:
        class Field:
            def contents(self):
                return "Qwen/Qwen3.5-0.8B"

        self.assertEqual(_field_value(Field()), "Qwen/Qwen3.5-0.8B")

    def test_infers_hf_source_from_url_metadata(self) -> None:
        source = infer_source_model(
            {
                "general.source.url": "https://huggingface.co/Qwen/Qwen3.5-0.8B",
                "general.architecture": "qwen3",
            }
        )

        self.assertEqual(source, "Qwen/Qwen3.5-0.8B")

    def test_prefers_base_model_repo_url_over_publisher_org(self) -> None:
        source = infer_source_model(
            {
                "general.repo_url": "https://huggingface.co/unsloth",
                "general.base_model.0.repo_url": "https://huggingface.co/google/gemma-4-E2B-it",
            }
        )

        self.assertEqual(source, "google/gemma-4-E2B-it")

    def test_infers_hf_source_from_repository_metadata(self) -> None:
        source = infer_source_model(
            {
                "general.source.huggingface.repository": "google/gemma-4-1b-it",
            }
        )

        self.assertEqual(source, "google/gemma-4-1b-it")

    def test_summarizes_large_metadata_values(self) -> None:
        summary = summarize_metadata(
            {
                "tokenizer.ggml.tokens": [str(index) for index in range(40)],
                "tokenizer.chat_template": "x" * 1200,
            }
        )

        self.assertEqual(summary["tokenizer.ggml.tokens"]["length"], 40)
        self.assertEqual(summary["tokenizer.chat_template"]["length"], 1200)

    def test_command_uses_coreml_llm_converter(self) -> None:
        command = coreml_llm_command(
            tool_dir=Path("/tmp/coreml-llm"),
            source_model="Qwen/Qwen3.5-0.8B",
            output_dir=Path("/tmp/out"),
            context_length="1024",
            quantize="int4",
            trust_remote_code=True,
        )

        self.assertIn("/tmp/coreml-llm/conversion/convert.py", command)
        self.assertIn("Qwen/Qwen3.5-0.8B", command)
        self.assertIn("--trust-remote-code", command)

    def test_plan_blocks_when_no_source_model_is_available(self) -> None:
        original = sys.modules["gguf_coreml_bridge"].read_gguf_summary
        try:
            sys.modules["gguf_coreml_bridge"].read_gguf_summary = lambda _: (
                {"general.architecture": "llama"},
                {"count": 1, "type_counts": {"Q4_K": 1}},
            )

            plan = build_plan(
                gguf=Path("/tmp/model.gguf"),
                source_model=None,
                output_dir=None,
                tools_root=Path("/tmp/tools"),
                context_length="2048",
                quantize="int4",
                trust_remote_code=False,
            )
        finally:
            sys.modules["gguf_coreml_bridge"].read_gguf_summary = original

        self.assertEqual(plan["state"], "blocked")
        self.assertFalse(plan["direct_gguf_to_coreml"])
        self.assertEqual(plan["command"], [])


if __name__ == "__main__":
    unittest.main()
