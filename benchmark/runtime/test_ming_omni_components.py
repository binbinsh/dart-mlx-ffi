from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


RUNTIME_DIR = Path(__file__).resolve().parent
CONVERTERS_DIR = RUNTIME_DIR / "converters"
sys.path.insert(0, str(CONVERTERS_DIR))

from ming_omni_components import (  # noqa: E402
    copy_official_source,
    extract_llm_hf,
    split_component_weights,
)
from ming_omni_native_components import (  # noqa: E402
    export_stop_head_onnx,
    read_ming_tts_shapes,
)


class MingOmniComponentsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dmf_ming_components_test_"))
        self.source = self.tmp / "source"
        self.source.mkdir()
        (self.source / "config.json").write_text(
            json.dumps(
                {
                    "transformers_version": "4.52.4",
                    "llm_config": {
                        "architectures": ["Qwen2ForCausalLM"],
                        "bos_token_id": 151643,
                        "eos_token_id": 151645,
                        "hidden_size": 2,
                        "model_type": "qwen2",
                        "tie_word_embeddings": True,
                        "vocab_size": 4,
                    },
                    "audio_tokenizer_config": {
                        "enc_kwargs": {
                            "latent_dim": 3,
                        },
                        "dec_kwargs": {
                            "output_dim": 8,
                            "backbone": {
                                "hidden_size": 2,
                                "num_hidden_layers": 1,
                            },
                        },
                    },
                    "ditar_config": {
                        "patch_size": 4,
                        "history_patch_size": 5,
                    },
                    "aggregator_config": {
                        "hidden_size": 8,
                        "depth": 1,
                        "num_heads": 2,
                    },
                }
            ),
            encoding="utf-8",
        )
        (self.source / "tokenizer.json").write_text("{}", encoding="utf-8")
        save_file(
            {
                "model.model.embed_tokens.weight": torch.ones(4, 2),
                "model.lm_head.weight": torch.zeros(4, 2),
                "audio.decoder.weight": torch.ones(1),
                "flowloss.cfm.weight": torch.ones(2),
                "linear_proj_audio.blocks.0.weight": torch.ones(3),
                "stop_head.weight": torch.ones(2, 2),
                "stop_head.bias": torch.zeros(2),
                "spk_head.bias": torch.ones(2),
            },
            self.source / "model.safetensors",
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_extract_llm_hf_strips_ming_model_prefix(self) -> None:
        out = self.tmp / "llm"

        report = extract_llm_hf(source_dir=self.source, output_dir=out)

        config = json.loads((out / "config.json").read_text(encoding="utf-8"))
        tensors = load_file(out / "model.safetensors")
        self.assertEqual(config["model_type"], "qwen2")
        self.assertEqual(config["architectures"], ["Qwen2ForCausalLM"])
        self.assertIn("model.embed_tokens.weight", tensors)
        self.assertIn("lm_head.weight", tensors)
        self.assertNotIn("model.model.embed_tokens.weight", tensors)
        self.assertEqual(report["tensor_count"], 2)
        self.assertEqual(report["tokenizer_files"], ["tokenizer.json"])

    def test_split_component_weights_writes_real_component_tensors(self) -> None:
        out = self.tmp / "components"

        report = split_component_weights(source_dir=self.source, output_dir=out)

        self.assertEqual(report["audio"]["tensor_count"], 1)
        self.assertEqual(report["flowloss"]["tensor_count"], 1)
        self.assertEqual(report["linear_proj_audio"]["tensor_count"], 1)
        self.assertEqual(report["stop_head"]["tensor_count"], 2)
        self.assertEqual(report["spk_head"]["tensor_count"], 1)
        manifest = json.loads(
            (out / "ming_omni_components.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            manifest["format"],
            "dart_mlx_ffi.ming_omni_tts_torch_components.v1",
        )

    def test_copy_official_source_copies_known_modules_and_dirs(self) -> None:
        official = self.tmp / "official"
        (official / "audio_tokenizer").mkdir(parents=True)
        (official / "modeling_bailingmm.py").write_text("model", encoding="utf-8")
        (official / "audio_tokenizer" / "modeling_audio_vae.py").write_text(
            "audio",
            encoding="utf-8",
        )

        copied = copy_official_source(
            official_source=official,
            target_dir=self.source,
        )

        self.assertIn("modeling_bailingmm.py", copied)
        self.assertIn("audio_tokenizer/", copied)
        self.assertTrue((self.source / "modeling_bailingmm.py").exists())
        self.assertTrue(
            (self.source / "audio_tokenizer" / "modeling_audio_vae.py").exists()
        )

    def test_read_ming_tts_shapes_uses_runtime_config(self) -> None:
        shapes = read_ming_tts_shapes(self.source / "config.json")

        self.assertEqual(shapes.hidden_size, 2)
        self.assertEqual(shapes.latent_dim, 3)
        self.assertEqual(shapes.patch_size, 4)
        self.assertEqual(shapes.history_patch_size, 5)
        self.assertEqual(shapes.audio_output_dim, 8)
        self.assertEqual(shapes.audio_decoder_config["num_hidden_layers"], 1)
        self.assertEqual(shapes.aggregator_config["hidden_size"], 8)

    def test_export_stop_head_onnx_writes_executable_component(self) -> None:
        try:
            import onnx  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("onnx is not installed")
        split_component_weights(
            source_dir=self.source,
            output_dir=self.tmp / "components",
        )

        report = export_stop_head_onnx(
            component_path=self.tmp / "components" / "stop_head.safetensors",
            output_path=self.tmp / "onnx" / "stop_head.onnx",
            hidden_size=2,
            opset=18,
        )

        self.assertTrue((self.tmp / "onnx" / "stop_head.onnx").exists())
        self.assertEqual(report["input_names"], ["z_diff"])
        self.assertEqual(report["output_shape"], [1, 1, 2])


if __name__ == "__main__":
    unittest.main()
