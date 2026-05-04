from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def load_module():
    path = (
        Path(__file__).resolve().parent / "convert_unsloth_mlx.py"
    )
    spec = importlib.util.spec_from_file_location("convert_unsloth_mlx", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = load_module()


class ConvertUnslothMlxTest(unittest.TestCase):
    def test_resolve_cli_prefix_falls_back_to_npx(self):
        with mock.patch.object(mod.shutil, "which") as which:
            which.side_effect = lambda cmd: None if cmd == "mlx" else "/usr/bin/npx"
            self.assertEqual(
                mod.resolve_cli_prefix("mlx"),
                ["npx", "--yes", mod.DEFAULT_MLX_NODE_PACKAGE],
            )

    def test_resolve_input_path_uses_local_dir(self):
        with tempfile.TemporaryDirectory() as td:
            path, source = mod.resolve_input_path(
                td,
                cache_dir=None,
                revision=None,
                token=None,
            )
            self.assertEqual(path, Path(td).resolve())
            self.assertEqual(source, "local")

    def test_resolve_input_path_downloads_hub_model(self):
        with mock.patch.object(mod, "snapshot_download", return_value="/tmp/model") as download:
            path, source = mod.resolve_input_path(
                "org/model",
                cache_dir="/tmp/cache",
                revision="main",
                token="tok",
            )
            self.assertEqual(path, Path("/tmp/model").resolve())
            self.assertEqual(source, "huggingface")
            download.assert_called_once()

    def test_resolve_imatrix_path_downloads_repo_file(self):
        args = argparse.Namespace(
            imatrix_path=None,
            imatrix_repo="org/imatrix",
            imatrix_file="imatrix.gguf",
            cache_dir="/tmp/cache",
            imatrix_revision="main",
            token="tok",
        )
        with mock.patch.object(mod, "hf_hub_download", return_value="/tmp/imatrix.gguf") as download:
            path, source = mod.resolve_imatrix_path(args)
            self.assertEqual(path, Path("/tmp/imatrix.gguf").resolve())
            self.assertEqual(source, "hf:org/imatrix")
            download.assert_called_once()

    def test_build_convert_command_includes_unsloth_recipe(self):
        command = mod.build_convert_command(
            cli_prefix=["npx", "--yes", "@mlx-node/cli"],
            input_path=Path("/tmp/model"),
            output_dir=Path("/tmp/out"),
            imatrix_path=Path("/tmp/imatrix.gguf"),
            model_type="qwen3_5",
            dtype="bfloat16",
            q_bits=3,
            q_group_size=64,
            quantize=True,
        )
        self.assertEqual(command[:5], ["npx", "--yes", "@mlx-node/cli", "convert", "--input"])
        self.assertIn("unsloth", command)
        self.assertIn("--imatrix-path", command)
        self.assertIn("qwen3_5", command)

    def test_build_convert_command_supports_non_quantized(self):
        command = mod.build_convert_command(
            cli_prefix=["mlx"],
            input_path=Path("/tmp/model"),
            output_dir=Path("/tmp/out"),
            imatrix_path=None,
            model_type="qwen3_5",
            dtype="bfloat16",
            q_bits=None,
            q_group_size=None,
            quantize=False,
        )
        self.assertNotIn("--quantize", command)
        self.assertNotIn("--imatrix-path", command)
        self.assertIn("--dtype", command)

    def test_verify_output_dir_requires_quantization(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            (out / "config.json").write_text("{}", encoding="utf-8")
            (out / "model.safetensors").write_bytes(b"x")
            with self.assertRaises(ValueError):
                mod.verify_output_dir(out, quantized=True)

    def test_verify_output_dir_allows_non_quantized(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            (out / "config.json").write_text("{}", encoding="utf-8")
            (out / "model.safetensors").write_bytes(b"x")
            mod.verify_output_dir(out, quantized=False)

    def test_copy_sidecar_files(self):
        with tempfile.TemporaryDirectory() as src_td, tempfile.TemporaryDirectory() as out_td:
            src = Path(src_td)
            out = Path(out_td)
            (src / "tokenizer.json").write_text("{}", encoding="utf-8")
            (src / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (src / "chat_template.jinja").write_text("{{ prompt }}", encoding="utf-8")
            mod.copy_sidecar_files(src, out)
            self.assertTrue((out / "tokenizer.json").exists())
            self.assertTrue((out / "tokenizer_config.json").exists())
            self.assertTrue((out / "chat_template.jinja").exists())

    def test_matches_any_prefix(self):
        self.assertTrue(mod._matches_any_prefix("visual.encoder.weight", ["visual."]))
        self.assertFalse(mod._matches_any_prefix("model.embed.weight", ["visual."]))
        self.assertTrue(
            mod._matches_any_prefix("vision_tower.x", ["visual.", "vision_tower."])
        )

    def test_stage_filtered_input_drops_matching_tensors(self):
        try:
            import numpy as np
            from safetensors.numpy import save_file
            from safetensors import safe_open
        except ImportError:
            self.skipTest("safetensors/numpy not available")

        with tempfile.TemporaryDirectory() as src_td, tempfile.TemporaryDirectory() as stage_td:
            src = Path(src_td)
            tensors = {
                "model.embed.weight": np.zeros((4, 4), dtype=np.float32),
                "model.layers.0.weight": np.ones((4, 4), dtype=np.float32),
                "visual.patch_embed.weight": np.zeros((8, 8), dtype=np.float32),
                "visual.proj.weight": np.zeros((8, 8), dtype=np.float32),
            }
            save_file(tensors, str(src / "model.safetensors"))
            (src / "config.json").write_text('{"model_type":"x"}', encoding="utf-8")
            (src / "tokenizer.json").write_text("{}", encoding="utf-8")
            index = {
                "metadata": {"total_size": 999},
                "weight_map": {k: "model.safetensors" for k in tensors},
            }
            (src / "model.safetensors.index.json").write_text(
                __import__("json").dumps(index), encoding="utf-8"
            )

            staged = Path(stage_td) / "staged"
            result = mod.stage_filtered_input(
                src, staged_dir=staged, prefixes=("visual.",)
            )
            self.assertEqual(result, staged)

            with safe_open(str(staged / "model.safetensors"), framework="numpy") as f:
                kept_keys = set(f.keys())
            self.assertEqual(
                kept_keys, {"model.embed.weight", "model.layers.0.weight"}
            )

            self.assertTrue((staged / "config.json").exists())
            self.assertTrue((staged / "tokenizer.json").exists())

            new_index = __import__("json").loads(
                (staged / "model.safetensors.index.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                set(new_index["weight_map"].keys()),
                {"model.embed.weight", "model.layers.0.weight"},
            )
            self.assertNotIn("total_size", new_index.get("metadata", {}))

    def test_make_plan_without_skip_prefix_uses_input_directly(self):
        with tempfile.TemporaryDirectory() as src_td, tempfile.TemporaryDirectory() as out_td:
            args = argparse.Namespace(
                input=src_td,
                output_dir=out_td,
                imatrix_path=None,
                imatrix_repo=None,
                imatrix_file="x.gguf",
                model_type=None,
                dtype=None,
                q_bits=None,
                q_group_size=None,
                mlx_cli="mlx",
                cache_dir=None,
                revision=None,
                imatrix_revision=None,
                token=None,
                dry_run=True,
                no_quantize=True,
                skip_prefix=None,
            )
            with mock.patch.object(mod, "resolve_cli_prefix", return_value=["mlx"]):
                plan = mod.make_plan(args)
            self.assertIsNone(plan.staged_input_path)
            self.assertEqual(plan.skip_prefixes, ())
            # Command must reference the original input path verbatim.
            self.assertIn(str(Path(src_td).resolve()), plan.command)

    def test_make_plan_with_skip_prefix_routes_command_to_staged_dir(self):
        with tempfile.TemporaryDirectory() as src_td, tempfile.TemporaryDirectory() as out_td:
            args = argparse.Namespace(
                input=src_td,
                output_dir=out_td,
                imatrix_path=None,
                imatrix_repo=None,
                imatrix_file="x.gguf",
                model_type=None,
                dtype=None,
                q_bits=None,
                q_group_size=None,
                mlx_cli="mlx",
                cache_dir=None,
                revision=None,
                imatrix_revision=None,
                token=None,
                dry_run=True,
                no_quantize=True,
                skip_prefix=["visual.", "vision_tower."],
            )
            with mock.patch.object(mod, "resolve_cli_prefix", return_value=["mlx"]):
                plan = mod.make_plan(args)
            self.assertIsNotNone(plan.staged_input_path)
            self.assertEqual(plan.skip_prefixes, ("visual.", "vision_tower."))
            self.assertIn(str(plan.staged_input_path), plan.command)
            # Original input path must not appear as the converter --input value.
            input_idx = plan.command.index("--input")
            self.assertEqual(plan.command[input_idx + 1], str(plan.staged_input_path))


if __name__ == "__main__":
    unittest.main()
