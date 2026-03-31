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
        )
        self.assertEqual(command[:5], ["npx", "--yes", "@mlx-node/cli", "convert", "--input"])
        self.assertIn("unsloth", command)
        self.assertIn("--imatrix-path", command)
        self.assertIn("qwen3_5", command)

    def test_verify_output_dir_requires_quantization(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            (out / "config.json").write_text("{}", encoding="utf-8")
            (out / "model.safetensors").write_bytes(b"x")
            with self.assertRaises(ValueError):
                mod.verify_output_dir(out)


if __name__ == "__main__":
    unittest.main()
