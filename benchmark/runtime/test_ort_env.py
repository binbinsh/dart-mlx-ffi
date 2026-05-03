from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


RUNTIME_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNTIME_DIR))

import ort_env


class OrtEnvTest(unittest.TestCase):
    def test_normalize_android_abi(self) -> None:
        self.assertEqual(ort_env._normalize_android_abi("arm64"), "arm64-v8a")
        self.assertEqual(ort_env._normalize_android_abi("x64"), "x86_64")
        self.assertEqual(ort_env._normalize_android_abi("armeabi-v7a"), "armeabi-v7a")

    def test_resolve_android_prefers_explicit_env_library(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            library = Path(tmp) / "libonnxruntime.so"
            library.write_bytes(b"dmf")
            with (
                mock.patch.dict(
                    os.environ,
                    {"DART_MLX_ORT_LIBRARY": str(library)},
                    clear=False,
                ),
                mock.patch("ort_env._onnxruntime_package", return_value=(None, "unknown")),
            ):
                env = ort_env.resolve_ort_environment(
                    target_os="android",
                    target_arch="arm64-v8a",
                )
        self.assertEqual(env.library, library.resolve())

    def test_tools_dir_fallback_when_primary_has_broken_symlink_ancestor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            broken = root / "broken"
            broken.symlink_to(root / "missing-target")
            primary = broken / "tools" / "onnxruntime"
            fallback = root / "artifacts_local" / "tools" / "onnxruntime"
            with (
                mock.patch.dict(os.environ, {}, clear=False),
                mock.patch.object(ort_env, "TOOLS_DIR", primary),
                mock.patch.object(ort_env, "TOOLS_DIR_FALLBACK", fallback),
            ):
                resolved = ort_env._tools_dir(ensure_exists=True)
                self.assertEqual(resolved, fallback)
                self.assertTrue(fallback.exists())

    def test_tools_dir_prefers_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            override = Path(tmp) / "custom-ort-tools"
            with mock.patch.dict(
                os.environ,
                {ort_env.TOOLS_DIR_ENV: str(override)},
                clear=False,
            ):
                resolved = ort_env._tools_dir(ensure_exists=True)
                self.assertEqual(resolved, override)
                self.assertTrue(override.exists())


if __name__ == "__main__":
    unittest.main()
