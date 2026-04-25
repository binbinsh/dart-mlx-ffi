from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


RUNTIME_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNTIME_DIR))

import litert_env


class LiteRtEnvTest(unittest.TestCase):
    def test_normalize_android_abi(self) -> None:
        self.assertEqual(litert_env._normalize_android_abi("arm64"), "arm64-v8a")
        self.assertEqual(litert_env._normalize_android_abi("x64"), "x86_64")
        self.assertEqual(litert_env._normalize_android_abi("armeabi-v7a"), "armeabi-v7a")

    def test_resolve_version_android_defaults_to_litert_android_track(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            version = litert_env._resolve_version(
                package_version="2.16.1",
                target_os="android",
            )
        self.assertEqual(version, litert_env.DEFAULT_ANDROID_LITERT_VERSION)

    def test_resolve_version_android_respects_explicit_override(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"DART_MLX_LITERT_ANDROID_VERSION": "2.16.1"},
            clear=True,
        ):
            version = litert_env._resolve_version(
                package_version="unknown",
                target_os="android",
            )
        self.assertEqual(version, "2.16.1")

    def test_resolve_android_prefers_explicit_env_library(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            library = Path(tmp) / "libtensorflowlite_jni.so"
            library.write_bytes(b"dmf")
            with (
                mock.patch.dict(
                    os.environ,
                    {"DART_MLX_LITERT_LIBRARY": str(library)},
                    clear=False,
                ),
                mock.patch("litert_env._litert_package", return_value=(None, "unknown")),
            ):
                env = litert_env.resolve_litert_environment(
                    target_os="android",
                    target_arch="arm64-v8a",
                )
        self.assertTrue(env.ready)
        self.assertEqual(env.library, library.resolve())
        values = env.to_env()
        self.assertEqual(values["DART_MLX_LITERT_LIBRARY"], str(library.resolve()))
        self.assertEqual(values["DART_MLX_TFLITE_LIBRARY"], str(library.resolve()))

    def test_resolve_android_fetches_when_cache_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            extracted = root / "android" / "arm64-v8a" / "libtensorflowlite_jni.so"
            extracted.parent.mkdir(parents=True, exist_ok=True)
            extracted.write_bytes(b"dmf")
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch.object(litert_env, "TOOLS_DIR", root),
                mock.patch("litert_env._litert_package", return_value=(None, "unknown")),
                mock.patch("litert_env._download_android_aar"),
                mock.patch(
                    "litert_env._extract_android_library",
                    return_value=extracted,
                ) as extract_mock,
            ):
                env = litert_env.resolve_litert_environment(
                    target_os="android",
                    target_arch="arm64-v8a",
                    fetch_library=True,
                )
        self.assertTrue(env.ready)
        self.assertEqual(env.library, extracted.resolve())
        extract_mock.assert_called_once()

    def test_resolve_android_fetch_ignores_missing_optional_select_ops(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            extracted = root / "android" / "arm64-v8a" / "libtensorflowlite_jni.so"
            extracted.parent.mkdir(parents=True, exist_ok=True)
            extracted.write_bytes(b"dmf")
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch.object(litert_env, "TOOLS_DIR", root),
                mock.patch("litert_env._litert_package", return_value=(None, "unknown")),
                mock.patch("litert_env._download_android_aar"),
                mock.patch(
                    "litert_env._extract_android_library",
                    return_value=extracted,
                ),
                mock.patch(
                    "litert_env._download_android_select_ops_aar",
                    side_effect=RuntimeError("not found"),
                ),
            ):
                env = litert_env.resolve_litert_environment(
                    target_os="android",
                    target_arch="arm64-v8a",
                    fetch_library=True,
                )
        self.assertTrue(env.ready)
        self.assertEqual(env.extra_libraries, ())


if __name__ == "__main__":
    unittest.main()
