from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sys


RUNTIME_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNTIME_DIR))

import ort_smoke


class _Env:
    include_dir = Path("/tmp/include")
    library = Path("/tmp/libonnxruntime.so")


class OrtSmokeTest(unittest.TestCase):
    def test_android_configure_command_includes_ndk_toolchain_and_abi(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ndk = Path(tmp) / "ndk"
            toolchain = ndk / "build" / "cmake" / "android.toolchain.cmake"
            toolchain.parent.mkdir(parents=True)
            toolchain.write_text("# toolchain\n", encoding="utf-8")

            command = ort_smoke._cmake_configure_command(
                env=_Env(),
                build_dir=Path("/tmp/build"),
                target_os="android",
                target_arch="arm64-v8a",
                android_ndk_home=ndk,
            )

        self.assertIn(f"-DCMAKE_TOOLCHAIN_FILE={toolchain}", command)
        self.assertIn("-DANDROID_ABI=arm64-v8a", command)
        self.assertIn("-DANDROID_PLATFORM=android-26", command)

    def test_host_configure_command_omits_android_toolchain(self) -> None:
        command = ort_smoke._cmake_configure_command(
            env=_Env(),
            build_dir=Path("/tmp/build"),
            target_os="host",
            target_arch=None,
            android_ndk_home=None,
        )

        self.assertFalse(any(item.startswith("-DCMAKE_TOOLCHAIN_FILE=") for item in command))
        self.assertFalse(any(item.startswith("-DANDROID_ABI=") for item in command))


if __name__ == "__main__":
    unittest.main()
