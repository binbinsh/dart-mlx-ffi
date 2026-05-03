from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock


RUNTIME_DIR = Path(__file__).resolve().parent
RUNNERS_DIR = RUNTIME_DIR / "runners"
sys.path.insert(0, str(RUNNERS_DIR))

import litert_runner  # noqa: E402
import ort_runner  # noqa: E402


class RuntimeRunnersEnvTest(unittest.TestCase):
    def test_ort_runner_uses_prepared_environment(self) -> None:
        argv = [
            "ort_runner.py",
            "--model-id",
            "qwen3_5",
            "--artifact",
            "model.onnx",
            "--platform",
            "linux",
            "--input-json",
            "input.json",
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch(
                "ort_runner.prepare_runtime_environment",
                return_value=(
                    {
                        "TEST_RUNTIME_ENV": "1",
                        "DART_MLX_ENABLE_ORT": "1",
                    },
                    {"ort_env": {"ready": True}},
                ),
            ) as prepare,
            mock.patch(
                "ort_runner.write_runtime_env_file",
                return_value=Path("/tmp/runtime_env.json"),
            ) as write_env_file,
            mock.patch("ort_runner.clear_runtime_env_file") as clear_env_file,
            mock.patch("ort_runner.subprocess.run") as run,
        ):
            ort_runner.main()

        self.assertEqual(run.call_args.kwargs["env"]["TEST_RUNTIME_ENV"], "1")
        self.assertEqual(
            run.call_args.kwargs["env"]["DART_MLX_RUNTIME_ENV_FILE"],
            "/tmp/runtime_env.json",
        )
        write_env_file.assert_called_once_with({"DART_MLX_ENABLE_ORT": "1"})
        clear_env_file.assert_called_once_with(Path("/tmp/runtime_env.json"))
        prepare.assert_called_once()

    def test_litert_runner_uses_prepared_environment(self) -> None:
        argv = [
            "litert_runner.py",
            "--model-id",
            "silero_vad",
            "--artifact",
            "model.tflite",
            "--platform",
            "android",
            "--input-json",
            "input.json",
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch(
                "litert_runner.prepare_runtime_environment",
                return_value=(
                    {
                        "TEST_RUNTIME_ENV": "1",
                        "DART_MLX_LITERT_LIBRARY": "/tmp/libtensorflowlite_c.so",
                    },
                    {"litert_env": {"ready": True}},
                ),
            ) as prepare,
            mock.patch(
                "litert_runner.write_runtime_env_file",
                return_value=Path("/tmp/runtime_env.json"),
            ) as write_env_file,
            mock.patch("litert_runner.clear_runtime_env_file") as clear_env_file,
            mock.patch("litert_runner.subprocess.run") as run,
        ):
            litert_runner.main()

        self.assertEqual(run.call_args.kwargs["env"]["TEST_RUNTIME_ENV"], "1")
        self.assertEqual(
            run.call_args.kwargs["env"]["DART_MLX_RUNTIME_ENV_FILE"],
            "/tmp/runtime_env.json",
        )
        write_env_file.assert_called_once_with(
            {"DART_MLX_LITERT_LIBRARY": "/tmp/libtensorflowlite_c.so"}
        )
        clear_env_file.assert_called_once_with(Path("/tmp/runtime_env.json"))
        prepare.assert_called_once()


if __name__ == "__main__":
    unittest.main()
