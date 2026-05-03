from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

RUNTIME_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNTIME_DIR))

from runtime_backend_env import prepare_runtime_environment


class _FakeResolvedEnv:
    def __init__(
        self,
        *,
        ready: bool,
        env: dict[str, str],
        payload: dict[str, object],
    ) -> None:
        self.ready = ready
        self._env = env
        self._payload = payload

    def to_env(self) -> dict[str, str]:
        return dict(self._env)

    def to_json(self) -> dict[str, object]:
        return dict(self._payload)


class RuntimeBackendEnvTest(unittest.TestCase):
    def test_prepare_onnx_environment_merges_resolved_values(self) -> None:
        resolved = _FakeResolvedEnv(
            ready=True,
            env={"DART_MLX_ENABLE_ORT": "1", "DART_MLX_ORT_LIBRARY": "/tmp/libort.so"},
            payload={"ready": True},
        )
        with mock.patch(
            "runtime_backend_env.resolve_ort_environment",
            return_value=resolved,
        ) as resolve:
            env, metadata = prepare_runtime_environment(
                engine="onnx",
                platform="linux",
                base_env={"EXISTING": "1"},
                fetch_dependencies=False,
            )

        self.assertEqual(env["EXISTING"], "1")
        self.assertEqual(env["DART_MLX_ENABLE_ORT"], "1")
        self.assertEqual(env["DART_MLX_ORT_LIBRARY"], "/tmp/libort.so")
        self.assertEqual(metadata, {"ort_env": {"ready": True}})
        resolve.assert_called_once_with(
            fetch_headers=False,
            target_os="host",
            target_arch=None,
        )

    def test_prepare_litert_environment_skips_unready_values(self) -> None:
        resolved = _FakeResolvedEnv(
            ready=False,
            env={"DART_MLX_LITERT_LIBRARY": "/tmp/libtensorflowlite.so"},
            payload={"ready": False},
        )
        with mock.patch(
            "runtime_backend_env.resolve_litert_environment",
            return_value=resolved,
        ) as resolve:
            env, metadata = prepare_runtime_environment(
                engine="litert",
                platform="android",
                base_env={"EXISTING": "1"},
                fetch_dependencies=True,
            )

        self.assertEqual(env, {"EXISTING": "1"})
        self.assertEqual(metadata, {"litert_env": {"ready": False}})
        resolve.assert_called_once_with(
            fetch_library=True,
            target_os="host",
            target_arch=None,
        )

    def test_prepare_unknown_engine_keeps_original_environment(self) -> None:
        env, metadata = prepare_runtime_environment(
            engine="coreml",
            platform="macos",
            base_env={"EXISTING": "1"},
        )

        self.assertEqual(env, {"EXISTING": "1"})
        self.assertEqual(metadata, {})


if __name__ == "__main__":
    unittest.main()
